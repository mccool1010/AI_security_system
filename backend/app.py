# app.py
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import time
import signal
import threading
import os
import detector
import height_estimator
from action_recognizer import ActionRecognizer
from camera_calibrator import CameraCalibrator, list_presets as list_camera_presets
from depth_estimator import DepthEstimator
import image_quality
from datetime import datetime
from pymongo import MongoClient
from flask import request
import base64
import uuid
import numpy as np
import cv2 as cv2_local
from face_utils import get_faces_and_embeddings, cosine_similarity, compute_blur_score, compute_brightness

app = Flask(__name__)
CORS(app)

# ---------- MongoDB setup ----------
# Use a local MongoDB server. If you run `mongod` locally the default host/port is fine.
# If you're using a custom host/port, replace the URI below.
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["security_system"]
events_collection = db["events"]
users_collection = db["users"]
# create an index on timestamp to help queries (optional)
events_collection.create_index("timestamp")
# in-memory cache of user embeddings to avoid scanning DB each frame
cached_users = []
# ---- Event session tracking ----
current_event = None
event_start_time = None
last_seen_time = None
event_detections = []
# Auto-calibrated height scale: metres_per_pixel
reference_scale = {
    "meters_per_pixel": None,
    "reference_height_m": None,
    "reference_pixels": None,
}
AVG_PERSON_HEIGHT_M = 1.7  # default assumption for auto-calibration

# ── SlowFast action recognizer (Kinetics-400) ────
activity_recognizer = ActionRecognizer()

# ── Camera calibrator & depth estimator ────
# Per-camera calibrators stored in dict; default local camera uses "cam_local"
camera_calibrators: dict[str, CameraCalibrator] = {}

def get_calibrator(cam_id: str = "cam_local") -> CameraCalibrator:
    """Get or create a CameraCalibrator for a given camera ID."""
    if cam_id not in camera_calibrators:
        camera_calibrators[cam_id] = CameraCalibrator(cam_id)
    return camera_calibrators[cam_id]

# Initialize default calibrator
_default_calibrator = get_calibrator("cam_local")

# Depth estimator (shared across all cameras — single MiDaS model)
# Uses MiDaS_small by default for faster inference; change to "DPT_Large" for best accuracy
depth_est = DepthEstimator(model_type="MiDaS_small")

def load_users_cache():
    global cached_users
    cached_users = []
    try:
        for u in users_collection.find({}):
            emb = u.get("embedding")
            if emb:
                cached_users.append({
                    "user_id": u.get("user_id") or str(u.get("_id")),
                    "name": u.get("name"),
                    "embedding": emb,
                    "embeddings": u.get("embeddings", []),  # multi-embedding support
                })
        print(f"✓ Loaded {len(cached_users)} users: {[u['name'] for u in cached_users]}", flush=True)
    except Exception as e:
        print("Failed to load users cache:", e, flush=True)

# initial cache load
load_users_cache()

# ---------- Camera & detection setup ----------
from demo_video import DEMO_MODE, get_demo_camera

if DEMO_MODE:
    print("🎬 DEMO MODE — Using sample video instead of live camera", flush=True)

def get_camera():
    if DEMO_MODE:
        return get_demo_camera()
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cam

camera = get_camera()
background = None

motion_counter = 0
motion_threshold = 5
alert_cooldown = 0

# Detection filtering
CONSECUTIVE_REQUIRED = 2
CONF_THRESHOLD = 0.6
consecutive_detections = 0
# allow runtime tuning via endpoints (will update these globals)
# motion_threshold and alert_cooldown are already declared earlier

# allow pausing the camera loop so external consumers (browser) can use the device
camera_paused = False

activity_state = {
    "active": False,
    "last_event": None,
    "last_detections": [],
    "active_until": None,
    "current_session": None,
    "risk_level": "LOW",
}

SESSION_TIMEOUT = 5

# ── Screenshots ──────────────────────────────────────
SCREENSHOT_DIR = os.path.join(os.path.dirname(__file__), "static", "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


# ── Frame overlay helper ─────────────────────────────
def _estimate_height_new(det, frame_shape, frame=None, cam_id="cam_local"):
    """Estimate height using the full precision pipeline."""
    kpts = det.get("keypoints")
    box = det.get("box", [])
    if len(box) < 4:
        return None
    kpts_np = np.array(kpts) if kpts else None
    calibrator = get_calibrator(cam_id)
    # Auto-calibrate camera from frame shape if not yet calibrated
    if not calibrator.is_calibrated and frame is not None:
        calibrator.auto_calibrate_from_frame(frame.shape)
    result = height_estimator.estimate_height(
        kpts_np, box, frame_shape,
        pose_action=det.get("pose_action"),
        camera_calibrator=calibrator,
        depth_estimator=depth_est,
        quality_analyzer=image_quality,
        frame=frame,
    )
    return result


def _auto_calibrate_new(det, frame_shape):
    """Auto-calibrate height estimator on first standing person."""
    if height_estimator.is_calibrated():
        return
    if det.get("pose_action") == "standing":
        kpts = det.get("keypoints")
        box = det.get("box", [])
        if len(box) < 4:
            return
        kpts_np = np.array(kpts) if kpts else None
        height_estimator.auto_calibrate(
            kpts_np, box, frame_shape,
            known_height_m=AVG_PERSON_HEIGHT_M
        )


def draw_overlays(frame, detections, ref_scale, faces=None, risk_level=None, slowfast_actions=None):
    """
    No-op: overlays are now rendered in the frontend side panel.
    Returns the raw frame untouched for maximum FPS.
    """
    return frame

# ── Person Tracker (per-camera) ──────────────────────
# Tracks persons across frames using bounding box IoU
# Key: camera_id -> { person_idx: { first_seen, last_seen, box, action } }
person_trackers = {}

def _iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1, (boxA[2]-boxA[0]) * (boxA[3]-boxA[1]))
    areaB = max(1, (boxB[2]-boxB[0]) * (boxB[3]-boxB[1]))
    return inter / (areaA + areaB - inter)

def _update_tracker(cam_id, detections, now):
    """Update person tracker for a camera. Returns action classifications."""
    if cam_id not in person_trackers:
        person_trackers[cam_id] = {}
    tracker = person_trackers[cam_id]

    # Match current detections to tracked persons via IoU
    used = set()
    actions = []
    for det in detections:
        if det.get("class") != "person":
            continue
        box = det.get("box", [])
        if len(box) < 4:
            continue
        best_id, best_iou = None, 0.3  # minimum IoU threshold
        for pid, p in tracker.items():
            if pid in used:
                continue
            iou = _iou(box, p["box"])
            if iou > best_iou:
                best_id, best_iou = pid, iou
        if best_id is not None:
            # Update existing person
            tracker[best_id]["last_seen"] = now
            tracker[best_id]["box"] = box
            used.add(best_id)
        else:
            # New person
            best_id = str(uuid.uuid4())[:8]
            tracker[best_id] = {
                "first_seen": now, "last_seen": now, "box": box
            }
            used.add(best_id)
        # Classify action based on duration
        duration = now - tracker[best_id]["first_seen"]
        if duration < 5:
            action = "passing"
        elif duration < 15:
            action = "visitor"
        elif duration < 60:
            action = "lingering"
        else:
            action = "loitering"
        tracker[best_id]["action"] = action
        actions.append(action)

    # Expire stale persons (not seen for 5s)
    stale = [pid for pid, p in tracker.items() if now - p["last_seen"] > 5]
    for pid in stale:
        del tracker[pid]

    return actions


def _compute_risk(faces, actions):
    """Compute risk score and level from face results + actions."""
    score = 0
    # Face signals
    if faces:
        all_unknown = True
        for f in faces:
            if f.get("name") and f["name"] != "unknown" and f.get("user_id"):
                all_unknown = False  # known person = no penalty
        if all_unknown:
            score += 1  # unknown person = mild risk
    else:
        score += 1  # no face detected = mild risk
    # Action signals — only high-risk temporal behaviors
    if actions:
        worst = actions[0] if len(actions) == 1 else (
            "loitering" if "loitering" in actions else
            "lingering" if "lingering" in actions else
            "visitor" if "visitor" in actions else "passing"
        )
        if worst == "loitering":
            score += 3  # loitering 60s+ is genuinely suspicious
        elif worst == "lingering":
            score += 1  # lingering 15s is normal, slight bump
    # Risk level (more conservative thresholds)
    if score >= 4:
        level = "HIGH"
    elif score >= 2:
        level = "MEDIUM"
    else:
        level = "LOW"
    return score, level

@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": DEMO_MODE}

@app.get("/api/system/info")
def system_info():
    """System info for the landing/portfolio page."""
    return jsonify({
        "demo_mode": DEMO_MODE,
        "models": {
            "pose": "YOLOv8n-Pose (COCO 17-keypoint)",
            "activity": "SlowFast R50 (Kinetics-400)",
            "depth": "MiDaS v3.1",
            "face": "ArcFace (DeepFace)",
        },
        "features": [
            "Real-time pose estimation & action recognition",
            "Precision height measurement (±3-5cm)",
            "Monocular depth estimation",
            "Face recognition & enrollment",
            "Multi-camera support (USB/IP/RTSP)",
            "Per-person Kalman-filtered tracking",
            "Camera lens distortion correction",
            "Image quality-gated confidence scoring",
            "Risk level assessment",
        ],
    })

@app.get("/activity_status")
def activity_status():
    return jsonify(activity_state)

# ── Shared state between streaming loop and ML worker ──────────
_latest_frame = None          # raw BGR frame from camera, updated every read
_latest_frame_lock = threading.Lock()
_cached_overlay_data = {      # ML results cached for API + overlay
    "detections": [],
    "risk_level": "LOW",
    "sf_actions": [],
    "height_data": [],           # per-person height/distance/visibility
}

def _ml_worker():
    """
    Background thread: runs YOLO, motion detection, face matching, event
    generation, etc.  Reads _latest_frame, writes _cached_overlay_data.
    This frees the streaming loop to run at camera-native FPS (~30).
    """
    global camera, background, motion_counter, alert_cooldown, activity_state
    global consecutive_detections

    local_bg = None
    print("🧠 ML worker thread started", flush=True)

    while True:
        # Grab the latest raw frame
        with _latest_frame_lock:
            raw = _latest_frame
        if raw is None:
            time.sleep(0.01)
            continue

        frame = raw.copy()

        # ── Motion detection ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if local_bg is None:
            local_bg = gray.copy().astype("float")
            background = local_bg
            time.sleep(0.01)
            continue

        frame_delta = cv2.absdiff(cv2.convertScaleAbs(local_bg), gray)
        _, thresh = cv2.threshold(frame_delta, 35, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = any(cv2.contourArea(c) > 4000 for c in contours)
        motion_counter = motion_counter + 1 if motion else max(0, motion_counter - 1)
        cv2.accumulateWeighted(gray, local_bg, 0.1)
        background = local_bg

        # ── YOLO-Pose detection (~88ms — the expensive part) ──
        _, detections = detector.detect(frame, conf=CONF_THRESHOLD)

        detections = [
            {
                "class": d.get("class"),
                "confidence": float(d.get("confidence", 0.0)),
                "box": [float(v) for v in d.get("box", [])],
                "pose_action": d.get("pose_action", "unknown"),
                "height_px": d.get("height_px", 0),
                "keypoints": d.get("keypoints"),
            }
            for d in detections
            if float(d.get("confidence", 0.0)) >= CONF_THRESHOLD and d.get("class") == "person"
        ]

        for d in detections:
            _auto_calibrate_new(d, frame.shape)

        # Push to SlowFast buffer and depth estimator
        activity_recognizer.push_frame(frame)
        depth_est.push_frame(frame)

        # Get SlowFast actions (returns cached, non-blocking)
        sf_actions = activity_recognizer.get_actions()
        risk_lvl = activity_state.get("risk_level", "LOW")

        # Compute height/distance for each person (for API)
        height_data = []
        for d in detections:
            try:
                h_res = _estimate_height_new(d, frame.shape, frame=frame, cam_id="cam_local")
                if h_res:
                    height_data.append(h_res)
                else:
                    height_data.append({})
            except Exception:
                height_data.append({})

        # Update cached overlay data for streaming loop + API
        _cached_overlay_data["detections"] = detections
        _cached_overlay_data["risk_level"] = risk_lvl
        _cached_overlay_data["sf_actions"] = sf_actions
        _cached_overlay_data["height_data"] = height_data

        # ── Consecutive detection tracking ──
        if len(detections) > 0:
            consecutive_detections += 1
        else:
            consecutive_detections = 0

        now_ts = time.time()
        actions = _update_tracker("camera_0", detections, now_ts)

        # ── Event generation (only when activity threshold met) ──
        if (
            len(detections) > 0
            and motion_counter > motion_threshold
            and alert_cooldown == 0
            and consecutive_detections >= CONSECUTIVE_REQUIRED
        ):
            print("⚠️ REAL ACTIVITY DETECTED", flush=True)
            alert_cooldown = 30
            motion_counter = 0

            duration_label = (
                "loitering" if "loitering" in actions else
                "lingering" if "lingering" in actions else
                "visitor" if "visitor" in actions else "passing"
            ) if actions else "passing"

            pose_actions = [d.get("pose_action", "unknown") for d in detections if d.get("pose_action")]
            primary_pose = pose_actions[0] if pose_actions else "unknown"

            sf_label = ""
            if sf_actions and sf_actions[0].get("action") and sf_actions[0]["action"] != "Loading model...":
                sf_label = sf_actions[0]["action"].capitalize()

            parts = [p for p in [sf_label, primary_pose.capitalize(), duration_label] if p]
            worst_action = " — ".join(parts) if parts else "passing"

            screenshot_name = None
            try:
                screenshot_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_camera_0.jpg"
                screenshot_path = os.path.join(SCREENSHOT_DIR, screenshot_name)
                cv2.imwrite(screenshot_path, frame)
            except Exception as e:
                print(f"Screenshot save error: {e}", flush=True)

            iso_ts = datetime.utcnow().isoformat()
            event_doc = {
                "timestamp": iso_ts,
                "unix_ts": now_ts,
                "detections": detections,
                "source": "camera_0",
                "camera_name": "Local Camera",
                "action": worst_action,
            }
            if screenshot_name:
                event_doc["screenshot_path"] = screenshot_name

            try:
                if len(detections) > 0:
                    h_res = _estimate_height_new(detections[0], frame.shape, frame=frame)
                    if h_res and h_res.get("height_m"):
                        event_doc["height_m"] = h_res["height_m"]
                        event_doc["height_confidence"] = h_res.get("confidence", 0)
                        event_doc["height_visibility"] = h_res.get("visibility", "unknown")
                        if h_res.get("distance_m"):
                            event_doc["distance_m"] = h_res["distance_m"]
            except Exception as e:
                print("Height estimation error:", e, flush=True)

            # Face matching
            try:
                matched = []
                users = cached_users
                for idx, det in enumerate(detections):
                    if det.get("class") != "person":
                        continue
                    box = det.get("box")
                    if not box:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    h_box = max(1, y2 - y1)
                    w_box = max(1, x2 - x1)
                    pad_y = int(h_box * 0.35)
                    pad_x = int(w_box * 0.25)
                    xa = max(0, x1 - pad_x)
                    ya = max(0, y1 - pad_y)
                    xb = min(frame.shape[1], x2 + pad_x)
                    yb = min(frame.shape[0], y2 + pad_y)
                    crop = frame[ya:yb, xa:xb]
                    if crop is None or crop.size == 0:
                        continue
                    try:
                        face_list = get_faces_and_embeddings(crop)
                    except Exception:
                        face_list = []
                    if not face_list:
                        continue
                    for f in face_list:
                        emb = f.get("embedding")
                        best = {"user_id": None, "name": None, "score": 0.0}
                        for u in users:
                            multi_embs = u.get("embeddings", [])
                            stored_single = u.get("embedding")
                            best_user_score = 0.0
                            if multi_embs:
                                for se in multi_embs:
                                    s = cosine_similarity(emb, se)
                                    if s > best_user_score:
                                        best_user_score = s
                            elif stored_single:
                                best_user_score = cosine_similarity(emb, stored_single)
                            else:
                                continue
                            if best_user_score > best["score"]:
                                best = {"user_id": u.get("user_id"), "name": u.get("name"), "score": best_user_score}
                        if best["score"] >= 0.45:
                            matched.append(best)
                        else:
                            matched.append({"user_id": None, "name": "unknown", "score": best["score"]})
                if matched:
                    event_doc["faces"] = matched
                else:
                    event_doc["faces"] = [{"user_id": None, "name": "unknown", "score": 0.0}]
            except Exception as e:
                print("Face processing error:", e, flush=True)

            risk_score, risk_level = _compute_risk(
                event_doc.get("faces", []), actions
            )
            event_doc["risk_score"] = risk_score
            event_doc["risk_level"] = risk_level
            print(f"   Risk: {risk_level} (score={risk_score}), Action: {worst_action}", flush=True)

            try:
                events_collection.insert_one(event_doc)
            except Exception:
                pass

            activity_state["active"] = True
            activity_state["last_event"] = iso_ts
            activity_state["last_detections"] = detections
            activity_state["risk_level"] = risk_level
            activity_state["active_until"] = time.time() + 5

        # Housekeeping
        if activity_state.get("active_until") and time.time() > activity_state.get("active_until"):
            activity_state["active"] = False
            activity_state["active_until"] = None
        if alert_cooldown > 0:
            alert_cooldown -= 1


# Start ML worker thread once
_ml_thread = threading.Thread(target=_ml_worker, daemon=True)
_ml_thread.start()


# ── Shared display frame for MJPEG streaming ──────────────────────
# Only ONE thread (the camera reader) reads from the camera and composes
# the display frame.  All MJPEG Response generators just copy this buffer,
# preventing the "split/mirror" glitch caused by multiple concurrent readers.
_display_frame = None
_display_frame_lock = threading.Lock()


def _camera_reader():
    """
    Single background thread that reads the camera at native FPS,
    draws cached ML overlays, and stores the composed frame in
    _display_frame.  MJPEG generators read from there.
    """
    global camera, _latest_frame, _display_frame
    global camera_paused

    print("⚙️  Camera reader thread started", flush=True)

    while True:
        if camera_paused:
            try:
                if camera and camera.isOpened():
                    camera.release()
            except Exception:
                pass
            time.sleep(0.05)
            continue

        if not camera.isOpened():
            camera = get_camera()

        success, frame = camera.read()
        if not success or frame is None:
            camera.release()
            camera = get_camera()
            continue

        # Share raw frame with ML worker
        with _latest_frame_lock:
            _latest_frame = frame

        # Encode raw frame (no overlays) for maximum FPS
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with _display_frame_lock:
            _display_frame = buffer.tobytes()


# Start the single camera reader thread
_cam_reader_thread = threading.Thread(target=_camera_reader, daemon=True)
_cam_reader_thread.start()


def generate_frames():
    """
    MJPEG generator — yields the latest composed display frame.
    Multiple connections can call this safely; they all read from
    the same shared buffer (no camera contention).
    """
    while True:
        with _display_frame_lock:
            frame_bytes = _display_frame

        if frame_bytes is None:
            time.sleep(0.03)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )
        time.sleep(0.016)  # ~60 FPS cap per client

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/detections")
def get_detections():
    """
    Return real-time detection data as JSON for the frontend side panel.
    Uses SlowFast activity when confident, falls back to YOLO pose otherwise.
    """
    overlay = _cached_overlay_data
    detections = overlay.get("detections", [])
    sf_actions = overlay.get("sf_actions", [])
    height_data = overlay.get("height_data", [])
    risk_level = overlay.get("risk_level", "LOW")

    persons = []
    for i, det in enumerate(detections):
        if det.get("class") != "person":
            continue
        h_info = height_data[i] if i < len(height_data) else {}
        person = {
            "confidence": det.get("confidence", 0),
            "pose": det.get("pose_action", "unknown"),
            "height_m": h_info.get("height_m"),
            "raw_height_m": h_info.get("raw_height_m"),
            "height_confidence": h_info.get("confidence"),
            "margin_m": h_info.get("margin_m"),
            "distance_m": h_info.get("distance_m"),
            "visibility": h_info.get("visibility", "unknown"),
            "quality_score": h_info.get("quality_score"),
            "depth_method": h_info.get("depth_method"),
            "calibration": h_info.get("calibration"),
            "method": h_info.get("method"),
            "kalman_updates": h_info.get("kalman_updates", 0),
        }
        persons.append(person)

    # SlowFast now returns ONLY security-relevant actions (pre-filtered).
    # Its own MIN_CONFIDENCE already gates garbage, so we trust its output.
    ACTIVITY_CONF_THRESHOLD = 0.08
    activities = []
    for a in sf_actions:
        if (a.get("action")
            and a["action"] != "Loading model..."
            and a.get("confidence", 0) >= ACTIVITY_CONF_THRESHOLD):
            activities.append({
                "action": a["action"],
                "confidence": a.get("confidence", 0),
            })

    # Fallback: if no confident SlowFast activity, use YOLO pose as the activity
    if not activities and persons:
        pose = persons[0].get("pose", "unknown")
        if pose and pose != "unknown":
            activities.append({
                "action": pose.replace("_", " "),
                "confidence": 0.9,  # pose-based, high confidence
            })
        else:
            activities.append({"action": "idle", "confidence": 1.0})

    return jsonify({
        "persons": persons,
        "activities": activities,
        "risk_level": risk_level,
        "person_count": len(persons),
        "model_ready": activity_recognizer.is_ready,
    })


@app.post("/camera/pause")
def pause_camera():
    """Pause the internal camera loop and release the device so other apps (browser) can use it."""
    global camera_paused, camera
    try:
        camera_paused = True
        if camera and camera.isOpened():
            camera.release()
        return jsonify({"ok": True, "paused": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/camera/resume")
def resume_camera():
    """Resume internal camera loop and reacquire the device."""
    global camera_paused, camera
    try:
        camera_paused = False
        # re-open camera immediately
        try:
            if camera is None or not camera.isOpened():
                camera = get_camera()
        except Exception:
            pass
        return jsonify({"ok": True, "paused": False})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/camera/state")
def camera_state():
    return jsonify({"paused": bool(camera_paused), "camera_open": bool(camera.isOpened())})


@app.route("/enroll", methods=["POST"])
def enroll_user():
    """Enroll a user by posting form-data `name` and `image` file, or JSON with `name` and `image_base64`."""
    name = None
    img_bgr = None
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        name = request.form.get("name")
        img_file = request.files.get("image")
        if img_file:
            arr = np.frombuffer(img_file.read(), np.uint8)
            img_bgr = cv2_local.imdecode(arr, cv2_local.IMREAD_COLOR)
    else:
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        img_b64 = data.get("image_base64")
        if img_b64:
            img_bytes = base64.b64decode(img_b64.split(",")[-1])
            arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2_local.imdecode(arr, cv2_local.IMREAD_COLOR)

    if img_bgr is None or name is None:
        return jsonify({"error": "Provide 'name' and image (file or image_base64)"}), 400

    faces = get_faces_and_embeddings(img_bgr)
    if len(faces) == 0:
        return jsonify({"error": "No face found in image"}), 400

    emb = faces[0]["embedding"].tolist()
    user_id = str(uuid.uuid4())
    users_collection.insert_one({
        "user_id": user_id,
        "name": name,
        "embedding": emb,
        "created_at": datetime.utcnow().isoformat()
    })
    # update in-memory cache so matching picks this up immediately
    try:
        cached_users.append({"user_id": user_id, "name": name, "embedding": emb})
    except Exception:
        # fallback: reload entire cache
        load_users_cache()
    return jsonify({"user_id": user_id, "name": name})


@app.route("/api/register", methods=["POST"])
def register_user():
    """Register a user with several base64 images. JSON: { name: str, images_base64: [dataurl,...] }
    Returns: { user_id, name, samples, rejected, quality_scores }
    """
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    images = data.get("images_base64") or []
    if not name or not images:
        return jsonify({"error": "Provide 'name' and 'images_base64' list"}), 400

    embeddings = []
    quality_scores = []
    rejected = 0
    samples = 0

    for idx, img_b64 in enumerate(images):
        score = {"index": idx, "accepted": False, "reason": None, "blur": 0.0, "brightness": 128.0}
        try:
            if "," in img_b64:
                img_b64 = img_b64.split(",")[1]
            img_bytes = base64.b64decode(img_b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            img_bgr = cv2_local.imdecode(arr, cv2_local.IMREAD_COLOR)

            if img_bgr is None or img_bgr.size == 0:
                score["reason"] = "decode_failed"
                rejected += 1
                quality_scores.append(score)
                continue

            # Resize if too small
            h, w = img_bgr.shape[:2]
            if h < 150 or w < 150:
                scale = max(150 / h, 150 / w)
                img_bgr = cv2_local.resize(img_bgr, (int(w * scale), int(h * scale)))

            faces = get_faces_and_embeddings(img_bgr)
            if not faces:
                score["reason"] = "no_face"
                rejected += 1
                quality_scores.append(score)
                continue

            # Quality checks on the face crop
            box = faces[0].get("box", [0, 0, img_bgr.shape[1], img_bgr.shape[0]])
            x1, y1, x2, y2 = box
            crop = img_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size > 0:
                blur = compute_blur_score(crop)
                bright = compute_brightness(crop)
                score["blur"] = round(blur, 1)
                score["brightness"] = round(bright, 1)

                if blur < 20.0:
                    score["reason"] = "too_blurry"
                    rejected += 1
                    quality_scores.append(score)
                    print(f"register: image {idx} rejected (blur={blur:.1f})", flush=True)
                    continue
                if bright < 40.0 or bright > 230.0:
                    score["reason"] = "bad_lighting"
                    rejected += 1
                    quality_scores.append(score)
                    print(f"register: image {idx} rejected (brightness={bright:.1f})", flush=True)
                    continue

            emb = faces[0].get("embedding")
            if emb is not None:
                embeddings.append(np.array(emb).tolist())
                samples += 1
                score["accepted"] = True
                print(f"register: image {idx} accepted (blur={score['blur']}, bright={score['brightness']})", flush=True)
        except Exception as e:
            score["reason"] = str(e)
            rejected += 1
            print(f"register: image {idx} error: {e}", flush=True)
        quality_scores.append(score)

    if len(embeddings) == 0:
        return jsonify({"error": "No usable faces found in provided images", "quality_scores": quality_scores}), 400

    # Store individual embeddings (not averaged) for best-of-N matching
    try:
        user_id = str(uuid.uuid4())
        # Also compute average for backward compatibility
        avg = np.mean(np.stack([np.array(e) for e in embeddings], axis=0), axis=0).tolist()
        users_collection.insert_one({
            "user_id": user_id,
            "name": name,
            "embedding": avg,
            "embeddings": embeddings,  # store all individual embeddings
            "samples": samples,
            "created_at": datetime.utcnow().isoformat(),
        })
        # Update cache with all embeddings
        try:
            cached_users.append({"user_id": user_id, "name": name, "embedding": avg, "embeddings": embeddings})
        except Exception:
            load_users_cache()
        print(f"register: user {name} registered with {samples} samples, {rejected} rejected", flush=True)
        return jsonify({"user_id": user_id, "name": name, "samples": samples, "rejected": rejected, "quality_scores": quality_scores})
    except Exception as e:
        print(f"register: storage error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/set_reference", methods=["POST"])
def set_reference():
    """Set a reference object height so the system can compute meters_per_pixel.

    POST JSON: { "reference_height_m": 1.95, "reference_pixels": 400 }
    - reference_height_m: real world height of the reference object in meters
    - reference_pixels: pixel height measured in the same camera frame (pixels)
    """
    data = request.get_json(silent=True) or {}
    try:
        h_m = float(data.get("reference_height_m"))
        pixels = float(data.get("reference_pixels"))
        if pixels <= 0 or h_m <= 0:
            return jsonify({"error": "reference_height_m and reference_pixels must be positive"}), 400
        meters_per_pixel = h_m / pixels
        reference_scale["meters_per_pixel"] = meters_per_pixel
        reference_scale["reference_height_m"] = h_m
        reference_scale["reference_pixels"] = pixels
        return jsonify({"meters_per_pixel": meters_per_pixel, "reference_height_m": h_m, "reference_pixels": pixels})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/users")
def list_users():
    """Return enrolled users (id + name). Does NOT return embeddings."""
    try:
        users = [{"user_id": u.get("user_id"), "name": u.get("name"), "created_at": u.get("created_at")} for u in users_collection.find({}, {"embedding": 0}).sort("created_at", -1)]
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/events")
def get_events():
    """
    Return recent events from MongoDB (latest first).
    """
    docs = list(events_collection.find().sort("unix_ts", -1).limit(50))
    results = []
    for d in docs:
        results.append({
            "timestamp": d.get("timestamp"),
            "detections": d.get("detections", []),
            "faces": d.get("faces", []),
            "source": d.get("source"),
            "camera_name": d.get("camera_name", ""),
            "action": d.get("action", ""),
            "risk_level": d.get("risk_level", ""),
            "risk_score": d.get("risk_score"),
            "screenshot_path": d.get("screenshot_path", ""),
            "height_m": d.get("height_m"),
        })
    return jsonify(results)


@app.get("/status")
def status():
    """Return basic backend health: Flask up, Mongo ping, camera state."""
    mongo_ok = False
    mongo_msg = ""
    try:
        client.admin.command("ping")
        mongo_ok = True
    except Exception as e:
        mongo_msg = str(e)

    cam_ok = False
    try:
        cam_ok = camera.isOpened()
    except Exception as e:
        cam_ok = False
        mongo_msg = mongo_msg or str(e)

    return jsonify({
        "service": "ok",
        "mongo": {"ok": mongo_ok, "error": mongo_msg},
        "camera_open": bool(cam_ok),
    })


@app.get("/config")
def get_config():
    """Return current runtime configuration for tunables."""
    return jsonify({
        "motion_threshold": motion_threshold,
        "alert_cooldown": alert_cooldown,
        "consecutive_required": CONSECUTIVE_REQUIRED,
        "confidence_threshold": CONF_THRESHOLD,
    })


@app.post("/config")
def set_config():
    """Update runtime configuration. POST JSON with any of: motion_threshold, alert_cooldown, consecutive_required, confidence_threshold"""
    global motion_threshold, alert_cooldown, CONSECUTIVE_REQUIRED, CONF_THRESHOLD
    data = request.get_json(silent=True) or {}
    try:
        if "motion_threshold" in data:
            motion_threshold = int(data.get("motion_threshold"))
        if "alert_cooldown" in data:
            alert_cooldown = int(data.get("alert_cooldown"))
        if "consecutive_required" in data:
            CONSECUTIVE_REQUIRED = int(data.get("consecutive_required"))
        if "confidence_threshold" in data:
            CONF_THRESHOLD = float(data.get("confidence_threshold"))
        return jsonify({"ok": True, "config": {"motion_threshold": motion_threshold, "alert_cooldown": alert_cooldown, "consecutive_required": CONSECUTIVE_REQUIRED, "confidence_threshold": CONF_THRESHOLD}})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.get("/stats")
def stats():
    """Return simple stats: events last hour and today."""
    now_ts = time.time()
    one_hour = now_ts - 3600
    try:
        events_last_hour = events_collection.count_documents({"unix_ts": {"$gte": one_hour}})
    except Exception:
        events_last_hour = 0

    # start of today (UTC)
    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        events_today = events_collection.count_documents({"unix_ts": {"$gte": today_start}})
    except Exception:
        events_today = 0

    return jsonify({"events_last_hour": events_last_hour, "events_today": events_today})


# =====================================================================
#  CAMERA CALIBRATION API
# =====================================================================

@app.get("/api/calibrate/presets")
def calibrate_presets():
    """List available camera presets."""
    return jsonify(list_camera_presets())


@app.post("/api/calibrate/preset")
def calibrate_preset():
    """Apply a preset camera profile. JSON: { camera_id?, preset_name }"""
    data = request.get_json(silent=True) or {}
    cam_id = data.get("camera_id", "cam_local")
    preset_name = data.get("preset_name", "")
    if not preset_name:
        return jsonify({"error": "Provide 'preset_name'"}), 400
    try:
        calibrator = get_calibrator(cam_id)
        info = calibrator.load_preset(preset_name)
        return jsonify(info)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/calibrate/manual")
def calibrate_manual():
    """Set camera parameters manually.
    JSON: { camera_id?, frame_width, frame_height, fov_h_deg?, focal_mm?, sensor_w_mm?, dist_coeffs? }
    """
    data = request.get_json(silent=True) or {}
    cam_id = data.get("camera_id", "cam_local")
    try:
        calibrator = get_calibrator(cam_id)
        info = calibrator.set_manual_params(
            frame_width=int(data.get("frame_width", 640)),
            frame_height=int(data.get("frame_height", 480)),
            fov_h_deg=data.get("fov_h_deg"),
            focal_mm=data.get("focal_mm"),
            sensor_w_mm=data.get("sensor_w_mm"),
            dist_coeffs=data.get("dist_coeffs"),
        )
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/calibrate/checkerboard")
def calibrate_checkerboard():
    """Calibrate from checkerboard images.
    JSON: { camera_id?, images_base64: [dataurl,...], board_size?: [9,6], square_size_mm?: 25 }
    """
    data = request.get_json(silent=True) or {}
    cam_id = data.get("camera_id", "cam_local")
    images_b64 = data.get("images_base64", [])
    board_size = tuple(data.get("board_size", [9, 6]))
    square_size = float(data.get("square_size_mm", 25.0))

    if len(images_b64) < 3:
        return jsonify({"error": "Need at least 3 checkerboard images"}), 400

    # Decode images
    images = []
    for b64 in images_b64:
        try:
            if "," in b64:
                b64 = b64.split(",")[1]
            img_bytes = base64.b64decode(b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2_local.imdecode(arr, cv2_local.IMREAD_COLOR)
            if img is not None:
                images.append(img)
        except Exception:
            continue

    if len(images) < 3:
        return jsonify({"error": f"Only {len(images)} valid images decoded (need ≥ 3)"}), 400

    try:
        calibrator = get_calibrator(cam_id)
        info = calibrator.calibrate_from_checkerboard(images, board_size, square_size)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/calibrate/status")
def calibrate_status():
    """Get calibration status for a camera. Query param: ?camera_id=cam_local"""
    cam_id = request.args.get("camera_id", "cam_local")
    calibrator = get_calibrator(cam_id)
    info = calibrator.get_calibration_info()
    info["depth_estimator"] = depth_est.get_status()
    return jsonify(info)


@app.post("/api/calibrate/reset")
def calibrate_reset():
    """Reset calibration for a camera. JSON: { camera_id? }"""
    data = request.get_json(silent=True) or {}
    cam_id = data.get("camera_id", "cam_local")
    calibrator = get_calibrator(cam_id)
    calibrator.reset()
    # Also reset height estimator's legacy calibration
    height_estimator.reset_calibration()
    # Reset depth scale
    depth_est.reset_scale()
    return jsonify({"ok": True, "camera_id": cam_id})


from flask import send_from_directory

@app.route("/api/screenshots/<path:filename>")
def serve_screenshot(filename):
    """Serve a saved event screenshot."""
    return send_from_directory(SCREENSHOT_DIR, filename)


@app.get("/api/metrics")
def get_metrics():
    """Return aggregated metrics for the dashboard charts."""
    now = time.time()
    twenty_four_h = now - 86400
    try:
        recent = list(events_collection.find(
            {"unix_ts": {"$gte": twenty_four_h}},
            {"_id": 0, "unix_ts": 1, "action": 1, "risk_level": 1,
             "faces": 1, "detections": 1}
        ))
    except Exception:
        recent = []

    # 1. Events by hour (last 24h, bucketed)
    from collections import Counter
    hour_buckets = Counter()
    for ev in recent:
        h = int((ev.get("unix_ts", 0) - twenty_four_h) / 3600)
        h = max(0, min(23, h))
        hour_buckets[h] += 1
    events_by_hour = [{"hour": h, "count": hour_buckets.get(h, 0)} for h in range(24)]

    # 2. Detection distribution (action breakdown)
    action_counts = Counter()
    for ev in recent:
        action_counts[ev.get("action", "unknown")] += 1
    detection_distribution = [{"action": a, "count": c} for a, c in action_counts.items()]

    # 3. Confidence histogram (10% buckets)
    conf_buckets = Counter()
    for ev in recent:
        for d in ev.get("detections", []):
            conf = d.get("confidence", 0)
            bucket = min(9, int(conf * 10))
            conf_buckets[bucket] += 1
    confidence_histogram = [
        {"range": f"{b*10}-{b*10+10}%", "count": conf_buckets.get(b, 0)}
        for b in range(10)
    ]

    # 4. Risk distribution
    risk_counts = Counter()
    for ev in recent:
        risk_counts[ev.get("risk_level", "MEDIUM")] += 1
    risk_distribution = [{"level": l, "count": risk_counts.get(l, 0)}
                         for l in ["LOW", "MEDIUM", "HIGH"]]

    # 5. Face match rate
    total_faces = 0
    known_faces = 0
    for ev in recent:
        for f in ev.get("faces", []):
            total_faces += 1
            if f.get("name") and f["name"] != "unknown":
                known_faces += 1
    face_match_rate = round(known_faces / total_faces * 100, 1) if total_faces > 0 else 0

    return jsonify({
        "events_by_hour": events_by_hour,
        "detection_distribution": detection_distribution,
        "confidence_histogram": confidence_histogram,
        "risk_distribution": risk_distribution,
        "face_match_rate": face_match_rate,
        "total_events_24h": len(recent),
    })

# =====================================================================
#  MULTI-CAMERA MANAGEMENT
# =====================================================================
import threading

# Camera registry: list of camera dicts
# Each: { id, name, source, type, status, capture, lock }
camera_registry = []
_cam_id_counter = 0
_cam_lock = threading.Lock()


def _next_cam_id():
    global _cam_id_counter
    _cam_id_counter += 1
    return f"cam_{_cam_id_counter}"


def _open_capture(source):
    """Open a VideoCapture for the given source (int index or URL string)."""
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def _register_camera(name, source, cam_type="local"):
    """Register a camera and return its dict."""
    cam_id = _next_cam_id()
    cap = _open_capture(source)
    status = "online" if cap.isOpened() else "offline"
    entry = {
        "id": cam_id,
        "name": name,
        "source": source,
        "type": cam_type,            # local | ip | rtsp
        "status": status,
        "capture": cap,
        "lock": threading.Lock(),
    }
    with _cam_lock:
        camera_registry.append(entry)
    print(f"📷 Registered camera '{name}' (id={cam_id}, source={source}, status={status})", flush=True)
    return entry


def _find_cam(cam_id):
    for c in camera_registry:
        if c["id"] == cam_id:
            return c
    return None


def _generate_cam_feed(cam_entry):
    """MJPEG generator for a single camera — with full ML pipeline."""
    global activity_state

    cam_bg = None          # per-camera background for motion detection
    cam_motion_ctr = 0
    cam_consec = 0
    cam_cooldown = 0
    cam_frame_count = 0
    cam_id = cam_entry.get("id", "unknown")
    is_network = not isinstance(cam_entry.get("source"), int)

    print(f"⚙️  ML pipeline started for camera '{cam_entry.get('name')}' ({cam_id})", flush=True)

    while True:
        if cam_entry.get("removed"):
            break
        cap = cam_entry.get("capture")
        if cap is None or not cap.isOpened():
            # try to reopen
            try:
                cap = _open_capture(cam_entry["source"])
                cam_entry["capture"] = cap
                cam_entry["status"] = "online" if cap.isOpened() else "offline"
                # Minimize internal buffer for network streams
                if is_network and cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                cam_entry["status"] = "offline"
            if not cap or not cap.isOpened():
                time.sleep(1)
                continue

        # For network cameras, drain stale buffered frames before reading
        # cap.grab() discards the oldest buffered frame without decoding
        if is_network:
            cap.grab()

        with cam_entry["lock"]:
            ok, frame = cap.read()

        if not ok or frame is None:
            cam_entry["status"] = "offline"
            time.sleep(0.5)
            continue

        cam_entry["status"] = "online"
        cam_frame_count += 1

        # ── Motion detection ──────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if cam_bg is None:
            cam_bg = gray.copy().astype("float")
            continue

        frame_delta = cv2.absdiff(cv2.convertScaleAbs(cam_bg), gray)
        _, thresh = cv2.threshold(frame_delta, 35, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = any(cv2.contourArea(c) > 4000 for c in contours)
        cam_motion_ctr = cam_motion_ctr + 1 if motion else max(0, cam_motion_ctr - 1)
        cv2.accumulateWeighted(gray, cam_bg, 0.1)

        # ── YOLO-Pose detection ────────────────────────────
        frame, detections = detector.detect(frame, conf=CONF_THRESHOLD)
        detections = [
            {
                "class": d.get("class"),
                "confidence": float(d.get("confidence", 0.0)),
                "box": [float(v) for v in d.get("box", [])],
                "pose_action": d.get("pose_action", "unknown"),
                "height_px": d.get("height_px", 0),
                "keypoints": d.get("keypoints"),
            }
            for d in detections
            if float(d.get("confidence", 0.0)) >= CONF_THRESHOLD and d.get("class") == "person"
        ]

        # Auto-calibrate height on first standing person
        for d in detections:
            _auto_calibrate_new(d, frame.shape)

        # Push frame to SlowFast buffer for activity recognition
        activity_recognizer.push_frame(frame)
        depth_est.push_frame(frame)

        if len(detections) > 0:
            cam_consec += 1
        else:
            cam_consec = 0

        now_ts = time.time()

        # ── Track persons across frames for action classification ──
        actions = _update_tracker(cam_id, detections, now_ts)

        if (
            len(detections) > 0
            and cam_motion_ctr > motion_threshold
            and cam_cooldown == 0
            and cam_consec >= CONSECUTIVE_REQUIRED
        ):
            print(f"⚠️ REAL ACTIVITY on {cam_entry.get('name')} ({cam_id})", flush=True)
            cam_cooldown = 30
            cam_motion_ctr = 0

            # Determine worst action: combine pose ML + time-based tracking
            duration_label = (
                "loitering" if "loitering" in actions else
                "lingering" if "lingering" in actions else
                "visitor" if "visitor" in actions else "passing"
            ) if actions else "passing"
            pose_actions = [d.get("pose_action", "unknown") for d in detections if d.get("pose_action")]
            primary_pose = pose_actions[0] if pose_actions else "unknown"

            # Get SlowFast activity (Kinetics-400)
            sf_actions = activity_recognizer.get_actions()
            sf_label = ""
            if sf_actions and sf_actions[0].get("action") and sf_actions[0]["action"] != "Loading model...":
                sf_label = sf_actions[0]["action"].capitalize()

            # Combined: "Climbing — Standing — lingering"
            parts = [p for p in [sf_label, primary_pose.capitalize(), duration_label] if p]
            worst_action = " — ".join(parts) if parts else "passing"

            # Save screenshot
            screenshot_name = None
            try:
                screenshot_name = f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{cam_id}.jpg"
                screenshot_path = os.path.join(SCREENSHOT_DIR, screenshot_name)
                cv2.imwrite(screenshot_path, frame)
            except Exception as e:
                print(f"Screenshot save error: {e}", flush=True)

            iso_ts = datetime.utcnow().isoformat()
            event_doc = {
                "timestamp": iso_ts,
                "unix_ts": now_ts,
                "detections": detections,
                "source": cam_id,
                "camera_name": cam_entry.get("name", ""),
                "action": worst_action,
            }
            if screenshot_name:
                event_doc["screenshot_path"] = screenshot_name

            # Estimate height
            try:
                if len(detections) > 0:
                    h_res = _estimate_height_new(detections[0], frame.shape, frame=frame, cam_id=cam_id)
                    if h_res and h_res.get("height_m"):
                        event_doc["height_m"] = h_res["height_m"]
                        event_doc["height_confidence"] = h_res.get("confidence", 0)
                        event_doc["height_visibility"] = h_res.get("visibility", "unknown")
                        if h_res.get("distance_m"):
                            event_doc["distance_m"] = h_res["distance_m"]
            except Exception as e:
                print(f"Height estimation error: {e}", flush=True)

            # ── Face recognition ──────────────────────────
            try:
                matched = []
                users = cached_users
                for idx, det in enumerate(detections):
                    if det.get("class") != "person":
                        continue
                    box = det.get("box")
                    if not box:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    h = max(1, y2 - y1)
                    w = max(1, x2 - x1)
                    pad_y = int(h * 0.35)
                    pad_x = int(w * 0.25)
                    xa = max(0, x1 - pad_x)
                    ya = max(0, y1 - pad_y)
                    xb = min(frame.shape[1], x2 + pad_x)
                    yb = min(frame.shape[0], y2 + pad_y)
                    crop = frame[ya:yb, xa:xb]

                    if crop is None or crop.size == 0:
                        continue

                    try:
                        faces = get_faces_and_embeddings(crop)
                    except Exception:
                        faces = []
                    if not faces:
                        continue

                    for f in faces:
                        emb = f.get("embedding")
                        best = {"user_id": None, "name": None, "score": 0.0}
                        for u in users:
                            multi_embs = u.get("embeddings", [])
                            stored_single = u.get("embedding")
                            best_user_score = 0.0
                            if multi_embs:
                                for se in multi_embs:
                                    s = cosine_similarity(emb, se)
                                    if s > best_user_score:
                                        best_user_score = s
                            elif stored_single:
                                best_user_score = cosine_similarity(emb, stored_single)
                            else:
                                continue
                            if best_user_score > best["score"]:
                                best = {"user_id": u.get("user_id"), "name": u.get("name"), "score": best_user_score}
                        if best["score"] >= 0.45:
                            matched.append(best)
                        else:
                            matched.append({"user_id": None, "name": "unknown", "score": best["score"]})

                if matched:
                    event_doc["faces"] = matched
                else:
                    event_doc["faces"] = [{"user_id": None, "name": "unknown", "score": 0.0}]
            except Exception as e:
                print(f"Face processing error on {cam_id}: {e}", flush=True)

            # ── Compute risk score ──
            risk_score, risk_level = _compute_risk(
                event_doc.get("faces", []), actions
            )
            event_doc["risk_score"] = risk_score
            event_doc["risk_level"] = risk_level
            print(f"   Risk: {risk_level} (score={risk_score}), Action: {worst_action}", flush=True)

            # ── Store event ─────────────────────────────────────────
            try:
                events_collection.insert_one(event_doc)
            except Exception as e:
                print(f"Mongo insert error: {e}", flush=True)

            activity_state["active"] = True
            activity_state["last_event"] = iso_ts
            activity_state["last_detections"] = detections
            activity_state["risk_level"] = risk_level
            activity_state["active_until"] = time.time() + 5

        # Decay activity and cooldown
        if activity_state.get("active_until") and time.time() > activity_state.get("active_until"):
            activity_state["active"] = False
            activity_state["active_until"] = None
        if cam_cooldown > 0:
            cam_cooldown -= 1

        # ── Encode raw frame (no overlays) for max FPS ──
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# Auto-register the local webcam (index 0) — but DON'T open it if paused
# We register it lazily so it doesn't conflict with the existing generate_frames
_local_cam = {
    "id": "cam_local",
    "name": "Local Camera",
    "source": 0,
    "type": "local",
    "status": "online",
    "capture": None,   # uses the existing 'camera' global
    "lock": threading.Lock(),
    "is_default": True,
}
camera_registry.append(_local_cam)
_cam_id_counter = 1  # start additional cameras from cam_2


@app.route("/api/cameras", methods=["GET"])
def list_cameras():
    """List all registered cameras."""
    result = []
    for c in camera_registry:
        # refresh status
        cap = c.get("capture")
        if c.get("is_default"):
            c["status"] = "online" if (not camera_paused and camera and camera.isOpened()) else "paused" if camera_paused else "offline"
        elif cap and cap.isOpened():
            c["status"] = "online"
        else:
            c["status"] = "offline"
        result.append({
            "id": c["id"],
            "name": c["name"],
            "source": str(c["source"]),
            "type": c["type"],
            "status": c["status"],
        })
    return jsonify(result)


@app.route("/api/cameras", methods=["POST"])
def add_camera():
    """Add a new IP / RTSP camera. JSON: { name, source }
    source can be a URL (http://..., rtsp://...) or an integer device index.
    """
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    source = data.get("source", "").strip()
    if not name or not source:
        return jsonify({"error": "Provide 'name' and 'source'"}), 400

    # auto-detect type
    cam_type = "ip"
    if source.startswith("rtsp://"):
        cam_type = "rtsp"
    elif source.isdigit():
        source = int(source)
        cam_type = "local"

    entry = _register_camera(name, source, cam_type)
    return jsonify({
        "id": entry["id"],
        "name": entry["name"],
        "source": str(entry["source"]),
        "type": entry["type"],
        "status": entry["status"],
    }), 201


@app.route("/api/cameras/<cam_id>", methods=["DELETE"])
def remove_camera(cam_id):
    """Remove a camera from the registry."""
    if cam_id == "cam_local":
        return jsonify({"error": "Cannot remove the default local camera"}), 400
    with _cam_lock:
        cam = _find_cam(cam_id)
        if not cam:
            return jsonify({"error": "Camera not found"}), 404
        cam["removed"] = True
        try:
            if cam.get("capture") and cam["capture"].isOpened():
                cam["capture"].release()
        except Exception:
            pass
        camera_registry[:] = [c for c in camera_registry if c["id"] != cam_id]
    print(f"📷 Removed camera '{cam.get('name')}' (id={cam_id})", flush=True)
    return jsonify({"ok": True})


@app.route("/api/cameras/<cam_id>/feed")
def camera_feed(cam_id):
    """MJPEG stream for a specific camera."""
    # For the default local camera, use the existing generate_frames
    if cam_id == "cam_local":
        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    cam = _find_cam(cam_id)
    if not cam:
        return jsonify({"error": "Camera not found"}), 404
    return Response(_generate_cam_feed(cam), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/cameras/<cam_id>/test", methods=["POST"])
def test_camera(cam_id):
    """Test if a camera connection is working."""
    cam = _find_cam(cam_id)
    if not cam:
        return jsonify({"error": "Camera not found"}), 404

    if cam.get("is_default"):
        ok = camera and camera.isOpened() and not camera_paused
        return jsonify({"ok": ok, "status": "online" if ok else "paused" if camera_paused else "offline"})

    cap = cam.get("capture")
    if cap is None or not cap.isOpened():
        try:
            cap = _open_capture(cam["source"])
            cam["capture"] = cap
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)})
    ok = cap.isOpened()
    if ok:
        ret, _ = cap.read()
        ok = ret
    cam["status"] = "online" if ok else "offline"
    return jsonify({"ok": ok, "status": cam["status"]})


def _clean_exit(*_):
    """Force clean exit to avoid Python 3.13 thread cleanup crash on Windows."""
    print("\n🛑 Shutting down cleanly...", flush=True)
    # Release cameras
    try:
        if camera and camera.isOpened():
            camera.release()
    except Exception:
        pass
    for c in camera_registry:
        try:
            cap = c.get("capture")
            if cap and cap.isOpened():
                cap.release()
        except Exception:
            pass
    os._exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _clean_exit)
    signal.signal(signal.SIGTERM, _clean_exit)
    # debug help: list routes so you can confirm endpoints exist
    print("Registered routes:")
    for rule in app.url_map.iter_rules():
        print(f"  {rule}")
    app.run(host="0.0.0.0", port=5000)
