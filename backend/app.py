"""SecureVision backend: Flask API + per-camera ML pipelines.

Environment:
  MONGO_URI, MONGO_DB          database (default mongodb://localhost:27017/, security_system)
  CAMERA_SOURCE                default camera: device index or URL (default 0)
  DEMO_MODE, DEMO_VIDEO        loop a video file instead of the default camera
  FACE_BACKEND                 facenet (default) | deepface | none
  ENABLE_ACTIVITY              1 (default) | 0   SlowFast activity recognition
  ENABLE_DEPTH                 0 (default) | 1   MiDaS relative depth
  ENABLE_CLIPS                 1 (default) | 0   record a short video per event
  OVERLAYS                     1 (default) | 0   draw boxes/skeletons on the stream
  DASHBOARD_PASSWORD           dashboard password (generated on first run if unset)
  AUTH_DISABLED                1 disables the password gate entirely
  PORT                         default 5000
"""
import base64
import os
import queue
import signal
import sys
import threading
import time

# Log lines contain non-ASCII symbols; never let a legacy console encoding crash a thread.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
from collections import Counter
from datetime import datetime, timezone

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from pymongo import DESCENDING, MongoClient

import clips
import detector
import face_utils
from action_recognizer import ActionRecognizer
from auth import Auth
from camera_calibrator import CameraCalibrator, list_presets as list_camera_presets
from demo_video import DEMO_MODE, get_demo_camera
from depth_estimator import DepthEstimator
from event_bus import EventBus
from identity import IdentityStore
from pipeline import PIPELINE_VERSION, CameraPipeline, FaceWorker, Settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOT_DIR = os.path.join(BASE_DIR, "static", "screenshots")
CLIP_DIR = os.path.join(BASE_DIR, "static", "clips")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(CLIP_DIR, exist_ok=True)


def _flag(name, default):
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")


app = Flask(__name__)
CORS(app, supports_credentials=True)
auth = Auth().install(app)

# ── Database ───────────────────────────────────────────────────────
client = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017/"), serverSelectionTimeoutMS=5000)
db = client[os.environ.get("MONGO_DB", "security_system")]
events_collection = db["events"]
users_collection = db["users"]
cameras_collection = db["cameras"]
try:
    events_collection.create_index([("unix_ts", DESCENDING)])
    users_collection.create_index("name")
except Exception as e:
    print(f"⚠ MongoDB not reachable at startup ({e}); events will not be stored until it is.", flush=True)


# ── Shared context ─────────────────────────────────────────────────
class AppContext:
    def __init__(self):
        self.settings = Settings()
        self.bus = EventBus()
        self.events = events_collection
        self.screenshot_dir = SCREENSHOT_DIR
        self.clip_dir = CLIP_DIR
        self.clips_enabled = _flag("ENABLE_CLIPS", "1")
        self.face_backend = face_utils.BACKEND_ID
        self.faces_enabled = face_utils.is_available()
        self.identity = IdentityStore(users_collection, face_utils.BACKEND_ID)
        self.identity.reload()
        self.faces = FaceWorker(face_utils, self.identity)
        self.action = ActionRecognizer(enabled=_flag("ENABLE_ACTIVITY", "1"))
        self.depth = DepthEstimator(enabled=_flag("ENABLE_DEPTH", "0"))


ctx = AppContext()
auth.announce()
detector.warmup()
for _cam in os.listdir(CLIP_DIR):
    clips.prune(os.path.join(CLIP_DIR, _cam))
calibrators = {}
pipelines = {}                 # cam_id -> CameraPipeline (insertion order = display order)
_cam_lock = threading.Lock()
DEFAULT_CAM = "cam_local"


def get_calibrator(cam_id):
    if cam_id not in calibrators:
        calibrators[cam_id] = CameraCalibrator(cam_id)
    return calibrators[cam_id]


def open_source(source):
    if source == "__demo__":
        return get_demo_camera()
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def parse_source(value):
    value = str(value).strip()
    return int(value) if value.isdigit() else value


def add_pipeline(cam_id, name, source, cam_type):
    p = CameraPipeline(cam_id, name, source, cam_type, open_source, ctx)
    with _cam_lock:
        pipelines[cam_id] = p
    p.start()
    print(f"📷 Camera '{name}' ({cam_id}) started, source={source}", flush=True)
    return p


def get_pipeline(cam_id=None):
    return pipelines.get(cam_id or DEFAULT_CAM)


if DEMO_MODE:
    print("🎬 DEMO MODE — looping a video file instead of the camera", flush=True)
    add_pipeline(DEFAULT_CAM, "Demo Video", "__demo__", "demo")
else:
    add_pipeline(DEFAULT_CAM, "Local Camera", parse_source(os.environ.get("CAMERA_SOURCE", "0")), "local")

try:
    for c in cameras_collection.find({}):
        add_pipeline(c["cam_id"], c["name"], parse_source(c["source"]), c.get("type", "ip"))
except Exception as e:
    print(f"⚠ Could not restore saved cameras: {e}", flush=True)


def _json_error(msg, code=400, **extra):
    return jsonify({"error": msg, **extra}), code


def _decode_image(b64):
    if not b64:
        return None
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# =====================================================================
#  Health / system
# =====================================================================

@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok", "demo_mode": DEMO_MODE}


@app.get("/status")
@app.get("/api/status")
def status():
    mongo_ok, mongo_msg = False, ""
    try:
        client.admin.command("ping")
        mongo_ok = True
    except Exception as e:
        mongo_msg = str(e)
    cam = get_pipeline()
    return jsonify({
        "service": "ok",
        "mongo": {"ok": mongo_ok, "error": mongo_msg},
        "camera_open": bool(cam and cam.status == "online"),
    })


def models_status():
    import torch
    return {
        "device": detector.DEVICE,
        "gpu": torch.cuda.get_device_name(0) if detector.DEVICE == "cuda" else None,
        "torch": torch.__version__,
        "pose": {"model": os.path.basename(detector.MODEL_PATH), "ready": True, "device": detector.DEVICE},
        "activity": ctx.action.status(),
        "depth": ctx.depth.status(),
        "face": {"backend": face_utils.BACKEND_ID, "available": face_utils.is_available(),
                 "error": face_utils.BACKEND_ERROR},
        "clips": {"enabled": ctx.clips_enabled, "pre_s": clips.PRE_S, "post_s": clips.POST_S},
        "auth": {"required": not auth.disabled},
    }


@app.get("/api/system/info")
def system_info():
    return jsonify({
        "demo_mode": DEMO_MODE,
        "pipeline_version": PIPELINE_VERSION,
        "models": {
            "pose": "YOLOv8n-Pose (COCO 17 keypoints)",
            "activity": "SlowFast R50 (Kinetics-400, curated categories)" if ctx.action.enabled else "disabled",
            "depth": f"MiDaS {ctx.depth.model_type} (relative depth only)" if ctx.depth.enabled else "disabled",
            "face": {"facenet-vggface2": "FaceNet InceptionResnetV1 (VGGFace2) + MTCNN",
                     "deepface-arcface": "ArcFace (DeepFace)"}.get(face_utils.BACKEND_ID, "disabled"),
        },
        "runtime": models_status(),
        "features": [
            "Pose-based posture: walking, running, falling, lying down, fighting",
            "Activity recognition on person-centred clips (fighting, falls, running, climbing)",
            "Dwell-time loitering detection",
            "Height from calibrated ground-plane geometry",
            "Face recognition with backend-safe enrollment",
            "Multi-camera (USB / IP / RTSP), one pipeline per camera",
            "Live push updates (Server-Sent Events)",
            "Overlays drawn on the live stream (boxes, skeletons, identity, risk)",
            "A short video clip saved around every event",
            "Password-protected API, streams and enrollment",
            "Measured performance at /api/perf",
        ],
    })


@app.get("/api/perf")
def get_perf():
    cams = {cid: p.perf.report() for cid, p in list(pipelines.items())}
    default = cams.get(DEFAULT_CAM, {})
    return jsonify({
        **models_status(),
        "torch_cuda_available": detector.DEVICE == "cuda",
        "cameras": cams,
        # top-level fields mirror the default camera for simple consumers
        **{k: default.get(k) for k in ("capture_fps", "pipeline_fps", "frames_captured",
                                       "frames_processed", "events_written", "stages", "latency")},
        "stream_subscribers": ctx.bus.subscriber_count,
    })


# =====================================================================
#  Live data
# =====================================================================

@app.get("/api/stream")
def stream():
    """Server-Sent Events: `detections` (≤10 Hz per camera) and `event` messages."""
    q = ctx.bus.subscribe()

    def gen():
        import json
        try:
            yield "retry: 2000\n\n"
            for p in list(pipelines.values()):
                yield f"event: detections\ndata: {json.dumps(p.snapshot(), default=str)}\n\n"
            while True:
                try:
                    yield q.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            ctx.bus.unsubscribe(q)

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/detections")
def get_detections():
    p = get_pipeline(request.args.get("camera_id"))
    if p is None:
        return _json_error("camera not found", 404)
    return jsonify(p.snapshot())


@app.get("/activity_status")
@app.get("/api/activity_status")
def activity_status():
    p = get_pipeline(request.args.get("camera_id"))
    snap = p.snapshot() if p else {}
    last = p.last_event if p else None
    return jsonify({
        "active": bool(last and time.time() - last["unix_ts"] < 5),
        "last_event": last["timestamp"] if last else None,
        "risk_level": snap.get("risk_level", "LOW"),
        "person_count": snap.get("person_count", 0),
    })


def _pipeline_video(cam_id):
    p = get_pipeline(cam_id)
    if p is None:
        return _json_error("camera not found", 404)
    return Response(p.mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/video_feed")
@app.route("/api/video_feed")
def video_feed():
    return _pipeline_video(DEFAULT_CAM)


@app.post("/camera/pause")
@app.post("/api/camera/pause")
def pause_camera():
    get_pipeline().pause()
    return jsonify({"ok": True, "paused": True})


@app.post("/camera/resume")
@app.post("/api/camera/resume")
def resume_camera():
    get_pipeline().resume()
    return jsonify({"ok": True, "paused": False})


@app.get("/camera/state")
@app.get("/api/camera/state")
def camera_state():
    p = get_pipeline()
    return jsonify({"paused": p.paused, "camera_open": p.status == "online"})


# =====================================================================
#  Events / stats / metrics
# =====================================================================

def _event_out(d):
    d.pop("_id", None)
    return d


@app.get("/events")
@app.get("/api/events")
def get_events():
    limit = min(500, int(request.args.get("limit", 50)))
    q = {}
    if request.args.get("camera_id"):
        q["source"] = request.args["camera_id"]
    if request.args.get("since"):
        q["unix_ts"] = {"$gt": float(request.args["since"])}
    try:
        docs = list(events_collection.find(q).sort("unix_ts", -1).limit(limit))
    except Exception as e:
        return _json_error(f"database unavailable: {e}", 503)
    return jsonify([_event_out(d) for d in docs])


@app.get("/stats")
@app.get("/api/stats")
def stats():
    now = time.time()
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    try:
        return jsonify({
            "events_last_hour": events_collection.count_documents({"unix_ts": {"$gte": now - 3600}}),
            "events_today": events_collection.count_documents({"unix_ts": {"$gte": today}}),
        })
    except Exception:
        return jsonify({"events_last_hour": 0, "events_today": 0})


def _is_real_face(f):
    # Pre-2.0 events stored a placeholder {"name": "unknown", "score": 0.0} when no face was visible.
    return not (f.get("name") == "unknown" and not f.get("score"))


@app.get("/api/metrics")
def get_metrics():
    now = time.time()
    since = now - 86400
    q = {"unix_ts": {"$gte": since}}
    if request.args.get("version"):
        q["pipeline_version"] = request.args["version"]
    try:
        recent = list(events_collection.find(q, {"_id": 0, "unix_ts": 1, "action": 1, "risk_level": 1,
                                                  "faces": 1, "detections": 1, "trigger": 1,
                                                  "activities": 1}))
    except Exception:
        recent = []

    hours = Counter(max(0, min(23, int((ev.get("unix_ts", 0) - since) / 3600))) for ev in recent)
    triggers = Counter()
    for ev in recent:
        cats = [a.get("category") for a in ev.get("activities") or [] if a.get("category")]
        triggers[ev.get("trigger") or (cats[0] if cats else None) or "activity"] += 1
    conf = Counter()
    for ev in recent:
        for d in ev.get("detections", []):
            conf[min(9, int(float(d.get("confidence", 0)) * 10))] += 1
    risks = Counter(ev.get("risk_level", "LOW") for ev in recent)
    faces = [f for ev in recent for f in (ev.get("faces") or []) if _is_real_face(f)]
    known = sum(1 for f in faces if f.get("name") and f["name"] != "unknown")

    return jsonify({
        "events_by_hour": [{"hour": h, "count": hours.get(h, 0)} for h in range(24)],
        "detection_distribution": [{"action": a, "count": c} for a, c in triggers.most_common()],
        "confidence_histogram": [{"range": f"{b * 10}-{b * 10 + 10}%", "count": conf.get(b, 0)} for b in range(10)],
        "risk_distribution": [{"level": l, "count": risks.get(l, 0)} for l in ("LOW", "MEDIUM", "HIGH")],
        "face_match_rate": round(known / len(faces) * 100, 1) if faces else 0,
        "face_observations": len(faces),
        "face_match_rate_note": "share of visible faces matched to an enrolled person — not accuracy",
        "total_events_24h": len(recent),
    })


@app.route("/api/screenshots/<path:filename>")
def serve_screenshot(filename):
    return send_from_directory(SCREENSHOT_DIR, filename)


@app.route("/api/clips/<cam_id>/<path:filename>")
def serve_clip(cam_id, filename):
    """Event video. Sent with Range support so the browser can seek."""
    return send_from_directory(os.path.join(CLIP_DIR, cam_id), filename, conditional=True)


# =====================================================================
#  Runtime config
# =====================================================================

@app.get("/config")
@app.get("/api/config")
def get_config():
    return jsonify(ctx.settings.as_dict())


@app.post("/config")
@app.post("/api/config")
def set_config():
    try:
        ctx.settings.update(request.get_json(silent=True) or {})
    except (TypeError, ValueError) as e:
        return _json_error(str(e))
    return jsonify({"ok": True, "config": ctx.settings.as_dict()})


# =====================================================================
#  Identities
# =====================================================================

def _embed_images(images, check_quality):
    """Returns (embeddings, quality_scores, rejected)."""
    embeddings, scores, rejected = [], [], 0
    for idx, img in enumerate(images):
        score = {"index": idx, "accepted": False, "reason": None, "blur": 0.0, "brightness": 128.0}
        try:
            if img is None or img.size == 0:
                raise ValueError("decode_failed")
            h, w = img.shape[:2]
            if min(h, w) < 150:
                s = 150 / min(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            faces = face_utils.get_faces_and_embeddings(img)
            if not faces:
                raise ValueError("no_face")
            if len(faces) > 1:
                raise ValueError("multiple_faces")
            x1, y1, x2, y2 = faces[0]["box"]
            crop = img[y1:y2, x1:x2]
            score["blur"] = round(face_utils.compute_blur_score(crop), 1)
            score["brightness"] = round(face_utils.compute_brightness(crop), 1)
            if check_quality and score["blur"] < 20.0:
                raise ValueError("too_blurry")
            if check_quality and not 40.0 <= score["brightness"] <= 230.0:
                raise ValueError("bad_lighting")
            embeddings.append(faces[0]["embedding"])
            score["accepted"] = True
        except ValueError as e:
            score["reason"] = str(e)
            rejected += 1
        except Exception as e:
            score["reason"] = f"error: {e}"
            rejected += 1
        scores.append(score)
    return embeddings, scores, rejected


def _enroll(name, images, check_quality):
    if not face_utils.is_available():
        return _json_error(f"face recognition is disabled ({face_utils.BACKEND_ERROR or 'FACE_BACKEND=none'})", 503)
    if not name or not name.strip():
        return _json_error("Provide a non-empty 'name'")
    embeddings, scores, rejected = _embed_images(images, check_quality)
    if not embeddings:
        return _json_error("No usable faces found in provided images", quality_scores=scores)
    user_id, total, created = ctx.identity.enroll(name, embeddings)
    return jsonify({"user_id": user_id, "name": name.strip(), "samples": len(embeddings),
                    "total_samples": total, "created": created, "rejected": rejected,
                    "quality_scores": scores, "face_backend": face_utils.BACKEND_ID})


@app.route("/enroll", methods=["POST"])
@app.route("/api/enroll", methods=["POST"])
def enroll_user():
    """multipart: name + image, or JSON: {name, image_base64}"""
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        name = request.form.get("name")
        f = request.files.get("image")
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR) if f else None
    else:
        data = request.get_json(silent=True) or {}
        name, img = data.get("name"), _decode_image(data.get("image_base64"))
    if img is None:
        return _json_error("Provide 'name' and an image (file or image_base64)")
    return _enroll(name, [img], check_quality=False)


@app.route("/api/register", methods=["POST"])
def register_user():
    """JSON: {name, images_base64: [dataurl, ...]}. Adds to an existing user with the same name."""
    data = request.get_json(silent=True) or {}
    images = []
    for b64 in data.get("images_base64") or []:
        try:
            images.append(_decode_image(b64))
        except Exception:
            images.append(None)
    if not images:
        return _json_error("Provide 'name' and 'images_base64' list")
    return _enroll(data.get("name"), images, check_quality=True)


@app.get("/users")
@app.get("/api/users")
def list_users():
    return jsonify(ctx.identity.summary())


@app.delete("/api/users/<user_id>")
def delete_user(user_id):
    res = users_collection.delete_one({"user_id": user_id})
    if not res.deleted_count:
        return _json_error("user not found", 404)
    ctx.identity.reload()
    return jsonify({"ok": True})


# =====================================================================
#  Camera intrinsics calibration
# =====================================================================

@app.get("/api/calibrate/presets")
def calibrate_presets():
    return jsonify(list_camera_presets())


@app.post("/api/calibrate/preset")
def calibrate_preset():
    data = request.get_json(silent=True) or {}
    if not data.get("preset_name"):
        return _json_error("Provide 'preset_name'")
    try:
        return jsonify(get_calibrator(data.get("camera_id", DEFAULT_CAM)).load_preset(data["preset_name"]))
    except ValueError as e:
        return _json_error(str(e))


@app.post("/api/calibrate/manual")
def calibrate_manual():
    data = request.get_json(silent=True) or {}
    try:
        info = get_calibrator(data.get("camera_id", DEFAULT_CAM)).set_manual_params(
            frame_width=int(data.get("frame_width", 640)),
            frame_height=int(data.get("frame_height", 480)),
            fov_h_deg=data.get("fov_h_deg"),
            focal_mm=data.get("focal_mm"),
            sensor_w_mm=data.get("sensor_w_mm"),
            dist_coeffs=data.get("dist_coeffs"),
        )
        return jsonify(info)
    except Exception as e:
        return _json_error(str(e))


@app.post("/api/calibrate/checkerboard")
def calibrate_checkerboard():
    data = request.get_json(silent=True) or {}
    images = []
    for b64 in data.get("images_base64", []):
        try:
            img = _decode_image(b64)
            if img is not None:
                images.append(img)
        except Exception:
            continue
    if len(images) < 3:
        return _json_error(f"Need at least 3 valid checkerboard images (got {len(images)})")
    try:
        info = get_calibrator(data.get("camera_id", DEFAULT_CAM)).calibrate_from_checkerboard(
            images, tuple(data.get("board_size", [9, 6])), float(data.get("square_size_mm", 25.0)))
        return jsonify(info)
    except Exception as e:
        return _json_error(str(e))


@app.get("/api/calibrate/status")
def calibrate_status():
    cam_id = request.args.get("camera_id", DEFAULT_CAM)
    info = get_calibrator(cam_id).get_calibration_info()
    p = get_pipeline(cam_id)
    info["ground"] = p.ground.status() if p else None
    info["depth_estimator"] = ctx.depth.status()
    return jsonify(info)


@app.post("/api/calibrate/reset")
def calibrate_reset():
    data = request.get_json(silent=True) or {}
    cam_id = data.get("camera_id", DEFAULT_CAM)
    get_calibrator(cam_id).reset()
    return jsonify({"ok": True, "camera_id": cam_id})


# =====================================================================
#  Height calibration (ground plane)
# =====================================================================

def _ground_intrinsics(cam_id, frame_shape):
    cal = get_calibrator(cam_id)
    fy = cal.get_focal_length_px(frame_shape)
    cy = cal.get_principal_point(frame_shape)[1]
    return fy, cy


@app.get("/api/calibrate/ground")
def ground_status():
    p = get_pipeline(request.args.get("camera_id"))
    if p is None:
        return _json_error("camera not found", 404)
    return jsonify(p.ground.status())


@app.post("/api/calibrate/ground/sample")
def ground_sample():
    """Record a reference: a person of known height standing in view.

    JSON: {camera_id?, height_m, track_id?}. With several people in view, pass track_id
    (from /api/detections). Record samples at several distances from the camera.
    """
    import height_geometry
    data = request.get_json(silent=True) or {}
    p = get_pipeline(data.get("camera_id"))
    if p is None:
        return _json_error("camera not found", 404)
    try:
        height_m = float(data["height_m"])
    except (KeyError, TypeError, ValueError):
        return _json_error("Provide 'height_m' (metres)")
    if not 0.5 < height_m < 2.6:
        return _json_error("height_m must be between 0.5 and 2.6")
    with p._frame_lock:
        frame = p._frame
    if frame is None:
        return _json_error("camera has no frame yet", 409)
    tracks = list(p.tracker.tracks.values())
    if data.get("track_id"):
        tracks = [t for t in tracks if t.id == data["track_id"]]
    tracks = [t for t in tracks if time.time() - t.last_seen < 1.0]
    if len(tracks) != 1:
        return _json_error(f"need exactly one person in view (found {len(tracks)}); pass track_id", 409)
    ok, why = height_geometry.is_measurable(tracks[0], frame.shape)
    if not ok:
        return _json_error(f"person is not measurable: {why}", 409)
    rows = height_geometry.rows_from_track(tracks[0])
    sample = {**rows, "height_m": height_m, "resolution": [frame.shape[1], frame.shape[0]],
              "ts": time.time()}
    n = p.ground.add_sample(sample)
    return jsonify({"ok": True, "sample": sample, "samples": n})


@app.post("/api/calibrate/ground/solve")
def ground_solve():
    """JSON: {camera_id?, camera_height_m?}. Fits camera height + tilt; fixes the height if given."""
    data = request.get_json(silent=True) or {}
    p = get_pipeline(data.get("camera_id"))
    if p is None:
        return _json_error("camera not found", 404)
    with p._frame_lock:
        frame = p._frame
    if frame is None:
        return _json_error("camera has no frame yet", 409)
    fy, cy = _ground_intrinsics(p.cam_id, frame.shape)
    cam_h = data.get("camera_height_m")
    try:
        result = p.ground.solve(fy, cy, (frame.shape[1], frame.shape[0]),
                                float(cam_h) if cam_h not in (None, "") else None)
    except ValueError as e:
        return _json_error(str(e))
    return jsonify({"ok": True, "params": result})


@app.post("/api/calibrate/ground/manual")
def ground_manual():
    """JSON: {camera_id?, camera_height_m, tilt_deg} — for installers who measured the mount."""
    data = request.get_json(silent=True) or {}
    p = get_pipeline(data.get("camera_id"))
    if p is None:
        return _json_error("camera not found", 404)
    with p._frame_lock:
        frame = p._frame
    if frame is None:
        return _json_error("camera has no frame yet", 409)
    try:
        fy, cy = _ground_intrinsics(p.cam_id, frame.shape)
        p.ground.params = {"camera_height_m": float(data["camera_height_m"]),
                           "tilt_deg": float(data["tilt_deg"]), "fy": fy, "cy": cy,
                           "resolution": [frame.shape[1], frame.shape[0]], "rms_cm": None,
                           "loo_rms_cm": None, "n_samples": 0, "method": "manual"}
        p.ground._save()
    except (KeyError, TypeError, ValueError):
        return _json_error("Provide 'camera_height_m' and 'tilt_deg'")
    return jsonify({"ok": True, "params": p.ground.params})


@app.post("/api/calibrate/ground/reset")
def ground_reset():
    data = request.get_json(silent=True) or {}
    p = get_pipeline(data.get("camera_id"))
    if p is None:
        return _json_error("camera not found", 404)
    p.ground.clear()
    for t in p.tracker.tracks.values():
        t.heights.clear()
    return jsonify({"ok": True})


# =====================================================================
#  Cameras
# =====================================================================

@app.get("/api/cameras")
def list_cameras():
    return jsonify([p.info() for p in list(pipelines.values())])


@app.post("/api/cameras")
def add_camera():
    """JSON: {name, source} — source is a device index or http:// / rtsp:// URL."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    source = str(data.get("source", "")).strip()
    if not name or not source:
        return _json_error("Provide 'name' and 'source'")
    source = parse_source(source)
    cam_type = "local" if isinstance(source, int) else "rtsp" if source.startswith("rtsp://") else "ip"
    with _cam_lock:
        n = 2
        while f"cam_{n}" in pipelines:
            n += 1
        cam_id = f"cam_{n}"
    try:
        cameras_collection.insert_one({"cam_id": cam_id, "name": name, "source": str(source), "type": cam_type})
    except Exception as e:
        print(f"⚠ Camera '{name}' will not survive a restart: {e}", flush=True)
    p = add_pipeline(cam_id, name, source, cam_type)
    return jsonify(p.info()), 201


@app.delete("/api/cameras/<cam_id>")
def remove_camera(cam_id):
    if cam_id == DEFAULT_CAM:
        return _json_error("Cannot remove the default camera")
    with _cam_lock:
        p = pipelines.pop(cam_id, None)
    if p is None:
        return _json_error("Camera not found", 404)
    p.stop()
    try:
        cameras_collection.delete_one({"cam_id": cam_id})
    except Exception:
        pass
    return jsonify({"ok": True})


@app.route("/api/cameras/<cam_id>/feed")
def camera_feed(cam_id):
    return _pipeline_video(cam_id)


@app.post("/api/cameras/<cam_id>/test")
def test_camera(cam_id):
    p = get_pipeline(cam_id)
    if p is None:
        return _json_error("Camera not found", 404)
    fresh = bool(p.perf.capture_ts) and time.time() - p.perf.capture_ts[-1] < 2.0
    return jsonify({"ok": fresh and not p.paused, "status": p.status, "error": p.error})


# =====================================================================

def _clean_exit(*_):
    print("\n🛑 Shutting down...", flush=True)
    for p in list(pipelines.values()):
        p.stop()
    os._exit(0)


def serve():
    """Waitress in front by default; Flask's dev server only when asked for.

    send_bytes=1 stops waitress buffering the live event stream. Measured on the
    demo video, camera->client latency by send_bytes: 1 -> 29 ms, 4096 -> 1.4 s,
    18000 (waitress default) -> 6.3 s. MJPEG throughput is unaffected (~11.9 fps
    either way), so the small value costs nothing here.

    Each camera stream and each SSE subscriber holds a thread, hence the
    generous thread count.
    """
    host, port = "0.0.0.0", int(os.environ.get("PORT", "5000"))
    if _flag("FLASK_DEV", "0"):
        print(f"⚠ Flask development server on {port} (FLASK_DEV=1)", flush=True)
        app.run(host=host, port=port, threaded=True)
        return
    from waitress import serve as waitress_serve
    threads = int(os.environ.get("SERVER_THREADS", "24"))
    print(f"✓ Serving on http://{host}:{port} (waitress, {threads} threads)", flush=True)
    waitress_serve(app, host=host, port=port, threads=threads,
                   send_bytes=int(os.environ.get("SERVER_SEND_BYTES", "1")),
                   channel_timeout=86400, ident="SecureVision")


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _clean_exit)
    signal.signal(signal.SIGTERM, _clean_exit)
    serve()
