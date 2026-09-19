"""Per-camera processing pipeline.

One CameraPipeline per camera, created when the camera is registered. It owns:

  reader thread  - reads the source at native rate, keeps only the newest frame,
                   feeds the clip buffer and draws overlays for viewers
  worker thread  - motion gate, YOLO-Pose, tracking, posture, height, activity
                   feed, risk scoring, event generation and live publishing

ML runs whether or not anyone is viewing, exactly once per camera. Face
recognition runs on a shared background FaceWorker so it never stalls a
camera's frame loop.
"""
import os
import queue
import threading
import time
from collections import deque
from datetime import datetime, timezone

import cv2
import numpy as np

import clips
import detector
import height_geometry
import overlay

PIPELINE_VERSION = "2.0"
SEVERITY_POINTS = {"HIGH": 3, "MEDIUM": 1}
POSE_RISK = {"falling": (4, "person falling"), "lying_down": (3, "person on the ground"),
             "fighting": (4, "fighting posture"), "kicking": (2, "kicking")}
URGENT_POSES = {"falling", "lying_down", "fighting"}


class Settings:
    """Runtime-tunable thresholds shared by all cameras (POST /config)."""

    def __init__(self):
        self.conf_threshold = float(os.environ.get("CONF_THRESHOLD", "0.5"))
        self.motion_threshold = 5          # frames of motion required before an event
        self.consecutive_required = 2      # frames with a person required before an event
        self.alert_cooldown_s = float(os.environ.get("ALERT_COOLDOWN_S", "5"))
        self.urgent_cooldown_s = 15.0      # per track + reason, for falls/fights/loitering
        self.face_retry_s = 2.0
        self.face_refresh_s = 15.0
        self.publish_hz = 10.0

    def as_dict(self):
        return {
            "confidence_threshold": self.conf_threshold,
            "motion_threshold": self.motion_threshold,
            "consecutive_required": self.consecutive_required,
            "alert_cooldown": self.alert_cooldown_s,
            "alert_cooldown_unit": "seconds",
        }

    def update(self, data):
        """Validate everything first so a bad value never leaves a half-applied config."""
        new = {}
        if "confidence_threshold" in data:
            new["conf_threshold"] = min(0.95, max(0.05, float(data["confidence_threshold"])))
        if "motion_threshold" in data:
            new["motion_threshold"] = int(data["motion_threshold"])
        if "consecutive_required" in data:
            new["consecutive_required"] = max(1, int(data["consecutive_required"]))
        if "alert_cooldown" in data:
            new["alert_cooldown_s"] = max(0.0, float(data["alert_cooldown"]))
        for k, v in new.items():
            setattr(self, k, v)


def compute_risk(persons, activities):
    """Risk from per-person evidence + stream activities. Returns (score, level, reasons)."""
    score, reasons = 0, []
    for p in persons:
        ident = p.get("identity")
        if ident and ident.get("name") == "unknown":
            score += 1
            reasons.append("unrecognised face")
        if p.get("dwell") == "loitering":
            score += 3
            reasons.append(f"loitering {int(p.get('dwell_s', 0))}s")
        elif p.get("dwell") == "lingering":
            score += 1
            reasons.append("lingering")
        pose = p.get("stable_pose", p.get("pose"))
        if pose in POSE_RISK:
            pts, why = POSE_RISK[pose]
            score += pts
            reasons.append(why)
    for a in activities:
        pts = SEVERITY_POINTS.get(a.get("severity"), 0)
        if pts:
            score += pts
            reasons.append(f"{a['action'].lower()} ({a['confidence']:.2f})")
    level = "HIGH" if score >= 4 else "MEDIUM" if score >= 2 else "LOW"
    return score, level, reasons


class Perf:
    def __init__(self):
        self.lock = threading.Lock()
        self.capture_ts = deque(maxlen=300)
        self.worker_ts = deque(maxlen=300)
        self.stages = {k: deque(maxlen=300) for k in ("motion", "detect", "track", "height", "face")}
        self.frame_to_result = deque(maxlen=300)
        self.frame_to_event = deque(maxlen=100)
        self.frames_captured = 0
        self.frames_processed = 0
        self.events_written = 0

    def stage(self, name, ms):
        with self.lock:
            self.stages[name].append(ms)

    @staticmethod
    def _summary(arr):
        if not arr:
            return None
        s = sorted(arr)
        return {"n": len(s), "mean_ms": round(sum(s) / len(s), 2),
                "p50_ms": round(s[len(s) // 2], 2), "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 2),
                "max_ms": round(s[-1], 2)}

    @staticmethod
    def _rate(ts):
        if len(ts) < 2 or ts[-1] - ts[0] <= 0:
            return None
        return round((len(ts) - 1) / (ts[-1] - ts[0]), 2)

    def report(self):
        with self.lock:
            return {
                "capture_fps": self._rate(list(self.capture_ts)),
                "pipeline_fps": self._rate(list(self.worker_ts)),
                "frames_captured": self.frames_captured,
                "frames_processed": self.frames_processed,
                "events_written": self.events_written,
                "stages": {k: self._summary(list(v)) for k, v in self.stages.items()},
                "latency": {"frame_to_result": self._summary(list(self.frame_to_result)),
                            "frame_to_event": self._summary(list(self.frame_to_event))},
            }


class FaceWorker:
    """Shared background face recognition. Jobs are (pipeline, track, crop)."""

    def __init__(self, face_module, identity):
        self.face = face_module
        self.identity = identity
        self.q = queue.Queue(maxsize=8)
        if face_module.is_available():
            threading.Thread(target=self._run, daemon=True, name="faces").start()

    def submit(self, pipeline, track, crop):
        try:
            self.q.put_nowait((pipeline, track, crop))
            return True
        except queue.Full:
            return False

    def recognise(self, crop):
        faces = self.face.get_faces_and_embeddings(crop)
        if not faces:
            return None
        # largest face in the person crop
        f = max(faces, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))
        return self.identity.match(f["embedding"])

    def _run(self):
        while True:
            pipeline, track, crop = self.q.get()
            t0 = time.perf_counter()
            try:
                result = self.recognise(crop)
            except Exception as e:
                print(f"Face worker error: {e}", flush=True)
                result = None
            pipeline.perf.stage("face", (time.perf_counter() - t0) * 1000)
            track.identity_ts = time.time()
            if result is not None:
                track.identity = result


def person_crop(frame, box):
    """Upper part of the person box, padded — where the face is."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    bh, bw = max(1, y2 - y1), max(1, x2 - x1)
    xa, xb = max(0, x1 - bw // 4), min(w, x2 + bw // 4)
    ya, yb = max(0, y1 - bh // 6), min(h, y1 + int(bh * 0.6))
    return frame[ya:yb, xa:xb]


class CameraPipeline:
    def __init__(self, cam_id, name, source, cam_type, open_fn, ctx):
        self.cam_id = cam_id
        self.name = name
        self.source = source
        self.cam_type = cam_type
        self._open_fn = open_fn
        self.ctx = ctx                    # AppContext: models, settings, stores, bus
        self.perf = Perf()
        self.status = "starting"
        self.paused = False
        self.removed = False
        self.viewers = 0
        self.error = None

        self._cap = None
        self._cap_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._frame = None
        self._frame_ts = 0.0
        self._jpeg = None
        self._draw_state = None
        self.overlays = os.environ.get("OVERLAYS", "1").strip().lower() in ("1", "true", "yes", "on")
        self.clips = clips.ClipRecorder(os.path.join(ctx.clip_dir, cam_id),
                                        enabled=ctx.clips_enabled)
        self._snapshot_lock = threading.Lock()
        self._snapshot = self._empty_snapshot()
        self.last_event = None

        from tracking import PersonTracker
        self.tracker = PersonTracker()
        self.ground = height_geometry.GroundCalibration(cam_id)
        self._urgent_sent = {}

        self._reader = threading.Thread(target=self._read_loop, daemon=True, name=f"read-{cam_id}")
        self._worker = threading.Thread(target=self._work_loop, daemon=True, name=f"ml-{cam_id}")

    # ── lifecycle ──────────────────────────────────────────
    def start(self):
        self._reader.start()
        self._worker.start()
        return self

    def stop(self):
        self.removed = True
        self._release()
        self.ctx.action.drop_stream(self.cam_id)
        self.ctx.depth.drop_stream(self.cam_id)

    def pause(self):
        self.paused = True
        self._release()
        self.status = "paused"

    def resume(self):
        self.paused = False

    def _release(self):
        with self._cap_lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception:
                    pass
                self._cap = None

    def info(self):
        return {"id": self.cam_id, "name": self.name, "source": str(self.source),
                "type": self.cam_type, "status": self.status, "paused": self.paused,
                "error": self.error, "viewers": self.viewers,
                "height_calibrated": self.ground.is_calibrated}

    # ── capture ────────────────────────────────────────────
    def _read_loop(self):
        backoff = 0.5
        while not self.removed:
            if self.paused:
                time.sleep(0.1)
                continue
            with self._cap_lock:
                if self._cap is None or not self._cap.isOpened():
                    try:
                        self._cap = self._open_fn(self.source)
                    except Exception as e:
                        self._cap, self.error = None, str(e)
                cap = self._cap
            if cap is None or not cap.isOpened():
                self.status = "offline"
                time.sleep(backoff)
                backoff = min(5.0, backoff * 2)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                self.status = "offline"
                self._release()
                time.sleep(backoff)
                backoff = min(5.0, backoff * 2)
                continue
            backoff = 0.5
            self.status, self.error = "online", None
            ts = time.time()
            with self._frame_lock:
                self._frame, self._frame_ts = frame, ts
            with self.perf.lock:
                self.perf.capture_ts.append(ts)
                self.perf.frames_captured += 1
            # Clips keep the raw frame (evidence), the live stream gets the overlay.
            if self.clips.enabled or self.viewers > 0:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    raw_jpeg = buf.tobytes()
                    self.clips.add(ts, raw_jpeg)
                    if self.viewers > 0:
                        self._jpeg = self._encode_for_viewers(frame, raw_jpeg)

    def _encode_for_viewers(self, frame, raw_jpeg):
        state = self._draw_state
        if not (self.overlays and state):
            return raw_jpeg
        # never draw on the array the worker analyses
        ok, buf = cv2.imencode(".jpg", overlay.draw(frame.copy(), state),
                               [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes() if ok else raw_jpeg

    def mjpeg(self):
        """Generator for multipart MJPEG; counts viewers so frames are only encoded when needed."""
        self.viewers += 1
        try:
            last = None
            while not self.removed:
                jpeg = self._jpeg
                if jpeg is None or jpeg is last:
                    time.sleep(0.01)
                    continue
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        finally:
            self.viewers -= 1

    # ── processing ─────────────────────────────────────────
    def _empty_snapshot(self):
        return {"camera_id": self.cam_id, "ts": None, "persons": [], "person_count": 0,
                "activities": [], "risk_level": "LOW", "risk_score": 0, "risk_reasons": [],
                "motion": False, "model_ready": False}

    def snapshot(self):
        with self._snapshot_lock:
            return dict(self._snapshot)

    def _work_loop(self):
        ctx = self.ctx
        bg = None
        motion_ctr = 0
        consecutive = 0
        last_ts = 0.0
        last_publish = 0.0
        last_event_ts = 0.0

        while not self.removed:
            with self._frame_lock:
                frame, ts = self._frame, self._frame_ts
            if frame is None or ts == last_ts or self.paused:
                time.sleep(0.005)
                continue
            last_ts = ts

            # motion gate
            t = time.perf_counter()
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            if bg is None or bg.shape != gray.shape:
                bg = gray.astype("float")
            delta = cv2.absdiff(cv2.convertScaleAbs(bg), gray)
            _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = frame.shape[0] * frame.shape[1] * 0.002    # 0.2% of the frame
            motion = any(cv2.contourArea(c) > min_area for c in contours)
            motion_ctr = motion_ctr + 1 if motion else max(0, motion_ctr - 1)
            cv2.accumulateWeighted(gray, bg, 0.1)
            self.perf.stage("motion", (time.perf_counter() - t) * 1000)

            # detection + tracking
            t = time.perf_counter()
            dets = detector.detect(frame, conf=ctx.settings.conf_threshold)
            self.perf.stage("detect", (time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            tracks = self.tracker.update(dets, ts, frame)
            self.perf.stage("track", (time.perf_counter() - t) * 1000)

            boxes = [tr.box for tr in tracks if not tr.truncated]
            ctx.action.push_frame(frame, self.cam_id, boxes, ts)
            ctx.depth.push_frame(frame, self.cam_id, ts)
            activities = ctx.action.get_actions(self.cam_id)

            # height + face scheduling
            t = time.perf_counter()
            persons = [self._person_state(tr, frame) for tr in tracks]
            self.perf.stage("height", (time.perf_counter() - t) * 1000)
            now = time.time()
            if ctx.faces_enabled:
                for tr in tracks:
                    due = ctx.settings.face_refresh_s if tr.identity else ctx.settings.face_retry_s
                    if now - tr.identity_ts >= due and (tr.box[3] - tr.box[1]) >= 60:
                        tr.identity_ts = now   # mark as attempted even if the queue is full
                        ctx.faces.submit(self, tr, person_crop(frame, tr.box).copy())

            score, level, reasons = compute_risk(persons, activities)
            done = time.time()
            with self.perf.lock:
                self.perf.worker_ts.append(done)
                self.perf.frames_processed += 1
                self.perf.frame_to_result.append((done - ts) * 1000)

            snap = {"camera_id": self.cam_id, "ts": ts, "persons": persons, "person_count": len(persons),
                    "activities": activities, "risk_level": level, "risk_score": score,
                    "risk_reasons": reasons, "motion": motion, "model_ready": ctx.action.is_ready}
            with self._snapshot_lock:
                self._snapshot = snap
            if self.overlays:
                self._draw_state = overlay.build_state(tracks, persons, snap, self.name)
            # 0.75 of the interval: with a loop period that does not divide the
            # interval, an exact gate aliases to half the intended rate.
            if done - last_publish >= 0.75 / ctx.settings.publish_hz:
                last_publish = done
                ctx.bus.publish("detections", snap)

            # event generation
            consecutive = consecutive + 1 if tracks else 0
            trigger = self._urgent_trigger(tracks, activities, now)
            routine = (tracks and motion_ctr > ctx.settings.motion_threshold
                       and consecutive >= ctx.settings.consecutive_required
                       and now - last_event_ts >= ctx.settings.alert_cooldown_s)
            if trigger or routine:
                last_event_ts = now
                motion_ctr = 0
                self._emit_event(frame, ts, tracks, persons, activities, trigger or "activity")

    def _person_state(self, tr, frame):
        shape = frame.shape
        p = {"track_id": tr.id, "box": [round(v, 1) for v in tr.box],
             "confidence": round(tr.confidence, 3),
             "pose": tr.pose, "stable_pose": tr.stable_pose, "truncated": tr.truncated,
             "dwell": tr.dwell, "dwell_s": round(tr.dwell_s, 1),
             "identity": tr.identity, "height_m": None, "height_spread_m": None,
             "height_samples": 0, "distance_m": None, "height_status": None, "relative_depth": None}
        if not self.ground.is_calibrated:
            p["height_status"] = "camera not calibrated"
        else:
            ok, why = height_geometry.is_measurable(tr, shape)
            if ok:
                rows = height_geometry.rows_from_track(tr, window_s=0.3)
                h, d = self.ground.measure(rows["v_foot"], rows["v_head"], shape)
                if h is not None and 0.5 < h < 2.6:
                    tr.heights.append((tr.last_seen, h))
                    p["distance_m"] = round(d, 2)
                else:
                    why = "outside measurable range"
            summary = tr.height_summary()
            if summary:
                p.update({"height_m": summary["height_m"], "height_spread_m": summary["spread_m"],
                          "height_samples": summary["samples"]})
            p["height_status"] = "measured" if summary else why
        if self.ctx.depth.is_ready:
            foot = ((tr.box[0] + tr.box[2]) / 2, tr.box[3])
            p["relative_depth"] = self.ctx.depth.relative_depth(self.cam_id, [foot], shape)[0]
        return p

    def _urgent_trigger(self, tracks, activities, now):
        """Falls, fights and new loitering raise an event immediately (per-reason cooldown)."""
        cool = self.ctx.settings.urgent_cooldown_s
        for key in [k for k, t in self._urgent_sent.items() if now - t > 600]:
            del self._urgent_sent[key]
        candidates = []
        for tr in tracks:
            if tr.stable_pose in URGENT_POSES:
                candidates.append((tr.id, tr.stable_pose))
            if tr.dwell == "loitering":
                candidates.append((tr.id, "loitering"))
        for a in activities:
            if a.get("severity") == "HIGH":
                candidates.append(("stream", a["category"]))
        for key in candidates:
            if now - self._urgent_sent.get(key, 0) >= cool:
                self._urgent_sent[key] = now
                return key[1]
        return None

    def _clip_done(self, stem, filename):
        """Mark the stored event ready (or failed) once its clip has been written."""
        status = "ready" if filename else "failed"
        try:
            self.ctx.events.update_one({"clip_path": f"{stem}.mp4"},
                                       {"$set": {"clip_status": status}})
        except Exception as e:
            print(f"Could not update clip status for {stem}: {e}", flush=True)
        self.ctx.bus.publish("clip", {"source": self.cam_id, "clip_path": f"{stem}.mp4", "status": status})

    def _emit_event(self, frame, frame_ts, tracks, persons, activities, trigger):
        ctx = self.ctx
        now = time.time()
        # Recognise faces synchronously for people never attempted, so the event has them.
        if ctx.faces_enabled:
            for tr, p in zip(tracks, persons):
                if tr.identity is None and tr.face_attempts == 0 and (tr.box[3] - tr.box[1]) >= 60:
                    tr.face_attempts += 1
                    t = time.perf_counter()
                    try:
                        res = ctx.faces.recognise(person_crop(frame, tr.box))
                    except Exception:
                        res = None
                    self.perf.stage("face", (time.perf_counter() - t) * 1000)
                    if res:
                        tr.identity = p["identity"] = res
        faces = [dict(p["identity"], track_id=p["track_id"]) for p in persons if p.get("identity")]
        score, level, reasons = compute_risk(persons, activities)

        dt = datetime.now(timezone.utc)
        stem = f"{dt.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_{self.cam_id}"
        shot = f"{stem}.jpg"
        try:
            cv2.imwrite(os.path.join(ctx.screenshot_dir, shot), frame)
        except Exception as e:
            print(f"Screenshot save error: {e}", flush=True)
            shot = None
        clip = self.clips.capture(stem, frame_ts, on_done=lambda name: self._clip_done(stem, name))

        top = persons[0] if persons else {}
        parts = [a["action"] for a in activities[:1]]
        if top.get("pose") not in (None, "unknown"):
            parts.append(top["pose"].replace("_", " ").capitalize())
        if top.get("dwell"):
            parts.append(top["dwell"])
        measured = [p for p in persons if p.get("height_m")]
        doc = {
            "timestamp": dt.isoformat(),
            "unix_ts": now,
            "source": self.cam_id,
            "camera_name": self.name,
            "trigger": trigger,
            "action": " — ".join(parts) or "person detected",
            "persons": persons,
            "activities": activities,
            "faces": faces,
            "face_status": ("disabled" if not ctx.faces_enabled else
                            "matched" if any(f["name"] != "unknown" for f in faces) else
                            "unknown" if faces else "no_face_visible"),
            "detections": [{"class": "person", "confidence": p["confidence"], "box": p["box"]} for p in persons],
            "height_m": measured[0]["height_m"] if measured else None,
            "height_spread_m": measured[0]["height_spread_m"] if measured else None,
            "distance_m": measured[0]["distance_m"] if measured else None,
            "risk_score": score,
            "risk_level": level,
            "risk_reasons": reasons,
            "screenshot_path": shot,
            "clip_path": clip,
            "clip_status": "recording" if clip else "disabled",
            "pipeline_version": PIPELINE_VERSION,
            "face_backend": ctx.face_backend,
        }
        try:
            ctx.events.insert_one(doc)
            with self.perf.lock:
                self.perf.events_written += 1
                self.perf.frame_to_event.append((time.time() - frame_ts) * 1000)
        except Exception as e:
            print(f"Mongo insert error on {self.cam_id}: {e}", flush=True)
        doc.pop("_id", None)
        self.last_event = doc
        ctx.bus.publish("event", doc)
        print(f"⚠️  [{self.cam_id}] {trigger}: {doc['action']} | risk {level} ({score}) {reasons}", flush=True)
