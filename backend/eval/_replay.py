"""Offline replay of a video through the live pipeline's components.

Uses video time (frame index / FPS) instead of wall-clock time, so results are
reproducible and independent of how fast this machine is.
"""
import os
import sys

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

import cv2  # noqa: E402

import detector  # noqa: E402
from tracking import PersonTracker  # noqa: E402

_activity = None


def activity_model():
    """Loaded once; used synchronously (no background thread)."""
    global _activity
    if _activity is None:
        from action_recognizer import ActionRecognizer
        _activity = ActionRecognizer(enabled=False)
        _activity._load()
    return _activity


def replay(path, use_activity=True, max_width=640, conf=0.5, stride=1):
    """Yields dicts per processed frame: ts, frame, tracks, activities."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = PersonTracker()
    act = activity_model() if use_activity else None
    import action_recognizer as act_mod
    stream = f"replay:{path}"
    if act:
        act.drop_stream(stream)
    last_inf = -1e9
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            if frame.shape[1] > max_width:
                frame = cv2.resize(frame, (max_width, int(frame.shape[0] * max_width / frame.shape[1])))
            ts = idx / fps
            tracks = tracker.update(detector.detect(frame, conf=conf), ts, frame)
            activities = []
            if act:
                act.push_frame(frame, stream, [t.box for t in tracks if not t.truncated], ts)
                st = act._streams.get(stream)
                if st and ts - last_inf >= act_mod.INFERENCE_INTERVAL and act.clip_ready(st):
                    act._infer(st, list(st.frames))
                    last_inf = ts
                activities = act.get_actions(stream)
            yield {"ts": ts, "index": idx, "fps": fps, "frame": frame, "tracks": tracks,
                   "activities": activities}
    finally:
        cap.release()
        if act:
            act.drop_stream(stream)
