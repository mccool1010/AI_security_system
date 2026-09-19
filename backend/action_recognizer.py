"""SlowFast R50 (Kinetics-400) activity recognition for CCTV, one stream per camera.

Design choices, each fixing a problem in the previous version:

* Curated categories. Kinetics-400 is a sports/everyday dataset; mapping 145 of
  its classes to "security" labels (hugging -> HIGH, archery -> CRITICAL) made
  most alerts noise. Only a handful of classes with a clear CCTV meaning are
  kept, grouped into categories whose score is the summed probability of their
  member classes.
* Person-centred crops. SlowFast sees a square crop around the people the pose
  detector found, not the whole wide frame squashed to 256x256.
* Time-based clip sampling. The model was trained on ~2.1 s windows at 30 fps
  (32 frames, stride 2). Frames are timestamped and the clip is resampled from
  the last CLIP_SECONDS, so results do not change with pipeline FPS.
* Nobody in view -> no activity. Buffers and results are cleared. People too
  small to classify (upscaling a 40 px figure produces confident nonsense) are
  ignored; callers also leave out people cut off by the frame edge.
* Agreement over time. Smoothing starts from zero, so a category must score
  in at least two consecutive clips before it is reported.
* Stricter bar for HIGH severity. Kinetics-trained SlowFast has a domain gap
  on overhead CCTV (an overhead walker scored 0.48 "drop kicking" on the test
  clip), so HIGH categories need a higher smoothed score than MEDIUM ones.
  Measure it on your own footage with eval/eval_activity.py.
* Per-stream state. Each camera has its own buffer, smoothing and results; one
  background thread schedules inference across streams.
"""
import json
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch

# category -> severity + Kinetics-400 member labels (validated against the label file)
CATEGORIES = {
    "fighting": {"display": "Fighting", "severity": "HIGH", "labels": [
        "slapping", "punching person (boxing)", "headbutting", "wrestling",
        "drop kicking", "high kick", "side kick", "sword fighting",
    ]},
    "falling": {"display": "Fall", "severity": "HIGH", "labels": ["faceplanting"]},
    "running": {"display": "Running", "severity": "MEDIUM", "labels": [
        "jogging", "running on treadmill", "hurdling", "parkour",
    ]},
    "climbing": {"display": "Climbing", "severity": "MEDIUM", "labels": [
        "climbing a rope", "climbing ladder", "climbing tree", "rock climbing", "abseiling", "vault",
    ]},
    "vandalism": {"display": "Spray painting", "severity": "MEDIUM", "labels": ["spray painting"]},
}

CLIP_LEN = 32                 # frames fed to the fast pathway
CLIP_SECONDS = 32 * 2 / 30.0  # temporal extent of a Kinetics training clip
SIDE_SIZE = 256
INFERENCE_INTERVAL = float(os.environ.get("ACTIVITY_INTERVAL_S", "0.5"))
MIN_SCORE = float(os.environ.get("ACTIVITY_MIN_SCORE", "0.30"))
MIN_SCORE_HIGH = float(os.environ.get("ACTIVITY_MIN_SCORE_HIGH", "0.50"))
EMA_ALPHA = 0.5
IDLE_CLEAR_S = 1.0            # no person for this long -> clear stream
MIN_PERSON_PX = int(os.environ.get("ACTIVITY_MIN_PERSON_PX", "100"))   # longer box side
MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)


def load_kinetics_labels(path):
    with open(path, "r") as f:
        label_map = json.load(f)
    labels = [""] * 400
    for name, idx in label_map.items():
        if idx < 400:
            labels[idx] = name.strip().strip('"').replace("_", " ")
    return labels


def build_category_index(labels):
    """category -> np.array of label indices. Raises if a configured label is missing."""
    lookup = {l: i for i, l in enumerate(labels)}
    index = {}
    for cat, spec in CATEGORIES.items():
        missing = [l for l in spec["labels"] if l not in lookup]
        if missing:
            raise ValueError(f"category '{cat}' references unknown Kinetics labels: {missing}")
        index[cat] = np.array([lookup[l] for l in spec["labels"]], dtype=np.int64)
    return index


def square_roi(boxes, frame_shape, margin=1.3):
    """Square region (x1,y1,x2,y2) covering all boxes with a margin, clipped to the frame."""
    h, w = frame_shape[:2]
    b = np.asarray(boxes, dtype=np.float32)
    x1, y1 = b[:, 0].min(), b[:, 1].min()
    x2, y2 = b[:, 2].max(), b[:, 3].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * margin
    side = min(max(side, 64.0), float(max(w, h)))
    x1, y1 = cx - side / 2, cy - side / 2
    x1 = min(max(0.0, x1), max(0.0, w - side))
    y1 = min(max(0.0, y1), max(0.0, h - side))
    x2, y2 = min(float(w), x1 + side), min(float(h), y1 + side)
    return int(x1), int(y1), int(x2), int(y2)


class _Stream:
    def __init__(self):
        self.frames = deque(maxlen=160)   # (timestamp, 256x256 RGB uint8)
        self.last_person_ts = 0.0
        self.last_inference = 0.0
        self.last_frame_ts = None      # newest frame used by the last inference
        self.ema = None
        self.results = []
        self.raw_top = []


class ActionRecognizer:
    def __init__(self, enabled=True):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._labels = []
        self._cat_index = {}
        self._streams = {}
        self._lock = threading.Lock()
        self._ready = False
        self._loading = enabled
        self.enabled = enabled
        self.load_error = None
        self.last_inference_ms = None
        if enabled:
            threading.Thread(target=self._run, daemon=True, name="slowfast").start()

    # ── Status ───────────────────────────────────────────
    @property
    def is_ready(self):
        return self._ready

    def status(self):
        return {"enabled": self.enabled, "ready": self._ready, "loading": self._loading,
                "device": self._device, "error": self.load_error,
                "interval_s": INFERENCE_INTERVAL, "min_score": MIN_SCORE,
                "min_score_high": MIN_SCORE_HIGH, "min_person_px": MIN_PERSON_PX,
                "last_inference_ms": self.last_inference_ms,
                "categories": {k: v["severity"] for k, v in CATEGORIES.items()}}

    # ── Input ────────────────────────────────────────────
    def push_frame(self, frame_bgr, stream="default", person_boxes=(), ts=None):
        """Buffer a person-centred crop. Cheap (one resize); call from the camera worker."""
        if not self._ready:
            return
        ts = ts or time.time()
        person_boxes = [b for b in person_boxes if max(b[2] - b[0], b[3] - b[1]) >= MIN_PERSON_PX]
        with self._lock:
            st = self._streams.setdefault(stream, _Stream())
        if not person_boxes:
            if ts - st.last_person_ts > IDLE_CLEAR_S:
                with self._lock:
                    st.frames.clear()
                    st.ema, st.results, st.raw_top, st.last_frame_ts = None, [], [], None
            return
        x1, y1, x2, y2 = square_roi(person_boxes, frame_bgr.shape)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return
        crop = cv2.cvtColor(cv2.resize(crop, (SIDE_SIZE, SIDE_SIZE)), cv2.COLOR_BGR2RGB)
        with self._lock:
            st.last_person_ts = ts
            st.frames.append((ts, crop))

    def get_actions(self, stream="default"):
        with self._lock:
            st = self._streams.get(stream)
            return list(st.results) if st else []

    def get_raw_top(self, stream="default"):
        with self._lock:
            st = self._streams.get(stream)
            return list(st.raw_top) if st else []

    def drop_stream(self, stream):
        with self._lock:
            self._streams.pop(stream, None)

    # ── Model + scheduler ────────────────────────────────
    def _load(self):
        print("🧠 Loading SlowFast R50 (Kinetics-400)...", flush=True)
        model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
        self._model = model.to(self._device).eval()
        self._labels = load_kinetics_labels(os.path.join(os.path.dirname(__file__), "kinetics_400_labels.json"))
        self._cat_index = build_category_index(self._labels)
        self._ready = True
        print(f"  ✅ SlowFast R50 on {self._device}; categories: {list(CATEGORIES)}", flush=True)

    def _run(self):
        try:
            self._load()
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            print(f"  ❌ SlowFast load error — activity recognition DISABLED: {self.load_error}", flush=True)
            return
        finally:
            self._loading = False

        while True:
            now = time.time()
            due = None
            with self._lock:
                for name, st in self._streams.items():
                    if (self.clip_ready(st) and now - st.last_inference >= INFERENCE_INTERVAL
                            and (due is None or st.last_inference < due[1].last_inference)):
                        due = (name, st)
                if due:
                    frames = list(due[1].frames)
                    due[1].last_inference = now
            if not due:
                time.sleep(0.02)
                continue
            try:
                self._infer(due[1], frames)
            except Exception as e:
                print(f"SlowFast inference error on {due[0]}: {e}", flush=True)

    @staticmethod
    def clip_ready(st):
        """Enough history for a clip, and new frames since the last inference."""
        return (len(st.frames) >= 2 and st.frames[-1][0] - st.frames[0][0] >= CLIP_SECONDS * 0.5
                and st.frames[-1][0] != st.last_frame_ts)

    @staticmethod
    def sample_clip(frames, end_ts):
        """Pick CLIP_LEN frames evenly over [end_ts - CLIP_SECONDS, end_ts] (nearest frame, repeats allowed)."""
        ts = np.array([t for t, _ in frames])
        targets = np.linspace(end_ts - CLIP_SECONDS, end_ts, CLIP_LEN)
        idx = np.clip(np.searchsorted(ts, targets), 0, len(ts) - 1)
        return np.stack([frames[i][1] for i in idx])

    def _infer(self, st, frames):
        t0 = time.perf_counter()
        clip = self.sample_clip(frames, frames[-1][0]).astype(np.float32) / 255.0   # T,H,W,C
        clip = (clip - MEAN) / STD
        fast = torch.from_numpy(clip).permute(3, 0, 1, 2).contiguous()              # C,T,H,W
        slow = fast[:, ::4]
        with torch.no_grad():
            logits = self._model([slow.unsqueeze(0).to(self._device), fast.unsqueeze(0).to(self._device)])
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        self.last_inference_ms = round((time.perf_counter() - t0) * 1000, 1)

        scores = np.array([probs[self._cat_index[c]].sum() for c in CATEGORIES], dtype=np.float32)
        top = np.argsort(probs)[::-1][:3]
        with self._lock:
            st.last_frame_ts = frames[-1][0]
            prev = st.ema if st.ema is not None else np.zeros_like(scores)
            st.ema = EMA_ALPHA * scores + (1 - EMA_ALPHA) * prev
            results = []
            for (cat, spec), score in zip(CATEGORIES.items(), st.ema):
                if score < (MIN_SCORE_HIGH if spec["severity"] == "HIGH" else MIN_SCORE):
                    continue
                member = spec["labels"][int(np.argmax(probs[self._cat_index[cat]]))]
                results.append({"action": spec["display"], "category": cat, "severity": spec["severity"],
                                "confidence": round(float(score), 3), "raw_label": member})
            st.results = sorted(results, key=lambda r: -r["confidence"])
            st.raw_top = [{"label": self._labels[i], "p": round(float(probs[i]), 3)} for i in top]
