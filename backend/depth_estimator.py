"""MiDaS monocular depth, per camera stream. Opt-in (ENABLE_DEPTH=1).

MiDaS predicts *relative inverse depth* that is only defined up to an unknown
scale and shift per image, so it cannot give metres on its own. The previous
version min-max normalised every frame (so the scale changed frame to frame)
and then "calibrated" it from an assumed 1.70 m person. Metric distance and
height now come from ground-plane geometry (height_geometry.py); this module
only reports a per-person relative depth rank (0 = nearest in the frame,
1 = farthest), which is what MiDaS actually supports.
"""
import os
import threading
import time

import cv2
import numpy as np
import torch

MODEL_TYPE = os.environ.get("DEPTH_MODEL", "MiDaS_small")
INTERVAL = float(os.environ.get("DEPTH_INTERVAL_S", "0.3"))


class DepthEstimator:
    def __init__(self, enabled=True, model_type=MODEL_TYPE):
        self.enabled = enabled
        self.model_type = model_type
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._transform = None
        self._ready = False
        self._loading = enabled
        self.load_error = None
        self.last_inference_ms = None
        self._lock = threading.Lock()
        self._latest = {}      # stream -> (ts, frame)
        self._maps = {}        # stream -> (ts, disparity map HxW float32)
        self._last_run = {}    # stream -> ts
        if enabled:
            threading.Thread(target=self._run, daemon=True, name="midas").start()

    @property
    def is_ready(self):
        return self._ready

    def status(self):
        return {"enabled": self.enabled, "ready": self._ready, "loading": self._loading,
                "model_type": self.model_type, "device": self._device, "error": self.load_error,
                "interval_s": INTERVAL, "last_inference_ms": self.last_inference_ms,
                "output": "relative depth rank (not metric)"}

    def push_frame(self, frame, stream="default", ts=None):
        if self._ready:
            with self._lock:
                self._latest[stream] = (ts or time.time(), frame)

    def drop_stream(self, stream):
        with self._lock:
            for d in (self._latest, self._maps, self._last_run):
                d.pop(stream, None)

    def relative_depth(self, stream, points, frame_shape):
        """Depth rank in [0, 1] (0 = nearest) for each (x, y) point, or None."""
        with self._lock:
            entry = self._maps.get(stream)
        if entry is None:
            return [None] * len(points)
        disp = entry[1]
        mh, mw = disp.shape
        lo, hi = np.percentile(disp, [2, 98])
        out = []
        for p in points:
            if p is None:
                out.append(None)
                continue
            x = int(round(p[0] * mw / frame_shape[1]))
            y = int(round(p[1] * mh / frame_shape[0]))
            if not (0 <= x < mw and 0 <= y < mh) or hi - lo < 1e-6:
                out.append(None)
                continue
            v = float(np.median(disp[max(0, y - 2):y + 3, max(0, x - 2):x + 3]))
            # larger disparity = nearer
            out.append(round(float(np.clip((hi - v) / (hi - lo), 0, 1)), 3))
        return out

    def _load(self):
        print(f"🔍 Loading MiDaS {self.model_type} ...", flush=True)
        self._model = torch.hub.load("intel-isl/MiDaS", self.model_type, trust_repo=True).to(self._device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self._transform = (transforms.dpt_transform if self.model_type.startswith("DPT")
                           else transforms.small_transform)
        self._ready = True
        print(f"  ✅ MiDaS {self.model_type} on {self._device}", flush=True)

    def _run(self):
        try:
            self._load()
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            print(f"  ❌ MiDaS load error — depth DISABLED: {self.load_error}", flush=True)
            return
        finally:
            self._loading = False
        while True:
            now = time.time()
            job = None
            with self._lock:
                for s, (ts, frame) in self._latest.items():
                    last = self._last_run.get(s, 0.0)
                    if now - last >= INTERVAL and (job is None or last < self._last_run.get(job[0], 0.0)):
                        job = (s, ts, frame)
                if job:
                    self._last_run[job[0]] = now
            if job is None:
                time.sleep(0.02)
                continue
            try:
                t0 = time.perf_counter()
                inp = self._transform(cv2.cvtColor(job[2], cv2.COLOR_BGR2RGB)).to(self._device)
                with torch.no_grad():
                    pred = self._model(inp)
                disp = pred.squeeze().float().cpu().numpy()
                self.last_inference_ms = round((time.perf_counter() - t0) * 1000, 1)
                with self._lock:
                    self._maps[job[0]] = (job[1], disp)
            except Exception as e:
                print(f"  ⚠ MiDaS inference error on {job[0]}: {e}", flush=True)
