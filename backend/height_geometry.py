"""Person height from ground-plane geometry (fixed camera, zero roll, pinhole).

Why: the previous estimator derived distance from an *assumed* 1.70 m person
and then height from that distance, so every output was 1.70 m times a pose
factor. Metric height needs real information about the scene. For a fixed
camera that information is its mounting height H and downward tilt θ:

    φ(v)  = θ + atan((v - cy) / fy)      angle below horizontal of image row v
    D     = H / tan(φ_foot)              ground distance to the feet
    h     = H - D · tan(φ_head)          height of the head above the ground

H and θ are fitted once per camera from a few reference observations of a
person of known height standing at different distances. After that, every
person's height is measured independently. Calibration quality is reported as
the leave-one-out RMS error over the reference samples.
"""
import json
import math
import os
import threading

import numpy as np

CALIB_DIR = os.path.join(os.path.dirname(__file__), "calibration_data")
MIN_BOX_PX = 40
EDGE_MARGIN_PX = 3
MEASURABLE_POSES = {"standing", "walking"}


def ray_angle(v, theta, fy, cy):
    return theta + math.atan((v - cy) / fy)


def height_from_rows(v_foot, v_head, H, theta, fy, cy):
    """Returns (height_m, distance_m) or (None, None) when the feet are at/above the horizon."""
    phi_f = ray_angle(v_foot, theta, fy, cy)
    if phi_f <= math.radians(0.5) or phi_f >= math.radians(89.5):
        return None, None
    D = H / math.tan(phi_f)
    h = H - D * math.tan(ray_angle(v_head, theta, fy, cy))
    return h, D


def fit(samples, fy, cy, camera_height_m=None):
    """Fit (H, θ) to samples [{v_foot, v_head, height_m}]. Fixes H when camera_height_m is given.

    Returns dict with camera_height_m, tilt_deg, rms_cm, loo_rms_cm (None if < 3 samples).
    """
    from scipy.optimize import least_squares

    need = 1 if camera_height_m else 2
    if len(samples) < need:
        raise ValueError(f"need at least {need} reference sample(s), got {len(samples)}")

    def solve(subset):
        def resid(p):
            H, th = (camera_height_m, p[0]) if camera_height_m else (p[0], p[1])
            out = []
            for s in subset:
                h, _ = height_from_rows(s["v_foot"], s["v_head"], H, th, fy, cy)
                out.append(10.0 if h is None else h - s["height_m"])
            return out
        if camera_height_m:
            r = least_squares(resid, [math.radians(15)], bounds=([math.radians(-30)], [math.radians(89)]))
            return camera_height_m, float(r.x[0])
        r = least_squares(resid, [2.5, math.radians(15)],
                          bounds=([0.3, math.radians(-30)], [30.0, math.radians(89)]))
        return float(r.x[0]), float(r.x[1])

    H, th = solve(samples)
    errs = []
    for s in samples:
        h, _ = height_from_rows(s["v_foot"], s["v_head"], H, th, fy, cy)
        errs.append((h - s["height_m"]) if h is not None else float("nan"))
    rms = float(np.sqrt(np.nanmean(np.square(errs))))

    loo = None
    if len(samples) >= need + 2:
        loo_errs = []
        for i, s in enumerate(samples):
            Hi, thi = solve(samples[:i] + samples[i + 1:])
            h, _ = height_from_rows(s["v_foot"], s["v_head"], Hi, thi, fy, cy)
            if h is not None:
                loo_errs.append(h - s["height_m"])
        if loo_errs:
            loo = float(np.sqrt(np.mean(np.square(loo_errs))))

    return {
        "camera_height_m": round(H, 4),
        "tilt_deg": round(math.degrees(th), 3),
        "rms_cm": round(rms * 100, 2),
        "loo_rms_cm": round(loo * 100, 2) if loo is not None else None,
        "residuals_cm": [round(e * 100, 2) for e in errs],
    }


def rows_from_track(track, window_s=1.0):
    """Median foot/head rows and x over the last window of a track's boxes."""
    if not track.history:
        return None
    t_end = track.history[-1][0]
    boxes = np.array([b for t, b, _ in track.history if t_end - t <= window_s])
    return {
        "v_head": float(np.median(boxes[:, 1])),
        "v_foot": float(np.median(boxes[:, 3])),
        "x": float(np.median((boxes[:, 0] + boxes[:, 2]) / 2)),
        "n": int(len(boxes)),
    }


def is_measurable(track, frame_shape):
    """Upright, fully visible, large enough to measure."""
    x1, y1, x2, y2 = track.box
    h, _ = frame_shape[:2]
    if track.pose not in MEASURABLE_POSES:
        return False, f"pose '{track.pose}'"
    if y2 - y1 < MIN_BOX_PX:
        return False, "too small"
    if y1 <= EDGE_MARGIN_PX or y2 >= h - EDGE_MARGIN_PX:
        return False, "cut off by frame edge"
    k = track.history[-1][2] if track.history else None
    if k is None or max(k[15][2], k[16][2]) < 0.3 or k[0][2] < 0.3:
        return False, "head or feet not visible"
    return True, None


class GroundCalibration:
    """Per-camera ground-plane calibration, persisted to calibration_data/<cam>_ground.json."""

    def __init__(self, cam_id):
        self.cam_id = cam_id
        self._lock = threading.Lock()
        self.samples = []
        self.params = None      # fit() result + fy, cy, resolution
        self._load()

    @property
    def path(self):
        return os.path.join(CALIB_DIR, f"{self.cam_id}_ground.json")

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.samples = data.get("samples", [])
            self.params = data.get("params")
        except Exception as e:
            print(f"⚠ Could not load ground calibration for {self.cam_id}: {e}", flush=True)

    def _save(self):
        os.makedirs(CALIB_DIR, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"samples": self.samples, "params": self.params}, f, indent=2)

    @property
    def is_calibrated(self):
        return self.params is not None

    def status(self):
        with self._lock:
            return {"calibrated": self.is_calibrated, "params": self.params,
                    "samples": list(self.samples)}

    def add_sample(self, sample):
        with self._lock:
            self.samples.append(sample)
            self._save()
            return len(self.samples)

    def clear(self):
        with self._lock:
            self.samples, self.params = [], None
            if os.path.exists(self.path):
                os.remove(self.path)

    def solve(self, fy, cy, resolution, camera_height_m=None):
        with self._lock:
            # samples were recorded at their own resolution; rescale rows to the current one
            scaled = []
            for s in self.samples:
                ry = resolution[1] / s["resolution"][1]
                scaled.append({"v_foot": s["v_foot"] * ry, "v_head": s["v_head"] * ry,
                               "height_m": s["height_m"]})
            result = fit(scaled, fy, cy, camera_height_m)
            result.update({"fy": fy, "cy": cy, "resolution": list(resolution),
                           "fixed_camera_height": camera_height_m is not None,
                           "n_samples": len(scaled)})
            self.params = result
            self._save()
            return result

    def measure(self, v_foot, v_head, frame_shape):
        p = self.params
        if p is None:
            return None, None
        h_img, w_img = frame_shape[:2]
        ry = h_img / p["resolution"][1]
        fy, cy = p["fy"] * ry, p["cy"] * ry
        return height_from_rows(v_foot, v_head, p["camera_height_m"], math.radians(p["tilt_deg"]), fy, cy)
