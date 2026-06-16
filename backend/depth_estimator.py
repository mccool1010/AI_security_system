# depth_estimator.py — Monocular depth estimation via MiDaS for metric distance
"""
Provides per-pixel relative depth maps using MiDaS v3.1, then converts
to metric depth using a calibration scale factor.

Replaces the broken foot_y / frame_h heuristic with real neural-network
depth estimation that works for ANY camera angle (level, elevated, angled).

Usage:
    depth_est = DepthEstimator(model_type="DPT_Large")
    depth_est.push_frame(frame)                # async inference
    depth_map = depth_est.get_depth_map()       # cached result
    dist = depth_est.get_depth_at_points([(x, y)], frame_shape)
"""

import threading
import time

import cv2
import numpy as np
import torch

# ── Configuration ────────────────────────────────────────────────
# Model variants (accuracy vs speed):
#   "DPT_Large"    — best accuracy, ~300MB, ~50ms/frame on GPU
#   "DPT_Hybrid"   — good accuracy, ~200MB, ~35ms/frame on GPU
#   "MiDaS_small"  — fastest, ~80MB, ~15ms/frame on GPU, ~40ms CPU
SUPPORTED_MODELS = ["DPT_Large", "DPT_Hybrid", "MiDaS_small"]

# How often to run depth inference (seconds)
# Depth doesn't change as fast as pose, so we can afford a lower rate
DEPTH_INFERENCE_INTERVAL = 0.3

# Minimum time between full inference runs (prevents overload)
MIN_INFERENCE_GAP = 0.1


class DepthEstimator:
    """
    Monocular depth estimation using MiDaS v3.1.

    Thread-safe: push_frame() is cheap (stores frame ref).
    Inference runs in a dedicated background thread.
    """

    def __init__(self, model_type: str = "DPT_Large"):
        if model_type not in SUPPORTED_MODELS:
            print(f"  ⚠ Unknown MiDaS model '{model_type}', falling back to MiDaS_small",
                  flush=True)
            model_type = "MiDaS_small"

        self._model_type = model_type
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._transform = None
        self._ready = False
        self._loading = True
        self._load_error = None

        # Frame buffer and cached results
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._depth_map = None          # Raw relative depth (H×W float32)
        self._depth_map_shape = None    # Shape of source frame
        self._depth_lock = threading.Lock()
        self._last_inference_time = 0
        self._inference_running = False

        # Metric calibration
        self._scale_factor = None       # multiplier: metric_depth = relative * scale
        self._scale_calibrated = False

        # Load model in background
        t = threading.Thread(target=self._load_model, daemon=True)
        t.start()

    # ── Model loading ────────────────────────────────────────────

    def _load_model(self):
        """Load MiDaS model from torch.hub (downloads on first use)."""
        try:
            print(f"🔍 Loading MiDaS {self._model_type} for depth estimation...", flush=True)

            self._model = torch.hub.load(
                "intel-isl/MiDaS",
                self._model_type,
                trust_repo=True,
            )
            self._model.to(self._device)
            self._model.eval()

            # Load the appropriate transform
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
            )

            if self._model_type == "DPT_Large" or self._model_type == "DPT_Hybrid":
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform

            self._ready = True
            self._loading = False
            print(f"  ✅ MiDaS {self._model_type} loaded on {self._device}", flush=True)

        except Exception as e:
            self._load_error = str(e)
            self._loading = False
            print(f"  ❌ MiDaS load error: {e}", flush=True)
            import traceback
            traceback.print_exc()

    # ── Frame input ──────────────────────────────────────────────

    def push_frame(self, bgr_frame: np.ndarray):
        """Store latest frame for async depth inference."""
        if not self._ready:
            return
        with self._frame_lock:
            self._latest_frame = bgr_frame

        # Trigger inference if enough time has passed
        now = time.time()
        if (
            not self._inference_running
            and (now - self._last_inference_time) >= DEPTH_INFERENCE_INTERVAL
        ):
            self._inference_running = True
            t = threading.Thread(target=self._run_inference, daemon=True)
            t.start()

    # ── Inference ────────────────────────────────────────────────

    def _run_inference(self):
        """Run MiDaS depth inference on the latest frame."""
        try:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is None:
                return

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self._transform(frame_rgb).to(self._device)

            with torch.no_grad():
                prediction = self._model(input_batch)

                # Resize to original frame size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            # MiDaS outputs inverse depth (closer = larger values)
            # Normalize so that larger = further (intuitive metric depth direction)
            # We keep the relative scale — metric conversion uses scale_factor
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max - depth_min > 1e-6:
                # Invert: MiDaS gives large values for close objects
                # We want large values = far, small = close (like real distance)
                depth_map = (depth_max - depth_map) / (depth_max - depth_min)
            else:
                depth_map = np.ones_like(depth_map) * 0.5

            with self._depth_lock:
                self._depth_map = depth_map.astype(np.float32)
                self._depth_map_shape = frame.shape[:2]

            self._last_inference_time = time.time()

        except Exception as e:
            print(f"  ⚠ MiDaS inference error: {e}", flush=True)
        finally:
            self._inference_running = False

    # ── Depth queries ────────────────────────────────────────────

    def get_depth_map(self) -> np.ndarray:
        """Return the latest relative depth map (H×W float32, 0=close, 1=far)."""
        with self._depth_lock:
            return self._depth_map.copy() if self._depth_map is not None else None

    def get_depth_at_points(
        self,
        points: list,
        frame_shape: tuple = None,
    ) -> list:
        """
        Get relative depth values at specific (x, y) pixel locations.

        Args:
            points: list of (x, y) tuples
            frame_shape: (H, W) of the source frame (for scaling if depth map size differs)

        Returns:
            list of float depth values (0=close, 1=far), or None for invalid points
        """
        with self._depth_lock:
            depth_map = self._depth_map
            map_shape = self._depth_map_shape

        if depth_map is None:
            return [None] * len(points)

        results = []
        h, w = depth_map.shape[:2]

        for pt in points:
            if pt is None or len(pt) < 2:
                results.append(None)
                continue

            px, py = float(pt[0]), float(pt[1])

            # Scale coordinates if frame shape differs from depth map
            if frame_shape is not None and map_shape is not None:
                px *= w / frame_shape[1] if frame_shape[1] > 0 else 1
                py *= h / frame_shape[0] if frame_shape[0] > 0 else 1

            ix = int(round(px))
            iy = int(round(py))

            if 0 <= ix < w and 0 <= iy < h:
                # Sample a small region (3×3) for stability
                y_lo = max(0, iy - 1)
                y_hi = min(h, iy + 2)
                x_lo = max(0, ix - 1)
                x_hi = min(w, ix + 2)
                region = depth_map[y_lo:y_hi, x_lo:x_hi]
                results.append(float(np.median(region)))
            else:
                results.append(None)

        return results

    def get_metric_depth_at_points(
        self,
        points: list,
        frame_shape: tuple = None,
    ) -> list:
        """
        Get metric depth (meters) at specific pixel locations.
        Requires scale calibration via calibrate_scale().

        Returns list of float (meters) or None for uncalibrated/invalid.
        """
        if not self._scale_calibrated or self._scale_factor is None:
            return [None] * len(points)

        relative = self.get_depth_at_points(points, frame_shape)
        return [
            round(r * self._scale_factor, 2) if r is not None else None
            for r in relative
        ]

    # ── Scale calibration ────────────────────────────────────────

    def calibrate_scale(
        self,
        ref_points: list,
        ref_distance_m: float,
    ):
        """
        Calibrate the relative-to-metric depth scale factor.

        Args:
            ref_points: list of (x, y) pixel coordinates at known distance
            ref_distance_m: the real-world distance in meters to those points
        """
        relative_depths = self.get_depth_at_points(ref_points)
        valid = [d for d in relative_depths if d is not None and d > 0.01]

        if not valid:
            print("  ⚠ Depth calibration failed: no valid depth at reference points",
                  flush=True)
            return False

        avg_relative = sum(valid) / len(valid)
        self._scale_factor = ref_distance_m / avg_relative
        self._scale_calibrated = True

        print(f"  ✅ Depth scale calibrated: {avg_relative:.4f} relative "
              f"= {ref_distance_m:.2f}m (scale={self._scale_factor:.2f})", flush=True)
        return True

    def auto_calibrate_from_person(
        self,
        ankle_points: list,
        person_height_px: float,
        person_height_m: float,
        focal_length_px: float,
    ):
        """
        Auto-calibrate depth scale using a detected person of known/estimated height.

        Uses pinhole model: distance = focal * real_height / pixel_height
        Then maps that distance to the relative depth at the person's feet.
        """
        if self._scale_calibrated:
            return True

        if person_height_px < 30 or focal_length_px < 50:
            return False

        # Estimate distance via pinhole model
        estimated_distance = focal_length_px * person_height_m / person_height_px

        # Get relative depth at ankles
        relative_depths = self.get_depth_at_points(ankle_points)
        valid = [d for d in relative_depths if d is not None and d > 0.01]

        if not valid:
            return False

        avg_relative = sum(valid) / len(valid)
        self._scale_factor = estimated_distance / avg_relative
        self._scale_calibrated = True

        print(f"  ✅ Depth auto-calibrated from person: "
              f"distance≈{estimated_distance:.1f}m, scale={self._scale_factor:.2f}",
              flush=True)
        return True

    def get_person_distance(
        self,
        ankle_points: list,
        frame_shape: tuple = None,
    ) -> float:
        """
        Get estimated distance (meters) to a person using depth at their feet.

        Returns float in meters, or None if not available.
        """
        if not self._scale_calibrated:
            return None

        depths = self.get_metric_depth_at_points(ankle_points, frame_shape)
        valid = [d for d in depths if d is not None and d > 0.1]

        if not valid:
            return None

        # Use median for robustness
        return round(float(np.median(valid)), 2)

    # ── Fallback distance estimation (no MiDaS) ─────────────────

    def fallback_distance(
        self,
        person_height_px: float,
        person_height_m: float,
        focal_length_px: float,
    ) -> float:
        """
        Pure pinhole distance estimate (no depth network needed).
        Used when MiDaS isn't ready or depth map unavailable.

        distance = focal × real_height / pixel_height
        """
        if person_height_px < 5 or focal_length_px < 10:
            return None

        dist = focal_length_px * person_height_m / person_height_px
        return round(max(0.3, min(30.0, dist)), 2)

    # ── Status ───────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def is_scale_calibrated(self) -> bool:
        return self._scale_calibrated

    @property
    def model_type(self) -> str:
        return self._model_type

    def get_status(self) -> dict:
        return {
            "ready": self._ready,
            "loading": self._loading,
            "model_type": self._model_type,
            "device": self._device,
            "scale_calibrated": self._scale_calibrated,
            "scale_factor": self._scale_factor,
            "has_depth_map": self._depth_map is not None,
            "load_error": self._load_error,
        }

    def reset_scale(self):
        """Reset metric depth scale calibration."""
        self._scale_factor = None
        self._scale_calibrated = False
