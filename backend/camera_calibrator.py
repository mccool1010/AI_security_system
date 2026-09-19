# camera_calibrator.py — Intrinsic camera calibration, lens distortion, perspective correction
"""
Manages per-camera intrinsic parameters so pixel measurements can be
converted to real-world metric values accurately.

Three calibration modes (best → worst accuracy):
  1. Checkerboard — OpenCV calibrateCamera() from checkerboard images
  2. Manual       — user provides FOV, focal length, or sensor dimensions
  3. Preset       — pre-configured profiles for common camera types

All modes produce:
  - camera_matrix  (3×3):  [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  - dist_coeffs    (1×5):  [k1, k2, p1, p2, k3]  (radial + tangential)
  - fov_h_deg, fov_v_deg

Usage:
    calib = CameraCalibrator("cam_local")
    calib.load_preset("webcam_720p")
    undistorted_kpts = calib.undistort_points(raw_kpts, frame_shape)
    focal_px = calib.get_focal_length_px()
"""

import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

# ── Storage directory for calibration JSON files ──────────────────
_CALIB_DIR = os.path.join(os.path.dirname(__file__), "calibration_data")
os.makedirs(_CALIB_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
#  CAMERA PRESETS
#  Pre-configured intrinsic profiles for common camera types.
#  Values are based on published specs and empirical measurements.
# ═══════════════════════════════════════════════════════════════════

CAMERA_PRESETS = {
    # ── Laptop / USB webcams ─────────────────────────────────────
    "webcam_720p": {
        "description": "Generic 720p laptop webcam (~60° H-FOV)",
        "fov_h_deg": 60.0,
        "resolution": (1280, 720),
        "dist_coeffs": [0.05, -0.08, 0.0, 0.0, 0.02],  # mild barrel
    },
    "webcam_1080p": {
        "description": "Generic 1080p USB webcam (~65° H-FOV)",
        "fov_h_deg": 65.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.06, -0.10, 0.0, 0.0, 0.03],
    },
    "webcam_4k": {
        "description": "4K USB webcam (~70° H-FOV)",
        "fov_h_deg": 70.0,
        "resolution": (3840, 2160),
        "dist_coeffs": [0.04, -0.06, 0.0, 0.0, 0.01],
    },
    "logitech_c920": {
        "description": "Logitech C920 / C922 (78° H-FOV, 1080p)",
        "fov_h_deg": 78.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.08, -0.15, 0.001, 0.001, 0.05],
    },
    "logitech_brio": {
        "description": "Logitech Brio 4K (90° H-FOV)",
        "fov_h_deg": 90.0,
        "resolution": (3840, 2160),
        "dist_coeffs": [0.12, -0.20, 0.001, 0.001, 0.08],
    },

    # ── CCTV / IP cameras ────────────────────────────────────────
    "cctv_dome_2mp": {
        "description": "Standard 2MP dome CCTV (2.8mm lens, ~100° H-FOV)",
        "fov_h_deg": 100.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.15, -0.25, 0.002, 0.002, 0.10],
    },
    "cctv_dome_4mp": {
        "description": "4MP dome CCTV (2.8mm lens, ~100° H-FOV)",
        "fov_h_deg": 100.0,
        "resolution": (2560, 1440),
        "dist_coeffs": [0.15, -0.25, 0.002, 0.002, 0.10],
    },
    "cctv_bullet_4mm": {
        "description": "Bullet CCTV (4mm lens, ~80° H-FOV)",
        "fov_h_deg": 80.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.10, -0.18, 0.001, 0.001, 0.06],
    },
    "cctv_varifocal": {
        "description": "Varifocal CCTV (2.8-12mm, ~60° at 6mm)",
        "fov_h_deg": 60.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.08, -0.12, 0.001, 0.001, 0.04],
    },
    "hikvision_dome": {
        "description": "Hikvision DS-2CD series dome (2.8mm, ~102° H-FOV)",
        "fov_h_deg": 102.0,
        "resolution": (2560, 1440),
        "dist_coeffs": [0.16, -0.28, 0.002, 0.002, 0.12],
    },
    "dahua_dome": {
        "description": "Dahua IPC-HDBW series (2.8mm, ~104° H-FOV)",
        "fov_h_deg": 104.0,
        "resolution": (2560, 1440),
        "dist_coeffs": [0.17, -0.30, 0.002, 0.002, 0.13],
    },

    # ── PTZ cameras ──────────────────────────────────────────────
    "ptz_wide": {
        "description": "PTZ camera at widest zoom (~60° H-FOV)",
        "fov_h_deg": 60.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.03, -0.04, 0.0, 0.0, 0.01],
    },
    "ptz_zoomed": {
        "description": "PTZ camera at moderate zoom (~20° H-FOV)",
        "fov_h_deg": 20.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.01, -0.01, 0.0, 0.0, 0.0],  # near zero at zoom
    },

    # ── Doorbell / wide-angle ────────────────────────────────────
    "ring_doorbell": {
        "description": "Ring Video Doorbell (~155° diagonal FOV, ~140° H-FOV)",
        "fov_h_deg": 140.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.35, -0.60, 0.005, 0.005, 0.30],  # heavy barrel
    },
    "nest_doorbell": {
        "description": "Google Nest Doorbell (~145° diagonal FOV, ~130° H-FOV)",
        "fov_h_deg": 130.0,
        "resolution": (1600, 1200),
        "dist_coeffs": [0.30, -0.50, 0.004, 0.004, 0.25],
    },
    "wide_angle_action": {
        "description": "GoPro / action cam style (~120° H-FOV)",
        "fov_h_deg": 120.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.25, -0.45, 0.003, 0.003, 0.20],
    },

    # ── Phone cameras ────────────────────────────────────────────
    "phone_main": {
        "description": "Typical smartphone main camera (~75° H-FOV)",
        "fov_h_deg": 75.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.04, -0.06, 0.0, 0.0, 0.01],
    },
    "phone_ultrawide": {
        "description": "Smartphone ultrawide camera (~120° H-FOV)",
        "fov_h_deg": 120.0,
        "resolution": (1920, 1080),
        "dist_coeffs": [0.22, -0.40, 0.003, 0.003, 0.18],
    },
}


# ═══════════════════════════════════════════════════════════════════
#  CAMERA CALIBRATOR
# ═══════════════════════════════════════════════════════════════════

class CameraCalibrator:
    """Per-camera intrinsic calibration manager."""

    def __init__(self, camera_id: str = "default"):
        self.camera_id = camera_id
        self._calibrated = False
        self._method = "uncalibrated"

        # Intrinsic parameters
        self._camera_matrix = None   # 3×3 np.array
        self._dist_coeffs = None     # 1×5 np.array
        self._fov_h_deg = None
        self._fov_v_deg = None
        self._resolution = None      # (width, height) the calibration was done at
        self._rms_error = None       # reprojection error from checkerboard

        # Cached undistortion maps (computed once per resolution)
        self._undist_map1 = None
        self._undist_map2 = None
        self._undist_resolution = None
        self._new_camera_matrix = None

        # Try to load saved calibration
        self._try_load()

    # ── Preset calibration ───────────────────────────────────────

    def load_preset(self, preset_name: str) -> dict:
        """Load a pre-configured camera profile."""
        if preset_name not in CAMERA_PRESETS:
            available = list(CAMERA_PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        preset = CAMERA_PRESETS[preset_name]
        fov_h = preset["fov_h_deg"]
        res_w, res_h = preset["resolution"]
        dist = preset["dist_coeffs"]

        # Compute focal length in pixels from FOV
        fx = res_w / (2.0 * math.tan(math.radians(fov_h / 2.0)))
        fy = fx  # assume square pixels
        cx, cy = res_w / 2.0, res_h / 2.0

        self._camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)

        self._dist_coeffs = np.array(dist, dtype=np.float64).reshape(1, -1)
        self._fov_h_deg = fov_h
        self._fov_v_deg = 2.0 * math.degrees(math.atan(res_h / (2.0 * fy)))
        self._resolution = (res_w, res_h)
        self._calibrated = True
        self._method = f"preset:{preset_name}"
        self._rms_error = None

        # Invalidate cached maps
        self._undist_map1 = None
        self._undist_map2 = None

        self.save()
        info = self.get_calibration_info()
        print(f"  📷 Camera '{self.camera_id}' calibrated via preset '{preset_name}' "
              f"(FOV={fov_h:.0f}°, fx={fx:.1f}px)", flush=True)
        return info

    # ── Manual parameter calibration ─────────────────────────────

    def set_manual_params(
        self,
        frame_width: int,
        frame_height: int,
        fov_h_deg: float = None,
        focal_mm: float = None,
        sensor_w_mm: float = None,
        dist_coeffs: list = None,
    ) -> dict:
        """
        Set camera parameters manually.

        Provide one of:
          - fov_h_deg: horizontal field of view in degrees
          - focal_mm + sensor_w_mm: focal length and sensor width in mm

        Optionally provide dist_coeffs [k1, k2, p1, p2, k3].
        """
        if fov_h_deg is None and (focal_mm is None or sensor_w_mm is None):
            raise ValueError("Provide either fov_h_deg, or both focal_mm and sensor_w_mm")

        if fov_h_deg is None:
            # Convert mm to FOV
            fov_h_deg = 2.0 * math.degrees(math.atan(sensor_w_mm / (2.0 * focal_mm)))

        # Clamp to sane range
        fov_h_deg = max(5.0, min(180.0, fov_h_deg))

        fx = frame_width / (2.0 * math.tan(math.radians(fov_h_deg / 2.0)))
        fy = fx
        cx, cy = frame_width / 2.0, frame_height / 2.0

        self._camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)

        if dist_coeffs and len(dist_coeffs) >= 4:
            while len(dist_coeffs) < 5:
                dist_coeffs.append(0.0)
            self._dist_coeffs = np.array(dist_coeffs[:5], dtype=np.float64).reshape(1, -1)
        else:
            self._dist_coeffs = np.zeros((1, 5), dtype=np.float64)

        self._fov_h_deg = fov_h_deg
        self._fov_v_deg = 2.0 * math.degrees(math.atan(frame_height / (2.0 * fy)))
        self._resolution = (frame_width, frame_height)
        self._calibrated = True
        self._method = "manual"
        self._rms_error = None

        self._undist_map1 = None
        self._undist_map2 = None

        self.save()
        info = self.get_calibration_info()
        print(f"  📷 Camera '{self.camera_id}' calibrated manually "
              f"(FOV={fov_h_deg:.1f}°, fx={fx:.1f}px)", flush=True)
        return info

    # ── Checkerboard calibration ─────────────────────────────────

    def calibrate_from_checkerboard(
        self,
        images: list,
        board_size: tuple = (9, 6),
        square_size_mm: float = 25.0,
    ) -> dict:
        """
        Full OpenCV camera calibration from checkerboard images.

        Args:
            images: list of BGR np.ndarray frames containing the checkerboard
            board_size: (columns, rows) of internal corners
            square_size_mm: size of each checkerboard square in mm
        """
        if len(images) < 3:
            raise ValueError("Need at least 3 checkerboard images for calibration")

        # Prepare 3D object points
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size_mm

        obj_points = []  # 3D points in real world
        img_points = []  # 2D points in image plane
        img_size = None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            if img_size is None:
                img_size = (gray.shape[1], gray.shape[0])

            found, corners = cv2.findChessboardCorners(gray, board_size, None)
            if found:
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )
                obj_points.append(objp)
                img_points.append(corners_refined)
                print(f"    ✓ Checkerboard found in image {i + 1}", flush=True)
            else:
                print(f"    ✗ No checkerboard in image {i + 1}", flush=True)

        if len(obj_points) < 3:
            raise ValueError(
                f"Only {len(obj_points)} valid checkerboard images found (need ≥ 3)"
            )

        # Calibrate
        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None
        )

        self._camera_matrix = mtx
        self._dist_coeffs = dist
        self._resolution = img_size
        self._rms_error = rms

        # Extract FOV from focal lengths
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        self._fov_h_deg = 2.0 * math.degrees(math.atan(img_size[0] / (2.0 * fx)))
        self._fov_v_deg = 2.0 * math.degrees(math.atan(img_size[1] / (2.0 * fy)))

        self._calibrated = True
        self._method = "checkerboard"

        self._undist_map1 = None
        self._undist_map2 = None

        self.save()
        info = self.get_calibration_info()
        print(f"  📷 Camera '{self.camera_id}' calibrated via checkerboard "
              f"(RMS={rms:.4f}, FOV={self._fov_h_deg:.1f}°, "
              f"fx={fx:.1f}, fy={fy:.1f})", flush=True)
        return info

    # ── Auto-calibrate from FOV guess based on frame shape ───────

    def auto_calibrate_from_frame(self, frame_shape: tuple) -> dict:
        """
        Fallback: estimate parameters from frame dimensions assuming
        a typical webcam FOV (~62°). Better than nothing.
        """
        if self._calibrated:
            return self.get_calibration_info()

        h, w = frame_shape[:2]
        fov_h = 62.0  # conservative webcam default
        fx = w / (2.0 * math.tan(math.radians(fov_h / 2.0)))
        fy = fx
        cx, cy = w / 2.0, h / 2.0

        self._camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float64)
        # Unknown lens: assume no distortion rather than "correcting" with invented coefficients.
        self._dist_coeffs = np.zeros((1, 5), dtype=np.float64)
        self._fov_h_deg = fov_h
        self._fov_v_deg = 2.0 * math.degrees(math.atan(h / (2.0 * fy)))
        self._resolution = (w, h)
        self._calibrated = True
        self._method = "auto_default"
        return self.get_calibration_info()

    # ── Point undistortion ───────────────────────────────────────

    def undistort_points(self, points: np.ndarray, frame_shape: tuple = None) -> np.ndarray:
        """
        Correct lens distortion on a set of (x, y) points.

        Args:
            points: shape (N, 2) or (N, 3) — if 3 cols, col[2] is confidence (preserved)
            frame_shape: (H, W, ...) — used to auto-calibrate if needed

        Returns:
            Corrected points with same shape as input.
        """
        if not self._calibrated:
            if frame_shape is not None:
                self.auto_calibrate_from_frame(frame_shape)
            else:
                return points  # can't do anything

        if points is None or len(points) == 0:
            return points

        pts = np.array(points, dtype=np.float64)
        has_conf = pts.shape[1] >= 3 if pts.ndim == 2 else False

        # Extract x, y
        xy = pts[:, :2].copy() if pts.ndim == 2 else pts.reshape(-1, 2)

        # Scale points if current frame resolution differs from calibration resolution
        scale_x, scale_y = 1.0, 1.0
        if frame_shape is not None and self._resolution is not None:
            actual_w = frame_shape[1] if len(frame_shape) >= 2 else self._resolution[0]
            actual_h = frame_shape[0] if len(frame_shape) >= 2 else self._resolution[1]
            cal_w, cal_h = self._resolution
            scale_x = cal_w / actual_w
            scale_y = cal_h / actual_h
            xy[:, 0] *= scale_x
            xy[:, 1] *= scale_y

        # OpenCV undistortPoints wants shape (N, 1, 2)
        xy_cv = xy.reshape(-1, 1, 2).astype(np.float64)

        corrected = cv2.undistortPoints(
            xy_cv,
            self._camera_matrix,
            self._dist_coeffs,
            P=self._camera_matrix,  # re-project to pixel coords
        )

        corrected = corrected.reshape(-1, 2)

        # Scale back
        if scale_x != 1.0 or scale_y != 1.0:
            corrected[:, 0] /= scale_x
            corrected[:, 1] /= scale_y

        # Reassemble with confidence if present
        if has_conf:
            result = pts.copy()
            result[:, :2] = corrected
            return result
        return corrected

    def undistort_keypoints(self, kpts: np.ndarray, frame_shape: tuple) -> np.ndarray:
        """
        Undistort COCO-format keypoints: shape (17, 3) where col 2 = confidence.
        Only corrects points with confidence > 0.
        """
        if kpts is None or len(kpts) == 0:
            return kpts
        if not self._calibrated:
            self.auto_calibrate_from_frame(frame_shape)

        result = kpts.copy()
        # Find points with nonzero confidence
        valid = result[:, 2] > 0
        if not np.any(valid):
            return result

        corrected = self.undistort_points(result[valid, :2], frame_shape)
        result[valid, :2] = corrected
        return result

    # ── Frame undistortion ───────────────────────────────────────

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """Undistort an entire frame (expensive — use sparingly)."""
        if not self._calibrated:
            return frame

        h, w = frame.shape[:2]
        current_res = (w, h)

        # Rebuild maps if resolution changed
        if self._undist_map1 is None or self._undist_resolution != current_res:
            # Scale camera matrix if resolution differs from calibration
            mtx = self._camera_matrix.copy()
            if self._resolution and self._resolution != current_res:
                sx = w / self._resolution[0]
                sy = h / self._resolution[1]
                mtx[0, :] *= sx
                mtx[1, :] *= sy

            self._new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                mtx, self._dist_coeffs, (w, h), 1, (w, h)
            )
            self._undist_map1, self._undist_map2 = cv2.initUndistortRectifyMap(
                mtx, self._dist_coeffs, None, self._new_camera_matrix, (w, h), cv2.CV_32FC1
            )
            self._undist_resolution = current_res

        return cv2.remap(frame, self._undist_map1, self._undist_map2, cv2.INTER_LINEAR)

    # ── Accessors ────────────────────────────────────────────────

    def get_focal_length_px(self, frame_shape: tuple = None) -> float:
        """Return focal length in pixels (fx), scaled to current frame if needed."""
        if not self._calibrated:
            if frame_shape is not None:
                self.auto_calibrate_from_frame(frame_shape)
            else:
                return 554.0  # fallback: 640px * 0.866

        fx = float(self._camera_matrix[0, 0])

        # Scale if resolution differs
        if frame_shape is not None and self._resolution is not None:
            actual_w = frame_shape[1]
            cal_w = self._resolution[0]
            fx *= (actual_w / cal_w)

        return fx

    def get_principal_point(self, frame_shape: tuple = None) -> tuple:
        """Return (cx, cy) principal point, scaled to current frame."""
        if not self._calibrated:
            if frame_shape is not None:
                self.auto_calibrate_from_frame(frame_shape)
                return (frame_shape[1] / 2.0, frame_shape[0] / 2.0)
            return (320.0, 240.0)

        cx = float(self._camera_matrix[0, 2])
        cy = float(self._camera_matrix[1, 2])

        if frame_shape is not None and self._resolution is not None:
            cx *= frame_shape[1] / self._resolution[0]
            cy *= frame_shape[0] / self._resolution[1]

        return (cx, cy)

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def calibration_method(self) -> str:
        return self._method

    @property
    def distortion_magnitude(self) -> float:
        """Return a scalar magnitude of lens distortion (|k1| + |k2|)."""
        if self._dist_coeffs is None:
            return 0.0
        d = self._dist_coeffs.flatten()
        return abs(d[0]) + abs(d[1])

    def get_calibration_info(self) -> dict:
        """Return full calibration state as a dict (JSON-safe)."""
        info = {
            "camera_id": self.camera_id,
            "calibrated": self._calibrated,
            "method": self._method,
            "fov_h_deg": self._fov_h_deg,
            "fov_v_deg": self._fov_v_deg,
            "resolution": list(self._resolution) if self._resolution else None,
            "rms_error": self._rms_error,
            "distortion_magnitude": self.distortion_magnitude,
        }
        if self._camera_matrix is not None:
            info["fx"] = float(self._camera_matrix[0, 0])
            info["fy"] = float(self._camera_matrix[1, 1])
            info["cx"] = float(self._camera_matrix[0, 2])
            info["cy"] = float(self._camera_matrix[1, 2])
        if self._dist_coeffs is not None:
            info["dist_coeffs"] = self._dist_coeffs.flatten().tolist()
        return info

    # ── Edge proximity scoring ───────────────────────────────────

    def point_distortion_severity(self, px: float, py: float, frame_shape: tuple) -> float:
        """
        Estimate how much distortion affects a point based on its
        distance from the principal point. Returns 0.0 (center, no distortion)
        to 1.0 (corner, maximum distortion).

        Useful for modulating height confidence — points near frame edges
        are less reliable even after undistortion.
        """
        if not self._calibrated:
            self.auto_calibrate_from_frame(frame_shape)

        cx, cy = self.get_principal_point(frame_shape)
        h, w = frame_shape[:2]

        # Normalized distance from center (0 = center, 1 = corner)
        dx = (px - cx) / (w / 2.0)
        dy = (py - cy) / (h / 2.0)
        r = math.sqrt(dx * dx + dy * dy)

        # Weight by distortion magnitude
        severity = min(1.0, r * self.distortion_magnitude * 2.0)
        return severity

    # ── Persistence ──────────────────────────────────────────────

    def save(self):
        """Save calibration to JSON file."""
        if not self._calibrated:
            return

        path = os.path.join(_CALIB_DIR, f"{self.camera_id}.json")
        data = {
            "camera_id": self.camera_id,
            "method": self._method,
            "fov_h_deg": self._fov_h_deg,
            "fov_v_deg": self._fov_v_deg,
            "resolution": list(self._resolution) if self._resolution else None,
            "rms_error": self._rms_error,
            "camera_matrix": self._camera_matrix.tolist() if self._camera_matrix is not None else None,
            "dist_coeffs": self._dist_coeffs.tolist() if self._dist_coeffs is not None else None,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"  ⚠ Failed to save calibration for '{self.camera_id}': {e}", flush=True)

    def _try_load(self):
        """Load calibration from saved JSON if available."""
        path = os.path.join(_CALIB_DIR, f"{self.camera_id}.json")
        if not os.path.exists(path):
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            if data.get("camera_matrix") is not None:
                self._camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
                self._dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
                self._fov_h_deg = data.get("fov_h_deg")
                self._fov_v_deg = data.get("fov_v_deg")
                self._resolution = tuple(data["resolution"]) if data.get("resolution") else None
                self._rms_error = data.get("rms_error")
                self._method = data.get("method", "loaded")
                self._calibrated = True
                print(f"  📷 Loaded saved calibration for '{self.camera_id}' "
                      f"(method={self._method})", flush=True)
        except Exception as e:
            print(f"  ⚠ Failed to load calibration for '{self.camera_id}': {e}", flush=True)

    def reset(self):
        """Reset calibration to uncalibrated state."""
        self._calibrated = False
        self._method = "uncalibrated"
        self._camera_matrix = None
        self._dist_coeffs = None
        self._fov_h_deg = None
        self._fov_v_deg = None
        self._resolution = None
        self._rms_error = None
        self._undist_map1 = None
        self._undist_map2 = None

        # Delete saved file
        path = os.path.join(_CALIB_DIR, f"{self.camera_id}.json")
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass

        print(f"  📷 Calibration reset for '{self.camera_id}'", flush=True)


# ── Module-level convenience ─────────────────────────────────────

def list_presets() -> list:
    """Return list of available camera preset names and descriptions."""
    return [
        {"name": name, "description": preset["description"],
         "fov_h_deg": preset["fov_h_deg"], "resolution": list(preset["resolution"])}
        for name, preset in CAMERA_PRESETS.items()
    ]
