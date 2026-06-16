# demo_video.py — Demo mode: loop sample video instead of live camera
"""
When DEMO_MODE=true, this module provides a video source that loops
a sample video file endlessly, replacing the live camera feed.

This lets the system run 24/7 for portfolio demos without needing
a physical camera.

Usage:
    # Set environment variable before running:
    # DEMO_MODE=true python app.py
    #
    # Or specify a custom video:
    # DEMO_VIDEO=path/to/video.mp4 DEMO_MODE=true python app.py
"""

import os
import threading
import time
import urllib.request

import cv2

# ── Configuration ────────────────────────────────────────────
DEMO_MODE = os.environ.get("DEMO_MODE", "").lower() in ("true", "1", "yes")

# Default sample video — a freely licensed pedestrian walking video
# from the VIRAT dataset (public domain surveillance footage)
_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "demo_data")
_DEFAULT_VIDEO = os.path.join(_SAMPLE_DIR, "sample_pedestrian.mp4")

# Custom video path from environment
DEMO_VIDEO_PATH = os.environ.get("DEMO_VIDEO", _DEFAULT_VIDEO)

# Publicly available sample pedestrian video (Creative Commons)
# This is a short clip of people walking — perfect for demonstrating the system
_SAMPLE_VIDEO_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
    "person-bicycle-car-detection.mp4"
)


def ensure_sample_video():
    """Download sample video if it doesn't exist."""
    os.makedirs(_SAMPLE_DIR, exist_ok=True)

    if os.path.exists(DEMO_VIDEO_PATH):
        return DEMO_VIDEO_PATH

    # If custom path doesn't exist and it's not the default, warn
    if DEMO_VIDEO_PATH != _DEFAULT_VIDEO:
        print(f"  ⚠ Demo video not found at '{DEMO_VIDEO_PATH}', "
              f"downloading sample...", flush=True)

    if os.path.exists(_DEFAULT_VIDEO):
        return _DEFAULT_VIDEO

    print(f"  📥 Downloading sample demo video...", flush=True)
    try:
        urllib.request.urlretrieve(_SAMPLE_VIDEO_URL, _DEFAULT_VIDEO)
        file_size = os.path.getsize(_DEFAULT_VIDEO)
        print(f"  ✅ Sample video downloaded ({file_size / 1024 / 1024:.1f} MB)",
              flush=True)
        return _DEFAULT_VIDEO
    except Exception as e:
        print(f"  ❌ Failed to download sample video: {e}", flush=True)
        print(f"  💡 Place a .mp4 file at: {_DEFAULT_VIDEO}", flush=True)
        return None


class DemoVideoCapture:
    """
    Drop-in replacement for cv2.VideoCapture that loops a video file.
    Matches the OpenCV VideoCapture API so it works with existing code.
    """

    def __init__(self, video_path: str = None):
        self.video_path = video_path or DEMO_VIDEO_PATH
        self._cap = None
        self._lock = threading.Lock()
        self._opened = False
        self._fps = 30
        self._frame_delay = 1.0 / self._fps
        self._last_read = 0

        self._open()

    def _open(self):
        """Open (or reopen) the video file."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass

        if not os.path.exists(self.video_path):
            print(f"  ❌ Demo video file not found: {self.video_path}", flush=True)
            self._opened = False
            return

        self._cap = cv2.VideoCapture(self.video_path)
        self._opened = self._cap.isOpened()

        if self._opened:
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30
            self._frame_delay = 1.0 / self._fps
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / self._fps if self._fps > 0 else 0
            print(f"  🎬 Demo video loaded: {self.video_path} "
                  f"({total_frames} frames, {duration:.1f}s, {self._fps:.0f} FPS)",
                  flush=True)

    def isOpened(self):
        return self._opened

    def read(self):
        """Read next frame, looping back to start when video ends."""
        if not self._opened or self._cap is None:
            return False, None

        # Throttle to video FPS (don't blast frames faster than real-time)
        now = time.time()
        elapsed = now - self._last_read
        if elapsed < self._frame_delay:
            time.sleep(self._frame_delay - elapsed)

        with self._lock:
            ret, frame = self._cap.read()

            if not ret:
                # End of video — loop back to start
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()

                if not ret:
                    # Video is broken, try reopening
                    self._open()
                    return False, None

        self._last_read = time.time()

        # Resize to standard resolution for consistent processing
        if frame is not None:
            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

        return True, frame

    def release(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._opened = False

    def set(self, prop, value):
        """Stub for cv2.VideoCapture.set() — mostly ignored in demo mode."""
        pass

    def get(self, prop):
        """Stub for cv2.VideoCapture.get()."""
        if self._cap is not None:
            return self._cap.get(prop)
        return 0

    def grab(self):
        """Stub for cv2.VideoCapture.grab()."""
        if self._cap is not None:
            return self._cap.grab()
        return False


def get_demo_camera():
    """Create a DemoVideoCapture with the configured video file."""
    video_path = ensure_sample_video()
    if video_path is None:
        print("  ⚠ No demo video available — falling back to camera 0", flush=True)
        return cv2.VideoCapture(0)
    return DemoVideoCapture(video_path)
