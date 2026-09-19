"""Short video clips around events.

A security event is much more useful as a few seconds of video than as one
still. Every camera keeps a rolling buffer of recently captured frames (already
JPEG-encoded, so the buffer costs a few MB rather than hundreds); when an event
fires, a background writer waits for the post-roll and muxes
[event - PRE, event + POST] into an .mp4 next to the screenshot.

Old clips are pruned by age and by total size, so a camera left running cannot
fill the disk.
"""
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

PRE_S = float(os.environ.get("CLIP_PRE_S", "4"))
POST_S = float(os.environ.get("CLIP_POST_S", "4"))
MAX_AGE_DAYS = float(os.environ.get("CLIP_MAX_AGE_DAYS", "30"))
MAX_TOTAL_MB = float(os.environ.get("CLIP_MAX_TOTAL_MB", "2048"))
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


def prune(directory, max_age_days=MAX_AGE_DAYS, max_total_mb=MAX_TOTAL_MB):
    """Delete the oldest clips beyond the age or size budget. Returns how many went."""
    try:
        files = [(os.path.getmtime(os.path.join(directory, f)), os.path.getsize(os.path.join(directory, f)),
                  os.path.join(directory, f))
                 for f in os.listdir(directory) if f.endswith(".mp4")]
    except FileNotFoundError:
        return 0
    files.sort()                                    # oldest first
    now = time.time()
    total = sum(size for _, size, _ in files)
    removed = 0
    for mtime, size, path in files:
        too_old = now - mtime > max_age_days * 86400
        too_big = total > max_total_mb * 1024 * 1024
        if not (too_old or too_big):
            break
        try:
            os.remove(path)
            total -= size
            removed += 1
        except OSError:
            pass
    return removed


class ClipRecorder:
    """Rolling JPEG buffer for one camera, plus background clip writing."""

    def __init__(self, directory, enabled=True, pre_s=PRE_S, post_s=POST_S):
        self.directory = directory
        self.enabled = enabled
        self.pre_s, self.post_s = pre_s, post_s
        self._buf = deque()                          # (ts, jpeg bytes)
        self._lock = threading.Lock()
        self._span = pre_s + post_s + 1.0
        self.last_error = None
        if enabled:
            os.makedirs(directory, exist_ok=True)

    def add(self, ts, jpeg):
        if not self.enabled:
            return
        with self._lock:
            self._buf.append((ts, jpeg))
            while self._buf and ts - self._buf[0][0] > self._span:
                self._buf.popleft()

    def buffered_seconds(self):
        with self._lock:
            return (self._buf[-1][0] - self._buf[0][0]) if len(self._buf) > 1 else 0.0

    def capture(self, name, event_ts, on_done=None):
        """Write [event_ts - pre, event_ts + post] to `name`.mp4 once the post-roll has passed."""
        if not self.enabled:
            return None
        filename = f"{name}.mp4"
        threading.Thread(target=self._write_later, args=(filename, event_ts, on_done),
                         daemon=True, name=f"clip-{name[:12]}").start()
        return filename

    def _write_later(self, filename, event_ts, on_done):
        wait = (event_ts + self.post_s) - time.time()
        if wait > 0:
            time.sleep(wait)
        with self._lock:
            frames = [(ts, j) for ts, j in self._buf if event_ts - self.pre_s <= ts <= event_ts + self.post_s]
        path = os.path.join(self.directory, filename)
        try:
            ok = self._write(path, frames)
        except Exception as e:                        # a clip must never take the pipeline down
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"Clip write error ({filename}): {e}", flush=True)
            ok = False
        if ok:
            prune(self.directory)
        if on_done:
            on_done(filename if ok else None)

    def _write(self, path, frames):
        if len(frames) < 2:
            return False
        gaps = np.diff([ts for ts, _ in frames])
        fps = float(np.clip(1.0 / max(np.median(gaps), 1e-3), 1, 60))
        first = cv2.imdecode(np.frombuffer(frames[0][1], np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return False
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(path, FOURCC, fps, (w, h))
        if not writer.isOpened():
            self.last_error = "VideoWriter could not open (missing mp4v codec?)"
            return False
        try:
            for _, jpeg in frames:
                img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None and img.shape[:2] == (h, w):
                    writer.write(img)
        finally:
            writer.release()
        return os.path.exists(path) and os.path.getsize(path) > 0
