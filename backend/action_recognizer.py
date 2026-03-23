# action_recognizer.py — SlowFast R50 (Kinetics-400) for CCTV activity recognition
"""
Advanced human activity recognition using PyTorchVideo's SlowFast R50 model,
pretrained on Kinetics-400 (400 action classes).

The model processes 32-frame video clips to understand temporal context:
what a person is *doing* over a short time window.

Usage:
    from action_recognizer import ActionRecognizer
    recognizer = ActionRecognizer()
    recognizer.push_frame(frame)          # feed BGR frames (very cheap)
    actions = recognizer.get_actions()    # returns cached top-3 predictions
"""
import threading
import time
import json
import os
from collections import deque

import cv2
import numpy as np
import torch

# ── Kinetics-400 class labels (subset most relevant to CCTV surveillance) ──
SURVEILLANCE_RELEVANT = {
    "arguing", "carrying", "climbing", "crawling", "crouching",
    "dancing", "drinking", "eating", "entering", "exiting",
    "falling down", "fighting", "gesturing", "holding object",
    "hugging", "jumping", "kicking", "knocking", "leaning",
    "looking around", "lying down", "opening door", "phoning",
    "picking up", "pointing", "pulling", "punching", "pushing",
    "reading", "running", "shaking hands", "sitting", "smoking",
    "standing", "stretching", "texting", "throwing", "walking",
    "waving", "writing",
}

# Number of frames the SlowFast model expects
CLIP_LEN = 32
# How often to run inference (seconds)
INFERENCE_INTERVAL = 2.0
# Spatial crop size expected by the model
CROP_SIZE = 256
SIDE_SIZE = 256
# Only buffer every Nth frame to reduce CPU cost in main loop
FRAME_SKIP = 3


class ActionRecognizer:
    """
    Wraps PyTorchVideo SlowFast R50 for real-time CCTV action recognition.
    Thread-safe: push_frame() is extremely cheap (just stores raw frame ref).
    Inference runs in a dedicated background thread, never blocking the camera loop.
    """

    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._labels = []
        self._raw_buffer = deque(maxlen=CLIP_LEN)  # stores raw BGR frames
        self._lock = threading.Lock()
        self._last_result = []                       # cached action predictions
        self._last_inference_time = 0
        self._loading = True
        self._ready = False
        self._frame_counter = 0
        self._inference_running = False

        # Load model + labels in background to avoid blocking startup
        t = threading.Thread(target=self._load, daemon=True)
        t.start()

    # ── Model loading ────────────────────────────────────
    def _load(self):
        """Load SlowFast R50 from torch.hub and download Kinetics-400 labels."""
        try:
            print("🧠 Loading SlowFast R50 (Kinetics-400)...", flush=True)
            self._model = torch.hub.load(
                "facebookresearch/pytorchvideo",
                "slowfast_r50",
                pretrained=True,
            )
            self._model = self._model.to(self._device)
            self._model.eval()

            # Load Kinetics-400 class label mapping
            self._labels = self._load_labels()
            self._ready = True
            self._loading = False
            print(f"  ✅ SlowFast R50 loaded — {len(self._labels)} action classes", flush=True)
        except Exception as e:
            print(f"  ❌ SlowFast load error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self._loading = False

    def _load_labels(self):
        """Load Kinetics-400 class labels from local cache or download."""
        labels_path = os.path.join(os.path.dirname(__file__), "kinetics_400_labels.json")

        if not os.path.exists(labels_path):
            print("  📥 Downloading Kinetics-400 labels...", flush=True)
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/pyslowfast/dataset/class_names/kinetics_classnames.json"
            urllib.request.urlretrieve(url, labels_path)

        with open(labels_path, "r") as f:
            label_map = json.load(f)

        # label_map is {class_name: class_id} — we need {class_id: class_name}
        labels = [""] * 400
        for name, idx in label_map.items():
            if idx < 400:
                labels[idx] = name.replace("_", " ")
        return labels

    # ── Frame buffer (CHEAP — no resize, no color convert) ──
    def push_frame(self, bgr_frame):
        """
        Push a BGR frame into the circular buffer (thread-safe).
        Only stores every Nth frame to reduce CPU overhead.
        No resize or color conversion here — that's done in the inference thread.
        """
        if not self._ready:
            return
        self._frame_counter += 1
        if self._frame_counter % FRAME_SKIP != 0:
            return
        with self._lock:
            self._raw_buffer.append(bgr_frame.copy())

    # ── Inference (BACKGROUND THREAD) ────────────────────
    def get_actions(self, top_k=3):
        """
        Get the latest action predictions (returns cached result immediately).
        Kicks off background inference if enough time has passed.
        NEVER blocks the calling thread.
        """
        if not self._ready:
            if self._loading:
                return [{"action": "Loading model...", "confidence": 0.0}]
            return []

        now = time.time()
        if (
            not self._inference_running
            and (now - self._last_inference_time) >= INFERENCE_INTERVAL
            and len(self._raw_buffer) >= CLIP_LEN
        ):
            self._inference_running = True
            t = threading.Thread(target=self._run_inference, args=(top_k,), daemon=True)
            t.start()

        return self._last_result

    def _run_inference(self, top_k=3):
        """Run SlowFast inference on the current frame buffer (in background thread)."""
        try:
            with self._lock:
                raw_frames = list(self._raw_buffer)

            if len(raw_frames) < CLIP_LEN:
                self._inference_running = False
                return

            # Resize + color-convert here (background thread, not main loop)
            frames = []
            for f in raw_frames:
                resized = cv2.resize(f, (SIDE_SIZE, SIDE_SIZE))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(rgb)

            # Convert frames to tensor: (T, H, W, C) -> (C, T, H, W)
            clip = np.array(frames, dtype=np.float32) / 255.0
            clip = torch.from_numpy(clip).permute(3, 0, 1, 2)  # (C, T, H, W)

            # Normalize (ImageNet mean/std)
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            for c in range(3):
                clip[c] = (clip[c] - mean[c]) / std[c]

            # Center crop
            _, _, h, w = clip.shape
            y_off = max(0, (h - CROP_SIZE) // 2)
            x_off = max(0, (w - CROP_SIZE) // 2)
            clip = clip[:, :, y_off:y_off + CROP_SIZE, x_off:x_off + CROP_SIZE]

            # SlowFast expects two pathways:
            # Slow: every 4th frame (8 frames), Fast: all 32 frames
            slow_pathway = clip[:, ::4, :, :]  # (C, 8, H, W)
            fast_pathway = clip                # (C, 32, H, W)

            # Add batch dimension
            slow_pathway = slow_pathway.unsqueeze(0).to(self._device)
            fast_pathway = fast_pathway.unsqueeze(0).to(self._device)

            with torch.no_grad():
                preds = self._model([slow_pathway, fast_pathway])

            probs = torch.nn.functional.softmax(preds, dim=1)
            top_probs, top_indices = probs.topk(top_k)

            results = []
            for i in range(top_k):
                idx = top_indices[0][i].item()
                conf = top_probs[0][i].item()
                label = self._labels[idx] if idx < len(self._labels) else f"class_{idx}"
                results.append({"action": label, "confidence": round(conf, 3)})

            self._last_result = results
            self._last_inference_time = time.time()

        except Exception as e:
            print(f"SlowFast inference error: {e}", flush=True)
        finally:
            self._inference_running = False

    # ── Status ───────────────────────────────────────────
    @property
    def is_ready(self):
        return self._ready

    @property
    def is_loading(self):
        return self._loading

    @property
    def buffer_fill(self):
        return len(self._raw_buffer)
