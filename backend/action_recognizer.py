# action_recognizer.py — SlowFast R50 (Kinetics-400) for CCTV activity recognition
"""
Security-focused activity recognition using PyTorchVideo's SlowFast R50 model.

Strategy:
  - SlowFast classifies 400 Kinetics actions, but most are irrelevant for CCTV
    (e.g. "shaving beard", "playing violin", "eating watermelon")
  - We ONLY report actions from a curated SECURITY_ACTIONS set
  - When none of those match confidently, the API falls back to YOLO pose
    (standing, walking, crouching, etc.) which is always accurate
  - This prevents garbage predictions while still catching real threats

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

# ── Security-relevant action classes from Kinetics-400 ──────────
# These are the ONLY actions the system will ever report.
# Everything else (shaving, cooking, yoga, etc.) is silently ignored.
# The display name maps Kinetics labels to cleaner CCTV-appropriate names.
SECURITY_ACTIONS = {
    # ═══════════════════════════════════════════════════════════════
    #  CRITICAL — MURDER / LETHAL ASSAULT PROXIES
    #  (Kinetics-400 has no "murder" label, so we map the closest
    #   physical proxies that a model would fire on such footage)
    # ═══════════════════════════════════════════════════════════════
    "headbutting": "⚠ CRITICAL — Violent headbutt",
    "slapping": "⚠ CRITICAL — Slapping assault",
    "punching person (boxing)": "⚠ CRITICAL — Punch / assault",
    "punching bag": "⚠ HIGH — Punching",
    "wrestling": "⚠ CRITICAL — Wrestling / grappling assault",
    "capoeira": "⚠ HIGH — Combat movement",
    "kickboxing": "⚠ CRITICAL — Kickboxing assault",
    "drop kicking": "⚠ CRITICAL — Kick to person",
    "side kick": "⚠ CRITICAL — Kick to person",
    "high kick": "⚠ CRITICAL — Kick to head level",
    "karate chop": "⚠ CRITICAL — Striking attack",
    "krumping": "⚠ HIGH — Aggressive rapid movement",

    # ═══════════════════════════════════════════════════════════════
    #  CRITICAL — STRANGULATION / CHOKING / NECK ATTACK PROXIES
    #  (SlowFast fires these labels on restraint/choking postures)
    # ═══════════════════════════════════════════════════════════════
    "arm wrestling": "⚠ CRITICAL — Restraint / hold",
    "massaging person's head": "⚠ HIGH — Hands on head/neck",
    "cracking neck": "⚠ CRITICAL — Neck manipulation",
    "tickling": "⚠ HIGH — Hands-on contact",
    "hugging": "⚠ HIGH — Close physical contact",

    # ═══════════════════════════════════════════════════════════════
    #  CRITICAL — ROBBERY / ARMED THREAT PROXIES
    # ═══════════════════════════════════════════════════════════════
    "sword fighting": "⚠ CRITICAL — Armed attack",
    "fencing (sport)": "⚠ CRITICAL — Armed confrontation",
    "sharpening knives": "⚠ CRITICAL — Knife handling",
    "chopping wood": "⚠ CRITICAL — Swinging bladed object",
    "throwing axe": "⚠ CRITICAL — Throwing weapon",
    "javelin throw": "⚠ HIGH — Throwing weapon",
    "hammer throw": "⚠ HIGH — Throwing heavy object",
    "shot put": "⚠ HIGH — Throwing heavy object",
    "throwing discus": "⚠ HIGH — Throwing object at person",
    "throwing ball": "⚠ HIGH — Throwing object",
    "catching or throwing baseball": "⚠ HIGH — Throwing object",
    "catching or throwing softball": "⚠ HIGH — Throwing object",
    "catching or throwing frisbee": "HIGH — Throwing object",
    "archery": "⚠ CRITICAL — Ranged weapon",
    "spray painting": "⚠ HIGH — Vandalism / substance spray",
    "spraying": "⚠ HIGH — Spraying substance",
    "counting money": "⚠ HIGH — Handling cash (robbery context)",
    "unboxing": "HIGH — Opening package",
    "ripping paper": "HIGH — Destroying property",
    "shredding paper": "HIGH — Destroying documents",

    # ═══════════════════════════════════════════════════════════════
    #  CRITICAL — FIRE / ARSON
    # ═══════════════════════════════════════════════════════════════
    "juggling fire": "⚠ CRITICAL — Fire hazard / arson risk",
    "extinguishing fire": "⚠ CRITICAL — Active fire emergency",
    "welding": "⚠ HIGH — Open flame / sparks",
    "blasting sand": "⚠ HIGH — Industrial hazard",

    # ═══════════════════════════════════════════════════════════════
    #  CRITICAL — BREAKING & SMASHING / PROPERTY DESTRUCTION
    # ═══════════════════════════════════════════════════════════════
    "bending metal": "⚠ HIGH — Tampering with structure",
    "sanding floor": "⚠ HIGH — Tampering / break-in activity",
    "plastering": "HIGH — Tampering with wall",
    "laying bricks": "HIGH — Construction / obstruction",
    "building shed": "HIGH — Unauthorised construction",
    "building cabinet": "HIGH — Construction activity",
    "digging": "⚠ HIGH — Digging (sabotage/burial risk)",
    "shoveling snow": "HIGH — Digging",

    # ═══════════════════════════════════════════════════════════════
    #  RUNNING / CHASING / FLEEING (fast movements)
    # ═══════════════════════════════════════════════════════════════
    "jogging": "HIGH — Running (possible flight/chase)",
    "running on treadmill": "HIGH — Running",
    "hurdling": "⚠ HIGH — Sprinting / obstacle evasion",
    "marching": "HIGH — Marching / coordinated movement",
    "parkour": "⚠ CRITICAL — Parkour / evasion",
    "vault": "⚠ HIGH — Vaulting obstacle",
    "pole vault": "⚠ HIGH — Vaulting",
    "triple jump": "HIGH — Jumping",
    "long jump": "HIGH — Jumping",
    "high jump": "HIGH — Jumping",
    "jumping into pool": "HIGH — Jumping from height",
    "bungee jumping": "HIGH — Jumping from height",
    "somersaulting": "HIGH — Acrobatic evasion",
    "cartwheeling": "HIGH — Fast acrobatic movement",
    "gymnastics tumbling": "HIGH — Fast acrobatic movement",
    "springboard diving": "HIGH — Jumping",
    "skipping rope": "HIGH — Fast repetitive movement",
    "bouncing on trampoline": "HIGH — Jumping",
    "hopscotch": "HIGH — Fast movement",

    # ═══════════════════════════════════════════════════════════════
    #  CLIMBING / INTRUSION / TRESPASSING
    # ═══════════════════════════════════════════════════════════════
    "climbing a rope": "⚠ HIGH — Climbing (intrusion)",
    "climbing tree": "⚠ HIGH — Climbing (intrusion)",
    "climbing ladder": "⚠ HIGH — Climbing",
    "rock climbing": "HIGH — Climbing",
    "ice climbing": "HIGH — Climbing",
    "abseiling": "⚠ HIGH — Descending wall (intrusion)",
    "crossing river": "⚠ HIGH — Trespassing",
    "diving cliff": "HIGH — Jumping from height",

    # ═══════════════════════════════════════════════════════════════
    #  FALLEN PERSON / MEDICAL EMERGENCY / DISTRESS
    # ═══════════════════════════════════════════════════════════════
    "faceplanting": "⚠ CRITICAL — Person collapsed / fallen",
    "stumbling": "⚠ HIGH — Person stumbling / falling",
    "falling off chair": "⚠ HIGH — Person fallen",
    "crying": "⚠ HIGH — Person in distress",
    "bandaging": "⚠ HIGH — Medical aid required",
    "situp": "HIGH — Person getting up from ground",
    "push up": "HIGH — Person on ground",
    "yawning": None,
    "sneezing": None,

    # ═══════════════════════════════════════════════════════════════
    #  CROWD VIOLENCE / RIOT / GANG ACTIVITY
    # ═══════════════════════════════════════════════════════════════
    "surfing crowd": "⚠ CRITICAL — Crowd disturbance / riot",
    "celebrating": "HIGH — Crowd gathering",
    "applauding": "HIGH — Crowd activity",
    "clapping": "HIGH — Crowd activity",
    "pumping fist": "⚠ HIGH — Aggressive crowd gesture",
    "sticking tongue out": "HIGH — Provocative gesture",
    "shaking head": "HIGH — Gesturing",
    "testifying": "HIGH — Speaking loudly",
    "finger snapping": "HIGH — Signaling",
    "sign language interpreting": "HIGH — Signaling",

    # ═══════════════════════════════════════════════════════════════
    #  CROUCHING / HIDING / SNEAKING
    # ═══════════════════════════════════════════════════════════════
    "army crawling": "⚠ HIGH — Crawling (tactical movement)",
    "crawling baby": None,
    "squat": "HIGH — Crouching / concealing",
    "lunge": "HIGH — Lunging movement",
    "bending back": "HIGH — Bending / concealing",

    # ═══════════════════════════════════════════════════════════════
    #  CARRYING / MOVING (theft of goods, body, weapon)
    # ═══════════════════════════════════════════════════════════════
    "carrying weight": "⚠ HIGH — Carrying heavy object",
    "clean and jerk": "HIGH — Lifting heavy object",
    "snatch weight lifting": "HIGH — Lifting heavy object",
    "deadlifting": "HIGH — Lifting heavy object",
    "moving furniture": "⚠ HIGH — Moving large object",
    "unloading truck": "HIGH — Moving cargo",
    "opening bottle": "HIGH — Handling object",
    "opening present": "HIGH — Opening package",
    "massaging back": "HIGH — Physical contact",
    "massaging legs": "HIGH — Physical contact",
    "massaging feet": "HIGH — Physical contact",
    "kissing": "HIGH — Close contact",
    "shaking hands": "HIGH — Contact (exchange?)",
    "lunge": "HIGH — Low lunging movement",

    # ═══════════════════════════════════════════════════════════════
    #  VEHICLE / CHASE / DANGEROUS DRIVING
    # ═══════════════════════════════════════════════════════════════
    "driving car": "HIGH — Driving",
    "driving tractor": "HIGH — Driving vehicle",
    "motorcycling": "⚠ HIGH — Motorcycling (pursuit risk)",
    "riding a bike": "HIGH — Cycling",
    "riding mountain bike": "HIGH — Cycling fast",
    "riding scooter": "HIGH — On scooter",
    "riding mechanical bull": "HIGH — Erratic movement",
    "hoverboarding": "HIGH — On hoverboard",
    "skateboarding": "HIGH — On skateboard",
    "using segway": "HIGH — On segway",
    "pushing car": "⚠ HIGH — Pushing vehicle forcefully",
    "pushing cart": "HIGH — Pushing",
    "pushing wheelchair": "HIGH — Pushing",

    # ═══════════════════════════════════════════════════════════════
    #  GENERAL MOVEMENT / SURVEILLANCE CONTEXT
    # ═══════════════════════════════════════════════════════════════
    "walking the dog": "LOW — Walking",
    "stretching arm": "LOW — Stretching",
    "stretching leg": "LOW — Stretching",
    "exercising arm": "LOW — Exercising",
    "exercising with an exercise ball": "LOW — Exercising",
    "doing aerobics": "LOW — Exercising",
    "front raises": "LOW — Arm movement",
    "pull ups": "HIGH — Pulling up",
    "bench pressing": "LOW — Lifting",
    "swinging legs": "LOW — Restless",
    "swinging on something": "LOW — Swinging",
    "waiting in line": "LOW — Waiting",
    "using computer": "LOW — Using computer",
    "texting": "LOW — Using phone",
    "smoking": "LOW — Smoking",
    "smoking hookah": "LOW — Smoking",
    "drinking": "LOW — Drinking",
    "drinking beer": "HIGH — Drinking alcohol",
    "drinking shots": "HIGH — Drinking alcohol",
    "tasting beer": "HIGH — Drinking alcohol",
    "swinging legs": "LOW — Restless",
}

# Number of frames the SlowFast model expects
CLIP_LEN = 32
# How often to run inference (seconds)
# Lowered to 0.5s so fast events (punches, grabs, sprints) are caught quickly
INFERENCE_INTERVAL = 0.5
# Spatial crop size expected by the model
CROP_SIZE = 256
SIDE_SIZE = 256
# Buffer EVERY frame so fast motions (punches, kicks, strangulation) are captured
FRAME_SKIP = 1
# EMA: lower alpha = faster response to sudden violent actions
EMA_ALPHA = 0.65
# Minimum confidence to report a security action (lowered for better recall on fast events)
MIN_CONFIDENCE = 0.06


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
        self._relevant_indices = {}   # {model_index: display_name}
        self._raw_buffer = deque(maxlen=CLIP_LEN)
        self._lock = threading.Lock()
        self._last_result = []
        self._last_inference_time = 0
        self._loading = True
        self._ready = False
        self._frame_counter = 0
        self._inference_running = False
        self._ema_probs = None

        # Load model + labels in background
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

            # Load labels and build security-action index
            self._labels = self._load_labels()
            self._relevant_indices = self._build_security_index()
            self._ready = True
            self._loading = False
            print(f"  ✅ SlowFast R50 loaded — {len(self._labels)} classes, "
                  f"{len(self._relevant_indices)} security-relevant", flush=True)
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
            clean = name.strip('"').replace("_", " ")
            if idx < 400:
                labels[idx] = clean
        return labels

    def _build_security_index(self):
        """Map security-relevant Kinetics labels to their model indices + display names."""
        index_map = {}  # {model_index: display_name}
        matched = []
        for idx, label in enumerate(self._labels):
            if label in SECURITY_ACTIONS:
                display = SECURITY_ACTIONS[label]
                if display is not None:  # None = explicitly excluded
                    index_map[idx] = display
                    matched.append(f"{label} → {display}")

        print(f"  📋 Matched {len(index_map)} security actions:", flush=True)
        for m in sorted(matched):
            print(f"      • {m}", flush=True)
        return index_map

    # ── Frame buffer ─────────────────────────────────────
    def push_frame(self, bgr_frame):
        """Push a BGR frame into the circular buffer (thread-safe)."""
        if not self._ready:
            return
        self._frame_counter += 1
        if self._frame_counter % FRAME_SKIP != 0:
            return
        with self._lock:
            self._raw_buffer.append(bgr_frame.copy())

    # ── Inference ────────────────────────────────────────
    def get_actions(self, top_k=3):
        """
        Get the latest action predictions (returns cached result immediately).
        Only returns SECURITY-RELEVANT actions. If nothing relevant is detected,
        returns [] so the API can fall back to YOLO pose classification.
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
        """Run SlowFast inference on the current frame buffer (background thread)."""
        try:
            with self._lock:
                raw_frames = list(self._raw_buffer)

            if len(raw_frames) < CLIP_LEN:
                self._inference_running = False
                return

            # Resize + color-convert (background thread, not main loop)
            frames = []
            for f in raw_frames:
                resized = cv2.resize(f, (SIDE_SIZE, SIDE_SIZE))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                frames.append(rgb)

            # Convert to tensor: (T, H, W, C) -> (C, T, H, W)
            clip = np.array(frames, dtype=np.float32) / 255.0
            clip = torch.from_numpy(clip).permute(3, 0, 1, 2)

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

            # SlowFast pathways
            slow_pathway = clip[:, ::4, :, :]   # (C, 8, H, W)
            fast_pathway = clip                 # (C, 32, H, W)

            slow_pathway = slow_pathway.unsqueeze(0).to(self._device)
            fast_pathway = fast_pathway.unsqueeze(0).to(self._device)

            with torch.no_grad():
                preds = self._model([slow_pathway, fast_pathway])

            probs = torch.nn.functional.softmax(preds, dim=1)
            probs_np = probs[0].cpu().numpy()  # shape (400,)

            # ── Extract ONLY security-relevant probabilities ──
            # Zero out everything that's not a security action
            security_probs = np.zeros_like(probs_np)
            for idx in self._relevant_indices:
                security_probs[idx] = probs_np[idx]

            # ── EMA smoothing on security-filtered probs ──
            if self._ema_probs is None:
                self._ema_probs = security_probs
            else:
                self._ema_probs = EMA_ALPHA * security_probs + (1 - EMA_ALPHA) * self._ema_probs

            # Pick top-k from smoothed, security-only distribution
            top_indices = np.argsort(self._ema_probs)[::-1][:top_k]

            results = []
            for idx in top_indices:
                conf = float(self._ema_probs[idx])
                if conf < MIN_CONFIDENCE:
                    continue
                # Use the clean display name, not the raw Kinetics label
                display_name = self._relevant_indices.get(idx, self._labels[idx])
                results.append({
                    "action": display_name,
                    "confidence": round(conf, 3),
                    "raw_label": self._labels[idx],  # keep for debugging
                })

            # Always update — clear stale predictions when nothing passes threshold
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
