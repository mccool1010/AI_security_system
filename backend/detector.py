"""YOLOv8-Pose person detection + keypoint-based posture classification.

detect() only runs the model. Posture is classified per *track* (see
tracking.py) because it needs a short keypoint history with timestamps.

All speeds are in body-heights per second, so thresholds do not depend on
camera resolution, subject distance or pipeline FPS.
"""
import math
import os

import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = os.environ.get("POSE_MODEL", "yolov8n-pose.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF = DEVICE == "cuda"

model = YOLO(MODEL_PATH)
model.to(DEVICE)

# COCO keypoint indices
KP_NOSE = 0
KP_L_SH, KP_R_SH = 5, 6
KP_L_EL, KP_R_EL = 7, 8
KP_L_WR, KP_R_WR = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KN, KP_R_KN = 13, 14
KP_L_ANK, KP_R_ANK = 15, 16

KP_CONF_MIN = 0.25
KP_CONF_STRICT = 0.5   # joints used for whole-body geometry must be this confident
MIN_POSE_PX = 48       # below this body height only locomotion is classified

POSES = ("lying_down", "falling", "fighting", "kicking", "hands_raised",
         "crouching", "bending", "running", "walking", "standing", "unknown")

# Thresholds (body-heights per second unless stated)
FALL_DOWN_SPEED = 1.0       # shoulders dropping at least this fast
FIGHT_WRIST_SPEED = 3.0     # wrists moving relative to the hips
FIGHT_LIMB_SPEED = 1.3      # mean relative speed of all visible limbs
KICK_ANKLE_SPEED = 1.8
RUN_SPEED = 1.6             # whole-body translation
WALK_SPEED = 0.25


def detect(frame, conf=0.5, imgsz=640):
    """Run YOLOv8-Pose on a BGR frame. Returns [{box, confidence, keypoints (17x3 list)}]."""
    r = model(frame, imgsz=imgsz, conf=conf, device=DEVICE, classes=[0], half=HALF, verbose=False)[0]
    out = []
    if r.boxes is None:
        return out
    kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
    for i, box in enumerate(r.boxes):
        out.append({
            "class": "person",
            "confidence": float(box.conf[0]),
            "box": [float(v) for v in box.xyxy[0].tolist()],
            "keypoints": kpts[i].tolist() if kpts is not None and i < len(kpts) else None,
        })
    return out


def warmup(shape=(480, 640, 3)):
    detect(np.zeros(shape, dtype=np.uint8))


# ── helpers ───────────────────────────────────────────────────────
def _pt(k, i, min_conf=KP_CONF_MIN):
    return k[i, :2] if k[i, 2] > min_conf else None


def _mid(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return (a + b) / 2


def _angle(a, b, c):
    ba, bc = a - b, c - b
    cos = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _speed(history, fn, scale):
    """Mean speed (body-heights/s) of point fn(kpts, box) over the history."""
    total, dt_sum = 0.0, 0.0
    for (t0, b0, k0), (t1, b1, k1) in zip(history, list(history)[1:]):
        p0, p1 = fn(k0, b0), fn(k1, b1)
        dt = t1 - t0
        if p0 is None or p1 is None or dt <= 0:
            continue
        total += float(np.linalg.norm(p1 - p0))
        dt_sum += dt
    return (total / dt_sum / scale) if dt_sum > 0 else 0.0


def _down_speed(history, fn, scale):
    """Signed vertical speed (positive = moving down), body-heights/s, over the history span."""
    pts = [(t, fn(k, b)) for t, b, k in history]
    pts = [(t, p) for t, p in pts if p is not None]
    if len(pts) < 2 or pts[-1][0] - pts[0][0] <= 0:
        return 0.0
    return float(pts[-1][1][1] - pts[0][1][1]) / (pts[-1][0] - pts[0][0]) / scale


def classify_pose(history, body_scale_px, truncated=False):
    """Classify posture from [(ts, box, kpts ndarray 17x3), ...] (oldest first).

    body_scale_px: the person's standing height in pixels (max recent box height).
    truncated: the box touches the frame edge, so whole-body geometry is unreliable.
    """
    if not history:
        return "unknown"
    _, box, k = history[-1]
    if k is None:
        return "unknown"
    scale = max(body_scale_px, 1.0)

    l_sh, r_sh = _pt(k, KP_L_SH), _pt(k, KP_R_SH)
    l_hip, r_hip = _pt(k, KP_L_HIP), _pt(k, KP_R_HIP)
    l_kn, r_kn = _pt(k, KP_L_KN), _pt(k, KP_R_KN)
    l_ank, r_ank = _pt(k, KP_L_ANK), _pt(k, KP_R_ANK)
    l_wr, r_wr = _pt(k, KP_L_WR), _pt(k, KP_R_WR)
    nose = _pt(k, KP_NOSE)
    mid_sh, mid_hip = _mid(l_sh, r_sh), _mid(l_hip, r_hip)
    whole_body = not truncated and scale >= MIN_POSE_PX

    torso_tilt = None
    strict_sh = _mid(_pt(k, KP_L_SH, KP_CONF_STRICT), _pt(k, KP_R_SH, KP_CONF_STRICT))
    strict_hip = _mid(_pt(k, KP_L_HIP, KP_CONF_STRICT), _pt(k, KP_R_HIP, KP_CONF_STRICT))
    if whole_body and strict_sh is not None and strict_hip is not None:
        mid_sh, mid_hip = strict_sh, strict_hip
        d = mid_hip - mid_sh
        torso_tilt = math.degrees(math.atan2(abs(d[0]), max(d[1], 1e-6)))  # 0 = upright

    box_w, box_h = box[2] - box[0], box[3] - box[1]
    hist = [h for h in history if h[2] is not None]

    def hip_center(kk, bb):
        return _mid(_pt(kk, KP_L_HIP), _pt(kk, KP_R_HIP))

    def rel(idx):
        def f(kk, bb):
            p, c = _pt(kk, idx), hip_center(kk, bb)
            return None if p is None or c is None else p - c
        return f

    def box_center(kk, bb):
        return np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])

    def shoulders(kk, bb):
        return _mid(_pt(kk, KP_L_SH), _pt(kk, KP_R_SH))

    # 1. Lying down: torso near horizontal, or box much wider than tall
    if whole_body and ((torso_tilt is not None and torso_tilt > 60)
                       or (box_w > 1.3 * box_h and box_h < 0.6 * scale and len(hist) >= 3)):
        return "lying_down"

    # 2. Falling: shoulders dropping fast while the body leans or shrinks
    if whole_body and len(hist) >= 3:
        down = _down_speed(hist[-6:], shoulders, scale)
        if down > FALL_DOWN_SPEED and ((torso_tilt or 0) > 30 or box_h < 0.75 * scale):
            return "falling"

    if len(hist) >= 3:
        wrist_speed = max(_speed(hist, rel(KP_L_WR), scale), _speed(hist, rel(KP_R_WR), scale))
        limb_ids = (KP_L_WR, KP_R_WR, KP_L_EL, KP_R_EL, KP_L_ANK, KP_R_ANK, KP_L_KN, KP_R_KN)
        limb_speed = float(np.mean([_speed(hist, rel(i), scale) for i in limb_ids]))
        ankle_speed = max(_speed(hist, rel(KP_L_ANK), scale), _speed(hist, rel(KP_R_ANK), scale))
        # a box clipped by the frame edge grows as the person enters; track the shoulders instead
        body_speed = _speed(hist, shoulders if truncated else box_center, scale)
    else:
        wrist_speed = limb_speed = ankle_speed = body_speed = 0.0

    # 3. Fighting: fast arm movement relative to the body, not explained by locomotion
    if whole_body and wrist_speed > FIGHT_WRIST_SPEED and limb_speed > FIGHT_LIMB_SPEED and body_speed < RUN_SPEED:
        return "fighting"

    # 4. Kicking: an ankle raised to hip height and moving fast
    if whole_body and mid_hip is not None:
        for ank in (l_ank, r_ank):
            if ank is not None and ank[1] < mid_hip[1] and ankle_speed > KICK_ANKLE_SPEED:
                return "kicking"

    # 5. Locomotion
    if body_speed > RUN_SPEED:
        return "running"

    if not whole_body:
        return "walking" if body_speed > WALK_SPEED else "standing"

    # 6. Hands raised above the head
    if l_wr is not None and r_wr is not None and nose is not None:
        if l_wr[1] < nose[1] and r_wr[1] < nose[1]:
            return "hands_raised"

    # 7. Crouching: bent knees
    angles = [_angle(h, kn, a) for h, kn, a in ((l_hip, l_kn, l_ank), (r_hip, r_kn, r_ank))
              if h is not None and kn is not None and a is not None]
    if angles and min(angles) < 110:
        return "crouching"

    # 8. Bending forward
    if torso_tilt is not None and torso_tilt > 35:
        return "bending"

    if body_speed > WALK_SPEED:
        return "walking"
    return "standing"
