# detector.py — YOLOv8-Pose + temporal motion tracking for rich action classification
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import math
from collections import deque

# ── Model setup ──────────────────────────────────────────────────
MODEL_PATH = "yolov8n-pose.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO(MODEL_PATH)
model.to(DEVICE)

# ── COCO keypoint indices ─────────────────────────────────────────
KP_NOSE           = 0
KP_LEFT_EYE       = 1
KP_RIGHT_EYE      = 2
KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
KP_LEFT_KNEE      = 13
KP_RIGHT_KNEE     = 14
KP_LEFT_ANKLE     = 15
KP_RIGHT_ANKLE    = 16

KP_CONF_MIN = 0.25  # lower than before to handle blurry/low-res footage

# ── Per-person keypoint history for velocity tracking ─────────────
# Key: (grid_cx, grid_cy) — coarse person identity from box center
# Value: deque of (kpts_xy, timestamp)
_motion_history: dict[tuple, deque] = {}
_HISTORY_LEN = 8      # frames to keep
_GRID_SIZE   = 60     # pixels — coarser grid tolerates jitter


def _box_key(box):
    cx = int((box[0] + box[2]) / 2 / _GRID_SIZE)
    cy = int((box[1] + box[3]) / 2 / _GRID_SIZE)
    return (cx, cy)


def _push_history(box, kpts_np):
    """Store current keypoints and return per-joint velocity (px/frame)."""
    key = _box_key(box)
    if key not in _motion_history:
        _motion_history[key] = deque(maxlen=_HISTORY_LEN)
    hist = _motion_history[key]
    hist.append(kpts_np[:, :2].copy())   # store x,y only (17,2)

    if len(hist) < 2:
        return np.zeros(17)

    # Mean absolute velocity over available history
    vel = np.zeros(17)
    for i in range(1, len(hist)):
        diff = np.linalg.norm(hist[i] - hist[i - 1], axis=1)  # (17,)
        vel += diff
    return vel / (len(hist) - 1)


# Prune stale entries every 200 calls
_prune_counter = 0
def _prune_history():
    global _prune_counter
    _prune_counter += 1
    if _prune_counter % 200 == 0 and len(_motion_history) > 50:
        # Keep only the 50 most recently updated (FIFO deque is already bounded)
        oldest = list(_motion_history.keys())[:len(_motion_history)//2]
        for k in oldest:
            del _motion_history[k]


# ── Geometry helpers ──────────────────────────────────────────────

def _angle(a, b, c):
    """Angle at b in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))


def _mid(a, b):
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]


# ── Rich pose / action classifier ────────────────────────────────

def _classify_pose(kpts, vel):
    """
    Classify action from 17 COCO keypoints + per-joint velocity.

    Priority order (most specific / dangerous first):
      1.  lying_down
      2.  falling          ← downward torso velocity + body tilt
      3.  choking_posture  ← both wrists at throat/neck area
      4.  fighting         ← high velocity + aggressive geometry
      5.  punching         ← extended arm with high wrist velocity
      6.  kicking          ← raised leg with high ankle/knee velocity
      7.  guard_stance     ← boxing/martial-arts guard position
      8.  struggling       ← erratic high-velocity full-body motion
      9.  hands_raised     ← both wrists above head (surrender/reach)
      10. waving           ← one wrist above shoulder, oscillating
      11. arm_extended     ← static extended arm (pointing/threatening)
      12. jumping          ← both ankles above hip level
      13. crouching
      14. bending
      15. running
      16. walking
      17. standing         ← default
    """
    def kp(idx):
        if kpts[idx][2] > KP_CONF_MIN:
            return kpts[idx][:2].tolist()
        return None

    # Gather all joints
    nose      = kp(KP_NOSE)
    l_sh      = kp(KP_LEFT_SHOULDER);  r_sh    = kp(KP_RIGHT_SHOULDER)
    l_el      = kp(KP_LEFT_ELBOW);     r_el    = kp(KP_RIGHT_ELBOW)
    l_wr      = kp(KP_LEFT_WRIST);     r_wr    = kp(KP_RIGHT_WRIST)
    l_hip     = kp(KP_LEFT_HIP);       r_hip   = kp(KP_RIGHT_HIP)
    l_kn      = kp(KP_LEFT_KNEE);      r_kn    = kp(KP_RIGHT_KNEE)
    l_ank     = kp(KP_LEFT_ANKLE);     r_ank   = kp(KP_RIGHT_ANKLE)

    mid_sh    = _mid(l_sh, r_sh)
    mid_hip   = _mid(l_hip, r_hip)
    mid_kn    = _mid(l_kn, r_kn)
    mid_ank   = _mid(l_ank, r_ank)

    # Velocities for key joints
    v_l_wr    = vel[KP_LEFT_WRIST]
    v_r_wr    = vel[KP_RIGHT_WRIST]
    v_l_ank   = vel[KP_LEFT_ANKLE]
    v_r_ank   = vel[KP_RIGHT_ANKLE]
    v_l_kn    = vel[KP_LEFT_KNEE]
    v_r_kn    = vel[KP_RIGHT_KNEE]
    v_torso   = (vel[KP_LEFT_SHOULDER] + vel[KP_RIGHT_SHOULDER]) / 2
    v_overall = float(np.mean(vel))

    body_w = 1.0
    if l_sh and r_sh:
        body_w = max(1.0, abs(l_sh[0] - r_sh[0]))

    # ── 1. Lying down ──────────────────────────────────────────────
    if mid_sh and mid_hip:
        dy = abs(mid_hip[1] - mid_sh[1])
        dx = abs(mid_hip[0] - mid_sh[0])
        if dx > 0 and dy / (dx + 1e-6) < 0.6:
            return "lying_down"

    # ── 2. Falling ─────────────────────────────────────────────────
    # Torso moving fast downward OR body severely tilted + motion
    if mid_sh and mid_hip and v_torso > 12:
        return "falling"
    if mid_sh and mid_hip:
        dy = mid_hip[1] - mid_sh[1]
        dx = abs(mid_hip[0] - mid_sh[0])
        torso_tilt = dx / (dy + 1e-6)
        if torso_tilt > 0.7 and v_overall > 6:
            return "falling"

    # ── 3. Choking / Neck-grab posture ─────────────────────────────
    # Both wrists at or above shoulder height, close to the neck/head region
    if l_wr and r_wr and mid_sh and nose:
        neck_y = (mid_sh[1] + nose[1]) / 2
        l_at_neck = l_wr[1] < neck_y + body_w * 0.3
        r_at_neck = r_wr[1] < neck_y + body_w * 0.3
        # Wrists converging inward (narrow gap = grabbing)
        wrists_close = abs(l_wr[0] - r_wr[0]) < body_w * 1.2
        if l_at_neck and r_at_neck and wrists_close:
            return "choking_posture"

    # ── 4. Fighting (high-velocity + aggressive upper body) ─────────
    wrist_vel_max = max(v_l_wr, v_r_wr)
    if wrist_vel_max > 18 and v_overall > 8:
        return "fighting"

    # ── 5. Punching / Striking ─────────────────────────────────────
    # Fast wrist + nearly straight arm extended forward or sideways
    def _is_punch(sh, el, wr, wrist_vel):
        if not (sh and el and wr):
            return False
        arm_angle = _angle(sh, el, wr)
        wrist_far = abs(wr[0] - sh[0]) > body_w * 0.8
        return arm_angle > 145 and wrist_vel > 10 and wrist_far

    if _is_punch(l_sh, l_el, l_wr, v_l_wr):
        return "punching"
    if _is_punch(r_sh, r_el, r_wr, v_r_wr):
        return "punching"

    # ── 6. Kicking ─────────────────────────────────────────────────
    # Ankle or knee raised above hip level with high velocity
    def _is_kick(hip, knee, ankle, v_kn, v_ank):
        if hip is None:
            return False
        if ankle and ankle[1] < hip[1] and v_ank > 10:  # ankle above hip (y inverted)
            return True
        if knee and knee[1] < hip[1] - 5 and v_kn > 8:
            return True
        return False

    if _is_kick(l_hip, l_kn, l_ank, v_l_kn, v_l_ank):
        return "kicking"
    if _is_kick(r_hip, r_kn, r_ank, v_r_kn, v_r_ank):
        return "kicking"

    # ── 7. Guard stance (boxing / martial arts) ─────────────────────
    # Elbows raised ≈ shoulder height, wrists near face/chin, feet wide
    if l_el and r_el and mid_sh:
        elbows_up = l_el[1] < mid_sh[1] + body_w * 0.5 and r_el[1] < mid_sh[1] + body_w * 0.5
        if elbows_up:
            # Check wrists in front of face
            if l_wr and r_wr and nose:
                wrists_near_face = (l_wr[1] < nose[1] + body_w or r_wr[1] < nose[1] + body_w)
                if wrists_near_face:
                    return "guard_stance"

    # ── 8. Struggling ──────────────────────────────────────────────
    # General high-velocity chaotic movement across multiple joints
    fast_joints = int(v_l_wr > 8) + int(v_r_wr > 8) + int(v_l_ank > 8) + int(v_r_ank > 8)
    if fast_joints >= 3 or (v_overall > 12):
        return "struggling"

    # ── 9. Hands raised (both wrists above head) ───────────────────
    if l_wr and r_wr and nose:
        if l_wr[1] < nose[1] and r_wr[1] < nose[1]:
            return "hands_raised"

    # ── 10. Waving ─────────────────────────────────────────────────
    # One wrist above shoulder with moderate velocity (oscillation)
    if mid_sh:
        sh_y = mid_sh[1]
        if l_wr and l_wr[1] < sh_y and 3 < v_l_wr < 18:
            return "waving"
        if r_wr and r_wr[1] < sh_y and 3 < v_r_wr < 18:
            return "waving"

    # ── 11. Arm extended (pointing, threatening, reaching) ──────────
    def _arm_extended(sh, el, wr):
        if not (sh and el and wr):
            return False
        ang = _angle(sh, el, wr)
        dist = abs(wr[0] - sh[0])
        return ang > 155 and dist > body_w * 1.1

    if _arm_extended(l_sh, l_el, l_wr) or _arm_extended(r_sh, r_el, r_wr):
        return "arm_extended"

    # ── 12. Jumping ────────────────────────────────────────────────
    if mid_ank and mid_hip:
        if mid_ank[1] < mid_hip[1] and (v_l_ank + v_r_ank) > 10:
            return "jumping"

    # ── 13. Crouching ──────────────────────────────────────────────
    def _knee_angle(hip, knee, ankle):
        if hip and knee and ankle:
            return _angle(hip, knee, ankle)
        return 180

    la = _knee_angle(l_hip, l_kn, l_ank)
    ra = _knee_angle(r_hip, r_kn, r_ank)
    if la < 120 or ra < 120:
        return "crouching"

    # ── 14. Bending (torso forward) ────────────────────────────────
    if mid_sh and mid_hip:
        dy = mid_hip[1] - mid_sh[1]
        dx = abs(mid_hip[0] - mid_sh[0])
        if dy > 0:
            tilt = math.degrees(math.atan2(dx, dy))
            if tilt > 35:
                return "bending"

    # ── 15. Running / Walking / Standing (stride width) ─────────────
    if l_ank and r_ank and l_kn and r_kn:
        ankle_spread = abs(l_ank[0] - r_ank[0])
        stride = ankle_spread / body_w
        # Also use velocity to disambiguate walk vs run
        leg_vel = (v_l_ank + v_r_ank) / 2
        if stride > 1.2 or leg_vel > 12:
            return "running"
        elif stride > 0.6 or leg_vel > 4:
            return "walking"

    return "standing"


# ── Main detect function ──────────────────────────────────────────

def detect(frame, imgsz=640, conf=0.5):
    """
    Run YOLOv8-Pose on a BGR frame.
    Returns (frame, detections) — each detection has:
      class, confidence, box, pose_action, height_px, keypoints
    """
    results = model(frame, imgsz=imgsz, conf=conf, device=DEVICE, classes=[0])
    r = results[0]
    detections = []

    boxes = getattr(r, "boxes", None)
    kpts_data = getattr(r, "keypoints", None)
    if boxes is None:
        return frame, detections

    _prune_history()

    for i, box in enumerate(boxes):
        xyxy = [float(v) for v in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = xyxy
        conf_score = float(box.conf[0].item())
        cls_name = r.names[int(box.cls[0].item())] if hasattr(r, "names") else "person"

        pose_action = "unknown"
        kpts_list = []

        if kpts_data is not None and i < len(kpts_data):
            kpts_np = kpts_data[i].data[0].cpu().numpy()  # (17,3)
            kpts_list = kpts_np.tolist()
            # Push to motion history and get velocities
            vel = _push_history(xyxy, kpts_np)
            pose_action = _classify_pose(kpts_np, vel)

        detections.append({
            "class": cls_name,
            "confidence": conf_score,
            "box": xyxy,
            "pose_action": pose_action,
            "height_px": round(abs(y2 - y1), 1),
            "keypoints": kpts_list if kpts_list else None,
        })

    return frame, detections
