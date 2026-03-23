# height_estimator.py — Keypoint-based height estimation with distance & partial body awareness
"""
Uses YOLOv8-Pose COCO keypoints to estimate a person's real height,
accounting for:
  - Which body parts are actually visible (partial body extrapolation)
  - Distance from camera (pinhole camera model)
  - Confidence scoring based on keypoint quality

Human body proportions (fraction of total height):
  Head top → Nose/Eyes:   ~10%
  Nose → Shoulder:        ~8%
  Shoulder → Hip (torso): ~30%
  Hip → Knee (thigh):     ~26%
  Knee → Ankle (shin):    ~26%
"""

import math
import numpy as np

# ── COCO keypoint indices (same as detector.py) ──────────────────
KP_NOSE            = 0
KP_LEFT_EYE        = 1
KP_RIGHT_EYE       = 2
KP_LEFT_EAR        = 3
KP_RIGHT_EAR       = 4
KP_LEFT_SHOULDER   = 5
KP_RIGHT_SHOULDER  = 6
KP_LEFT_ELBOW      = 7
KP_RIGHT_ELBOW     = 8
KP_LEFT_WRIST      = 9
KP_RIGHT_WRIST     = 10
KP_LEFT_HIP        = 11
KP_RIGHT_HIP       = 12
KP_LEFT_KNEE       = 13
KP_RIGHT_KNEE      = 14
KP_LEFT_ANKLE      = 15
KP_RIGHT_ANKLE     = 16

# ── Body segment proportions (fraction of total standing height) ─
# Based on anthropometric data (average adult).
PROP_HEAD_TO_NOSE      = 0.10   # top-of-head to nose level
PROP_NOSE_TO_SHOULDER  = 0.08   # nose to mid-shoulder
PROP_SHOULDER_TO_HIP   = 0.30   # mid-shoulder to mid-hip (torso)
PROP_HIP_TO_KNEE       = 0.26   # mid-hip to mid-knee (thigh)
PROP_KNEE_TO_ANKLE     = 0.26   # mid-knee to mid-ankle (shin)

# Summed groups (for partial body extrapolation)
PROP_HEAD_AND_TORSO    = PROP_HEAD_TO_NOSE + PROP_NOSE_TO_SHOULDER + PROP_SHOULDER_TO_HIP  # 0.48
PROP_LEGS              = PROP_HIP_TO_KNEE + PROP_KNEE_TO_ANKLE                              # 0.52
PROP_UPPER_LEG         = PROP_HIP_TO_KNEE                                                   # 0.26
PROP_LOWER_LEG         = PROP_KNEE_TO_ANKLE                                                 # 0.26
PROP_SHOULDER_DOWN     = PROP_SHOULDER_TO_HIP + PROP_LEGS                                   # 0.82
PROP_NOSE_DOWN         = PROP_NOSE_TO_SHOULDER + PROP_SHOULDER_TO_HIP + PROP_LEGS            # 0.90

# Minimum keypoint confidence to consider it "visible"
MIN_KP_CONF = 0.3

# ── Calibration state ───────────────────────────────────────────
_calibration = {
    "focal_length_px": None,      # estimated focal length in pixels
    "reference_height_m": None,
    "reference_height_px": None,
    "reference_foot_y": None,     # y-position of feet in calibration frame
    "frame_height": None,
}

DEFAULT_HEIGHT_M = 1.7  # assumed average height for auto-calibration


def _kp(kpts, idx):
    """Return (x, y) if keypoint confidence > threshold, else None."""
    if kpts is None or idx >= len(kpts):
        return None
    if kpts[idx][2] > MIN_KP_CONF:
        return (float(kpts[idx][0]), float(kpts[idx][1]))
    return None


def _mid(p1, p2):
    """Midpoint of two (x,y) points."""
    if p1 is None or p2 is None:
        return p1 or p2
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def _dist(p1, p2):
    """Euclidean distance between two (x,y) points."""
    if p1 is None or p2 is None:
        return 0.0
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _near_edge(pt, frame_w, frame_h, margin_frac=0.03):
    """Check if a point is near the edge of the frame."""
    if pt is None:
        return False
    mx = frame_w * margin_frac
    my = frame_h * margin_frac
    return pt[0] < mx or pt[0] > frame_w - mx or pt[1] < my or pt[1] > frame_h - my


# ═══════════════════════════════════════════════════════════════════
#  VISIBILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_visibility(kpts, box, frame_shape):
    """
    Determine which body segments are visible and whether the person
    is cut off by the frame edges.

    Returns dict: {
        "class": "full_body" | "upper_body" | "lower_body" | "torso_only" | "head_only" | "unknown",
        "has_head": bool,
        "has_shoulders": bool,
        "has_torso": bool,   (shoulders + hips)
        "has_upper_legs": bool,
        "has_lower_legs": bool,
        "cut_off_top": bool,
        "cut_off_bottom": bool,
        "visible_kp_count": int,
        "total_kp_count": 17,
    }
    """
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = box[:4]

    result = {
        "class": "unknown",
        "has_head": False,
        "has_shoulders": False,
        "has_torso": False,
        "has_upper_legs": False,
        "has_lower_legs": False,
        "cut_off_top": y1 < frame_h * 0.03,
        "cut_off_bottom": y2 > frame_h * 0.97,
        "visible_kp_count": 0,
        "total_kp_count": 17,
    }

    if kpts is None:
        return result

    # Count visible keypoints
    for i in range(min(17, len(kpts))):
        if kpts[i][2] > MIN_KP_CONF:
            result["visible_kp_count"] += 1

    # Check each body region
    nose = _kp(kpts, KP_NOSE)
    l_eye = _kp(kpts, KP_LEFT_EYE)
    r_eye = _kp(kpts, KP_RIGHT_EYE)
    l_ear = _kp(kpts, KP_LEFT_EAR)
    r_ear = _kp(kpts, KP_RIGHT_EAR)
    result["has_head"] = any(p is not None for p in [nose, l_eye, r_eye, l_ear, r_ear])

    l_sh = _kp(kpts, KP_LEFT_SHOULDER)
    r_sh = _kp(kpts, KP_RIGHT_SHOULDER)
    result["has_shoulders"] = l_sh is not None or r_sh is not None

    l_hip = _kp(kpts, KP_LEFT_HIP)
    r_hip = _kp(kpts, KP_RIGHT_HIP)
    result["has_torso"] = result["has_shoulders"] and (l_hip is not None or r_hip is not None)

    l_knee = _kp(kpts, KP_LEFT_KNEE)
    r_knee = _kp(kpts, KP_RIGHT_KNEE)
    result["has_upper_legs"] = (l_hip is not None or r_hip is not None) and \
                                (l_knee is not None or r_knee is not None)

    l_ankle = _kp(kpts, KP_LEFT_ANKLE)
    r_ankle = _kp(kpts, KP_RIGHT_ANKLE)
    result["has_lower_legs"] = (l_knee is not None or r_knee is not None) and \
                                (l_ankle is not None or r_ankle is not None)

    # Classify overall visibility
    if result["has_head"] and result["has_torso"] and result["has_lower_legs"]:
        result["class"] = "full_body"
    elif result["has_head"] and result["has_torso"] and result["has_upper_legs"]:
        result["class"] = "upper_body_plus"   # head + torso + thighs
    elif result["has_head"] and result["has_shoulders"]:
        if result["has_torso"]:
            result["class"] = "upper_body"
        else:
            result["class"] = "head_shoulders"
    elif result["has_torso"] and result["has_lower_legs"]:
        result["class"] = "lower_body"        # torso + legs but no head
    elif result["has_head"]:
        result["class"] = "head_only"
    elif result["has_torso"]:
        result["class"] = "torso_only"
    else:
        result["class"] = "unknown"

    return result


# ═══════════════════════════════════════════════════════════════════
#  KEYPOINT-BASED HEIGHT MEASUREMENT (in pixels)
# ═══════════════════════════════════════════════════════════════════

def _measure_segments(kpts):
    """
    Measure visible body segments using keypoints.
    Returns dict of { segment_name: pixel_length } and the proportion
    of total height those segments represent.
    """
    nose = _kp(kpts, KP_NOSE)
    l_eye = _kp(kpts, KP_LEFT_EYE)
    r_eye = _kp(kpts, KP_RIGHT_EYE)

    l_sh = _kp(kpts, KP_LEFT_SHOULDER)
    r_sh = _kp(kpts, KP_RIGHT_SHOULDER)
    mid_sh = _mid(l_sh, r_sh)

    l_hip = _kp(kpts, KP_LEFT_HIP)
    r_hip = _kp(kpts, KP_RIGHT_HIP)
    mid_hip = _mid(l_hip, r_hip)

    l_knee = _kp(kpts, KP_LEFT_KNEE)
    r_knee = _kp(kpts, KP_RIGHT_KNEE)
    mid_knee = _mid(l_knee, r_knee)

    l_ankle = _kp(kpts, KP_LEFT_ANKLE)
    r_ankle = _kp(kpts, KP_RIGHT_ANKLE)
    mid_ankle = _mid(l_ankle, r_ankle)

    segments = {}
    proportions_covered = 0.0

    # Nose → Shoulder
    if nose and mid_sh:
        d = _dist(nose, mid_sh)
        if d > 2:
            segments["nose_to_shoulder"] = d
            proportions_covered += PROP_NOSE_TO_SHOULDER

    # Shoulder → Hip (torso)
    if mid_sh and mid_hip:
        d = _dist(mid_sh, mid_hip)
        if d > 5:
            segments["shoulder_to_hip"] = d
            proportions_covered += PROP_SHOULDER_TO_HIP

    # Hip → Knee (thigh)
    if mid_hip and mid_knee:
        d = _dist(mid_hip, mid_knee)
        if d > 5:
            segments["hip_to_knee"] = d
            proportions_covered += PROP_HIP_TO_KNEE

    # Knee → Ankle (shin)
    if mid_knee and mid_ankle:
        d = _dist(mid_knee, mid_ankle)
        if d > 5:
            segments["knee_to_ankle"] = d
            proportions_covered += PROP_KNEE_TO_ANKLE

    # Add head height estimate (nose to top of head ≈ distance between eyes × 2.5)
    if nose and (l_eye or r_eye):
        eye = _mid(l_eye, r_eye) or l_eye or r_eye
        eye_to_nose = _dist(eye, nose)
        if eye_to_nose > 1:
            # Approximate head-top overshoot (forehead + crown ≈ 2× eye-to-nose distance)
            head_above_nose = eye_to_nose * 2.0
            segments["head_above_nose"] = head_above_nose
            proportions_covered += PROP_HEAD_TO_NOSE

    return segments, proportions_covered


def measure_height_px(kpts, box):
    """
    Estimate the person's full-body pixel height using keypoints.
    Falls back to bounding box height when keypoints are insufficient.

    Returns (height_px, confidence, method) where:
        - height_px: estimated full standing height in pixels
        - confidence: 0.0 - 1.0
        - method: "keypoints" | "keypoints_partial" | "bbox"
    """
    if kpts is None or len(kpts) < 17:
        bbox_h = abs(box[3] - box[1])
        return bbox_h, 0.2, "bbox"

    segments, prop_covered = _measure_segments(kpts)

    if prop_covered < 0.15 or len(segments) < 1:
        # Not enough keypoints — fall back to bounding box
        bbox_h = abs(box[3] - box[1])
        return bbox_h, 0.2, "bbox"

    # Sum measured pixel lengths
    measured_px = sum(segments.values())

    # Extrapolate to full height: measured pixels cover `prop_covered` of total
    full_height_px = measured_px / prop_covered

    # Confidence based on how much of the body we actually measured
    if prop_covered >= 0.85:
        conf = 0.95
        method = "keypoints"
    elif prop_covered >= 0.60:
        conf = 0.75
        method = "keypoints"
    elif prop_covered >= 0.40:
        conf = 0.55
        method = "keypoints_partial"
    else:
        conf = 0.35
        method = "keypoints_partial"

    # Penalize confidence if person is not standing upright
    # (leaning/bending makes keypoint distances unreliable for height)
    return full_height_px, conf, method


# ═══════════════════════════════════════════════════════════════════
#  DISTANCE ESTIMATION (pinhole camera model)
# ═══════════════════════════════════════════════════════════════════

def auto_calibrate(kpts, box, frame_shape, known_height_m=None):
    """
    Auto-calibrate using a full-body standing person.
    Uses pinhole camera model: focal_length = pixel_height × distance / real_height

    Since we don't know the distance on first calibration, we use
    the person's foot position (y-coordinate relative to frame) as a
    proxy distance reference. The first person sets the baseline.

    Returns True if calibration was performed.
    """
    if _calibration["focal_length_px"] is not None:
        return False  # already calibrated

    height_px, conf, method = measure_height_px(kpts, box)

    if conf < 0.5 or method == "bbox":
        return False  # need decent keypoint coverage

    ref_h = known_height_m or DEFAULT_HEIGHT_M
    frame_h = frame_shape[0]

    # For initial calibration, we assume the person is at a "standard" distance.
    # We estimate focal length as: f = h_px × assumed_distance / h_real
    # We use a heuristic: if the person's pixel height is ~60% of the frame,
    # they're roughly 2.5m away from a typical webcam (f ≈ 600px for 640px frames).
    # More precisely: f = h_px / h_real × assumed_distance
    # But we can simplify: just store the ratio h_px / h_real as our scale factor
    # and use the foot-y position as relative distance indicator.

    # Simple but effective: store meters_per_pixel at the calibrant's position
    # Then adjust for other people using their relative position/size.
    _calibration["focal_length_px"] = height_px  # proxy: px-height of a 1.7m person
    _calibration["reference_height_m"] = ref_h
    _calibration["reference_height_px"] = height_px
    _calibration["frame_height"] = frame_h

    # Store the foot position (lowest visible point)
    l_ankle = _kp(kpts, KP_LEFT_ANKLE)
    r_ankle = _kp(kpts, KP_RIGHT_ANKLE)
    foot = _mid(l_ankle, r_ankle)
    if foot:
        _calibration["reference_foot_y"] = foot[1]
    else:
        _calibration["reference_foot_y"] = box[3]  # bottom of bbox

    print(f"  ✅ Height calibrated (keypoints): {height_px:.0f}px ≈ {ref_h}m, "
          f"foot_y={_calibration['reference_foot_y']:.0f}", flush=True)
    return True


def estimate_height(kpts, box, frame_shape, pose_action=None):
    """
    Estimate a person's real height in meters using keypoints + calibration.

    Returns dict: {
        "height_m": float or None,
        "distance_m": float or None,    (estimated distance from camera)
        "confidence": float,             (0.0 - 1.0)
        "method": str,
        "visibility": str,               (visibility class)
        "height_px": float,              (estimated full-height in pixels)
    }
    """
    result = {
        "height_m": None,
        "distance_m": None,
        "confidence": 0.0,
        "method": "none",
        "visibility": "unknown",
        "height_px": 0.0,
    }

    vis = analyze_visibility(kpts, box, frame_shape)
    result["visibility"] = vis["class"]

    height_px, conf, method = measure_height_px(kpts, box)
    result["height_px"] = height_px
    result["method"] = method

    # Reduce confidence if person is crouching/bending (height from keypoints is unreliable)
    if pose_action in ("crouching", "bending"):
        conf *= 0.4  # heavily penalize — not standing upright
        result["method"] = method + "_pose_adjusted"

    result["confidence"] = round(conf, 2)

    if _calibration["focal_length_px"] is None:
        # Not calibrated yet — can only give pixel height
        return result

    ref_h_m = _calibration["reference_height_m"]
    ref_h_px = _calibration["reference_height_px"]

    # ── Scale-based height estimation ──
    # In a pinhole camera: real_size ∝ pixel_size × distance / focal_length
    # If we assume people are at roughly the same distance as the reference,
    # then: height_m = height_px × (ref_h_m / ref_h_px)
    # But we can do better by adjusting for distance.

    base_mpp = ref_h_m / ref_h_px  # meters per pixel at reference distance

    # ── Distance correction ──
    # People further from the camera appear smaller (fewer pixels).
    # We use the person's bounding box bottom (foot position) as a proxy for distance.
    # Further people have their feet higher in the frame (smaller y value).
    # Closer people have their feet near the bottom of the frame.
    ref_foot_y = _calibration["reference_foot_y"]
    frame_h = _calibration["frame_height"] or frame_shape[0]

    # Current person's foot position
    l_ankle = _kp(kpts, KP_LEFT_ANKLE) if kpts is not None else None
    r_ankle = _kp(kpts, KP_RIGHT_ANKLE) if kpts is not None else None
    foot = _mid(l_ankle, r_ankle)
    current_foot_y = foot[1] if foot else box[3]

    # Relative distance: ratio of foot y-positions (higher y = closer = lower distance)
    # Using horizon model: distance ∝ 1 / (foot_y / frame_h)
    # This is a simplification but works reasonably for level cameras.
    if ref_foot_y and ref_foot_y > frame_h * 0.1:
        # Normalize foot positions relative to frame
        ref_ratio = ref_foot_y / frame_h
        cur_ratio = current_foot_y / frame_h

        if cur_ratio > 0.05:
            # Distance ratio: if current person's feet are higher, they're further away
            distance_ratio = ref_ratio / cur_ratio

            # Clamp to reasonable range (0.5× to 3× reference distance)
            distance_ratio = max(0.5, min(3.0, distance_ratio))

            # Adjust mpp: further away means each pixel represents more meters
            adjusted_mpp = base_mpp * distance_ratio

            height_m = height_px * adjusted_mpp

            # Estimate distance (rough): Use the reference as ~2.5m as default
            # f = ref_h_px * d_ref / ref_h_m => d_ref = f * ref_h_m / ref_h_px
            # Since we don't know absolute distance, estimate from size ratio
            ref_distance_m = 2.5  # assumed reference distance
            result["distance_m"] = round(ref_distance_m * distance_ratio, 1)
        else:
            height_m = height_px * base_mpp
    else:
        height_m = height_px * base_mpp

    # Clamp to human range
    height_m = max(0.5, min(2.5, height_m))
    result["height_m"] = round(height_m, 2)

    # ── Error margin based on confidence ──
    # Higher confidence = tighter error margin
    if conf > 0.7:
        margin = 0.05
    elif conf > 0.5:
        margin = 0.10
    else:
        margin = 0.20
    result["margin_m"] = margin

    return result


def reset_calibration():
    """Reset calibration (e.g. when camera position changes)."""
    _calibration["focal_length_px"] = None
    _calibration["reference_height_m"] = None
    _calibration["reference_height_px"] = None
    _calibration["reference_foot_y"] = None
    _calibration["frame_height"] = None


def is_calibrated():
    """Check if the height estimator has been calibrated."""
    return _calibration["focal_length_px"] is not None
