# height_estimator.py — Precision height estimation with Kalman filtering,
#                       perspective correction, and multi-method consensus
"""
Estimates a person's real-world height from camera footage with high accuracy.

Pipeline:
  1. Undistort keypoints (remove lens distortion via CameraCalibrator)
  2. Measure body segments from keypoints (multiple independent methods)
  3. Get distance via DepthEstimator (or pinhole fallback)
  4. Convert pixel height → metric height using camera intrinsics
  5. Score image quality (noise, blur, resolution, lighting)
  6. Fuse multiple estimates via weighted consensus
  7. Feed into per-person Kalman filter for temporal smoothing
  8. Output filtered height with quality-adjusted confidence

Accuracy targets:
  - Calibrated camera + standing person at 2-4m: ±3-5cm
  - Uncalibrated webcam + standing person: ±8-12cm
  - Partial body visibility: ±10-15cm (with pose correction)
"""

import math
import time
import numpy as np
from collections import defaultdict

# ── COCO keypoint indices ─────────────────────────────────────────
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
# Based on composite anthropometric data (DeLeva 1996, NASA-STD-3000).
# These are averages; the adaptive ratio system adjusts per-person.
PROP_HEAD_TO_NOSE      = 0.10   # top-of-head to nose level
PROP_NOSE_TO_SHOULDER  = 0.08   # nose to mid-shoulder
PROP_SHOULDER_TO_HIP   = 0.30   # mid-shoulder to mid-hip (torso)
PROP_HIP_TO_KNEE       = 0.26   # mid-hip to mid-knee (thigh)
PROP_KNEE_TO_ANKLE     = 0.26   # mid-knee to mid-ankle (shin)

# Derived
PROP_NOSE_DOWN         = PROP_NOSE_TO_SHOULDER + PROP_SHOULDER_TO_HIP + \
                          PROP_HIP_TO_KNEE + PROP_KNEE_TO_ANKLE  # 0.90
PROP_SHOULDER_DOWN     = PROP_SHOULDER_TO_HIP + PROP_HIP_TO_KNEE + PROP_KNEE_TO_ANKLE  # 0.82
PROP_LEGS              = PROP_HIP_TO_KNEE + PROP_KNEE_TO_ANKLE   # 0.52

# Average human head height in meters (crown to chin ≈ 23cm)
AVG_HEAD_HEIGHT_M      = 0.23
# Fraction of total height the head represents
PROP_HEAD_TOTAL        = PROP_HEAD_TO_NOSE + 0.035  # ≈ 0.135

# Minimum keypoint confidence to use
MIN_KP_CONF = 0.25

# Default assumed height for calibration
DEFAULT_HEIGHT_M = 1.70


# ═══════════════════════════════════════════════════════════════════
#  KALMAN FILTER (1D height tracking per person)
# ═══════════════════════════════════════════════════════════════════

class HeightKalmanFilter:
    """
    Simple 1D Kalman filter for tracking a person's height over time.

    State: [height_m]
    A person's height doesn't change, so process noise Q is very small.
    Measurement noise R is scaled by quality_score (bad frames = high R).
    """

    def __init__(self, initial_height: float = None):
        # State estimate
        self.x = initial_height if initial_height else DEFAULT_HEIGHT_M
        # Estimate uncertainty (start wide)
        self.P = 0.5 ** 2  # ±50cm initial uncertainty

        # Process noise: height doesn't change, but we allow tiny drift
        # for model flexibility (different shoes, hair, posture)
        self.Q = 0.001 ** 2   # ≈ 1mm²/step process noise

        # Base measurement noise (will be scaled by quality)
        self.R_base = 0.08 ** 2   # ±8cm base measurement noise

        self.n_updates = 0
        self.last_update_time = time.time()

    def predict(self):
        """Predict step (height is constant, so just add process noise)."""
        self.P += self.Q

    def update(self, measurement: float, quality_score: float = 1.0):
        """
        Update with a new height measurement.

        quality_score: 0.0 (terrible) to 1.0 (excellent)
        Lower quality → higher measurement noise → less trust in this measurement.
        """
        # Scale measurement noise inversely with quality
        # quality=1.0 → R = R_base
        # quality=0.5 → R = R_base * 4
        # quality=0.1 → R = R_base * 100
        quality_clamped = max(0.05, min(1.0, quality_score))
        R = self.R_base / (quality_clamped ** 2)

        # Kalman gain
        K = self.P / (self.P + R)

        # Update state
        innovation = measurement - self.x
        self.x += K * innovation

        # Update uncertainty
        self.P = (1 - K) * self.P

        # Clamp to human range
        self.x = max(0.5, min(2.50, self.x))

        self.n_updates += 1
        self.last_update_time = time.time()

    @property
    def height(self) -> float:
        return round(self.x, 3)

    @property
    def uncertainty(self) -> float:
        """Return ±1σ uncertainty in meters."""
        return round(math.sqrt(self.P), 3)

    @property
    def is_converged(self) -> bool:
        """Has the filter converged to a stable estimate?"""
        return self.n_updates >= 8 and self.uncertainty < 0.05

    @property
    def confidence(self) -> float:
        """Convert uncertainty to a 0-1 confidence score."""
        # At ±3cm uncertainty → confidence ≈ 0.95
        # At ±15cm → confidence ≈ 0.5
        # At ±50cm → confidence ≈ 0.1
        return round(max(0.0, min(1.0, 1.0 - self.uncertainty / 0.5)), 3)


# ═══════════════════════════════════════════════════════════════════
#  PER-PERSON TRACKER (maps person identity to Kalman filter)
# ═══════════════════════════════════════════════════════════════════

# person_key → HeightKalmanFilter
_person_filters: dict[str, HeightKalmanFilter] = {}
_person_last_seen: dict[str, float] = {}
_PERSON_TIMEOUT = 10.0  # seconds before pruning a person
_MAX_TRACKED = 30

# Previous keypoints per person (for stability scoring)
_person_prev_kpts: dict[str, np.ndarray] = {}


def _person_key_from_box(box: list, grid_size: int = 50) -> str:
    """Generate a coarse spatial key from bounding box center."""
    cx = int((box[0] + box[2]) / 2 / grid_size)
    cy = int((box[1] + box[3]) / 2 / grid_size)
    return f"{cx}_{cy}"


def _get_filter(person_key: str, initial_height: float = None) -> HeightKalmanFilter:
    """Get or create a Kalman filter for a person."""
    if person_key not in _person_filters:
        _person_filters[person_key] = HeightKalmanFilter(initial_height)
    _person_last_seen[person_key] = time.time()
    return _person_filters[person_key]


def _prune_old_persons():
    """Remove person filters that haven't been updated recently."""
    now = time.time()
    stale = [k for k, t in _person_last_seen.items() if now - t > _PERSON_TIMEOUT]
    for k in stale:
        _person_filters.pop(k, None)
        _person_last_seen.pop(k, None)
        _person_prev_kpts.pop(k, None)

    # Also cap total tracked
    if len(_person_filters) > _MAX_TRACKED:
        oldest = sorted(_person_last_seen.items(), key=lambda x: x[1])
        for k, _ in oldest[:len(_person_filters) - _MAX_TRACKED]:
            _person_filters.pop(k, None)
            _person_last_seen.pop(k, None)
            _person_prev_kpts.pop(k, None)


# ═══════════════════════════════════════════════════════════════════
#  KEYPOINT HELPERS
# ═══════════════════════════════════════════════════════════════════

def _kp(kpts, idx):
    """Return (x, y) if keypoint confidence > threshold, else None."""
    if kpts is None or idx >= len(kpts):
        return None
    if len(kpts[idx]) >= 3 and kpts[idx][2] > MIN_KP_CONF:
        return (float(kpts[idx][0]), float(kpts[idx][1]))
    return None


def _mid(p1, p2):
    """Midpoint of two points, handling None."""
    if p1 is None and p2 is None:
        return None
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def _dist(p1, p2):
    """Euclidean distance between two points."""
    if p1 is None or p2 is None:
        return 0.0
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _vert_dist(p1, p2):
    """Vertical (y-axis only) distance between two points. More robust for height."""
    if p1 is None or p2 is None:
        return 0.0
    return abs(p1[1] - p2[1])


# ═══════════════════════════════════════════════════════════════════
#  VISIBILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def analyze_visibility(kpts, box, frame_shape):
    """
    Determine which body segments are visible and whether the person
    is cut off by the frame edges.
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

    for i in range(min(17, len(kpts))):
        if len(kpts[i]) >= 3 and kpts[i][2] > MIN_KP_CONF:
            result["visible_kp_count"] += 1

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

    if result["has_head"] and result["has_torso"] and result["has_lower_legs"]:
        result["class"] = "full_body"
    elif result["has_head"] and result["has_torso"] and result["has_upper_legs"]:
        result["class"] = "upper_body_plus"
    elif result["has_head"] and result["has_shoulders"]:
        result["class"] = "upper_body" if result["has_torso"] else "head_shoulders"
    elif result["has_torso"] and result["has_lower_legs"]:
        result["class"] = "lower_body"
    elif result["has_head"]:
        result["class"] = "head_only"
    elif result["has_torso"]:
        result["class"] = "torso_only"
    else:
        result["class"] = "unknown"

    return result


# ═══════════════════════════════════════════════════════════════════
#  METHOD 1: KEYPOINT SEGMENT SUMMATION
#  Measure visible segments, extrapolate to full height
# ═══════════════════════════════════════════════════════════════════

def _measure_segments(kpts):
    """
    Measure visible body segments using keypoints.
    Uses VERTICAL distance (y-axis only) for segments that should be
    mostly vertical (torso, legs) to reduce perspective error.
    Uses Euclidean for head (which is small enough not to matter).
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

    l_ankle = _kp(kpts, KP_LEFT_ANKLE)
    r_ankle = _kp(kpts, KP_RIGHT_ANKLE)

    segments = {}
    proportions_covered = 0.0

    # Head above nose (estimated from eye-nose distance)
    if nose and (l_eye or r_eye):
        eye = _mid(l_eye, r_eye) or l_eye or r_eye
        eye_to_nose = _dist(eye, nose)
        if eye_to_nose > 1:
            head_above_nose = eye_to_nose * 2.0
            segments["head_above_nose"] = head_above_nose
            proportions_covered += PROP_HEAD_TO_NOSE

    # Nose → Shoulder
    if nose and mid_sh:
        d = _vert_dist(nose, mid_sh)
        if d > 2:
            segments["nose_to_shoulder"] = d
            proportions_covered += PROP_NOSE_TO_SHOULDER

    # Shoulder → Hip (torso) — use vertical distance
    if mid_sh and mid_hip:
        d = _vert_dist(mid_sh, mid_hip)
        if d > 5:
            segments["shoulder_to_hip"] = d
            proportions_covered += PROP_SHOULDER_TO_HIP

    # Hip → Knee (thigh) — use BOTH sides and take average for robustness
    thigh_measurements = []
    if l_hip and l_knee:
        d = _dist(l_hip, l_knee)
        if d > 5:
            thigh_measurements.append(d)
    if r_hip and r_knee:
        d = _dist(r_hip, r_knee)
        if d > 5:
            thigh_measurements.append(d)
    if mid_hip:
        mid_knee = _mid(l_knee, r_knee)
        if mid_knee:
            d = _dist(mid_hip, mid_knee)
            if d > 5:
                thigh_measurements.append(d)

    if thigh_measurements:
        segments["hip_to_knee"] = sum(thigh_measurements) / len(thigh_measurements)
        proportions_covered += PROP_HIP_TO_KNEE

    # Knee → Ankle (shin) — same approach, average both sides
    shin_measurements = []
    if l_knee and l_ankle:
        d = _dist(l_knee, l_ankle)
        if d > 5:
            shin_measurements.append(d)
    if r_knee and r_ankle:
        d = _dist(r_knee, r_ankle)
        if d > 5:
            shin_measurements.append(d)
    mid_knee = _mid(l_knee, r_knee)
    mid_ankle = _mid(l_ankle, r_ankle)
    if mid_knee and mid_ankle:
        d = _dist(mid_knee, mid_ankle)
        if d > 5:
            shin_measurements.append(d)

    if shin_measurements:
        segments["knee_to_ankle"] = sum(shin_measurements) / len(shin_measurements)
        proportions_covered += PROP_KNEE_TO_ANKLE

    return segments, proportions_covered


def _adaptive_proportions(segments: dict) -> dict:
    """
    Adjust standard body proportions based on the ratios between
    actually measured segments. If measured torso/thigh ratio differs
    from average, we adjust the extrapolation accordingly.
    """
    props = {
        "head_above_nose": PROP_HEAD_TO_NOSE,
        "nose_to_shoulder": PROP_NOSE_TO_SHOULDER,
        "shoulder_to_hip": PROP_SHOULDER_TO_HIP,
        "hip_to_knee": PROP_HIP_TO_KNEE,
        "knee_to_ankle": PROP_KNEE_TO_ANKLE,
    }

    # Check torso:thigh ratio to detect body type deviation
    torso_px = segments.get("shoulder_to_hip")
    thigh_px = segments.get("hip_to_knee")

    if torso_px and thigh_px and torso_px > 10 and thigh_px > 10:
        measured_ratio = torso_px / thigh_px
        expected_ratio = PROP_SHOULDER_TO_HIP / PROP_HIP_TO_KNEE  # 0.30/0.26 ≈ 1.15

        if 0.5 < measured_ratio < 2.0:
            # Adjust proportions to match measured body type
            deviation = measured_ratio / expected_ratio
            # Clamp adjustment to ±20%
            deviation = max(0.80, min(1.20, deviation))
            props["shoulder_to_hip"] = PROP_SHOULDER_TO_HIP * deviation
            props["hip_to_knee"] = PROP_HIP_TO_KNEE / deviation
            # Renormalize so total stays ≈ 1.0
            total = sum(props.values())
            for k in props:
                props[k] /= total

    return props


def estimate_height_keypoints(kpts, box):
    """
    Method 1: Segment summation with adaptive proportions.

    Returns (height_px, confidence, proportions_covered)
    """
    if kpts is None or len(kpts) < 17:
        bbox_h = abs(box[3] - box[1])
        return bbox_h, 0.15, 0.0

    segments, prop_covered = _measure_segments(kpts)

    if prop_covered < 0.15 or len(segments) < 1:
        bbox_h = abs(box[3] - box[1])
        return bbox_h, 0.15, 0.0

    # Get adaptive proportions for this body type
    adapted_props = _adaptive_proportions(segments)

    # Recompute proportions_covered using adapted values
    adapted_covered = 0.0
    measured_px = 0.0
    for seg_name, seg_px in segments.items():
        prop = adapted_props.get(seg_name, 0.0)
        adapted_covered += prop
        measured_px += seg_px

    if adapted_covered < 0.1:
        adapted_covered = prop_covered  # fallback to standard

    full_height_px = measured_px / adapted_covered

    # Confidence from coverage
    if adapted_covered >= 0.85:
        conf = 0.92
    elif adapted_covered >= 0.60:
        conf = 0.75
    elif adapted_covered >= 0.40:
        conf = 0.55
    else:
        conf = 0.30

    return full_height_px, conf, adapted_covered


# ═══════════════════════════════════════════════════════════════════
#  METHOD 2: BOUNDING BOX + DEPTH
#  Box height × distance / focal length
# ═══════════════════════════════════════════════════════════════════

def estimate_height_bbox_depth(
    box, distance_m: float, focal_length_px: float
) -> tuple:
    """
    Method 2: Bounding box height with metric distance.

    height_m = bbox_height_px × distance_m / focal_length_px

    Lower accuracy than keypoints but independent measurement.
    Returns (height_m, confidence)
    """
    if distance_m is None or distance_m < 0.3 or focal_length_px < 10:
        return None, 0.0

    bbox_h = abs(box[3] - box[1])
    if bbox_h < 10:
        return None, 0.0

    height_m = bbox_h * distance_m / focal_length_px

    # Bbox overestimates due to padding — apply correction factor
    # YOLO bounding boxes are typically 5-15% larger than actual body
    height_m *= 0.92

    # Clamp to human range
    height_m = max(0.5, min(2.50, height_m))

    # Confidence: bbox is less reliable than keypoints
    conf = 0.45
    return height_m, conf


# ═══════════════════════════════════════════════════════════════════
#  METHOD 3: HEAD SIZE ESTIMATION
#  Known average head height ≈ 23cm → extrapolate total height
# ═══════════════════════════════════════════════════════════════════

def estimate_height_head_size(
    kpts, distance_m: float, focal_length_px: float
) -> tuple:
    """
    Method 3: Head size as a ruler.

    Average human head height (crown to chin) ≈ 23cm.
    If we can measure head pixel height and know distance → total height.
    Returns (height_m, confidence)
    """
    if distance_m is None or focal_length_px < 10:
        return None, 0.0

    nose = _kp(kpts, KP_NOSE)
    l_eye = _kp(kpts, KP_LEFT_EYE)
    r_eye = _kp(kpts, KP_RIGHT_EYE)
    l_ear = _kp(kpts, KP_LEFT_EAR)
    r_ear = _kp(kpts, KP_RIGHT_EAR)

    if nose is None:
        return None, 0.0

    eye = _mid(l_eye, r_eye)
    if eye is None:
        return None, 0.0

    # Eye-to-nose distance ≈ 40% of head height
    eye_nose_px = _dist(eye, nose)
    if eye_nose_px < 2:
        return None, 0.0

    head_px = eye_nose_px / 0.4  # extrapolate full head height in pixels

    # Convert head pixels to meters
    head_m = head_px * distance_m / focal_length_px

    # Total height from head height
    # Head is approximately 13.5% of total height (PROP_HEAD_TOTAL)
    height_m = head_m / PROP_HEAD_TOTAL

    # Clamp
    height_m = max(0.5, min(2.50, height_m))

    # Lower confidence — head size varies significantly between individuals
    conf = 0.35

    # But if distance is close (head is large in frame), confidence is better
    if eye_nose_px > 15:
        conf = 0.45
    if eye_nose_px > 25:
        conf = 0.50

    return height_m, conf


# ═══════════════════════════════════════════════════════════════════
#  POSE-AWARE HEIGHT CORRECTION
#  "Unfold" bent joints to estimate standing height
# ═══════════════════════════════════════════════════════════════════

def _angle_at_joint(p1, p2, p3):
    """Angle at p2 in degrees, or None if any point is missing."""
    if p1 is None or p2 is None or p3 is None:
        return None
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))


def pose_correction_factor(kpts, pose_action: str = None) -> float:
    """
    Compute a multiplicative correction factor for height when the person
    is not standing upright.

    Returns factor >= 1.0 — multiply pixel height by this to approximate
    the height they WOULD be if standing straight.

    Returns 1.0 for standing/walking (no correction needed).
    """
    if pose_action in ("standing", "walking", "running", None):
        return 1.0

    # Measure actual joint angles to compute geometric correction
    l_hip = _kp(kpts, KP_LEFT_HIP)
    r_hip = _kp(kpts, KP_RIGHT_HIP)
    l_knee = _kp(kpts, KP_LEFT_KNEE)
    r_knee = _kp(kpts, KP_RIGHT_KNEE)
    l_ankle = _kp(kpts, KP_LEFT_ANKLE)
    r_ankle = _kp(kpts, KP_RIGHT_ANKLE)
    l_sh = _kp(kpts, KP_LEFT_SHOULDER)
    r_sh = _kp(kpts, KP_RIGHT_SHOULDER)
    mid_sh = _mid(l_sh, r_sh)
    mid_hip = _mid(l_hip, r_hip)

    factor = 1.0

    if pose_action == "crouching":
        # Measure knee bend angle
        l_knee_angle = _angle_at_joint(l_hip, l_knee, l_ankle)
        r_knee_angle = _angle_at_joint(r_hip, r_knee, r_ankle)
        angles = [a for a in [l_knee_angle, r_knee_angle] if a is not None]
        if angles:
            avg_knee = sum(angles) / len(angles)
            # Straight leg = 180°, deep squat ≈ 50°
            # When knees are bent, measured height is shorter
            # Correction: height_standing ≈ height_measured × (180 / knee_angle) ^ 0.3
            if avg_knee < 170:
                factor = (180.0 / max(40, avg_knee)) ** 0.35
                factor = min(factor, 1.6)  # cap at 60% correction

    elif pose_action == "bending":
        # Measure torso tilt
        if mid_sh and mid_hip:
            dy = abs(mid_hip[1] - mid_sh[1])
            dx = abs(mid_hip[0] - mid_sh[0])
            if dy > 0:
                tilt_deg = math.degrees(math.atan2(dx, dy))
                # Bending forward reduces measured height
                # Approximate: height_standing ≈ height_measured / cos(tilt)
                if tilt_deg > 10:
                    tilt_rad = math.radians(min(tilt_deg, 75))
                    factor = 1.0 / max(0.3, math.cos(tilt_rad))
                    factor = min(factor, 1.5)

    elif pose_action in ("sitting", "crouching"):
        # Rough correction: sitting height ≈ 52% of standing height
        # So factor ≈ 1/0.52 ≈ 1.92
        factor = 1.85  # conservative estimate

    return min(factor, 2.0)


# ═══════════════════════════════════════════════════════════════════
#  MULTI-METHOD CONSENSUS FUSION
# ═══════════════════════════════════════════════════════════════════

def _fuse_estimates(estimates: list) -> tuple:
    """
    Fuse multiple height estimates using weighted average.

    Args:
        estimates: list of (height_m, confidence, method_name)

    Returns:
        (fused_height_m, fused_confidence)
    """
    valid = [(h, c, m) for h, c, m in estimates if h is not None and c > 0]

    if not valid:
        return None, 0.0

    if len(valid) == 1:
        return valid[0][0], valid[0][1]

    # Weighted average by confidence
    total_weight = sum(c for _, c, _ in valid)
    fused_h = sum(h * c for h, c, _ in valid) / total_weight

    # Consensus bonus: if methods agree, boost confidence
    heights = [h for h, _, _ in valid]
    max_diff = max(heights) - min(heights)

    if max_diff < 0.05:  # within 5cm
        consensus_boost = 1.2
    elif max_diff < 0.10:  # within 10cm
        consensus_boost = 1.1
    elif max_diff > 0.25:  # disagree by >25cm
        consensus_boost = 0.8  # penalize disagreement
    else:
        consensus_boost = 1.0

    fused_conf = min(0.98, (total_weight / len(valid)) * consensus_boost)

    return round(fused_h, 3), round(fused_conf, 3)


# ═══════════════════════════════════════════════════════════════════
#  CALIBRATION STATE (backward compatible)
# ═══════════════════════════════════════════════════════════════════

_calibration = {
    "focal_length_px": None,
    "reference_height_m": None,
    "reference_height_px": None,
    "reference_foot_y": None,
    "frame_height": None,
}


def auto_calibrate(kpts, box, frame_shape, known_height_m=None):
    """Auto-calibrate from the first standing person seen (backward compatible)."""
    if _calibration["focal_length_px"] is not None:
        return False

    height_px, conf, _ = estimate_height_keypoints(kpts, box)
    if conf < 0.15 or height_px < 10:
        return False

    ref_h = known_height_m or DEFAULT_HEIGHT_M
    frame_h = frame_shape[0]

    _calibration["focal_length_px"] = height_px
    _calibration["reference_height_m"] = ref_h
    _calibration["reference_height_px"] = height_px
    _calibration["frame_height"] = frame_h

    l_ankle = _kp(kpts, KP_LEFT_ANKLE)
    r_ankle = _kp(kpts, KP_RIGHT_ANKLE)
    foot = _mid(l_ankle, r_ankle)
    _calibration["reference_foot_y"] = foot[1] if foot else box[3]

    print(f"  ✅ Height auto-calibrated: {height_px:.0f}px ≈ {ref_h}m "
          f"(conf={conf:.2f})", flush=True)
    return True


def reset_calibration():
    """Reset height calibration."""
    _calibration["focal_length_px"] = None
    _calibration["reference_height_m"] = None
    _calibration["reference_height_px"] = None
    _calibration["reference_foot_y"] = None
    _calibration["frame_height"] = None


def is_calibrated():
    """Check if height estimator has been calibrated."""
    return _calibration["focal_length_px"] is not None


# ═══════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def estimate_height(
    kpts,
    box,
    frame_shape,
    pose_action: str = None,
    camera_calibrator=None,
    depth_estimator=None,
    quality_analyzer=None,
    frame: np.ndarray = None,
):
    """
    Estimate height (m) and distance with full pipeline:
      1. Undistort keypoints (if calibrator available)
      2. Measure via multiple methods
      3. Get distance (depth estimator or pinhole fallback)
      4. Convert pixel → metric
      5. Apply pose correction
      6. Compute quality score
      7. Fuse estimates
      8. Update per-person Kalman filter

    Returns dict with:
      height_m, raw_height_m, confidence, distance_m, quality_score,
      calibration, visibility, method, margin_m, height_px, estimates, ...
    """
    result = {
        "height_m": None,
        "raw_height_m": None,
        "confidence": 0.0,
        "distance_m": None,
        "depth_method": "none",
        "quality_score": 0.5,
        "calibration": "uncalibrated",
        "visibility": "unknown",
        "method": "none",
        "margin_m": 0.25,
        "height_px": 0.0,
        "estimates": {},
        "kalman_updates": 0,
        "kalman_uncertainty": 0.5,
    }

    if box is None or len(box) < 4:
        return result

    # ── Step 0: Periodic cleanup ─────────────────────────────────
    _prune_old_persons()

    kpts_np = np.array(kpts, dtype=np.float64) if kpts is not None else None

    # ── Step 1: Undistort keypoints ──────────────────────────────
    if camera_calibrator is not None and kpts_np is not None and len(kpts_np) >= 17:
        try:
            kpts_np = camera_calibrator.undistort_keypoints(kpts_np, frame_shape)
            result["calibration"] = camera_calibrator.calibration_method
        except Exception:
            pass
    elif camera_calibrator is not None:
        result["calibration"] = camera_calibrator.calibration_method

    # ── Step 2: Visibility analysis ──────────────────────────────
    vis = analyze_visibility(kpts_np, box, frame_shape)
    result["visibility"] = vis["class"]

    # ── Step 3: Get focal length ─────────────────────────────────
    if camera_calibrator is not None:
        focal_px = camera_calibrator.get_focal_length_px(frame_shape)
    elif _calibration["focal_length_px"] is not None:
        focal_px = float(_calibration["reference_height_px"])
        result["calibration"] = "auto_legacy"
    else:
        # Fallback: assume 62° H-FOV
        focal_px = frame_shape[1] * 0.88  # W / (2*tan(31°))
        result["calibration"] = "uncalibrated"

    # ── Step 4: Method 1 — Keypoint segment summation ────────────
    kp_height_px, kp_conf, kp_coverage = estimate_height_keypoints(kpts_np, box)
    result["height_px"] = round(kp_height_px, 1)

    # ── Step 5: Get distance ─────────────────────────────────────
    distance_m = None

    # Try depth estimator first (most accurate)
    if depth_estimator is not None and depth_estimator.is_ready:
        l_ankle = _kp(kpts_np, KP_LEFT_ANKLE) if kpts_np is not None else None
        r_ankle = _kp(kpts_np, KP_RIGHT_ANKLE) if kpts_np is not None else None
        ankle_pts = [p for p in [l_ankle, r_ankle] if p is not None]

        if not ankle_pts:
            # Fallback to bottom center of box
            ankle_pts = [((box[0] + box[2]) / 2, box[3])]

        # Try auto-calibrating depth scale if not done yet
        if not depth_estimator.is_scale_calibrated and kp_height_px > 50:
            depth_estimator.auto_calibrate_from_person(
                ankle_pts, kp_height_px, DEFAULT_HEIGHT_M, focal_px
            )

        distance_m = depth_estimator.get_person_distance(ankle_pts, frame_shape)
        if distance_m is not None:
            result["depth_method"] = "midas"

    # Fallback: pinhole model
    if distance_m is None:
        if depth_estimator is not None:
            distance_m = depth_estimator.fallback_distance(
                kp_height_px, DEFAULT_HEIGHT_M, focal_px
            )
        else:
            # Pure pinhole
            if kp_height_px > 5 and focal_px > 10:
                distance_m = focal_px * DEFAULT_HEIGHT_M / kp_height_px
                distance_m = max(0.3, min(25.0, distance_m))
        if distance_m is not None:
            result["depth_method"] = "pinhole_fallback"

    result["distance_m"] = round(distance_m, 2) if distance_m else None

    # ── Step 6: Convert pixel heights to metric ──────────────────
    estimates = []

    # Method 1: Keypoints
    if kp_height_px > 5 and distance_m and focal_px > 10:
        h_m = kp_height_px * distance_m / focal_px
        # Apply pose correction
        pose_factor = pose_correction_factor(kpts_np, pose_action)
        h_m *= pose_factor
        h_m = max(0.5, min(2.50, h_m))
        estimates.append((h_m, kp_conf, "keypoints"))
        result["estimates"]["keypoints"] = round(h_m, 3)

    # Method 2: Bounding box + depth
    if distance_m and focal_px > 10:
        h_bbox, c_bbox = estimate_height_bbox_depth(box, distance_m, focal_px)
        if h_bbox is not None:
            estimates.append((h_bbox, c_bbox, "bbox_depth"))
            result["estimates"]["bbox_depth"] = round(h_bbox, 3)

    # Method 3: Head size
    if kpts_np is not None and distance_m and focal_px > 10:
        h_head, c_head = estimate_height_head_size(kpts_np, distance_m, focal_px)
        if h_head is not None:
            estimates.append((h_head, c_head, "head_size"))
            result["estimates"]["head_size"] = round(h_head, 3)

    # ── Step 7: Fuse estimates ───────────────────────────────────
    raw_height, raw_conf = _fuse_estimates(estimates)

    if raw_height is None:
        return result

    result["raw_height_m"] = round(raw_height, 3)

    # ── Step 8: Quality scoring ──────────────────────────────────
    quality_score = 0.5
    person_key = _person_key_from_box(box)
    prev_kpts = _person_prev_kpts.get(person_key)

    if quality_analyzer is not None and frame is not None:
        try:
            from image_quality import compute_quality_score
            q_result = compute_quality_score(
                frame, box, kp_height_px, kpts_np, prev_kpts
            )
            quality_score = q_result.get("quality_score", 0.5)
        except Exception:
            pass
    else:
        # Basic quality estimation without the full analyzer
        if kp_height_px > 200:
            quality_score = 0.85
        elif kp_height_px > 100:
            quality_score = 0.65
        elif kp_height_px > 50:
            quality_score = 0.45
        else:
            quality_score = 0.25

        # Penalize if crouching/bending (pose correction adds uncertainty)
        if pose_action in ("crouching", "bending", "lying_down"):
            quality_score *= 0.6

    result["quality_score"] = round(quality_score, 3)

    # Store current keypoints for next frame's stability check
    if kpts_np is not None:
        _person_prev_kpts[person_key] = kpts_np.copy()

    # ── Step 9: Kalman filter update ─────────────────────────────
    kf = _get_filter(person_key, initial_height=raw_height)
    kf.predict()
    kf.update(raw_height, quality_score=quality_score * raw_conf)

    result["height_m"] = kf.height
    result["confidence"] = round(min(raw_conf * quality_score, kf.confidence), 3)
    result["kalman_updates"] = kf.n_updates
    result["kalman_uncertainty"] = kf.uncertainty

    # ── Step 10: Determine method label and margin ───────────────
    method_parts = []
    if kp_coverage >= 0.6:
        method_parts.append("keypoints")
    elif kp_coverage >= 0.3:
        method_parts.append("keypoints_partial")
    else:
        method_parts.append("bbox")

    if result["depth_method"] == "midas":
        method_parts.append("depth")
    else:
        method_parts.append("pinhole")

    if kf.n_updates >= 5:
        method_parts.append("kalman")

    if result["calibration"] not in ("uncalibrated", "auto_default"):
        method_parts.append("calibrated")

    result["method"] = "_".join(method_parts)

    # Margin from Kalman uncertainty + quality adjustment
    base_margin = kf.uncertainty
    if result["calibration"] == "uncalibrated":
        base_margin *= 1.5
    if result["depth_method"] != "midas":
        base_margin *= 1.3

    result["margin_m"] = round(max(0.02, min(0.30, base_margin)), 3)

    return result
