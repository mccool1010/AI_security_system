# image_quality.py — Frame and person-region quality scoring for height estimation
"""
Lightweight quality analysis module that scores each frame's suitability
for reliable height measurement. Bad frames get lower confidence, and
the Kalman filter weights them less.

Scores:
  - Noise level (sensor noise estimation)
  - Motion blur (Laplacian variance in person region)
  - Resolution adequacy (person pixel height vs minimum threshold)
  - Keypoint stability (jitter across consecutive frames)
  - Lighting quality (under/overexposure in person region)

Combined into a single quality_score (0.0 – 1.0).
"""

import math

import cv2
import numpy as np


# ── Thresholds ───────────────────────────────────────────────────
# Person must be at least this many pixels tall for reliable height
MIN_RELIABLE_HEIGHT_PX = 80
# Optimal person pixel height (at this height, resolution score = 1.0)
OPTIMAL_HEIGHT_PX = 300

# Laplacian variance thresholds for blur detection
BLUR_SHARP_THRESHOLD = 100.0    # above this = sharp
BLUR_USABLE_THRESHOLD = 30.0    # below this = too blurry

# Noise: median filter residual thresholds
NOISE_CLEAN_THRESHOLD = 5.0     # below = clean image
NOISE_NOISY_THRESHOLD = 20.0    # above = very noisy

# Lighting: mean brightness thresholds
BRIGHTNESS_MIN = 40.0           # below = underexposed
BRIGHTNESS_MAX = 220.0          # above = overexposed
BRIGHTNESS_OPTIMAL_LO = 80.0
BRIGHTNESS_OPTIMAL_HI = 180.0

# Keypoint stability: maximum expected jitter (pixels) at full confidence
MAX_STABLE_JITTER = 3.0         # ≤3px between frames = stable
MAX_TOLERABLE_JITTER = 15.0     # >15px = unstable


# ═══════════════════════════════════════════════════════════════════
#  INDIVIDUAL QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════

def estimate_noise_level(frame: np.ndarray, region: tuple = None) -> float:
    """
    Estimate image noise using the median filter residual method.

    Lower = cleaner. Returns absolute noise level (not a score).

    Args:
        frame: BGR image
        region: optional (x1, y1, x2, y2) to analyze specific area
    """
    if frame is None or frame.size == 0:
        return 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

    if region is not None:
        x1, y1, x2, y2 = [int(v) for v in region]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(gray.shape[1], x2)
        y2 = min(gray.shape[0], y2)
        gray = gray[y1:y2, x1:x2]

    if gray.size == 0:
        return 0.0

    # Median filter removes signal, residual is noise
    median = cv2.medianBlur(gray, 5)
    residual = cv2.absdiff(gray, median)
    noise = float(np.mean(residual))
    return noise


def estimate_motion_blur(crop: np.ndarray) -> float:
    """
    Estimate motion blur via Laplacian variance.
    Higher = sharper. Returns the raw Laplacian variance.
    """
    if crop is None or crop.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def score_resolution(person_height_px: float) -> float:
    """
    Score based on person pixel height (0.0 – 1.0).

    Below MIN_RELIABLE_HEIGHT_PX: rapidly drops to 0
    At OPTIMAL_HEIGHT_PX: returns 1.0
    """
    if person_height_px <= 0:
        return 0.0
    if person_height_px < MIN_RELIABLE_HEIGHT_PX:
        # Quadratic falloff below minimum
        ratio = person_height_px / MIN_RELIABLE_HEIGHT_PX
        return max(0.0, ratio * ratio * 0.5)  # max 0.5 below threshold
    if person_height_px >= OPTIMAL_HEIGHT_PX:
        return 1.0
    # Linear ramp from 0.5 to 1.0
    ratio = (person_height_px - MIN_RELIABLE_HEIGHT_PX) / (OPTIMAL_HEIGHT_PX - MIN_RELIABLE_HEIGHT_PX)
    return 0.5 + 0.5 * ratio


def score_blur(laplacian_var: float) -> float:
    """
    Convert Laplacian variance to a 0.0–1.0 score.
    """
    if laplacian_var >= BLUR_SHARP_THRESHOLD:
        return 1.0
    if laplacian_var <= BLUR_USABLE_THRESHOLD:
        # Below usable: rapid dropoff
        if laplacian_var <= 5.0:
            return 0.05
        ratio = laplacian_var / BLUR_USABLE_THRESHOLD
        return 0.1 + 0.2 * ratio
    # Between usable and sharp: linear
    ratio = (laplacian_var - BLUR_USABLE_THRESHOLD) / (BLUR_SHARP_THRESHOLD - BLUR_USABLE_THRESHOLD)
    return 0.3 + 0.7 * ratio


def score_noise(noise_level: float) -> float:
    """
    Convert noise level to a 0.0–1.0 score (1.0 = clean).
    """
    if noise_level <= NOISE_CLEAN_THRESHOLD:
        return 1.0
    if noise_level >= NOISE_NOISY_THRESHOLD:
        return max(0.1, 1.0 - (noise_level - NOISE_NOISY_THRESHOLD) / 30.0)
    # Linear between clean and noisy
    ratio = (noise_level - NOISE_CLEAN_THRESHOLD) / (NOISE_NOISY_THRESHOLD - NOISE_CLEAN_THRESHOLD)
    return 1.0 - 0.6 * ratio


def score_lighting(crop: np.ndarray) -> float:
    """
    Score lighting quality (0.0–1.0) based on mean brightness
    and histogram spread in the person region.
    """
    if crop is None or crop.size == 0:
        return 0.5

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    mean_bright = float(np.mean(gray))
    std_bright = float(np.std(gray))

    # Mean brightness scoring
    if mean_bright < BRIGHTNESS_MIN:
        bright_score = max(0.1, mean_bright / BRIGHTNESS_MIN * 0.4)
    elif mean_bright > BRIGHTNESS_MAX:
        bright_score = max(0.1, 1.0 - (mean_bright - BRIGHTNESS_MAX) / 35.0 * 0.6)
    elif BRIGHTNESS_OPTIMAL_LO <= mean_bright <= BRIGHTNESS_OPTIMAL_HI:
        bright_score = 1.0
    elif mean_bright < BRIGHTNESS_OPTIMAL_LO:
        ratio = (mean_bright - BRIGHTNESS_MIN) / (BRIGHTNESS_OPTIMAL_LO - BRIGHTNESS_MIN)
        bright_score = 0.4 + 0.6 * ratio
    else:
        ratio = (BRIGHTNESS_MAX - mean_bright) / (BRIGHTNESS_MAX - BRIGHTNESS_OPTIMAL_HI)
        bright_score = 0.4 + 0.6 * ratio

    # Low contrast penalty (flat histogram = bad for feature detection)
    contrast_score = min(1.0, std_bright / 40.0)

    return 0.7 * bright_score + 0.3 * contrast_score


def score_keypoint_stability(
    current_kpts: np.ndarray,
    previous_kpts: np.ndarray,
    min_conf: float = 0.3,
) -> float:
    """
    Measure keypoint stability between consecutive frames.
    Returns 0.0 (unstable) to 1.0 (perfectly stable).

    Only considers keypoints visible in both frames.
    """
    if current_kpts is None or previous_kpts is None:
        return 0.5  # unknown stability

    if len(current_kpts) != len(previous_kpts):
        return 0.5

    jitters = []
    for i in range(min(len(current_kpts), 17)):
        # Both must be visible (confidence > threshold)
        if (len(current_kpts[i]) >= 3 and len(previous_kpts[i]) >= 3
                and current_kpts[i][2] > min_conf and previous_kpts[i][2] > min_conf):
            dx = current_kpts[i][0] - previous_kpts[i][0]
            dy = current_kpts[i][1] - previous_kpts[i][1]
            jitter = math.sqrt(dx * dx + dy * dy)
            jitters.append(jitter)

    if not jitters:
        return 0.5

    avg_jitter = sum(jitters) / len(jitters)
    max_jitter = max(jitters)

    # Score based on average jitter
    if avg_jitter <= MAX_STABLE_JITTER:
        stability = 1.0
    elif avg_jitter >= MAX_TOLERABLE_JITTER:
        stability = 0.1
    else:
        ratio = (avg_jitter - MAX_STABLE_JITTER) / (MAX_TOLERABLE_JITTER - MAX_STABLE_JITTER)
        stability = 1.0 - 0.9 * ratio

    # Penalize if any single keypoint jumped a lot
    if max_jitter > MAX_TOLERABLE_JITTER * 2:
        stability *= 0.7

    return max(0.0, min(1.0, stability))


def score_edge_proximity(
    box: list,
    frame_shape: tuple,
    margin_frac: float = 0.05,
) -> float:
    """
    Score based on how close the person's bounding box is to frame edges.
    Edge proximity = higher lens distortion + possible body cutoff.
    Returns 1.0 (center) to 0.3 (touching edge).
    """
    if not box or len(box) < 4:
        return 1.0

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box[:4]
    margin_x = w * margin_frac
    margin_y = h * margin_frac

    penalties = []
    if x1 < margin_x:
        penalties.append(x1 / margin_x)
    if y1 < margin_y:
        penalties.append(y1 / margin_y)
    if x2 > w - margin_x:
        penalties.append((w - x2) / margin_x)
    if y2 > h - margin_y:
        penalties.append((h - y2) / margin_y)

    if not penalties:
        return 1.0

    worst = min(penalties)
    return max(0.3, 0.3 + 0.7 * worst)


# ═══════════════════════════════════════════════════════════════════
#  COMBINED QUALITY SCORE
# ═══════════════════════════════════════════════════════════════════

def compute_quality_score(
    frame: np.ndarray,
    person_box: list,
    person_height_px: float,
    current_kpts: np.ndarray = None,
    previous_kpts: np.ndarray = None,
) -> dict:
    """
    Compute a comprehensive quality score for a person detection.

    Returns dict with individual scores and combined quality_score (0.0–1.0).

    Args:
        frame: full BGR frame
        person_box: [x1, y1, x2, y2] bounding box
        person_height_px: height of person in pixels
        current_kpts: current frame keypoints (17, 3)
        previous_kpts: previous frame keypoints (17, 3) for stability check
    """
    result = {
        "quality_score": 0.5,
        "resolution_score": 0.5,
        "blur_score": 0.5,
        "noise_score": 0.5,
        "lighting_score": 0.5,
        "stability_score": 0.5,
        "edge_score": 1.0,
        "noise_level": 0.0,
        "laplacian_var": 0.0,
        "mean_brightness": 128.0,
    }

    if frame is None or frame.size == 0 or not person_box or len(person_box) < 4:
        return result

    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in person_box[:4]]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    # Extract person crop
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return result

    # ── Individual scores ────────────────────────────────────────

    # 1. Resolution
    result["resolution_score"] = score_resolution(person_height_px)

    # 2. Blur (on person region)
    laplacian_var = estimate_motion_blur(crop)
    result["laplacian_var"] = round(laplacian_var, 1)
    result["blur_score"] = score_blur(laplacian_var)

    # 3. Noise (on person region)
    noise_level = estimate_noise_level(frame, region=(x1, y1, x2, y2))
    result["noise_level"] = round(noise_level, 2)
    result["noise_score"] = score_noise(noise_level)

    # 4. Lighting
    result["lighting_score"] = score_lighting(crop)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    result["mean_brightness"] = round(float(np.mean(gray_crop)), 1)

    # 5. Keypoint stability
    result["stability_score"] = score_keypoint_stability(current_kpts, previous_kpts)

    # 6. Edge proximity
    result["edge_score"] = score_edge_proximity(person_box, frame.shape)

    # ── Weighted combination ─────────────────────────────────────
    # Weights reflect importance for height measurement accuracy
    weights = {
        "resolution": 0.25,     # Can't measure what you can't see
        "blur": 0.20,           # Blur shifts keypoint locations
        "noise": 0.10,          # Noise adds random jitter
        "lighting": 0.10,       # Bad lighting = bad keypoint detection
        "stability": 0.20,      # Unstable keypoints = unreliable height
        "edge": 0.15,           # Edge = distortion + body cutoff
    }

    combined = (
        weights["resolution"] * result["resolution_score"]
        + weights["blur"] * result["blur_score"]
        + weights["noise"] * result["noise_score"]
        + weights["lighting"] * result["lighting_score"]
        + weights["stability"] * result["stability_score"]
        + weights["edge"] * result["edge_score"]
    )

    # Catastrophic quality: if any critical metric is very low, cap overall
    critical_min = min(
        result["resolution_score"],
        result["blur_score"],
        result["lighting_score"],
    )
    if critical_min < 0.2:
        combined = min(combined, 0.3)

    result["quality_score"] = round(max(0.0, min(1.0, combined)), 3)
    return result
