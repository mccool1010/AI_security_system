# detector.py — YOLOv8-Pose based person detection + action classification + height estimation
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import math

# ── Model setup ──────────────────────────────────────────────────
MODEL_PATH = "yolov8n-pose.pt"  # Pose model: detection + 17 body keypoints
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO(MODEL_PATH)
model.to(DEVICE)

# ── COCO keypoint indices ────────────────────────────────────────
KP_NOSE         = 0
KP_LEFT_EYE     = 1
KP_RIGHT_EYE    = 2
KP_LEFT_EAR     = 3
KP_RIGHT_EAR    = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW   = 7
KP_RIGHT_ELBOW  = 8
KP_LEFT_WRIST   = 9
KP_RIGHT_WRIST  = 10
KP_LEFT_HIP     = 11
KP_RIGHT_HIP    = 12
KP_LEFT_KNEE    = 13
KP_RIGHT_KNEE   = 14
KP_LEFT_ANKLE   = 15
KP_RIGHT_ANKLE  = 16


def _angle(a, b, c):
    """Angle at point b given three 2D points (in degrees)."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))


def _classify_pose(kpts):
    """
    Classify a person's action from their 17 COCO keypoints.
    kpts: numpy array of shape (17, 3) — [x, y, confidence] per keypoint.

    Returns one of: "standing", "walking", "crouching", "hands_raised", "bending"
    """
    # Only use keypoints with decent confidence
    def kp(idx):
        """Return (x, y) if confidence > 0.3, else None."""
        if kpts[idx][2] > 0.3:
            return kpts[idx][:2].tolist()
        return None

    l_shoulder = kp(KP_LEFT_SHOULDER)
    r_shoulder = kp(KP_RIGHT_SHOULDER)
    l_hip = kp(KP_LEFT_HIP)
    r_hip = kp(KP_RIGHT_HIP)
    l_knee = kp(KP_LEFT_KNEE)
    r_knee = kp(KP_RIGHT_KNEE)
    l_ankle = kp(KP_LEFT_ANKLE)
    r_ankle = kp(KP_RIGHT_ANKLE)
    l_wrist = kp(KP_LEFT_WRIST)
    r_wrist = kp(KP_RIGHT_WRIST)
    nose = kp(KP_NOSE)

    # ── Check "hands raised" first (wrists above shoulders) ──
    if l_wrist and r_wrist and l_shoulder and r_shoulder:
        avg_shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
        if l_wrist[1] < avg_shoulder_y and r_wrist[1] < avg_shoulder_y:
            return "hands_raised"

    # ── Check crouching / bending (hip-knee-ankle angle) ──
    if l_hip and l_knee and l_ankle:
        knee_angle = _angle(l_hip, l_knee, l_ankle)
        if knee_angle < 120:
            return "crouching"
    if r_hip and r_knee and r_ankle:
        knee_angle = _angle(r_hip, r_knee, r_ankle)
        if knee_angle < 120:
            return "crouching"

    # ── Check torso angle for bending ──
    if l_shoulder and r_shoulder and l_hip and r_hip:
        mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2]
        mid_hip = [(l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2]
        torso_dy = mid_hip[1] - mid_shoulder[1]
        torso_dx = abs(mid_hip[0] - mid_shoulder[0])
        if torso_dy > 0:
            torso_angle = math.degrees(math.atan2(torso_dx, torso_dy))
            if torso_angle > 35:
                return "bending"

    # ── Check walking vs standing (ankle/knee separation) ──
    if l_ankle and r_ankle and l_knee and r_knee:
        ankle_spread = abs(l_ankle[0] - r_ankle[0])
        knee_spread = abs(l_knee[0] - r_knee[0])
        # Person's approximate width from shoulders
        body_width = 1.0
        if l_shoulder and r_shoulder:
            body_width = max(1.0, abs(l_shoulder[0] - r_shoulder[0]))
        # Normalized stride: large spread relative to body width means walking
        stride_ratio = ankle_spread / body_width
        if stride_ratio > 0.7:
            return "walking"

    return "standing"


def detect(frame, imgsz=640, conf=0.6):
    """
    Run YOLOv8-Pose on a single OpenCV BGR frame.
    Returns (annotated_frame, detections) where each detection has:
      - class: "person"
      - confidence: float
      - box: [x1, y1, x2, y2]
      - pose_action: "standing" | "walking" | "crouching" | "hands_raised" | "bending"
      - height_px: bounding box pixel height
      - keypoints: list of [x, y, conf] for 17 keypoints (optional)
    """
    results = model(
        frame,
        imgsz=imgsz,
        conf=conf,
        device=DEVICE,
        classes=[0],  # person class only
    )

    r = results[0]
    annotated_frame = r.plot()

    detections = []
    boxes = getattr(r, "boxes", None)
    keypoints_data = getattr(r, "keypoints", None)

    if boxes is None:
        return annotated_frame, detections

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        conf_score = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        cls_name = r.names[cls_id] if hasattr(r, "names") else str(cls_id)

        # Extract keypoints for this person
        pose_action = "unknown"
        kpts_list = []
        if keypoints_data is not None and i < len(keypoints_data):
            kpts = keypoints_data[i].data[0].cpu().numpy()  # shape (17, 3)
            kpts_list = kpts.tolist()
            pose_action = _classify_pose(kpts)

        height_px = abs(y2 - y1)

        det = {
            "class": cls_name,
            "confidence": conf_score,
            "box": [x1, y1, x2, y2],
            "pose_action": pose_action,
            "height_px": round(height_px, 1),
            "keypoints": kpts_list if kpts_list else None,
        }
        detections.append(det)

    # Debug
    for det in detections:
        print(f"{det['class']} {det['confidence']:.3f} | pose: {det['pose_action']} | h_px: {det['height_px']}", flush=True)

    # Draw YOLO-Pose marker
    cv2.putText(
        annotated_frame,
        "YOLO-POSE ON",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return annotated_frame, detections
