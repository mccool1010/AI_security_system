"""Face detection + embedding with an explicit, pinned backend.

The backend is chosen by the FACE_BACKEND environment variable, never by
whichever library happens to be importable. Embeddings from different models
are not comparable, so every enrollment is tagged with BACKEND_ID and the
identity store refuses to match across backends (see identity.py).

    FACE_BACKEND=facenet   (default) MTCNN + InceptionResnetV1/VGGFace2, 512-d
    FACE_BACKEND=deepface  DeepFace ArcFace (needs tensorflow), 512-d
    FACE_BACKEND=none      face recognition disabled

Public API:
    BACKEND_ID, BACKEND_ERROR, is_available()
    get_faces_and_embeddings(frame_bgr) -> [{"box": [x1,y1,x2,y2], "embedding": np.ndarray, "prob": float}]
    cosine_similarity(a, b) -> float
    compute_blur_score / compute_brightness / check_face_quality
"""
import os

import cv2
import numpy as np

REQUESTED_BACKEND = os.environ.get("FACE_BACKEND", "facenet").strip().lower()

# Faces smaller than this (pixels, shorter side) give unreliable embeddings.
MIN_FACE_PX = int(os.environ.get("FACE_MIN_PX", "40"))
# MTCNN detection probability required to accept a face.
MIN_FACE_PROB = float(os.environ.get("FACE_MIN_PROB", "0.92"))

BACKEND_ID = "none"
BACKEND_ERROR = None
_impl = None


def _load_facenet():
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=MIN_FACE_PX, post_process=True)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    def embed(frame_bgr):
        from PIL import Image

        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes, probs = mtcnn.detect(img)
        if boxes is None:
            return []
        keep = [
            i for i, (b, p) in enumerate(zip(boxes, probs))
            if p is not None and p >= MIN_FACE_PROB
            and min(b[2] - b[0], b[3] - b[1]) >= MIN_FACE_PX
        ]
        if not keep:
            return []
        boxes, probs = boxes[keep], probs[keep]
        # Reuse the boxes we already have instead of running detection again.
        crops = mtcnn.extract(img, boxes, None)
        if crops is None:
            return []
        if crops.dim() == 3:
            crops = crops.unsqueeze(0)
        with torch.no_grad():
            embs = resnet(crops.to(device)).cpu().numpy()
        h, w = frame_bgr.shape[:2]
        out = []
        for b, p, e in zip(boxes, probs, embs):
            x1, y1, x2, y2 = [int(round(v)) for v in b]
            out.append({
                "box": [max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                "embedding": e.astype(np.float32),
                "prob": float(p),
            })
        return out

    return "facenet-vggface2", embed


def _load_deepface():
    from deepface import DeepFace

    def embed(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            reps = DeepFace.represent(rgb, model_name="ArcFace", detector_backend="mtcnn",
                                      enforce_detection=True)
        except ValueError:  # no face found
            return []
        out = []
        for r in reps:
            fa = r.get("facial_area") or {}
            x, y, w, h = fa.get("x", 0), fa.get("y", 0), fa.get("w", 0), fa.get("h", 0)
            conf = float(r.get("face_confidence", 1.0) or 1.0)
            if min(w, h) < MIN_FACE_PX or conf < MIN_FACE_PROB:
                continue
            out.append({
                "box": [int(x), int(y), int(x + w), int(y + h)],
                "embedding": np.asarray(r["embedding"], dtype=np.float32),
                "prob": conf,
            })
        return out

    return "deepface-arcface", embed


_LOADERS = {"facenet": _load_facenet, "deepface": _load_deepface}

if REQUESTED_BACKEND in _LOADERS:
    try:
        BACKEND_ID, _impl = _LOADERS[REQUESTED_BACKEND]()
    except Exception as e:  # missing package, weight download failure, ...
        BACKEND_ERROR = f"{type(e).__name__}: {e}"
    if _impl is not None:
        print(f"Face backend: {BACKEND_ID}", flush=True)
    else:
        print(f"ERROR: face backend '{REQUESTED_BACKEND}' failed to load, face recognition "
              f"is DISABLED. {BACKEND_ERROR}", flush=True)
elif REQUESTED_BACKEND != "none":
    BACKEND_ERROR = f"unknown FACE_BACKEND '{REQUESTED_BACKEND}'"
    print(f"❌ {BACKEND_ERROR} — face recognition is DISABLED", flush=True)


def is_available():
    return _impl is not None


def get_faces_and_embeddings(frame_bgr):
    """Detect faces and embed them. Returns [] when disabled or nothing is found."""
    if _impl is None or frame_bgr is None or frame_bgr.size == 0 or frame_bgr.ndim != 3:
        return []
    return _impl(frame_bgr)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ─── Quality scoring utilities ────────────────────────────────────────────────

def compute_blur_score(face_crop_bgr):
    """Laplacian variance of the grayscale crop (higher = sharper)."""
    try:
        gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def compute_brightness(face_crop_bgr):
    """Mean grayscale brightness (0-255) of the crop."""
    try:
        return float(cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY).mean())
    except Exception:
        return 128.0


def check_face_quality(frame_bgr, min_blur=30.0, min_bright=50.0, max_bright=220.0, min_face_ratio=0.02):
    """Run all quality checks on a frame.

    Returns dict with: blur_score, brightness, face_size_ratio, has_face, overall_pass
    """
    result = {
        "blur_score": 0.0,
        "brightness": 128.0,
        "face_size_ratio": 0.0,
        "has_face": False,
        "overall_pass": False,
    }
    if frame_bgr is None or frame_bgr.size == 0:
        return result

    h, w = frame_bgr.shape[:2]
    faces = get_faces_and_embeddings(frame_bgr)
    if not faces:
        return result
    result["has_face"] = True

    x1, y1, x2, y2 = faces[0]["box"]
    result["face_size_ratio"] = (max(1, x2 - x1) * max(1, y2 - y1)) / (h * w)
    crop = frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return result

    result["blur_score"] = compute_blur_score(crop)
    result["brightness"] = compute_brightness(crop)
    result["overall_pass"] = (
        result["blur_score"] >= min_blur
        and min_bright <= result["brightness"] <= max_bright
        and result["face_size_ratio"] >= min_face_ratio
    )
    return result
