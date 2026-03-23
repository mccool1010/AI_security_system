"""Face utilities with optional DeepFace, falling back to facenet-pytorch or stubs.

Functions provided:
- `get_faces_and_embeddings(frame_bgr)` -> list of {"box": [x1,y1,x2,y2], "embedding": np.array}
- `cosine_similarity(a, b)` -> float similarity in [-1,1]
"""

try:
    # Prefer DeepFace (high-level; supports many backends). This is easier
    # to install on Windows than some other heavy deps if you opt into it.
    from deepface import DeepFace
    import cv2
    import numpy as np

    FACE_MODEL = "ArcFace"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


    def get_faces_and_embeddings(frame_bgr):
        """Detect faces (OpenCV Haar) and compute embeddings with DeepFace.

        Returns list of dicts: {"box": [x1,y1,x2,y2], "embedding": np.ndarray}
        """
        print(f"  FACE_UTILS: Called with frame shape {frame_bgr.shape if frame_bgr is not None else None}", flush=True)
        if frame_bgr is None or frame_bgr.size == 0:
            print(f"  FACE_UTILS: Empty frame, returning", flush=True)
            return []
        
        # Ensure frame is in proper format
        if len(frame_bgr.shape) != 3:
            print(f"  FACE_UTILS: Invalid shape, returning", flush=True)
            return []
        
        # Convert BGR to RGB for proper color space (OpenCV uses BGR, DeepFace expects RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with different scales - more aggressive to handle various conditions
        dets = []
        scales_to_try = [1.05, 1.1, 1.15, 1.2, 1.3]
        for scale in scales_to_try:
            dets = face_cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=5, minSize=(15, 15))
            if len(dets) > 0:
                print(f"    FACE_UTILS: Haar found {len(dets)} faces at scale {scale}", flush=True)
                break
        
        if len(dets) == 0:
            print(f"    FACE_UTILS: No faces detected in frame of size {frame_bgr.shape}", flush=True)
            # Try a last-resort approach: use DeepFace directly on the whole frame (use RGB)
            try:
                print(f"    FACE_UTILS: Attempting DeepFace detection on entire frame", flush=True)
                result = DeepFace.detect(frame_rgb, detector_backend='mtcnn', enforce_detection=False)
                if isinstance(result, list) and len(result) > 0:
                    print(f"    FACE_UTILS: DeepFace found {len(result)} faces", flush=True)
                    dets = []
                    for detection in result:
                        if isinstance(detection, dict) and 'facial_area' in detection:
                            fa = detection['facial_area']
                            dets.append((fa['x'], fa['y'], fa['w'], fa['h']))
            except Exception as e:
                print(f"    FACE_UTILS: DeepFace detection failed: {e}", flush=True)
        
        faces = []
        for idx, det in enumerate(dets):
            if isinstance(det, tuple) and len(det) == 4:
                x, y, w, h = det
            elif isinstance(det, tuple) and len(det) >= 4:
                x, y, w, h = det[0], det[1], det[2], det[3]
            elif isinstance(det, np.ndarray):
                x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            else:
                continue
            
            if w < 15 or h < 15:  # Skip very small detections
                print(f"    FACE_UTILS: Skipping detection {idx}: too small ({w}x{h})", flush=True)
                continue
            
            crop = frame_bgr[y : y + h, x : x + w]
            if crop.size == 0:
                print(f"    FACE_UTILS: Skipping detection {idx}: empty crop", flush=True)
                continue
            
            print(f"    FACE_UTILS: Processing detection {idx}: crop {crop.shape}", flush=True)
            
            # Convert crop from BGR to RGB for DeepFace
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            vec = None
            try:
                # Try DeepFace first (expects RGB)
                emb_result = DeepFace.represent(crop_rgb, model_name=FACE_MODEL, enforce_detection=False)
                print(f"    FACE_UTILS: DeepFace returned type {type(emb_result)}", flush=True)
                
                # Extract embedding from various response formats
                if isinstance(emb_result, dict) and "embedding" in emb_result:
                    vec = np.array(emb_result["embedding"]).astype(float)
                elif isinstance(emb_result, list) and len(emb_result) > 0:
                    if isinstance(emb_result[0], dict) and "embedding" in emb_result[0]:
                        vec = np.array(emb_result[0]["embedding"]).astype(float)
                    else:
                        vec = np.array(emb_result[0]).astype(float)
                else:
                    vec = np.array(emb_result).astype(float).flatten()
                    
                print(f"    FACE_UTILS: Detection {idx} extracted embedding shape {vec.shape}", flush=True)
                
            except Exception as e:
                print(f"    FACE_UTILS: DeepFace extraction error for detection {idx}: {e}", flush=True)
                try:
                    # Fallback: try again with slightly different settings
                    emb_result = DeepFace.represent(crop_rgb, model_name=FACE_MODEL, enforce_detection=False)
                    if emb_result:
                        vec = np.array(emb_result[0]["embedding"] if isinstance(emb_result, list) else emb_result["embedding"]).astype(float)
                        print(f"    FACE_UTILS: Fallback succeeded for detection {idx}", flush=True)
                except Exception as e2:
                    print(f"    FACE_UTILS: Fallback also failed for detection {idx}: {e2}", flush=True)
                    vec = None
            
            if vec is not None and vec.size > 0:
                faces.append({"box": [int(x), int(y), int(x + w), int(y + h)], "embedding": vec})
                print(f"    FACE_UTILS: Detection {idx} added to faces", flush=True)
            else:
                print(f"    FACE_UTILS: Detection {idx} failed to produce embedding", flush=True)

        print(f"  FACE_UTILS: Total faces extracted: {len(faces)}", flush=True)
        return faces


    def cosine_similarity(a, b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        if a.size == 0 or b.size == 0:
            return 0.0
        an = a / (np.linalg.norm(a) + 1e-12)
        bn = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(an, bn))

except Exception:
    # Try facenet-pytorch next (older fallback used previously).
    try:
        from facenet_pytorch import MTCNN, InceptionResnetV1
        import torch
        import numpy as np
        from PIL import Image

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        mtcnn = MTCNN(keep_all=True, device=DEVICE)
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


        def pil_from_bgr(frame_bgr):
            return Image.fromarray(frame_bgr[:, :, ::-1])


        def get_faces_and_embeddings(frame_bgr):
            img = pil_from_bgr(frame_bgr)
            boxes, probs = mtcnn.detect(img)
            if boxes is None:
                return []

            face_tensors = mtcnn(img)
            if face_tensors is None:
                return []

            if isinstance(face_tensors, torch.Tensor) and face_tensors.dim() == 3:
                face_tensors = face_tensors.unsqueeze(0)

            faces = []
            for i, t in enumerate(face_tensors):
                t = t.to(DEVICE)
                with torch.no_grad():
                    emb = resnet(t.unsqueeze(0)).cpu().numpy().flatten()
                box = boxes[i].tolist() if boxes is not None else None
                faces.append({"box": box, "embedding": emb})
            return faces


        def l2_normalize(x):
            x = np.array(x, dtype=float)
            norm = np.linalg.norm(x)
            if norm == 0:
                return x
            return x / norm


        def cosine_similarity(a, b):
            a = l2_normalize(a)
            b = l2_normalize(b)
            return float(np.dot(a, b))

    except Exception as e:
        # Final fallback: stubs so the app runs even without heavy face libs.
        print("Warning: no supported face library available; face functions are stubbed:", e, flush=True)


        def get_faces_and_embeddings(frame_bgr):
            return []


        def cosine_similarity(a, b):
            try:
                import math

                a = [float(x) for x in (a or [])]
                b = [float(x) for x in (b or [])]
                if not a or not b:
                    return 0.0

                def norm(v):
                    s = sum(x * x for x in v)
                    return math.sqrt(s) if s > 0 else 0.0

                na = [x / norm(a) if norm(a) > 0 else 0.0 for x in a]
                nb = [x / norm(b) if norm(b) > 0 else 0.0 for x in b]
                d = sum(x * y for x, y in zip(na, nb))
                return float(d)
            except Exception:
                return 0.0


# ─── Quality scoring utilities (always available) ─────────────────────────────
import cv2 as _cv2
import numpy as _np


def compute_blur_score(face_crop_bgr):
    """Return Laplacian variance of the grayscale face crop (higher = sharper)."""
    try:
        gray = _cv2.cvtColor(face_crop_bgr, _cv2.COLOR_BGR2GRAY)
        return float(_cv2.Laplacian(gray, _cv2.CV_64F).var())
    except Exception:
        return 0.0


def compute_brightness(face_crop_bgr):
    """Return mean pixel brightness (0-255) of the grayscale face crop."""
    try:
        gray = _cv2.cvtColor(face_crop_bgr, _cv2.COLOR_BGR2GRAY)
        return float(gray.mean())
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
    frame_area = h * w

    faces = get_faces_and_embeddings(frame_bgr)
    if not faces:
        return result
    result["has_face"] = True

    # Use the first (largest) face
    box = faces[0].get("box", [0, 0, w, h])
    x1, y1, x2, y2 = box
    face_w, face_h = max(1, x2 - x1), max(1, y2 - y1)
    result["face_size_ratio"] = (face_w * face_h) / frame_area

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
