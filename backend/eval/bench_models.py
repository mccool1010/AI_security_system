"""Per-model inference timing on the current machine.

    python eval/bench_models.py [--video ../test_video.mp4] [--frames 200] [--device cpu|cuda]

Times each model in isolation on real frames and prints a table plus JSON.
Run it with --device cpu and --device cuda to compare. Use eval/bench_live.py
for end-to-end numbers from the running system.
"""
import argparse
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)

ap = argparse.ArgumentParser()
ap.add_argument("--video", default=os.path.join(BACKEND, "..", "test_video.mp4"))
ap.add_argument("--frames", type=int, default=200)
ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
ap.add_argument("--face-image", default=os.path.join(BACKEND, "test_images", "alice.jpg"))
ap.add_argument("--out", default=None, help="write JSON results here")
args = ap.parse_args()
if args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # must happen before torch is imported

sys.path.insert(0, BACKEND)
os.chdir(BACKEND)
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import detector  # noqa: E402


def summary(ms):
    ms = sorted(ms)
    return {"n": len(ms), "mean_ms": round(float(np.mean(ms)), 2), "p50_ms": round(ms[len(ms) // 2], 2),
            "p95_ms": round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 2),
            "fps_equiv": round(1000 / float(np.mean(ms)), 1)}


def timeit(fn, n, warmup=5):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        if detector.DEVICE == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if detector.DEVICE == "cuda":
            torch.cuda.synchronize()
        out.append((time.perf_counter() - t) * 1000)
    return summary(out)


def cpu_name():
    if platform.system() == "Windows":
        try:
            import subprocess
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor | Select-Object -First 1).Name"],
                                 capture_output=True, text=True, timeout=10)
            return out.stdout.strip() or platform.processor()
        except Exception:
            pass
    return platform.processor()


def hardware():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "device": detector.DEVICE, "cpu": cpu_name(),
            "cpu_threads": os.cpu_count()}
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        pass
    if detector.DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_mem_gb"] = round(p.total_memory / 2**30, 1)
    return info


cap = cv2.VideoCapture(args.video)
frames = []
while len(frames) < args.frames:
    ok, f = cap.read()
    if not ok:
        break
    if f.shape[1] > 640:
        f = cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1])))
    frames.append(f)
if not frames:
    sys.exit(f"could not read frames from {args.video}")
H, W = frames[0].shape[:2]
results = {"hardware": hardware(), "input": {"video": os.path.abspath(args.video), "frames": len(frames),
                                             "resolution": [W, H]}, "models": {}}
print(json.dumps(results["hardware"], indent=1))

# YOLOv8n-Pose, every frame
it = iter(range(10**9))
results["models"]["yolov8n_pose"] = timeit(lambda: detector.detect(frames[next(it) % len(frames)]), len(frames))

# SlowFast R50 on a person-centred 32-frame clip
import action_recognizer as ar  # noqa: E402

model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True).to(detector.DEVICE).eval()
clip = np.stack([cv2.cvtColor(cv2.resize(f, (ar.SIDE_SIZE, ar.SIDE_SIZE)), cv2.COLOR_BGR2RGB)
                 for f in frames[:ar.CLIP_LEN]]).astype(np.float32) / 255.0
fast = torch.from_numpy((clip - ar.MEAN) / ar.STD).permute(3, 0, 1, 2).unsqueeze(0).to(detector.DEVICE)
slow = fast[:, :, ::4]


def run_slowfast():
    with torch.no_grad():
        model([slow, fast])


results["models"]["slowfast_r50"] = timeit(run_slowfast, 30)

# MiDaS small
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(detector.DEVICE).eval()
tf = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
inp = tf(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)).to(detector.DEVICE)


def run_midas():
    with torch.no_grad():
        midas(inp)


results["models"]["midas_small"] = timeit(run_midas, 30)

# Face detection + embedding on a person-sized crop that contains a face
os.environ.setdefault("FACE_BACKEND", "facenet")
import face_utils  # noqa: E402

if face_utils.is_available() and os.path.exists(args.face_image):
    img = cv2.imread(args.face_image)
    crop = cv2.resize(img, (320, int(img.shape[0] * 320 / img.shape[1])))
    n_faces = len(face_utils.get_faces_and_embeddings(crop))
    results["models"]["face_detect_embed"] = dict(timeit(lambda: face_utils.get_faces_and_embeddings(crop), 30),
                                                  crop=[crop.shape[1], crop.shape[0]], faces=n_faces,
                                                  backend=face_utils.BACKEND_ID)

print(f"\n{'model':<20}{'n':>5}{'mean ms':>10}{'p50':>9}{'p95':>9}{'≈fps':>8}")
for k, v in results["models"].items():
    print(f"{k:<20}{v['n']:>5}{v['mean_ms']:>10.1f}{v['p50_ms']:>9.1f}{v['p95_ms']:>9.1f}{v['fps_equiv']:>8.1f}")
if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
