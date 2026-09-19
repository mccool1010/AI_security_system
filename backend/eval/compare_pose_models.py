"""Compare YOLO pose model sizes on your own footage.

    python eval/compare_pose_models.py --video ../test_video.mp4

Bigger is not automatically better: the larger models are trained on
COCO-style imagery, and on small, overhead CCTV subjects they can find *fewer*
people with *less* confident keypoints — which is what posture and height
measurement actually depend on. Measure before upgrading.

Reported per model: speed, how many frames contained a person at all, mean
detection confidence, and mean keypoint confidence. Detection counts are not
ground truth (a false positive also counts), so read them together with the
confidences, and ideally alongside eval_activity.py on labelled clips.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
PYTHON = sys.executable

CHILD = r'''
import json, os, sys, time
sys.path.insert(0, r"{backend}")
os.chdir(r"{backend}")
import cv2, numpy as np, torch
import detector

cap = cv2.VideoCapture(r"{video}")
frames = []
while True:
    ok, f = cap.read()
    if not ok:
        break
    if f.shape[1] > {width}:
        f = cv2.resize(f, ({width}, int(f.shape[0] * {width} / f.shape[1])))
    frames.append(f)

for _ in range(5):
    detector.detect(frames[0])

times, n_person, confs, kconfs, frames_with = [], 0, [], [], 0
for f in frames:
    if detector.DEVICE == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    dets = detector.detect(f, conf={conf})
    if detector.DEVICE == "cuda":
        torch.cuda.synchronize()
    times.append((time.perf_counter() - t) * 1000)
    n_person += len(dets)
    frames_with += 1 if dets else 0
    for d in dets:
        confs.append(d["confidence"])
        k = np.asarray(d["keypoints"])
        kconfs.append(float(k[:, 2].mean()))

print("RESULT" + json.dumps({{
    "model": os.path.basename(detector.MODEL_PATH),
    "device": detector.DEVICE,
    "frames": len(frames),
    "mean_ms": round(float(np.mean(times)), 2),
    "p95_ms": round(float(np.percentile(times, 95)), 2),
    "detections": n_person,
    "frames_with_person": frames_with,
    "mean_confidence": round(float(np.mean(confs)), 3) if confs else None,
    "mean_keypoint_confidence": round(float(np.mean(kconfs)), 3) if kconfs else None,
}}))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=os.path.join(BACKEND, "..", "test_video.mp4"))
    ap.add_argument("--models", nargs="+",
                    default=["yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt"])
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = []
    for model in args.models:
        env = dict(os.environ, POSE_MODEL=model, PYTHONIOENCODING="utf-8")
        code = CHILD.format(backend=BACKEND, video=os.path.abspath(args.video),
                            width=args.width, conf=args.conf)
        out = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", env=env, cwd=BACKEND)
        line = [l for l in (out.stdout or "").splitlines() if l.startswith("RESULT")]
        if not line:
            print(f"{model}: FAILED\n{(out.stderr or '')[-800:]}")
            continue
        rows.append(json.loads(line[0][6:]))

    if not rows:
        return
    print(f"\n{'model':<20}{'mean ms':>9}{'p95':>8}{'detections':>12}"
          f"{'frames w/ person':>18}{'conf':>7}{'kpt conf':>10}")
    for r in rows:
        print(f"{r['model']:<20}{r['mean_ms']:>9.1f}{r['p95_ms']:>8.1f}{r['detections']:>12}"
              f"{r['frames_with_person']:>18}{r['mean_confidence'] or 0:>7.3f}"
              f"{r['mean_keypoint_confidence'] or 0:>10.3f}")
    print(f"\nRun the winner with POSE_MODEL=<file> (device: {rows[0]['device']}).")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
