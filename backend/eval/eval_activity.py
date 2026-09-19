"""Per-class activity recognition accuracy on labelled clips.

    python eval/eval_activity.py clips/ [--no-slowfast] [--loiter-seconds 60] [--out results.json]

clips/
  walking/*.mp4   running/*.mp4   falling/*.mp4   loitering/*.mp4
  fighting/*.mp4  standing/*.mp4  none/*.mp4        (any subset)

Each clip is replayed through the live components (YOLO-Pose, tracker,
posture, dwell time, SlowFast) and assigned ONE label by this rule, highest
priority first:

  falling    a person's confirmed posture was falling/lying_down, or SlowFast "falling"
  fighting   confirmed posture fighting, or SlowFast "fighting"
  loitering  a person stayed in view >= --loiter-seconds
  running    >= 30% of person-frames running, or SlowFast "running"
  walking    >= 30% of person-frames walking
  standing   a person was present
  none       nobody detected

Loitering clips must be longer than --loiter-seconds.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import _replay  # noqa: F401
import tracking  # noqa: E402
from _replay import replay  # noqa: E402

LABELS = ["falling", "fighting", "loitering", "running", "walking", "standing", "none"]
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")

ap = argparse.ArgumentParser()
ap.add_argument("clips")
ap.add_argument("--no-slowfast", action="store_true")
ap.add_argument("--loiter-seconds", type=float, default=tracking.LOITER_S)
ap.add_argument("--share", type=float, default=0.30)
ap.add_argument("--out")
args = ap.parse_args()
tracking.LOITER_S = args.loiter_seconds
tracking.DWELL_LABELS = ((5.0, "passing"), (15.0, "visitor"), (args.loiter_seconds, "lingering"))


def predict(path):
    stable = Counter()
    sf = set()
    max_dwell, person_frames, frames = 0.0, 0, 0
    for f in replay(path, use_activity=not args.no_slowfast):
        frames += 1
        for t in f["tracks"]:
            person_frames += 1
            stable[t.stable_pose] += 1
            max_dwell = max(max_dwell, t.dwell_s)
        sf.update(a["category"] for a in f["activities"])
    share = lambda p: stable[p] / person_frames if person_frames else 0  # noqa: E731
    if stable["falling"] or stable["lying_down"] or "falling" in sf:
        label = "falling"
    elif stable["fighting"] or "fighting" in sf:
        label = "fighting"
    elif max_dwell >= args.loiter_seconds:
        label = "loitering"
    elif share("running") >= args.share or "running" in sf:
        label = "running"
    elif share("walking") >= args.share:
        label = "walking"
    elif person_frames:
        label = "standing"
    else:
        label = "none"
    return label, frames, {"poses": dict(stable), "slowfast": sorted(sf), "max_dwell_s": round(max_dwell, 1)}


truths, preds, details, total_frames = [], [], [], 0
for label in sorted(os.listdir(args.clips)):
    folder = os.path.join(args.clips, label)
    if not os.path.isdir(folder):
        continue
    if label not in LABELS:
        print(f"skipping unknown label folder '{label}' (expected one of {LABELS})")
        continue
    for clip in sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(VIDEO_EXT)):
        pred, n, info = predict(clip)
        total_frames += n
        truths.append(label)
        preds.append(pred)
        details.append({"clip": clip, "truth": label, "pred": pred, "frames": n, **info})
        print(f"  [{'ok' if pred == label else 'XX'}] {label:<10} -> {pred:<10} {os.path.basename(clip)}")

classes = [c for c in LABELS if c in truths or c in preds]
per_class = {}
for c in classes:
    tp = sum(t == c and p == c for t, p in zip(truths, preds))
    fp = sum(t != c and p == c for t, p in zip(truths, preds))
    fn = sum(t == c and p != c for t, p in zip(truths, preds))
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else (0.0 if prec is not None and rec is not None else None)
    per_class[c] = {"clips": truths.count(c), "precision": prec and round(prec, 3),
                    "recall": rec and round(rec, 3), "f1": f1 if f1 is None else round(f1, 3)}
confusion = defaultdict(Counter)
for t, p in zip(truths, preds):
    confusion[t][p] += 1

result = {
    "slowfast": not args.no_slowfast,
    "loiter_seconds": args.loiter_seconds,
    "clips": len(truths),
    "frames": total_frames,
    "accuracy": round(sum(t == p for t, p in zip(truths, preds)) / len(truths), 3) if truths else None,
    "per_class": per_class,
    "confusion": {t: dict(c) for t, c in confusion.items()},
}
print(json.dumps(result, indent=2))
if args.out:
    json.dump({**result, "details": details}, open(args.out, "w"), indent=2)
