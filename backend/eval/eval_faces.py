"""Face identification accuracy against the enrolled identities.

    python eval/eval_faces.py probes/ [--threshold 0.6] [--out results.json]

probes/
  <enrolled name>/*.jpg     photos of enrolled people NOT used for enrollment
  _unknown/*.jpg            photos of people who are not enrolled

Enrollments are read from MongoDB (MONGO_URI / MONGO_DB) and matched with the
configured FACE_BACKEND, exactly as the live system does.

Reports, at the chosen threshold:
  accuracy            correct name for enrolled people, "unknown" for strangers
  true_accept_rate    enrolled probe matched to the right person
  false_reject_rate   enrolled probe returned as unknown
  misidentification   enrolled probe matched to the wrong person
  false_accept_rate   stranger matched to anyone
plus a threshold sweep so you can pick the operating point.
"""
import argparse
import glob
import json
import os

import numpy as np

import _replay  # noqa: F401  (sys.path / cwd)
import cv2  # noqa: E402
import face_utils  # noqa: E402
from identity import IdentityStore  # noqa: E402
from pymongo import MongoClient  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("probes")
ap.add_argument("--threshold", type=float, default=None)
ap.add_argument("--out")
args = ap.parse_args()

if not face_utils.is_available():
    raise SystemExit(f"face backend unavailable: {face_utils.BACKEND_ERROR}")
db = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017/"))[os.environ.get("MONGO_DB", "security_system")]
store = IdentityStore(db["users"], face_utils.BACKEND_ID)
store.reload()
if args.threshold is not None:
    store.threshold = args.threshold
enrolled = {u["name"] for u in store.summary()["matchable"]}
if not enrolled:
    raise SystemExit("no matchable enrolled identities for backend " + face_utils.BACKEND_ID)

records, no_face, skipped_dirs = [], [], []
for d in sorted(os.listdir(args.probes)):
    path = os.path.join(args.probes, d)
    if not os.path.isdir(path):
        continue
    truth = None if d == "_unknown" else d
    if truth is not None and truth not in enrolled:
        skipped_dirs.append(d)
        continue
    for img_path in sorted(glob.glob(os.path.join(path, "*"))):
        img = cv2.imread(img_path)
        if img is None:
            continue
        if max(img.shape[:2]) > 1280:
            s = 1280 / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        faces = face_utils.get_faces_and_embeddings(img)
        if not faces:
            no_face.append(img_path)
            continue
        f = max(faces, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))
        store_threshold = store.threshold
        store.threshold = -1.0            # get the raw best match; thresholds applied below
        best = store.match(f["embedding"])
        store.threshold = store_threshold
        records.append({"image": img_path, "truth": truth, "best": best["name"], "score": best["score"]})


def metrics(th):
    g = [r for r in records if r["truth"] is not None]
    s = [r for r in records if r["truth"] is None]
    pred = lambda r: r["best"] if r["score"] >= th else None  # noqa: E731
    correct = sum(pred(r) == r["truth"] for r in records)
    return {
        "threshold": th,
        "accuracy": round(correct / len(records), 4) if records else None,
        "true_accept_rate": round(sum(pred(r) == r["truth"] for r in g) / len(g), 4) if g else None,
        "false_reject_rate": round(sum(pred(r) is None for r in g) / len(g), 4) if g else None,
        "misidentification_rate": round(sum(pred(r) not in (None, r["truth"]) for r in g) / len(g), 4) if g else None,
        "false_accept_rate": round(sum(pred(r) is not None for r in s) / len(s), 4) if s else None,
    }


main = metrics(store.threshold)
sweep = [metrics(round(t, 2)) for t in np.arange(0.30, 0.91, 0.05)]
per_id = {}
for r in records:
    k = r["truth"] or "_unknown"
    per_id.setdefault(k, {"probes": 0, "correct": 0})
    per_id[k]["probes"] += 1
    per_id[k]["correct"] += (r["best"] if r["score"] >= store.threshold else None) == r["truth"]

result = {
    "face_backend": face_utils.BACKEND_ID,
    "identities_enrolled": len(enrolled),
    "identities_probed": len([k for k in per_id if k != "_unknown"]),
    "genuine_probes": sum(r["truth"] is not None for r in records),
    "impostor_probes": sum(r["truth"] is None for r in records),
    "images_without_detectable_face": len(no_face),
    "skipped_folders_not_enrolled": skipped_dirs,
    "at_threshold": main,
    "per_identity": per_id,
    "sweep": sweep,
}
print(json.dumps({k: v for k, v in result.items() if k != "sweep"}, indent=2))
print(f"\n{'thr':>5}{'acc':>8}{'TAR':>8}{'FRR':>8}{'misID':>8}{'FAR':>8}")
for m in sweep:
    print(f"{m['threshold']:>5.2f}" + "".join(f"{(m[k] if m[k] is not None else float('nan')):>8.3f}"
          for k in ("accuracy", "true_accept_rate", "false_reject_rate", "misidentification_rate", "false_accept_rate")))
if args.out:
    json.dump({**result, "records": records, "no_face": no_face}, open(args.out, "w"), indent=2)
