"""Height measurement error against people of known height.

    python eval/eval_height.py manifest.json [--out results.json]

manifest.json (all clips from the same, fixed camera):
{
  "camera_id": "cam_local",            // intrinsics from calibration_data/<id>.json if present
  "camera_height_m": null,              // optional: fix the mount height if you measured it
  "calibration": [                      // reference person walking/standing at several distances
    {"video": "clips/calib_ref.mp4", "height_m": 1.78}
  ],
  "clips": [                            // DIFFERENT people (or different recordings) to evaluate
    {"video": "clips/p1.mp4", "height_m": 1.64, "person": "p1"},
    {"video": "clips/p2.mp4", "height_m": 1.91, "person": "p2"}
  ]
}

Each clip should show one person, fully visible, standing or walking. The
calibration is fitted from the calibration clips only, then every evaluation
clip is measured independently; the error per clip is median(measurements) -
true height.
"""
import argparse
import json
import math
import os

import numpy as np

from _replay import replay  # noqa: E402  (sets up sys.path / cwd)
import height_geometry as hg  # noqa: E402
from camera_calibrator import CameraCalibrator  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("manifest")
ap.add_argument("--sample-every", type=float, default=1.0, help="seconds between calibration samples")
ap.add_argument("--out")
args = ap.parse_args()
base = os.path.dirname(os.path.abspath(args.manifest))
man = json.load(open(args.manifest))


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(base, p)


def main_track(tracks):
    tracks = [t for t in tracks if t.hits > 3]
    return max(tracks, key=lambda t: t.box[3] - t.box[1]) if tracks else None


# 1. calibration samples
samples, shape = [], None
for c in man["calibration"]:
    last = -1e9
    for f in replay(resolve(c["video"]), use_activity=False):
        shape = f["frame"].shape
        t = main_track(f["tracks"])
        if t is None or f["ts"] - last < args.sample_every:
            continue
        ok, _ = hg.is_measurable(t, shape)
        if ok:
            rows = hg.rows_from_track(t, window_s=0.5)
            samples.append({"v_foot": rows["v_foot"], "v_head": rows["v_head"], "height_m": c["height_m"]})
            last = f["ts"]
print(f"calibration samples: {len(samples)}")
cal = CameraCalibrator(man.get("camera_id", "cam_local"))
fy, cy = cal.get_focal_length_px(shape), cal.get_principal_point(shape)[1]
params = hg.fit(samples, fy, cy, man.get("camera_height_m"))
print(f"camera height {params['camera_height_m']} m, tilt {params['tilt_deg']}°, "
      f"fit RMS {params['rms_cm']} cm, leave-one-out RMS {params['loo_rms_cm']} cm")
H, th = params["camera_height_m"], math.radians(params["tilt_deg"])

# 2. evaluation clips
rows_out, n_meas, n_frames = [], 0, 0
for c in man["clips"]:
    vals = []
    for f in replay(resolve(c["video"]), use_activity=False):
        n_frames += 1
        t = main_track(f["tracks"])
        if t is None:
            continue
        ok, _ = hg.is_measurable(t, f["frame"].shape)
        if not ok:
            continue
        r = hg.rows_from_track(t, window_s=0.3)
        h, _ = hg.height_from_rows(r["v_foot"], r["v_head"], H, th, fy, cy)
        if h is not None and 0.5 < h < 2.6:
            vals.append(h)
    n_meas += len(vals)
    est = float(np.median(vals)) if vals else None
    err = None if est is None else est - c["height_m"]
    rows_out.append({"video": c["video"], "person": c.get("person"), "true_m": c["height_m"],
                     "measured_m": None if est is None else round(est, 3),
                     "error_cm": None if err is None else round(err * 100, 1), "measurements": len(vals)})
    print(f"  {c['video']}: true {c['height_m']:.3f}  measured {est if est is None else round(est, 3)}  "
          f"error {rows_out[-1]['error_cm']} cm  (n={len(vals)})")

errs = np.array([r["error_cm"] for r in rows_out if r["error_cm"] is not None])
result = {
    "calibration": params,
    "clips": rows_out,
    "clips_evaluated": int(len(errs)),
    "clips_without_measurement": sum(r["error_cm"] is None for r in rows_out),
    "people": len({r["person"] or r["video"] for r in rows_out}),
    "measurements": n_meas,
    "frames": n_frames,
    "mae_cm": round(float(np.mean(np.abs(errs))), 2) if len(errs) else None,
    "rmse_cm": round(float(np.sqrt(np.mean(errs ** 2))), 2) if len(errs) else None,
    "bias_cm": round(float(np.mean(errs)), 2) if len(errs) else None,
    "max_abs_cm": round(float(np.max(np.abs(errs))), 2) if len(errs) else None,
}
print(json.dumps({k: v for k, v in result.items() if k != "clips"}, indent=2))
if args.out:
    json.dump(result, open(args.out, "w"), indent=2)
