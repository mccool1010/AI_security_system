"""Height self-consistency on a clip with no known-height person.

    python eval/height_consistency.py --video ../test_video.mp4

Absolute accuracy needs someone whose height you know (use eval_height.py).
Without that, this measures whether the ground-plane geometry holds together:
it calibrates from half of one person's observations, then measures the same
person on the half held out, as they move nearer to and further from the camera.
Large held-out deviations mean the geometry does not generalise across the frame
— usually too small a subject, or calibration samples clustered at one distance.

The absolute scale comes from --assumed-height, so read the output as
repeatability, not accuracy.
"""
import argparse
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _replay import replay  # noqa: E402  (also sets up sys.path / cwd)
import height_geometry as hg  # noqa: E402
from camera_calibrator import CameraCalibrator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="../test_video.mp4")
    ap.add_argument("--assumed-height", type=float, default=1.75,
                    help="reference height that sets the absolute scale (default 1.75)")
    ap.add_argument("--camera-id", default="consistency-check")
    args = ap.parse_args()

    samples, shape = [], None
    for f in replay(args.video, use_activity=False):
        shape = f["frame"].shape
        for t in f["tracks"]:
            ok, _ = hg.is_measurable(t, shape)
            if ok and t.hits > 3:
                rows = hg.rows_from_track(t, window_s=0.3)
                samples.append({"v_foot": rows["v_foot"], "v_head": rows["v_head"],
                                "height_m": args.assumed_height, "track": t.id,
                                "px": rows["v_foot"] - rows["v_head"]})
    if not samples:
        raise SystemExit("no measurable observations: is anyone fully visible and upright?")

    counts = Counter(s["track"] for s in samples)
    print(f"measurable observations: {len(samples)} from {len(counts)} track(s)")
    # one track only: mixing two people while assuming a single height would fold
    # their height difference into the "error"
    main_track = counts.most_common(1)[0][0]
    samples = [s for s in samples if s["track"] == main_track]
    print(f"using the longest track ({main_track}): {len(samples)} observations")
    if len(samples) < 6:
        raise SystemExit("need at least 6 observations of one person")

    samples.sort(key=lambda s: s["px"])
    print(f"apparent size range: {samples[0]['px']:.0f}px (far) -> {samples[-1]['px']:.0f}px (near)")

    cal = CameraCalibrator(args.camera_id)
    fy, cy = cal.get_focal_length_px(shape), cal.get_principal_point(shape)[1]
    calib, held = samples[::2], samples[1::2]
    params = hg.fit(calib, fy, cy)
    H, th = params["camera_height_m"], math.radians(params["tilt_deg"])
    print(f"\nfitted camera: {H:.2f} m high, {params['tilt_deg']:.1f}deg down  "
          f"(fit RMS {params['rms_cm']} cm, leave-one-out {params['loo_rms_cm']} cm, "
          f"n={len(calib)})")

    errs = []
    for s in held:
        h, _ = hg.height_from_rows(s["v_foot"], s["v_head"], H, th, fy, cy)
        if h is not None:
            errs.append((h - args.assumed_height) * 100)
    errs = np.array(errs)
    print(f"\nheld-out observations: {len(errs)}")
    print(f"  mean absolute deviation : {np.mean(np.abs(errs)):.1f} cm")
    print(f"  RMS deviation           : {np.sqrt(np.mean(errs ** 2)):.1f} cm")
    print(f"  worst                   : {np.max(np.abs(errs)):.1f} cm")
    print(f"  bias                    : {np.mean(errs):+.1f} cm")

    vals = [hg.height_from_rows(s["v_foot"], s["v_head"], H, th, fy, cy)[0] for s in samples]
    vals = [v for v in vals if v is not None]
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    print(f"\nwhat the dashboard would show: {med:.2f} m +/-{(q3 - q1) / 2 * 100:.1f} cm "
          f"over {len(vals)} readings")
    print("\nThis is repeatability, not accuracy - the scale is assumed. "
          "Use eval_height.py with a person of known height for real error.")


if __name__ == "__main__":
    main()
