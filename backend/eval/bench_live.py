"""End-to-end measurement against a running backend (same machine).

    python eval/bench_live.py [--url http://127.0.0.1:5000] [--seconds 60] [--camera cam_local]

Subscribes to the live stream (/api/stream) exactly like the dashboard does.
Every `detections` message carries the capture timestamp of its frame, so
`receive time - capture time` is the true camera-frame-to-client latency.
Also reads /api/perf for per-stage timings and throughput.
"""
import argparse
import json
import os
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:5000")
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--camera", default="cam_local")
ap.add_argument("--warmup", type=float, default=10)
ap.add_argument("--out", default=None)
ap.add_argument("--password", default=os.environ.get("DASHBOARD_PASSWORD"),
                help="dashboard password (default: $DASHBOARD_PASSWORD; printed on first run "
                     "and stored in backend/.secrets/password)")
args = ap.parse_args()

session = requests.Session()


def pct(a, p):
    a = sorted(a)
    return round(a[min(len(a) - 1, int(p * len(a)))], 1) if a else None


def summary(a):
    return {"n": len(a), "mean_ms": round(sum(a) / len(a), 1) if a else None,
            "p50_ms": pct(a, .5), "p95_ms": pct(a, .95), "max_ms": round(max(a), 1) if a else None}


for _ in range(120):
    try:
        session.get(args.url + "/api/health", timeout=2)
        break
    except requests.RequestException:
        time.sleep(1)
else:
    raise SystemExit("backend not reachable")

if session.get(args.url + "/api/perf", timeout=10).status_code == 401:
    if not args.password:
        raise SystemExit("backend requires a password: pass --password or set DASHBOARD_PASSWORD")
    r = session.post(args.url + "/api/login", json={"password": args.password}, timeout=10)
    if r.status_code != 200:
        raise SystemExit(f"login failed: {r.text}")

print(f"warming up {args.warmup:.0f}s ...", flush=True)
time.sleep(args.warmup)
perf0 = session.get(args.url + "/api/perf", timeout=10).json()["cameras"][args.camera]

det_lat, event_lat, n_det, n_evt, persons = [], [], 0, 0, 0
t_end = time.time() + args.seconds
with session.get(args.url + "/api/stream", stream=True, timeout=(5, 30)) as r:
    kind = None
    for line in r.iter_lines(decode_unicode=True):
        now = time.time()
        if now > t_end:
            break
        if not line:
            continue
        if line.startswith("event:"):
            kind = line[6:].strip()
        elif line.startswith("data:"):
            msg = json.loads(line[5:])
            if kind == "detections" and msg.get("camera_id") == args.camera and msg.get("ts"):
                det_lat.append((now - msg["ts"]) * 1000)
                n_det += 1
                persons += msg.get("person_count", 0)
            elif kind == "event" and msg.get("source") == args.camera:
                n_evt += 1
                event_lat.append((now - msg["unix_ts"]) * 1000)

perf1 = session.get(args.url + "/api/perf", timeout=10).json()
cam = perf1["cameras"][args.camera]
frames_cap = cam["frames_captured"] - perf0["frames_captured"]
frames_proc = cam["frames_processed"] - perf0["frames_processed"]

report = {
    "device": perf1.get("device"), "gpu": perf1.get("gpu"), "torch": perf1.get("torch"),
    "window_s": args.seconds,
    "frames_captured": frames_cap,
    "frames_processed": frames_proc,
    "processed_share": round(frames_proc / frames_cap, 3) if frames_cap else None,
    "capture_fps": cam["capture_fps"],
    "pipeline_fps": cam["pipeline_fps"],
    "stages_ms": {k: v and v["mean_ms"] for k, v in cam["stages"].items()},
    "frame_to_result": cam["latency"]["frame_to_result"],
    "frame_to_event_saved": cam["latency"]["frame_to_event"],
    "frame_to_client_detections": summary(det_lat),
    "detection_messages": n_det,
    "person_observations": persons,
    "events_received": n_evt,
    "event_publish_to_client": summary(event_lat),
    "models": {"activity": perf1["activity"], "depth": perf1["depth"], "face": perf1["face"]},
}
print(json.dumps(report, indent=2))
if args.out:
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
