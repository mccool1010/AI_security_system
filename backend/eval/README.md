# Evaluation kit

Scripts that produce the numbers people ask about: speed, latency, and accuracy.
Speed and latency need nothing but a video file. Accuracy needs labelled data
that only you can record. Each section below says what to record.

All commands run from `backend/` with the virtual environment active.

| Question | Script | Needs |
|---|---|---|
| Inference speed per model, and the hardware | `eval/bench_models.py` | any video |
| Pipeline FPS and camera-to-dashboard latency | `eval/bench_live.py` | a running backend |
| Height error (± cm) against known heights | `eval/eval_height.py` | clips of people with known height |
| Face ID accuracy | `eval/eval_faces.py` | photos of enrolled people and of strangers |
| Activity accuracy per class | `eval/eval_activity.py` | labelled clips |
| Is a bigger pose model worth it? | `eval/compare_pose_models.py` | any video |
| Is the height geometry self-consistent? | `eval/height_consistency.py` | any video with one person |

Every script prints a summary and writes full results with `--out results.json`.

## Speed

```bash
python eval/bench_models.py --device cuda --out models_gpu.json
python eval/bench_models.py --device cpu  --out models_cpu.json
```

Times YOLOv8n-Pose on every frame, plus SlowFast R50 on a 32-frame clip,
MiDaS small, and face detection and embedding on a 320 px crop. The output
includes CPU, GPU, RAM and the PyTorch version.

## Live pipeline and latency

Start the backend, then measure it:

```bash
# terminal 1: replay a video as if it were a camera (DEMO_FPS=0 means as fast as possible)
DEMO_MODE=true DEMO_VIDEO=../test_video.mp4 MONGO_DB=securevision_bench python app.py
# terminal 2
python eval/bench_live.py --seconds 90 --out live.json
```

`bench_live.py` subscribes to `/api/stream` the same way the dashboard does.
Every detection message carries its frame's capture time, so
`frame_to_client_detections` is the true time from camera frame to a browser.
Use a throwaway `MONGO_DB` so benchmark events don't mix with real ones.

Laptops throttle when lightly loaded. On Windows, compare the *Balanced* and
*Best performance* power modes before quoting numbers.

## Pose model size

```bash
python eval/compare_pose_models.py --video ../test_video.mp4
```

Downloads each model on first use and reports speed, how many frames contained a
person, and detection/keypoint confidence. On the bundled overhead clip the nano
model beat both larger ones on detections and keypoint confidence, so the default
stays nano — check yours before upgrading, then set `POSE_MODEL`.

## Height error

1. Mount the camera where it will stay. Calibration is per camera position.
2. **Calibration clip.** Record one person whose height you know. They walk slowly
   towards and away from the camera, covering the whole floor area you care about.
3. **Evaluation clips.** Record at least five *other* people, or the same people on
   different days, one per clip, standing and walking fully in view. Measure each
   person's height with a tape measure, without shoes if possible.
4. Write a manifest:

```json
{
  "camera_id": "cam_local",
  "camera_height_m": null,
  "calibration": [{"video": "calib.mp4", "height_m": 1.78}],
  "clips": [
    {"video": "p1.mp4", "height_m": 1.64, "person": "p1"},
    {"video": "p2.mp4", "height_m": 1.91, "person": "p2"}
  ]
}
```

5. `python eval/eval_height.py manifest.json --out height.json`

You get MAE, RMSE, bias and max error in cm, plus the number of clips, people
and measurements. Report the error together with the number of people, e.g.
"±x cm RMSE over n people".
If you measured the camera's mount height, put it in `camera_height_m`; the fit
is then better constrained.

The live app shows the same calibration's leave-one-out error in
*Settings → Height calibration*.

## Height consistency (no known-height person needed)

```bash
python eval/height_consistency.py --video ../test_video.mp4
```

Calibrates from half of one person's observations and measures them on the other
half as they move nearer and further. It reports repeatability, not accuracy —
the scale comes from `--assumed-height`. Useful to sanity-check a new camera
position before anyone stands in front of it with a tape measure. On the bundled
clip: 6.2 cm leave-one-out, 7.3 cm RMS on held-out readings, on a subject only
68-93 px tall.

## Face ID accuracy

```
probes/
  Alice/      5-20 photos NOT used when Alice enrolled (different days, lighting, angles)
  Bob/        ...
  _unknown/   photos of people who are not enrolled
```

```bash
python eval/eval_faces.py probes/ --out faces.json
```

Enrolled identities are read from MongoDB, and matching uses the same face
model as the live system. You get accuracy, true-accept rate,
false-reject rate, misidentification rate and false-accept rate at the current
threshold. You also get a sweep from 0.30 to 0.90 to help pick a threshold, and
the number of identities and probes. Folders named after people who aren't
enrolled are skipped and listed in the output.

## Activity accuracy per class

```
clips/
  walking/    running/    falling/    loitering/    fighting/    standing/    none/
```

Use any subset of these folders. Aim for at least 20 clips per class, from the
real camera angle you deploy at. Loitering clips must be longer than
`--loiter-seconds` (default 60). Stage falls safely, on mats.

```bash
python eval/eval_activity.py clips/ --out activity.json
python eval/eval_activity.py clips/ --no-slowfast        # posture + dwell only, for comparison
```

You get precision, recall and F1 per class, a confusion matrix, and totals for
clips and frames. The rule that turns a clip into a single label is documented
at the top of the script.
SlowFast was trained on Kinetics (YouTube, mostly eye-level), so on overhead
CCTV its categories can misfire. Run both modes and keep whichever does
better on your footage.
