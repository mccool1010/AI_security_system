# 🏃‍♂️ How to Run SecureVision Locally

## 📦 Prerequisites
1. **Python 3.12** for the backend.
2. **Node.js 18+** for the dashboard.
3. **MongoDB**, either local or started by Docker (see below).
4. **A webcam or IP/RTSP camera.** Camera 0 is used by default; set `CAMERA_SOURCE` to change it.
5. **Optional: an NVIDIA GPU.** RTX 50-series cards need a recent driver. The installer detects the GPU and installs CUDA 12.8 PyTorch.

---

## 🛠️ Step 1: Install

### Windows
Double-click **`install.bat`**. It will:
- create `backend\venv`;
- install PyTorch 2.7.1 (CUDA 12.8 if `nvidia-smi` works, otherwise the CPU build);
- install the backend requirements and the dashboard packages.

Set `FORCE_CPU=1` before running it to force the CPU build.

### Manual (any OS)
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate    macOS/Linux: source venv/bin/activate

# 1. PyTorch for your hardware
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128   # NVIDIA GPU
# pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu  # CPU only

# 2. Everything else
pip install -r requirements.txt
pip install --no-deps -r requirements-nodeps.txt
```
`requirements-nodeps.txt` contains facenet-pytorch and pytorchvideo. Their package metadata pins old PyTorch versions, so they're installed without their declared dependencies. The code works with PyTorch 2.7, and the test suite checks this.

```bash
cd ../dashboard
npm install
```

Model weights download automatically on first start: YOLOv8-Pose, SlowFast, FaceNet, and MiDaS if enabled.

---

## 🚀 Step 2: Start

### Windows
Double-click **`run.bat`**. It starts MongoDB in Docker if Docker is running, then the backend and the dashboard, and opens http://localhost:5173.

### Manual
```bash
# terminal 1
cd backend
venv/Scripts/python app.py          # macOS/Linux: venv/bin/python app.py

# terminal 2
cd dashboard
npm run dev                          # http://localhost:5173
```

On startup the backend prints the device in use (`cuda` or `cpu`), the face backend, whether any enrolled faces need re-enrolling — and, on the very first run, the generated dashboard password:

```
==============================================================
  Dashboard password: 7Kd2p-QvXm9
  Generated on first run and saved in backend/.secrets/password
  Change it with the DASHBOARD_PASSWORD environment variable.
==============================================================
```

The dashboard asks for it once and keeps a session cookie for 30 days. Everything
except the health check requires it, including the camera streams. To run without
a password on a machine nothing else can reach, set `AUTH_DISABLED=1`.

The backend serves through **waitress**; use `FLASK_DEV=1` for Flask's reloading
development server.

### Demo mode (no camera)
```bash
DEMO_MODE=true DEMO_VIDEO=../test_video.mp4 venv/Scripts/python app.py
```

---

## 📏 Step 3: Calibrate height (once per camera position)
Go to **Settings → Height calibration**:
1. Enter the height of a person standing fully in view and press **Record sample**.
2. Repeat at three or more distances from the camera.
3. Press **Calibrate**. Enter the camera's mount height first if you measured it.

Until then, the dashboard shows "camera not calibrated" instead of a height.

## 🧑 Step 4: Enroll faces
Use **Enroll Face** in the dashboard. Registering the same name again adds samples to that person. The page lists anyone enrolled with an older face model who needs to register again.

---

## 🐳 Docker (MongoDB + dashboard)

The backend runs natively so it can reach USB cameras and the GPU. Docker runs MongoDB and a production build of the dashboard:

```bash
docker compose up -d --build        # dashboard: http://localhost:3000
cd backend && venv/Scripts/python app.py
```

`deploy.bat` / `deploy.sh` do both steps.

## 🌐 Portfolio demo build
```bash
cd dashboard
npm run build:demo                   # simulated data, landing page, "demo mode" banner
```
A normal `npm run build` never shows simulated data. If the backend is unreachable, the dashboard says so.

---

## ✅ Tests and benchmarks
```bash
cd backend
venv/Scripts/python -m pip install pytest
venv/Scripts/python -m pytest -q
```
See [backend/eval/README.md](backend/eval/README.md) to measure speed, latency and accuracy on your own hardware and footage.

## 🎥 Overlays and event clips
The live stream is drawn with boxes, skeletons, trails, identity, posture and a
risk badge; set `OVERLAYS=0` for a clean feed. Every event also saves an MP4 of
the seconds around it (`ENABLE_CLIPS=0` to disable) under
`backend/static/clips/<camera>/`, pruned after 30 days or 2 GB. Recorded clips are
never drawn on, so stored footage stays original.

## 🛑 Stopping
Press `Ctrl + C` in both terminals, or run `stop.bat`.
