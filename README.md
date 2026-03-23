<div align="center">

# 🛡️ SecureVision — AI Surveillance Platform

**Real-time security monitoring with advanced ML-powered detection, action recognition, and face matching.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-00FFFF?style=flat-square)](https://docs.ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 🎯 Overview

SecureVision is a full-stack AI surveillance system that combines **YOLOv8-Pose** for real-time person detection and pose classification, **SlowFast R50** for advanced activity recognition (400 action classes), and **DeepFace** for face enrollment and matching — all streamed through a professional React dashboard at **~30fps**.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎥 **Live MJPEG Streaming** | ~30fps video feed with ML overlays via dual-thread architecture |
| 🏃 **Pose Detection** | YOLOv8-Pose classifies standing, sitting, crouching, lying down |
| 🧠 **Activity Recognition** | SlowFast R50 (Kinetics-400) — 400 human activities with temporal context |
| 👤 **Face Recognition** | DeepFace enrollment + real-time matching with cosine similarity |
| 📏 **Height Estimation** | Auto-calibrated pixel-to-meter height estimation |
| ⚠️ **Risk Assessment** | Automatic risk scoring (LOW/MEDIUM/HIGH) based on behavior + identity |
| 📊 **Event Timeline** | MongoDB-backed event storage with screenshots, timestamps, and metadata |
| 🎨 **Professional Dashboard** | Dark-themed React UI with animations, metrics, and multi-camera layouts |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        React Dashboard                         │
│   (Vite + React 19 + TailwindCSS + Chart.js)                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────┐    │
│   │Dashboard │  │ Events   │  │ Settings │  │ Face      │    │
│   │  Page    │  │  Page    │  │  Page    │  │ Enroll    │    │
│   └──────────┘  └──────────┘  └──────────┘  └───────────┘    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP + MJPEG
┌──────────────────────────┴──────────────────────────────────────┐
│                     Flask Backend (Python)                       │
│                                                                  │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ Streaming   │  │   ML Worker     │  │   Face Utils     │   │
│  │ Thread      │  │   Thread        │  │   (DeepFace)     │   │
│  │ (~30fps)    │  │                 │  │                  │   │
│  │ camera.read │  │ YOLOv8-Pose     │  │ Enrollment       │   │
│  │ draw overlay│  │ Motion Detect   │  │ Embedding match  │   │
│  │ JPEG encode │  │ SlowFast R50    │  │ Cosine similarity│   │
│  └─────────────┘  │ Face matching   │  └──────────────────┘   │
│                    │ Event generation│                          │
│                    └─────────────────┘                          │
│                              │                                  │
│                    ┌─────────┴─────────┐                       │
│                    │    MongoDB        │                       │
│                    │  Events + Users   │                       │
│                    └───────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **MongoDB** (running locally or Atlas connection string)
- **Webcam** (built-in or USB)

### One-Click Install
```bash
# Double-click install.bat or run:
.\install.bat
```

### One-Click Run
```bash
# Double-click run.bat or run:
.\run.bat
```

This starts both the backend (port 5000) and dashboard (port 5173), then opens your browser.

### Manual Setup

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

# Dashboard (new terminal)
cd dashboard
npm install
npm run dev
```

## 🧠 ML Pipeline

| Model | Purpose | Pretrained On | Runs On |
|-------|---------|---------------|---------|
| **YOLOv8n-Pose** | Person detection + pose keypoints | COCO | Every ML cycle (~88ms) |
| **SlowFast R50** | Activity recognition (400 classes) | Kinetics-400 | Background thread (every 2s) |
| **DeepFace** | Face embedding extraction | VGGFace2 | On event trigger only |

## 📁 Project Structure

```
security-system/
├── backend/
│   ├── app.py                 # Flask server + dual-thread pipeline
│   ├── detector.py            # YOLOv8-Pose wrapper
│   ├── action_recognizer.py   # SlowFast R50 wrapper
│   ├── face_utils.py          # DeepFace enrollment + matching
│   └── requirements.txt       # Python dependencies
├── dashboard/
│   ├── src/
│   │   ├── pages/             # Dashboard, Events, Settings, etc.
│   │   ├── layout/            # DashboardLayout with sidebar
│   │   └── index.css          # Design system (animations, cards, etc.)
│   └── package.json           # Node dependencies
├── install.bat                # One-click installer
├── run.bat                    # One-click launcher
└── stop.bat                   # One-click shutdown
```

## 🖥️ Dashboard

The dashboard features a professional dark theme with:
- **Multi-camera layouts** — Focus (spotlight), Grid, and Stack views
- **Real-time metrics** — Status, events/hour, events/today, risk level
- **Event timeline** — Searchable with screenshots and ML metadata
- **Settings panel** — Tune detection thresholds, alert cooldown, confidence
- **Face enrollment** — Upload photos to enroll known users

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <sub>Built with ❤️ using PyTorch, YOLOv8, SlowFast, DeepFace, Flask, and React</sub>
</div>
"# AI_security_system" 
