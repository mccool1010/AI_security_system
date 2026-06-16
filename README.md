# SecureVision AI 🛡️

![SecureVision Hero](https://images.unsplash.com/photo-1557597774-9d273605dfa9?q=80&w=2070&auto=format&fit=crop)

> A full-stack, enterprise-grade intelligent surveillance system built from scratch with 5 deep learning models working in concert.

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://dist-alpha-ten-64.vercel.app)
[![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

SecureVision replaces legacy security cameras with an AI-driven pipeline capable of real-time pose estimation, precision height measurement, activity recognition, and face identification. 

---

## 🔗 Live Demo
Check out the fully interactive portfolio and mock-data demo dashboard here:
**[SecureVision Live Demo](https://dist-alpha-ten-64.vercel.app)**

---

## ✨ Features

- **🚶‍♂️ Real-time Activity Recognition**: Utilizes SlowFast R50 to detect complex human behaviors (walking, running, falling, loitering).
- **📏 Precision Height Measurement**: Fuses YOLOv8-Pose with MiDaS depth mapping for highly accurate human height estimation.
- **🧑 Face Identification**: Integrates DeepFace and RetinaFace for seamless personnel recognition.
- **📊 Real-time Dashboard**: A stunning, hardware-accelerated Vite/React dashboard for live monitoring.
- **🛡️ Risk Scoring**: Automated, programmable threat assessment logic based on posture and activity.

---

## 🏗️ System Architecture

SecureVision operates on a decoupled microservices architecture, utilizing WebSockets and MJPEG streams for ultra-low latency inference visualization.

```mermaid
graph TD
    %% Define Styles
    classDef client fill:#1A1A24,stroke:#4F46E5,stroke-width:2px,color:#fff
    classDef api fill:#1A1A24,stroke:#10B981,stroke-width:2px,color:#fff
    classDef ml fill:#1A1A24,stroke:#E11D48,stroke-width:2px,color:#fff
    
    subgraph Frontend [React / Vite Dashboard]
        UI[User Interface]:::client
        SocketClient[WebSocket Hook]:::client
    end
    
    subgraph Backend [Flask API Server]
        Streamer[MJPEG Streamer]:::api
        REST[REST API]:::api
        SocketServer[Socket.IO Server]:::api
    end
    
    subgraph MLPipeline [AI Inference Engine]
        YOLO[YOLOv8 Pose]:::ml
        MiDaS[MiDaS Depth]:::ml
        SlowFast[SlowFast R50]:::ml
        DeepFace[DeepFace]:::ml
    end
    
    Camera((Live RTSP/USB Camera)) --> |Frames| Streamer
    Streamer --> |Inference Request| YOLO
    YOLO --> |Bounding Boxes & Keypoints| MiDaS
    YOLO --> |Action Sequences| SlowFast
    YOLO --> |Face Cropping| DeepFace
    
    MiDaS --> |Height & Distance| SocketServer
    SlowFast --> |Activity Status| SocketServer
    DeepFace --> |Identity| SocketServer
    
    SocketServer --> |Real-time Events| SocketClient
    Streamer --> |Annotated Feed| UI
    REST --> |Metrics & History| UI
    SocketClient --> UI
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- (Optional) Docker

### 1. Start the Backend
The backend runs the complete ML pipeline via Flask.
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Start the Flask server
python app.py
```

### 2. Start the Frontend
The frontend is a modern React/Vite SPA.
```bash
cd dashboard
npm install

# Start the development server
npm run dev
```

### 3. All-in-One Launcher (Windows)
Simply run the provided batch script to launch both services simultaneously:
```cmd
run.bat
```

---

## 📸 Screenshots

### Live Monitoring Dashboard
<img width="1080" height="997" alt="Screenshot 2026-06-16 at 15-22-27 dashboard" src="https://github.com/user-attachments/assets/d54469cb-4f09-44b6-adb9-e21f259bb6ec" />


### Analytics & Events Panel
<img width="1080" height="997" alt="Screenshot 2026-06-16 at 15-22-39 dashboard" src="https://github.com/user-attachments/assets/a90a2901-08a4-4aee-93ce-81e2d0f04561" />


---

## 🛠️ Technology Stack

- **Frontend**: React 18, Vite, React Router, Recharts, Framer Motion
- **Backend**: Python, Flask, Flask-SocketIO, Waitress
- **AI/ML**: PyTorch, YOLOv8, MiDaS v3, SlowFast, DeepFace
- **Communication**: WebSockets (Socket.IO), MJPEG Streaming

---

## 📄 License
This project is proprietary. All rights reserved.
