# 🏃‍♂️ How to Run SecureVision Locally

This guide will walk you through setting up and running the SecureVision AI surveillance pipeline on your local machine.

## 📦 Prerequisites
Before you start, make sure you have the following installed on your machine:
1. **Python 3.9 or higher**: For running the deep learning models and Flask backend.
2. **Node.js (version 18+)**: For running the React/Vite dashboard.
3. **Git**: To clone the repository.
4. **A Webcam or CCTV stream**: The system automatically attempts to connect to your primary USB camera (Camera 0) by default.

---

## 🛠️ Step 1: Install Backend Dependencies

The backend manages all AI inference (YOLOv8, MiDaS, SlowFast, DeepFace) and serves the WebSocket streams.

1. Open your terminal and navigate to the backend folder:
   ```bash
   cd backend
   ```
2. Create a virtual environment to keep dependencies isolated:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - **Windows**: `venv\Scripts\activate`
   - **Mac/Linux**: `source venv/bin/activate`
4. Install all required ML libraries and Flask dependencies:
   ```bash
   pip install -r requirements.txt
   ```
*(Note: PyTorch, Ultralytics YOLO, and DeepFace will automatically download their pre-trained weights the very first time you run the system).*

---

## 🎨 Step 2: Install Frontend Dependencies

The frontend is a hardware-accelerated React/Vite Single Page Application (SPA).

1. Open a **new, separate terminal window** and navigate to the dashboard folder:
   ```bash
   cd dashboard
   ```
2. Install the Node packages:
   ```bash
   npm install
   ```

---

## 🚀 Step 3: Start the System

You have two options for starting the system.

### Option A: The Easy Way (Windows Only)
If you are on Windows, we have provided an all-in-one startup script. Simply double-click the `run.bat` file in the root directory.
This script will automatically:
- Activate your Python virtual environment.
- Start the Flask backend server.
- Start the Vite development server for the dashboard.
- Bypass the portfolio landing page and boot directly into the live monitoring dashboard.

### Option B: Manual Start (Mac/Linux/Windows)
You need to keep **two** terminal windows running simultaneously.

**Terminal 1 (The Backend):**
```bash
cd backend
# Make sure your venv is activated!
python app.py
```
*(Leave this window open and running. It will listen on port 5000).*

**Terminal 2 (The Frontend):**
```bash
cd dashboard
npm run dev
```
*(Leave this window open. It will provide a local URL, usually `http://localhost:5173`).*

## 🐳 Option C: Using Docker (Recommended for Production)

If you prefer containerization, this project includes a complete Docker setup that automatically spins up the Backend, Frontend, and a **MongoDB** database instance for persistent event storage.

1. Make sure you have Docker and Docker Compose installed.
2. From the root directory of the project, simply run:
   ```bash
   docker-compose up --build
   ```
3. Docker will automatically:
   - Build and start the Python Backend container.
   - Build and start the React Frontend container.
   - Pull and start the official **MongoDB** container.
   - Connect the backend seamlessly to the MongoDB database to log all security events.

Once the containers are running, you can access the dashboard at `http://localhost:80`.

---

## 🛑 What Needs to be Kept Running?
To use the actual ML pipeline with real cameras, **both** the frontend and the backend processes must be actively running at the same time. 
- If you close the Python backend, the video feeds and AI analytics will stop working.
- If you close the Vite frontend, you will lose access to the UI dashboard.

If you ever need to stop the system, simply press `Ctrl + C` in both terminal windows.
