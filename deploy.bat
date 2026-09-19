@echo off
:: ═══════════════════════════════════════════════════════════════
::  AI Security System — Deploy Script (Windows)
:: ═══════════════════════════════════════════════════════════════
echo.
echo  ╔══════════════════════════════════════════════════╗
echo  ║   AI Security System — One-Click Deploy          ║
echo  ╚══════════════════════════════════════════════════╝
echo.

:: ── Step 1: Check Docker ────────────────────────────────────
echo  [1/3] Checking Docker...
where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  X Docker is NOT installed!
    echo.
    echo  Download Docker Desktop (FREE) from:
    echo  https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)

docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  X Docker is not running! Start Docker Desktop first.
    pause
    exit /b 1
)
echo  OK Docker is ready

:: ── Step 2: Start MongoDB + Dashboard ───────────────────────
echo.
echo  [2/3] Starting MongoDB + Dashboard in Docker...
docker compose up -d --build

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  X Docker Compose failed. Check the error above.
    pause
    exit /b 1
)
echo  OK MongoDB + Dashboard running

:: ── Step 3: Check Python + start backend ────────────────────
echo.
echo  [3/3] Starting Backend (for camera access)...

cd backend

if not exist "venv\Scripts\python.exe" (
    echo  Installing the backend - first run only...
    call install_backend.bat || (pause & exit /b 1)
)

echo.
echo  ═══════════════════════════════════════════════════
echo.
echo  READY! Starting the backend now...
echo.
echo  Dashboard:  http://localhost:3000
echo  Backend:    http://localhost:5000
echo.
echo  Press Ctrl+C to stop the backend.
echo  To stop Docker services: docker compose down
echo.
echo  ═══════════════════════════════════════════════════
echo.

:: Start Flask backend (this blocks — camera needs to run)
venv\Scripts\python.exe app.py
