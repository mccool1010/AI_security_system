@echo off
:: ═══════════════════════════════════════════════════════════════
::  AI Security System — DEMO MODE Deploy
::  Run this to create a public URL for your resume/portfolio
:: ═══════════════════════════════════════════════════════════════
echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║   SecureVision AI — Portfolio Demo Deploy            ║
echo  ╚══════════════════════════════════════════════════════╝
echo.
echo  This script will:
echo    1. Start MongoDB in Docker
echo    2. Start the backend in DEMO MODE (sample video)
echo    3. Give you a public URL to put in your resume
echo.

:: ── Check Docker ────────────────────────────────────────────
echo  [1/4] Checking Docker...
where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  X Docker is NOT installed!
    echo  Download free: https://www.docker.com/products/docker-desktop/
    pause
    exit /b 1
)
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  X Docker is not running! Open Docker Desktop first.
    pause
    exit /b 1
)
echo  OK Docker ready

:: ── Start MongoDB ───────────────────────────────────────────
echo.
echo  [2/4] Starting MongoDB...
docker compose up -d mongodb 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo  X Failed to start MongoDB. Check Docker Desktop.
    pause
    exit /b 1
)
echo  OK MongoDB running

:: ── Install Python deps if needed ───────────────────────────
echo.
echo  [3/4] Checking Python environment...
cd backend

if not exist ".venv\Scripts\activate.bat" (
    echo  Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r requirements.txt -q 2>nul
echo  OK Dependencies ready

:: ── Start backend in DEMO MODE ──────────────────────────────
echo.
echo  [4/4] Starting backend in DEMO MODE...
echo.
echo  ═══════════════════════════════════════════════════════════
echo.
echo   Backend starting at:    http://localhost:5000
echo   Dashboard (dev) at:     http://localhost:5173
echo.
echo   To get a PUBLIC URL for your resume:
echo   ────────────────────────────────────────────────
echo   1. Open a NEW terminal
echo   2. Run: npx cloudflared tunnel --url http://localhost:5173
echo   3. Copy the URL it gives you (like https://xxx.trycloudflare.com)
echo   4. Put that URL in your resume!
echo.
echo   To use a PERMANENT custom URL (recommended):
echo   ────────────────────────────────────────────────
echo   1. Install: winget install Cloudflare.cloudflared
echo   2. Login:   cloudflared tunnel login
echo   3. Create:  cloudflared tunnel create securevision
echo   4. Run:     cloudflared tunnel --url http://localhost:5173 run securevision
echo.
echo  ═══════════════════════════════════════════════════════════
echo.
echo  Press Ctrl+C to stop.
echo.

:: Set DEMO_MODE and start Flask
set DEMO_MODE=true
python app.py
