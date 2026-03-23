@echo off
title SecureVision AI — Installer
color 0B

echo.
echo  ============================================================
echo    SecureVision AI Surveillance Platform — Full Installer
echo  ============================================================
echo.

:: ── Check Python ────────────────────────────────────────────
echo  [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Python is not installed or not in PATH.
    echo  Please install Python 3.10+ from https://python.org
    echo.
    pause
    exit /b 1
)
python --version
echo         OK
echo.

:: ── Check Node.js ───────────────────────────────────────────
echo  [2/5] Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  ERROR: Node.js is not installed or not in PATH.
    echo  Please install Node.js 18+ from https://nodejs.org
    echo.
    pause
    exit /b 1
)
node --version
echo         OK
echo.

:: ── Create Python virtual environment ───────────────────────
echo  [3/5] Setting up Python virtual environment...
cd /d "%~dp0backend"
if not exist "venv" (
    echo         Creating venv...
    python -m venv venv
)
echo         Activating venv...
call venv\Scripts\activate.bat

:: ── Install Python dependencies ─────────────────────────────
echo  [4/5] Installing Python dependencies (this may take a while)...
echo.
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo  WARNING: Some Python packages failed to install.
    echo  You may need to install PyTorch manually:
    echo  https://pytorch.org/get-started/locally/
    echo.
)
echo.
echo         Python dependencies installed.
echo.

:: ── Install Node.js dependencies ────────────────────────────
echo  [5/5] Installing dashboard dependencies...
cd /d "%~dp0dashboard"
call npm install
if %errorlevel% neq 0 (
    echo.
    echo  WARNING: npm install encountered errors.
    echo.
)
echo.
echo         Dashboard dependencies installed.
echo.

:: ── Create screenshots directory ────────────────────────────
if not exist "%~dp0backend\screenshots" mkdir "%~dp0backend\screenshots"

:: ── Done ────────────────────────────────────────────────────
echo.
echo  ============================================================
echo    Installation Complete!
echo  ============================================================
echo.
echo    To start the system, double-click:  run.bat
echo.
echo  ============================================================
echo.
pause
