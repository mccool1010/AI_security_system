@echo off
title SecureVision AI — Running
color 0A

echo.
echo  ============================================================
echo    SecureVision AI Surveillance Platform
echo  ============================================================
echo.
echo    Starting database + backend + dashboard...
echo.

:: ── Start MongoDB (Docker) ──────────────────────────────────
echo  [1/3] Starting MongoDB in Docker...
cd /d "%~dp0"
docker info >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo         Docker is not running! Please open Docker Desktop.
    echo         Trying to continue without Docker MongoDB...
    echo         Make sure MongoDB is running some other way.
) else (
    docker compose up -d mongodb >nul 2>&1
    echo         MongoDB started
)

:: ── Start Backend (Python Flask) ────────────────────────────
echo  [2/3] Starting backend server...
cd /d "%~dp0backend"

if not exist "venv\Scripts\python.exe" (
    echo         backend\venv not found - run install.bat first.
    pause
    exit /b 1
)
start "SecureVision Backend" cmd /k "title SecureVision Backend && color 0E && venv\Scripts\python.exe app.py"

:: Give backend a moment to boot
echo         Waiting for backend to initialize...
timeout /t 8 /nobreak >nul

:: ── Start Dashboard (Vite dev server) ───────────────────────
echo  [3/3] Starting dashboard...
cd /d "%~dp0dashboard"
start "SecureVision Dashboard" cmd /k "title SecureVision Dashboard && color 0B && npm run dev"

:: Give dashboard a moment to start
timeout /t 3 /nobreak >nul

:: ── Open browser ────────────────────────────────────────────
echo.
echo  Opening dashboard in browser...
start http://localhost:5173

echo.
echo  ============================================================
echo    System is running!
echo  ============================================================
echo.
echo    MongoDB:    localhost:27017  (Docker)
echo    Backend:    http://localhost:5000
echo    Dashboard:  http://localhost:5173
echo.
echo    Two terminal windows have been opened:
echo      - Yellow  = Backend  (Python / Flask)
echo      - Blue    = Dashboard (Vite dev server)
echo.
echo    Close those windows to stop the system.
echo    Press any key to close this launcher...
echo  ============================================================
echo.
pause

