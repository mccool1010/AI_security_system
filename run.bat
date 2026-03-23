@echo off
title SecureVision AI — Running
color 0A

echo.
echo  ============================================================
echo    SecureVision AI Surveillance Platform
echo  ============================================================
echo.
echo    Starting backend + dashboard...
echo.

:: ── Start Backend (Python Flask) ────────────────────────────
echo  [1/2] Starting backend server...
cd /d "%~dp0backend"

:: Activate venv if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

start "SecureVision Backend" cmd /k "title SecureVision Backend && color 0E && python app.py"

:: Give backend a moment to boot
echo         Waiting for backend to initialize...
timeout /t 4 /nobreak >nul

:: ── Start Dashboard (Vite dev server) ───────────────────────
echo  [2/2] Starting dashboard...
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
