@echo off
title SecureVision AI — Stopping
color 0C

echo.
echo  ============================================================
echo    Stopping SecureVision AI...
echo  ============================================================
echo.

:: Kill Python backend
taskkill /f /fi "WINDOWTITLE eq SecureVision Backend*" >nul 2>&1
taskkill /f /im python.exe /fi "WINDOWTITLE eq SecureVision Backend*" >nul 2>&1

:: Kill Node dashboard
taskkill /f /fi "WINDOWTITLE eq SecureVision Dashboard*" >nul 2>&1
taskkill /f /im node.exe /fi "WINDOWTITLE eq SecureVision Dashboard*" >nul 2>&1

echo.
echo    All services stopped.
echo.
pause
