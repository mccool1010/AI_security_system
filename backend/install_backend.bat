@echo off
:: ═══════════════════════════════════════════════════════════════
::  Creates backend\venv and installs the backend with the right
::  PyTorch build (CUDA 12.8 if an NVIDIA GPU is present, else CPU).
::  Safe to run again. Used by install.bat, deploy.bat, demo-deploy.bat.
:: ═══════════════════════════════════════════════════════════════
setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo         Creating backend\venv ...
    python -m venv venv || exit /b 1
)
set PY=venv\Scripts\python.exe
%PY% -m pip install --upgrade pip >nul

set TORCH_INDEX=https://download.pytorch.org/whl/cpu
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    echo         NVIDIA GPU found - installing CUDA 12.8 PyTorch
) else (
    echo         No NVIDIA GPU found - installing CPU PyTorch
)
if defined FORCE_CPU set TORCH_INDEX=https://download.pytorch.org/whl/cpu

%PY% -m pip install torch==2.7.1 torchvision==0.22.1 --index-url %TORCH_INDEX% || exit /b 1
%PY% -m pip install -r requirements.txt || exit /b 1
%PY% -m pip install --no-deps -r requirements-nodeps.txt || exit /b 1
%PY% -c "import torch, detector; print('        Backend ready - device:', detector.DEVICE)" || exit /b 1
endlocal
