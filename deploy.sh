#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  AI Security System — One-Click Docker Deploy (Linux/Mac)
# ═══════════════════════════════════════════════════════════════
set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   AI Security System — Docker Deploy         ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed!"
    echo ""
    echo "Install Docker:"
    echo "  Ubuntu/Debian:  curl -fsSL https://get.docker.com | sh"
    echo "  Mac:            brew install --cask docker"
    echo ""
    exit 1
fi

# Check Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running!"
    echo "Start Docker and try again."
    exit 1
fi

echo "✓ Docker is installed and running"
echo ""

# Build and start
echo "Building and starting MongoDB + dashboard..."
echo "(This will take a few minutes on first run)"
echo ""

docker compose up -d --build

# The backend runs natively so it can reach cameras and the GPU.
cd backend
if [ ! -x venv/bin/python ]; then
    echo "Installing the backend into backend/venv ..."
    python3 -m venv venv
    if command -v nvidia-smi &> /dev/null; then
        TORCH_INDEX=https://download.pytorch.org/whl/cu128
    else
        TORCH_INDEX=https://download.pytorch.org/whl/cpu
    fi
    venv/bin/pip install --upgrade pip
    venv/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url "$TORCH_INDEX"
    venv/bin/pip install -r requirements.txt
    venv/bin/pip install --no-deps -r requirements-nodeps.txt
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ MongoDB + dashboard are running"
echo ""
echo "  Dashboard:  http://localhost:3000"
echo "  Backend:    http://localhost:5000  (starting below)"
echo "  MongoDB:    localhost:27017"
echo ""
echo "  Stop Docker services with: docker compose down"
echo "═══════════════════════════════════════════════"
exec venv/bin/python app.py
