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
echo "Building and starting all services..."
echo "(This will take a few minutes on first run)"
echo ""

docker compose up -d --build

echo ""
echo "═══════════════════════════════════════════════"
echo "✅ All services are running!"
echo ""
echo "  Dashboard:  http://localhost:3000"
echo "  Backend:    http://localhost:5000"
echo "  MongoDB:    localhost:27017"
echo ""
echo "  Useful commands:"
echo "    docker compose logs -f backend    (watch ML logs)"
echo "    docker compose down               (stop everything)"
echo "    docker compose restart backend    (restart backend)"
echo ""

# USB camera hint for Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "  📷 USB Camera: Uncomment 'devices' in docker-compose.yml"
    echo "     Then: docker compose up -d --build backend"
    echo ""
fi

echo "═══════════════════════════════════════════════"
