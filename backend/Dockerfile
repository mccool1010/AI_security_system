# ═══════════════════════════════════════════════════════════════
#  AI Security System — Backend Dockerfile
#  Multi-stage build: slim Python image + CUDA-optional
# ═══════════════════════════════════════════════════════════════

# ── Stage 1: Base with system dependencies ────────────────────
FROM python:3.12-slim AS base

# System deps for OpenCV, video processing, and general ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: Install Python dependencies ─────────────────────
FROM base AS deps

COPY requirements.txt .

# Install PyTorch CPU-only (saves ~1.5GB vs CUDA version)
# For GPU support, replace with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Final image ─────────────────────────────────────
FROM deps AS final

# Copy application code
COPY . .

# Create directories for runtime data
RUN mkdir -p static/screenshots calibration_data

# Pre-download YOLOv8n-pose model (avoids download on first run)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')" 2>/dev/null || true

# Expose Flask port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run the Flask app
# Use 0.0.0.0 to accept connections from outside the container
CMD ["python", "app.py"]
