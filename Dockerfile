# ============================================================================
# Dockerfile — Shorts Generator
# ============================================================================
# Build:
#   docker build -t shorts-generator .
#
# Run (CPU):
#   docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output shorts-generator
#
# Run (CUDA — requires nvidia-docker):
#   docker run --rm --gpus all -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output shorts-generator --cuda
# ============================================================================

FROM python:3.11-slim AS base

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    fonts-dejavu-core \
    fonts-freefont-ttf \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Application code
COPY generate_shorts.py .

# Create input/output directories
RUN mkdir -p input output

# Volumes for source videos and generated shorts
VOLUME ["/app/input", "/app/output"]

ENTRYPOINT ["python", "generate_shorts.py"]
CMD []

# ============================================================================
# CUDA variant — build with:
#   docker build --target cuda -t shorts-generator:cuda .
# ============================================================================

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    fonts-dejavu-core \
    fonts-freefont-ttf \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY generate_shorts.py .

RUN mkdir -p input output

VOLUME ["/app/input", "/app/output"]

ENTRYPOINT ["python3", "generate_shorts.py", "--cuda"]
CMD []
