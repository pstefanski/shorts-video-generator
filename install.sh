#!/usr/bin/env bash
# ============================================================================
# install.sh — Shorts Generator installer for WSL (Ubuntu/Debian)
# Usage:  chmod +x install.sh && ./install.sh [--cuda]
# ============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

CUDA=false
if [[ "${1:-}" == "--cuda" ]]; then
    CUDA=true
fi

# ---------- system packages --------------------------------------------------

info "Updating apt package index…"
sudo apt-get update -qq

info "Installing system dependencies…"
sudo apt-get install -y -qq \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    git \
    curl \
    > /dev/null

# ---------- python venv ------------------------------------------------------

VENV_DIR=".venv"
if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating Python virtual environment in ${VENV_DIR}/…"
    python3 -m venv "$VENV_DIR"
fi

source "${VENV_DIR}/bin/activate"
info "Virtual environment activated."

# ---------- pip packages -----------------------------------------------------

info "Upgrading pip…"
pip install --upgrade pip -q

info "Installing Python dependencies…"
pip install -r requirements.txt -q

# ---------- CUDA (optional) --------------------------------------------------

if $CUDA; then
    info "Installing PyTorch with CUDA support…"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    info "CUDA-enabled PyTorch installed."
else
    info "Installing PyTorch (CPU only)…"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
fi

# ---------- directories ------------------------------------------------------

mkdir -p input output

# ---------- done -------------------------------------------------------------

echo ""
info "============================================"
info "  Installation complete!"
info "============================================"
info ""
info "  Activate the environment:"
info "    source .venv/bin/activate"
info ""
info "  Place your source videos in:  input/"
info ""
info "  Run the generator:"
if $CUDA; then
    info "    python generate_shorts.py --cuda"
else
    info "    python generate_shorts.py"
fi
info ""
info "  Generated shorts will appear in:  output/"
info "============================================"
