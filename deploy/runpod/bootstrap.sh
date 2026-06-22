#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/voxcpm-workbench}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MODEL_DIR="${VOXCPM_MODEL_DIR:-$ROOT_DIR/models/VoxCPM2}"
MODEL_ID="${VOXCPM_MODEL_ID:-openbmb/VoxCPM2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/5] Checking repository path: $ROOT_DIR"
if [ ! -d "$ROOT_DIR" ]; then
  echo "Repository directory not found: $ROOT_DIR" >&2
  echo "Clone the repo first, then rerun this script." >&2
  exit 1
fi

echo "[2/5] Creating virtual environment: $VENV_DIR"
$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/5] Installing Python dependencies"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements-gpu-cu128.txt" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$ROOT_DIR/requirements-app.txt"

echo "[4/5] Ensuring model directory exists: $MODEL_DIR"
mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  echo "[5/5] Downloading model weights from Hugging Face: $MODEL_ID"
  HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
else
  echo "[5/5] Model weights already exist, skipping download"
fi

echo
echo "Bootstrap complete."
echo "Next step:"
echo "  bash \"$ROOT_DIR/deploy/runpod/start.sh\""
