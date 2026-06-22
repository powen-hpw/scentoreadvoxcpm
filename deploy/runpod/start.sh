#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/voxcpm-workbench}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
export VOXCPM_MODEL_DIR="${VOXCPM_MODEL_DIR:-$ROOT_DIR/models/VoxCPM2}"
export VOXCPM_OUTPUT_DIR="${VOXCPM_OUTPUT_DIR:-$ROOT_DIR/output}"
export VOXCPM_LOG_DIR="${VOXCPM_LOG_DIR:-$ROOT_DIR/request_logs}"
export VOXCPM_STATIC_DIR="${VOXCPM_STATIC_DIR:-$ROOT_DIR/app/static}"
export VOXCPM_DEFAULT_DEVICE="${VOXCPM_DEFAULT_DEVICE:-cuda}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

echo "Starting VoxCPM workbench on ${HOST}:${PORT}"
echo "Model dir: $VOXCPM_MODEL_DIR"
echo "Default device: $VOXCPM_DEFAULT_DEVICE"

python -m uvicorn app.server:app --host "$HOST" --port "$PORT"
