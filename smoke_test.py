#!/usr/bin/env python3
"""Minimal VoxCPM2 smoke test for the standalone workbench."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import soundfile as sf
from voxcpm import VoxCPM


DEFAULT_TEXT = "今仔日天氣真好，咱來講一个故事。"
DEFAULT_MODEL_DIR = "models/VoxCPM2"
DEFAULT_OUTPUT = "output/voxcpm-smoke-test.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one WAV file with VoxCPM2."
    )
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize.")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("VOXCPM_MODEL_DIR", DEFAULT_MODEL_DIR),
        help="Local model directory or remote model id.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "mps", "cpu", "cuda"),
        help="Execution device.",
    )
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output WAV path.")
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable VoxCPM runtime optimization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = VoxCPM.from_pretrained(
        args.model_path,
        load_denoiser=False,
        device=args.device,
        optimize=not args.no_optimize,
    )
    wav = model.generate(
        text=args.text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
    )

    sf.write(output_path, wav, model.tts_model.sample_rate)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
