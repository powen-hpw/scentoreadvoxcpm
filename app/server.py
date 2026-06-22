#!/usr/bin/env python3
"""Standalone VoxCPM workbench server."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from voxcpm import VoxCPM


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(
    os.environ.get("VOXCPM_MODEL_DIR", ROOT_DIR / "models" / "VoxCPM2")
).expanduser()
OUTPUT_DIR = Path(os.environ.get("VOXCPM_OUTPUT_DIR", ROOT_DIR / "output")).expanduser()
LOG_DIR = Path(os.environ.get("VOXCPM_LOG_DIR", ROOT_DIR / "request_logs")).expanduser()
STATIC_DIR = Path(
    os.environ.get("VOXCPM_STATIC_DIR", ROOT_DIR / "app" / "static")
).expanduser()
DEFAULT_TEXT = os.environ.get(
    "VOXCPM_DEFAULT_TEXT",
    "今仔日天氣真好，咱來講一个故事。",
)
DEFAULT_DEVICE = os.environ.get("VOXCPM_DEFAULT_DEVICE", "auto")


for directory in (OUTPUT_DIR, LOG_DIR, STATIC_DIR):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelHandle:
    """Cached model plus creation metadata."""

    model: VoxCPM
    load_ms: float
    created_at: str


class GenerateRequest(BaseModel):
    """User-controlled generation settings."""

    text: str = Field(..., min_length=1, description="Text to synthesize.")
    voice_gender: str = Field(default="", max_length=80)
    voice_age: str = Field(default="", max_length=80)
    voice_tone: str = Field(default="", max_length=120)
    voice_pace: str = Field(default="", max_length=80)
    voice_extra: str = Field(default="", max_length=240)
    device: str = Field(default=DEFAULT_DEVICE, pattern="^(auto|mps|cpu|cuda)$")
    cfg_value: float = Field(default=2.0, ge=0.1, le=10.0)
    inference_timesteps: int = Field(default=10, ge=1, le=200)
    optimize: bool = Field(default=False)


class GenerateResponse(BaseModel):
    """Response payload returned to the browser."""

    request_id: str
    audio_url: str
    log_url: str
    log: dict[str, Any]


class VoxWorkbench:
    """Lazy model cache keyed by runtime configuration."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, bool], ModelHandle] = {}
        self._lock = threading.Lock()

    def get_model(self, *, device: str, optimize: bool) -> tuple[ModelHandle, bool]:
        key = (device, optimize)
        with self._lock:
            cached = self._models.get(key)
            if cached is not None:
                return cached, False

            started = time.perf_counter()
            model = VoxCPM.from_pretrained(
                str(MODEL_DIR),
                load_denoiser=False,
                device=device,
                optimize=optimize,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            handle = ModelHandle(
                model=model,
                load_ms=elapsed_ms,
                created_at=utc_now(),
            )
            self._models[key] = handle
            return handle, True


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist structured JSON logs."""

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_voice_description(payload: GenerateRequest) -> str:
    """Compose the optional voice prompt prefix."""

    parts = [
        payload.voice_gender.strip(),
        payload.voice_age.strip(),
        payload.voice_tone.strip(),
        payload.voice_pace.strip(),
        payload.voice_extra.strip(),
    ]
    return ", ".join(part for part in parts if part)


def read_history(limit: int = 20) -> list[dict[str, Any]]:
    """Load recent request logs, newest first."""

    logs: list[dict[str, Any]] = []
    for log_path in sorted(LOG_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            logs.append(json.loads(log_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return logs


workbench = VoxWorkbench()
app = FastAPI(title="VoxCPM Workbench")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/request_logs", StaticFiles(directory=str(LOG_DIR)), name="request_logs")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    """Serve the browser workbench."""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/history")
def history() -> dict[str, Any]:
    """Return recent generation runs."""

    return {"items": read_history()}


@app.get("/api/defaults")
def defaults() -> dict[str, Any]:
    """Expose default form values and notes."""

    return {
        "text": DEFAULT_TEXT,
        "voice_gender": "",
        "voice_age": "",
        "voice_tone": "",
        "voice_pace": "",
        "voice_extra": "",
        "device": DEFAULT_DEVICE,
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "optimize": False,
        "model_path": str(MODEL_DIR),
        "notes": [
            "First request on a new device/optimize combination is usually slower.",
            "When cold_start is true, internal warm-up is included in generate_ms.",
        ],
    }


@app.post("/api/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    """Generate one WAV file and store a structured timing log."""

    request_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    audio_path = OUTPUT_DIR / f"{request_id}.wav"
    log_path = LOG_DIR / f"{request_id}.json"

    request_started = time.perf_counter()
    request_received_at = utc_now()
    voice_description = build_voice_description(payload)
    effective_text = (
        f"({voice_description}){payload.text}" if voice_description else payload.text
    )

    timings: dict[str, float] = {}
    phases: list[dict[str, Any]] = []

    model_phase_started = time.perf_counter()
    model_handle, cold_start = workbench.get_model(
        device=payload.device,
        optimize=payload.optimize,
    )
    timings["model_ready_ms"] = round(
        (time.perf_counter() - model_phase_started) * 1000, 2
    )
    phases.append(
        {
            "name": "model_ready",
            "elapsed_ms": timings["model_ready_ms"],
            "cold_start": cold_start,
        }
    )

    generate_phase_started = time.perf_counter()
    try:
        wav = model_handle.model.generate(
            text=effective_text,
            cfg_value=payload.cfg_value,
            inference_timesteps=payload.inference_timesteps,
        )
    except Exception as exc:
        error_log = {
            "request_id": request_id,
            "request_received_at": request_received_at,
            "success": False,
            "error": str(exc),
            "parameters": payload.model_dump(),
            "voice_description": voice_description,
            "effective_text": effective_text,
            "cold_start": cold_start,
            "model_cache_created_at": model_handle.created_at,
            "model_load_ms": round(model_handle.load_ms, 2),
            "timings_ms": timings,
            "phases": phases,
            "total_ms": round((time.perf_counter() - request_started) * 1000, 2),
        }
        save_json(log_path, error_log)
        raise HTTPException(status_code=500, detail=error_log) from exc

    timings["generate_ms"] = round(
        (time.perf_counter() - generate_phase_started) * 1000, 2
    )
    phases.append(
        {
            "name": "generate",
            "elapsed_ms": timings["generate_ms"],
            "note": "Includes VoxCPM internal warm-up when cold_start is true.",
        }
    )

    write_phase_started = time.perf_counter()
    sf.write(audio_path, wav, model_handle.model.tts_model.sample_rate)
    timings["write_wav_ms"] = round(
        (time.perf_counter() - write_phase_started) * 1000, 2
    )
    phases.append({"name": "write_wav", "elapsed_ms": timings["write_wav_ms"]})

    total_ms = round((time.perf_counter() - request_started) * 1000, 2)
    log_payload = {
        "request_id": request_id,
        "request_received_at": request_received_at,
        "success": True,
        "parameters": payload.model_dump(),
        "voice_description": voice_description,
        "effective_text": effective_text,
        "audio_file": audio_path.name,
        "audio_url": f"/output/{audio_path.name}",
        "log_file": log_path.name,
        "log_url": f"/request_logs/{log_path.name}",
        "cold_start": cold_start,
        "model_cache_created_at": model_handle.created_at,
        "model_load_ms": round(model_handle.load_ms, 2),
        "timings_ms": {
            **timings,
            "total_ms": total_ms,
        },
        "phases": phases,
    }
    save_json(log_path, log_payload)

    return GenerateResponse(
        request_id=request_id,
        audio_url=log_payload["audio_url"],
        log_url=log_payload["log_url"],
        log=log_payload,
    )
