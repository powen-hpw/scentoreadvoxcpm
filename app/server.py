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

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
REFERENCE_VOICE_DIR = Path(
    os.environ.get("VOXCPM_REFERENCE_VOICE_DIR", ROOT_DIR / "reference_voices")
).expanduser()
DEFAULT_TEXT = os.environ.get(
    "VOXCPM_DEFAULT_TEXT",
    "今仔日天氣真好，咱來講一个故事。",
)
DEFAULT_DEVICE = os.environ.get("VOXCPM_DEFAULT_DEVICE", "auto")
MAX_REFERENCE_VOICES = 5
DEFAULT_CFG_VALUE = 2.0
DEFAULT_INFERENCE_TIMESTEPS = 10
DEFAULT_RETRY_BADCASE = True
DEFAULT_RETRY_BADCASE_MAX_TIMES = 3
DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD = 6.0
DEFAULT_MIN_LEN = 1
DEFAULT_MAX_LEN = 4096
DEFAULT_OUTPUT_MODE = "full"

PARAMETER_LIMITS = {
    "cfg_value": {
        "min": 0.1,
        "max": 10.0,
        "step": 0.1,
        "default": DEFAULT_CFG_VALUE,
        "recommended": "1.5 to 3.0",
    },
    "inference_timesteps": {
        "min": 1,
        "max": 200,
        "step": 1,
        "default": DEFAULT_INFERENCE_TIMESTEPS,
        "recommended": "8 to 20",
    },
    "retry_badcase_max_times": {
        "min": 0,
        "max": 10,
        "step": 1,
        "default": DEFAULT_RETRY_BADCASE_MAX_TIMES,
        "recommended": "2 to 4",
    },
    "retry_badcase_ratio_threshold": {
        "min": 1.0,
        "max": 20.0,
        "step": 0.1,
        "default": DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD,
        "recommended": "4.0 to 8.0",
    },
    "min_len": {
        "min": 1,
        "max": 10000,
        "step": 1,
        "default": DEFAULT_MIN_LEN,
        "recommended": "1 to 32",
    },
    "max_len": {
        "min": 1,
        "max": 10000,
        "step": 1,
        "default": DEFAULT_MAX_LEN,
        "recommended": "512 to 4096",
    },
}


for directory in (OUTPUT_DIR, LOG_DIR, STATIC_DIR, REFERENCE_VOICE_DIR):
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
    reference_voice_id: str = Field(default="", max_length=200)
    output_mode: str = Field(default=DEFAULT_OUTPUT_MODE, pattern="^(full|segmented)$")
    device: str = Field(default=DEFAULT_DEVICE, pattern="^(auto|mps|cpu|cuda)$")
    cfg_value: float = Field(default=DEFAULT_CFG_VALUE, ge=0.1, le=10.0)
    inference_timesteps: int = Field(default=DEFAULT_INFERENCE_TIMESTEPS, ge=1, le=200)
    normalize: bool = Field(default=False)
    denoise: bool = Field(default=False)
    retry_badcase: bool = Field(default=DEFAULT_RETRY_BADCASE)
    retry_badcase_max_times: int = Field(default=DEFAULT_RETRY_BADCASE_MAX_TIMES, ge=0, le=10)
    retry_badcase_ratio_threshold: float = Field(default=DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD, ge=1.0, le=20.0)
    min_len: int = Field(default=DEFAULT_MIN_LEN, ge=1, le=10000)
    max_len: int = Field(default=DEFAULT_MAX_LEN, ge=1, le=10000)
    optimize: bool = Field(default=False)


class ReferenceVoiceInfo(BaseModel):
    """Stored reference voice metadata exposed to the UI."""

    voice_id: str
    label: str
    note: str
    filename: str
    file_size_bytes: int
    created_at: str
    audio_url: str


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


def split_text_for_segmented_tts(text: str) -> list[str]:
    """Split text into medium-length chunks for segmented generation."""

    text = text.strip()
    if not text:
        return []

    primary_breaks = "。！？；"
    secondary_breaks = "，、"
    chunks: list[str] = []
    current = ""

    def flush(force: bool = False) -> None:
        nonlocal current
        candidate = current.strip()
        if candidate and (force or len(candidate) >= 10):
            chunks.append(candidate)
            current = ""

    for char in text:
        current += char
        if char in primary_breaks:
            flush(force=True)
        elif char in secondary_breaks and len(current.strip()) >= 18:
            flush(force=True)

    if current.strip():
        if chunks and len(current.strip()) < 10:
            chunks[-1] = f"{chunks[-1]}{current.strip()}"
        else:
            chunks.append(current.strip())

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < 10:
            merged[-1] = f"{merged[-1]}{chunk}"
        else:
            merged.append(chunk)
    return merged


def read_history(limit: int = 20) -> list[dict[str, Any]]:
    """Load recent request logs, newest first."""

    logs: list[dict[str, Any]] = []
    for log_path in sorted(LOG_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            logs.append(json.loads(log_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return logs


def slugify_label(value: str) -> str:
    """Create a stable filesystem-friendly slug."""

    cleaned = "".join(
        ch.lower() if ch.isalnum() else "-" for ch in value.strip()
    ).strip("-")
    compact = "-".join(part for part in cleaned.split("-") if part)
    return compact or "reference-voice"


def reference_voice_metadata_path(voice_id: str) -> Path:
    """Return the sidecar metadata path for one reference voice."""

    return REFERENCE_VOICE_DIR / f"{voice_id}.json"


def read_reference_voice_metadata(voice_id: str) -> dict[str, Any]:
    """Load reference voice metadata if present."""

    metadata_path = reference_voice_metadata_path(voice_id)
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_reference_voice_metadata(voice_id: str, *, label: str, note: str) -> None:
    """Persist editable metadata for a reference voice."""

    reference_voice_metadata_path(voice_id).write_text(
        json.dumps({"label": label, "note": note}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_reference_voices() -> list[ReferenceVoiceInfo]:
    """Return stored reference voices sorted newest first."""

    items: list[ReferenceVoiceInfo] = []
    for path in sorted(REFERENCE_VOICE_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True):
        voice_id = path.stem
        stat = path.stat()
        label = voice_id.split("--", 1)[1].replace("-", " ")
        metadata = read_reference_voice_metadata(voice_id)
        label = metadata.get("label", label)
        note = metadata.get("note", "")
        items.append(
            ReferenceVoiceInfo(
                voice_id=voice_id,
                label=label,
                note=note,
                filename=path.name,
                file_size_bytes=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                audio_url=f"/reference_voices/{path.name}",
            )
        )
    return items


def resolve_reference_voice(voice_id: str) -> tuple[str | None, str | None]:
    """Resolve a selected reference voice id to path and label."""

    if not voice_id:
        return None, None
    path = REFERENCE_VOICE_DIR / f"{voice_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"Unknown reference voice: {voice_id}")
    label = next((item.label for item in list_reference_voices() if item.voice_id == voice_id), voice_id)
    return str(path), label


def get_reference_voice_or_404(voice_id: str) -> tuple[Path, dict[str, Any]]:
    """Return the voice wav path and current metadata."""

    path = REFERENCE_VOICE_DIR / f"{voice_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Unknown reference voice: {voice_id}")
    metadata = read_reference_voice_metadata(voice_id)
    default_label = voice_id.split("--", 1)[1].replace("-", " ")
    metadata.setdefault("label", default_label)
    metadata.setdefault("note", "")
    return path, metadata


def sanitize_payload(payload: GenerateRequest) -> GenerateRequest:
    """Normalize incoming values and block invalid parameter combinations."""

    data = payload.model_dump()
    if data["max_len"] < data["min_len"]:
        raise HTTPException(
            status_code=400,
            detail="Max Len must be greater than or equal to Min Len.",
        )
    return GenerateRequest(**data)


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
app.mount("/reference_voices", StaticFiles(directory=str(REFERENCE_VOICE_DIR)), name="reference_voices")
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
        "reference_voice_id": "",
        "output_mode": DEFAULT_OUTPUT_MODE,
        "device": DEFAULT_DEVICE,
        "cfg_value": DEFAULT_CFG_VALUE,
        "inference_timesteps": DEFAULT_INFERENCE_TIMESTEPS,
        "normalize": False,
        "denoise": False,
        "retry_badcase": DEFAULT_RETRY_BADCASE,
        "retry_badcase_max_times": DEFAULT_RETRY_BADCASE_MAX_TIMES,
        "retry_badcase_ratio_threshold": DEFAULT_RETRY_BADCASE_RATIO_THRESHOLD,
        "min_len": DEFAULT_MIN_LEN,
        "max_len": DEFAULT_MAX_LEN,
        "optimize": False,
        "model_path": str(MODEL_DIR),
        "parameter_limits": PARAMETER_LIMITS,
        "reference_voices": [item.model_dump() for item in list_reference_voices()],
        "notes": [
            "First request on a new device/optimize combination is usually slower.",
            "When cold_start is true, internal warm-up is included in generate_ms.",
        ],
    }


@app.get("/api/reference-voices")
def reference_voices() -> dict[str, Any]:
    """Return available reference voice files for the UI."""

    return {"items": [item.model_dump() for item in list_reference_voices()]}


@app.post("/api/reference-voices")
async def upload_reference_voice(
    file: UploadFile = File(...),
    label: str = Form(default=""),
    note: str = Form(default=""),
) -> dict[str, Any]:
    """Upload and store one reference voice WAV file."""

    existing = list_reference_voices()
    if len(existing) >= MAX_REFERENCE_VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Reference voice limit reached ({MAX_REFERENCE_VOICES}). Delete an old one before uploading a new file.",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".wav"}:
        raise HTTPException(status_code=400, detail="Only .wav files are supported for reference voices.")

    requested_label = (label or Path(file.filename or "reference-voice").stem).strip()
    voice_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}--{slugify_label(requested_label)}"
    destination = REFERENCE_VOICE_DIR / f"{voice_id}.wav"
    contents = await file.read()
    destination.write_bytes(contents)
    write_reference_voice_metadata(
        voice_id,
        label=requested_label,
        note=note.strip(),
    )

    item = ReferenceVoiceInfo(
        voice_id=voice_id,
        label=requested_label,
        note=note.strip(),
        filename=destination.name,
        file_size_bytes=destination.stat().st_size,
        created_at=datetime.fromtimestamp(destination.stat().st_mtime, tz=timezone.utc).isoformat(),
        audio_url=f"/reference_voices/{destination.name}",
    )
    return {"item": item.model_dump(), "items": [voice.model_dump() for voice in list_reference_voices()]}


@app.put("/api/reference-voices/{voice_id}")
async def update_reference_voice(
    voice_id: str,
    label: str = Form(...),
    note: str = Form(default=""),
) -> dict[str, Any]:
    """Update one reference voice label and note."""

    _, metadata = get_reference_voice_or_404(voice_id)
    next_label = label.strip() or metadata["label"]
    next_note = note.strip()
    write_reference_voice_metadata(voice_id, label=next_label, note=next_note)
    item = next((voice for voice in list_reference_voices() if voice.voice_id == voice_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown reference voice: {voice_id}")
    return {"item": item.model_dump(), "items": [voice.model_dump() for voice in list_reference_voices()]}


@app.delete("/api/reference-voices/{voice_id}")
def delete_reference_voice(voice_id: str) -> dict[str, Any]:
    """Delete one reference voice wav and metadata."""

    path, _ = get_reference_voice_or_404(voice_id)
    metadata_path = reference_voice_metadata_path(voice_id)
    path.unlink(missing_ok=False)
    if metadata_path.exists():
        metadata_path.unlink()
    return {"deleted_voice_id": voice_id, "items": [voice.model_dump() for voice in list_reference_voices()]}


@app.post("/api/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    """Generate one WAV file and store a structured timing log."""

    payload = sanitize_payload(payload)

    request_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
    audio_path = OUTPUT_DIR / f"{request_id}.wav"
    log_path = LOG_DIR / f"{request_id}.json"

    request_started = time.perf_counter()
    request_received_at = utc_now()
    voice_description = build_voice_description(payload)
    reference_wav_path, reference_voice_label = resolve_reference_voice(payload.reference_voice_id)
    effective_text = (
        f"({voice_description}){payload.text}" if voice_description else payload.text
    )
    text_segments = (
        split_text_for_segmented_tts(payload.text)
        if payload.output_mode == "segmented"
        else [payload.text]
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
        segment_wavs: list[np.ndarray] = []
        segment_logs: list[dict[str, Any]] = []
        for index, segment_text in enumerate(text_segments, start=1):
            effective_segment_text = (
                f"({voice_description}){segment_text}" if voice_description else segment_text
            )
            segment_wav = model_handle.model.generate(
                text=effective_segment_text,
                reference_wav_path=reference_wav_path,
                cfg_value=payload.cfg_value,
                inference_timesteps=payload.inference_timesteps,
                normalize=payload.normalize,
                denoise=payload.denoise,
                retry_badcase=payload.retry_badcase,
                retry_badcase_max_times=payload.retry_badcase_max_times,
                retry_badcase_ratio_threshold=payload.retry_badcase_ratio_threshold,
                min_len=payload.min_len,
                max_len=payload.max_len,
            )
            segment_wavs.append(np.asarray(segment_wav))
            segment_logs.append(
                {
                    "index": index,
                    "text": segment_text,
                    "effective_text": effective_segment_text,
                }
            )
        wav = (
            np.concatenate(segment_wavs)
            if len(segment_wavs) > 1
            else segment_wavs[0]
        )
    except Exception as exc:
        error_log = {
            "request_id": request_id,
            "request_received_at": request_received_at,
            "success": False,
            "error": str(exc),
            "parameters": payload.model_dump(),
            "voice_description": voice_description,
            "reference_voice_label": reference_voice_label,
            "reference_wav_path": reference_wav_path,
            "output_mode": payload.output_mode,
            "segment_count": len(text_segments),
            "segments": text_segments,
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
            "output_mode": payload.output_mode,
            "segment_count": len(text_segments),
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
        "reference_voice_label": reference_voice_label,
        "reference_wav_path": reference_wav_path,
        "output_mode": payload.output_mode,
        "segment_count": len(text_segments),
        "segments": segment_logs,
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
