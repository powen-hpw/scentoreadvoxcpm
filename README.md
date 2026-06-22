# VoxCPM Workbench

Minimal standalone VoxCPM2 workbench for local and Runpod deployment.

## What this project does

- serves a small HTML interface for VoxCPM2 text-to-speech testing,
- records per-request timing and request history,
- saves generated WAV files locally,
- supports a repeatable Runpod deployment flow.

## What this project does not include

- model weights,
- generated audio files,
- request logs,
- any other ScanToRead code.

Model weights are downloaded separately from Hugging Face:

- `openbmb/VoxCPM2`

## Project layout

```text
voxcpm-workbench/
├── app/
│   ├── server.py
│   └── static/
├── deploy/
│   └── runpod/
├── requirements.txt
└── smoke_test.py
```

## Local setup

```bash
cd voxcpm-workbench
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m huggingface_hub download openbmb/VoxCPM2 --local-dir models/VoxCPM2
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

Then open:

- `http://127.0.0.1:8000`

## Runpod setup

See:

- [deploy/runpod/README.md](./deploy/runpod/README.md)

## Useful environment variables

- `VOXCPM_MODEL_DIR`
- `VOXCPM_OUTPUT_DIR`
- `VOXCPM_LOG_DIR`
- `VOXCPM_STATIC_DIR`
- `VOXCPM_DEFAULT_TEXT`
- `VOXCPM_DEFAULT_DEVICE`

## Smoke test

Generate a single WAV file without starting the web UI:

```bash
python smoke_test.py --device auto
```
