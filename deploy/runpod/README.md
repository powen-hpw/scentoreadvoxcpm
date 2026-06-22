# Runpod Deployment

This guide assumes you are deploying this standalone repository on a Runpod Pod.

## Recommended starting setup

- Template: `Runpod Pytorch 2.8.0`
- GPU: `A40`, `L4`, or `RTX A5000`
- Pricing: `On-Demand`
- Network volume: enabled
- Public port: `8888` if using the stock template setup

## First-time setup inside the Pod

```bash
cd /workspace
git clone <YOUR_REPO_URL> voxcpm-workbench
cd /workspace/voxcpm-workbench
bash deploy/runpod/bootstrap.sh
```

## Start the service

If the Pod template already exposes port `8888`, start the service there:

```bash
PORT=8888 VOXCPM_DEFAULT_DEVICE=cuda bash deploy/runpod/start.sh
```

Otherwise use the port you exposed for the Pod:

```bash
PORT=8000 VOXCPM_DEFAULT_DEVICE=cuda bash deploy/runpod/start.sh
```

## What persists between restarts

- model files in `models/VoxCPM2/`
- generated WAV files in `output/`
- request logs in `request_logs/`

## What does not persist in memory

- loaded model weights in GPU memory
- the running Python process

That means restarting the Pod avoids redownloading the model, but the first
request after each restart is still slower than a warm request.
