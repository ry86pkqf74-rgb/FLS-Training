# GPU Training Launch Guide

## Overview

This guide now reflects the path that actually worked on the live RunPod deployment, including the resume and watchdog fixes that prevented idle GPU time.

Primary references:
- `docs/RUNPOD_RUNBOOK.md` for the full operational record
- `deploy/runpod_launch.sh` for one-shot launch
- `deploy/runpod_watchdog.sh` for restartable continuous training

## Prerequisites

1. Training data already prepared locally and pushed to GitHub
2. RunPod account with payment method
3. A single high-VRAM GPU pod

## Step-by-Step

### 1. Prepare Training Data Locally

```bash
cd ~/projects/FLS-Training
python scripts/040_prepare_training_data.py --ver <version>
git add training/data/
git commit -m "prepare training dataset"
git push origin main
```

If you are reproducing the successful April 2026 run, use the matching dataset/config revision that produced the pod-side `training/data/v2` path.

### 2. Launch RunPod Instance

Recommended shape:
- single GPU only
- `A100 80GB` when you want the stable documented path
- `RTX PRO 6000 Blackwell Server Edition` is also proven and supported by the launcher
- PyTorch + CUDA 12.x image
- at least 30 GB persistent disk

### 3. Clone And Setup On The Pod

```bash
cd /workspace
git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git
cd FLS-Training
bash scripts/runpod_setup.sh
```

If the pod is reusing an older volume or stale repo state:

```bash
git fetch origin
git checkout main
git pull --ff-only origin main
```

### 4. Start Training

Normal one-shot path:

```bash
bash deploy/runpod_launch.sh <dataset_path> <config_path>
```

Example for the canonical Task5 baseline in this branch:

```bash
bash deploy/runpod_launch.sh training/data src/configs/finetune_task5_v2.yaml
```

Continuous/resume path:

```bash
nohup env \
  CONTINUOUS_HOURS=2 \
  WATCHDOG_POLL_SECONDS=15 \
  WATCHDOG_MAX_RESTARTS=20 \
  RUN_DIR_OVERRIDE=<checkpoint_run_dir> \
  TRAIN_LOG=/workspace/fls_train_continuous.log \
  WATCHDOG_LOG=/workspace/fls_watchdog.log \
  PYTHONUNBUFFERED=1 \
  bash deploy/runpod_watchdog.sh <dataset_path> <config_path> \
  > /workspace/fls_watchdog.stdout 2>&1 < /dev/null &
```

Why this matters:
- the launcher now skips `flash-attn` source builds during restart
- the watchdog injects `SKIP_DEP_INSTALL=1` automatically on resume
- this is the fix that restored real GPU usage on the live pod

### 5. Verify Real GPU Work

```bash
pgrep -af "runpod_watchdog|runpod_launch|finetune_vlm"
tail -n 100 /workspace/fls_train_continuous.log
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
nvidia-smi pmon -c 1
```

Healthy run:
- active `finetune_vlm` process
- non-idle GPU utilization
- nontrivial VRAM use

### 6. Push Artifacts Back To GitHub

```bash
git add models/ training/runs/ memory/model_checkpoints/
git commit -m "feat: training artifacts from RunPod"
git push origin main
```

### 7. Stop The Pod

Do not leave the pod idle after training or push.

## Troubleshooting

**GPU looks idle after a restart:** check for stuck dependency builds with:

```bash
pgrep -af "pip install|flash-attn|nvcc|cicc"
```

If you see `flash-attn` builds during resume, kill the stale launcher and restart through `deploy/runpod_watchdog.sh`.

**"Unsloth install failed"**: the launcher falls back to `hf_trainer` automatically.

**Blackwell host fails on stock Torch:** let `deploy/runpod_launch.sh` upgrade Torch/TorchVision; that path is now built into the launcher.

**Merged 16-bit export fails at the end:** keep the adapter checkpoint. The trainer now treats merged export failure as non-fatal.

See `docs/RUNPOD_RUNBOOK.md` for the full proven recovery path.

## Using Docker (Alternative)

If you prefer containerized training:

```bash
# Build locally
docker build -t fls-trainer -f deploy/Dockerfile.trainer .

# Run on GPU instance
docker run --gpus all -v $(pwd)/data:/workspace/data fls-trainer
```
