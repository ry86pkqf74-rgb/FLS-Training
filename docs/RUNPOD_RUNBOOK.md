# RunPod Runbook

This is the repo's durable record of the path that actually worked on the live RunPod deployment in April 2026.

Use this document when you need to stand up a fresh training pod, resume a stalled run, or verify that the server is doing real work instead of sitting idle.

## Proven Pod Profile

- Provider: RunPod
- Access pattern: SSH via `ssh.runpod.io`
- Proven live GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Also acceptable when available: single `A100 80GB`
- Image family: PyTorch + CUDA 12.x base image
- Disk: at least 30 GB persistent volume
- Working repo path on pod: `/workspace/FLS-Training`

Notes:
- Single GPU is the correct shape here. Multi-GPU adds complexity and does not help this dataset size.
- Blackwell hosts need the updated Torch stack that `deploy/runpod_launch.sh` now installs automatically.
- Resume paths must not rebuild optional dependencies every time. The watchdog/launcher pair now avoids that.

## Local Prep Before You Start a Pod

Run these locally before paying for GPU time:

```bash
python scripts/040_prepare_training_data.py --ver <version>
git status
git add training/data/ deploy/ docs/ README.md
git commit -m "prepare training dataset and launch docs"
git push origin main
```

If you are deploying the exact successful April 2026 flow, the pod used a `training/data/v2` dataset and a matching v2 training config on that checked-out revision.

## Initial Pod Setup

SSH to the pod and run:

```bash
cd /workspace
git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git
cd FLS-Training
bash scripts/runpod_setup.sh
```

Sanity checks:
- `torch.cuda.is_available()` is true
- the GPU name is printed correctly
- training JSONL files are present under `training/data/` or your selected dataset directory

If the pod was launched from an older image or stale volume, refresh the repo first:

```bash
cd /workspace/FLS-Training
git fetch origin
git checkout main
git pull --ff-only origin main
```

## Standard One-Shot Launch

Use this when you want the simplest supported path:

```bash
cd /workspace/FLS-Training
bash deploy/runpod_launch.sh <dataset_path> <config_path>
```

Example using the config that exists in this branch today:

```bash
bash deploy/runpod_launch.sh training/data src/configs/finetune_task5_v1.yaml
```

What the launcher now handles automatically:
- verifies GPU presence
- installs the project if needed
- installs the Unsloth stack if missing
- skips `flash-attn` source builds during launch if it is absent
- upgrades Torch/TorchVision automatically on Blackwell hosts
- writes a runtime config to `/tmp/finetune_runtime.yaml`
- supports `OUTPUT_DIR_OVERRIDE` and `RESUME_FROM_CHECKPOINT`

## Continuous / Resume Mode

Use the watchdog when you want the server to keep working for a fixed window and resume from the latest checkpoint after interruptions.

```bash
cd /workspace/FLS-Training
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

Behavior that is now proven and intentional:
- the watchdog polls every 15 seconds
- it detects both direct config launches and runtime-config launches
- it prefers the configured `output_dir` or `RUN_DIR_OVERRIDE`
- on restart it injects `SKIP_DEP_INSTALL=1` so resume does not waste time in dependency setup
- if a `checkpoint-*` directory exists, it resumes from the newest one automatically

## The Failure Mode We Hit And The Fix

The main source of idle GPU time was not the model itself. It was the relaunch path trying to compile `flash-attn` from source every time the watchdog restarted training.

The successful fix is already in the repo:
- `deploy/runpod_launch.sh` now supports `SKIP_DEP_INSTALL=1`
- watchdog-triggered resumes set that flag automatically
- the launcher now warns and continues if `flash-attn` is absent instead of compiling it during restart

Operational rule:
- for first launch, normal dependency setup is fine
- for resume or continuous mode, always go through `deploy/runpod_watchdog.sh` or explicitly set `SKIP_DEP_INSTALL=1`

## Verification Checklist

Do not trust only the existence of a shell process. Verify actual GPU work.

Use these checks on the pod:

```bash
pgrep -af "runpod_watchdog|runpod_launch|finetune_vlm"
tail -n 100 /workspace/fls_train_continuous.log
tail -n 100 /workspace/fls_watchdog.log
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
nvidia-smi pmon -c 1
```

Healthy signs:
- a `python -m src.training.finetune_vlm --config /tmp/finetune_runtime.yaml` process is present
- GPU utilization is materially above idle
- VRAM is well above idle baseline
- `nvidia-smi pmon` shows a compute process

On the successful live fix, the pod returned to:
- `100%` GPU utilization
- about `32 GiB` VRAM in use
- active `finetune_vlm` process after clearing the stalled relaunch path

## If Training Looks Stuck

Check whether the launcher is building dependencies instead of training:

```bash
pgrep -af "pip install|flash-attn|nvcc|cicc"
```

If you see `flash-attn` builds on a resumed run, kill the stale launcher and restart via the watchdog:

```bash
pkill -f "pip install --quiet --break-system-packages flash-attn" || true
pkill -f "bash /workspace/FLS-Training/deploy/runpod_launch.sh" || true
pkill -f "bash deploy/runpod_watchdog.sh" || true
```

Then relaunch continuous mode with the watchdog command above.

## Checkpoints And Export Behavior

Training artifacts are stored under `memory/model_checkpoints/` unless the config overrides the output directory.

Important detail:
- merged 16-bit export failures should no longer invalidate a completed run
- if Unsloth fails during merged export, the adapter checkpoint is still the artifact to keep

Treat adapter checkpoints as the primary durable output.

## Post-Training Deployment Checklist

From the pod:

```bash
cd /workspace/FLS-Training
git status
git add models/ training/runs/ memory/model_checkpoints/
git commit -m "feat: training artifacts from RunPod"
git push origin main
```

Before pushing, also capture:
- training log location
- watchdog log location if continuous mode was used
- run directory under `memory/model_checkpoints/`
- evaluation outputs and MAE summary if generated

## Shutdown Checklist

After push and verification:
- stop the pod
- do not leave it idling after training
- if using a persistent volume, keep only the artifacts you actually need

Minimal final review before shutdown:

```bash
git log -1 --stat
ls memory/model_checkpoints
```

## Recommended Future Deployment Path

For future runs, use this order:
1. Prepare and commit the dataset locally.
2. Launch a single-GPU RunPod pod.
3. Clone fresh on the pod.
4. Run `bash scripts/runpod_setup.sh`.
5. Start with `deploy/runpod_launch.sh` for a normal run.
6. If you need a multi-hour or interruption-tolerant session, use `deploy/runpod_watchdog.sh` immediately.
7. Verify with `nvidia-smi`, `pmon`, and process checks.
8. Push artifacts back to GitHub.
9. Stop the pod.