# GPU Training Launch Guide

## Overview

This guide now reflects the path that actually worked on the live RunPod deployment, including the resume and watchdog fixes that prevented idle GPU time.

It also includes the repo-native S1 harvest-and-score path for CPU hosts that are being used to download YouTube targets, run the dual-teacher scoring pass, generate consensus artifacts, and push data outputs back to GitHub.

Primary references:
- `docs/RUNPOD_RUNBOOK.md` for the full operational record
- `deploy/runpod_launch.sh` for one-shot launch
- `deploy/runpod_watchdog.sh` for restartable continuous training
- `deploy/s1_harvest_score_deploy.sh` for Hetzner S1 harvest + scoring runs

## Hetzner S1 Harvest + Score

Use this when the goal is not GPU training, but bulk YouTube harvest plus teacher scoring on a CPU box.

### Prerequisites

1. Root access on the target host
2. `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` exported in the shell before launch
3. `GITHUB_TOKEN` exported if the host should push new score artifacts back to `main`
4. Optional but often needed for YouTube extraction: `YT_DLP_COOKIES_FILE` or `YT_DLP_COOKIES_FROM_BROWSER`

### Launch Sequence

Copy the script to the host and run it inside `tmux`:

```bash
scp deploy/s1_harvest_score_deploy.sh root@<host>:/root/
ssh root@<host>
tmux new -s fls

export ANTHROPIC_API_KEY='sk-ant-...'
export OPENAI_API_KEY='sk-...'
export GITHUB_TOKEN='github_pat_...'

bash /root/s1_harvest_score_deploy.sh
```

Detach with `Ctrl-B D` and later reconnect with:

```bash
tmux attach -t fls
```

### What The Script Does

1. Installs `ffmpeg`, `git`, `nodejs`, `python3-venv`, `tmux`, and `yt-dlp`
2. Clones or fast-forwards the repo at `/opt/FLS-Training`
3. Creates a dedicated virtual environment at `/opt/fls-training-venv`
4. Probes a configurable subset of harvest targets before downloading so blocked runs fail early
5. Runs `scripts/011c_harvest_from_csv.py` against `data/harvest_targets.csv`
6. Runs `scripts/021_batch_score.py` with the repo's current task-routing fixes
7. Runs `scripts/030_run_consensus.py --with-coach-feedback`
8. Runs `scripts/026_auto_validate.py --scores-dir memory/scores`
9. Commits and pushes updated `harvest_log.jsonl`, `memory/scores`, `memory/comparisons`, `memory/feedback`, and `memory/validation_results.jsonl`

### Useful Overrides

All of these are optional environment variables:

- `WORK_DIR` to change the checkout path
- `HARVEST_MAX` to limit how many CSV rows are downloaded in one run
- `HARVEST_PROBE_FIRST=0` to skip the pre-download accessibility probe
- `HARVEST_PROBE_MAX` to control how many CSV rows are checked during the probe
- `SCORE_MAX` to limit how many videos are scored in one run
- `PROMPT_VERSION` to select a prompt family such as `v002`
- `SCORER_DELAY` to increase or reduce the pause between API calls
- `HARVEST_INCLUDE_UNCLASSIFIED=1` to include CSV rows whose task is still unclassified
- `RUN_CONSENSUS=0`, `RUN_VALIDATION=0`, or `PUSH_RESULTS=0` to disable later pipeline stages
- `YT_DLP_COOKIES_FILE=/root/youtube-cookies.txt` or `YT_DLP_COOKIES_FROM_BROWSER=firefox` to let `yt-dlp` use an authenticated session

### Notes

- Downloaded source videos land under the executing user's home directory because `scripts/011c_harvest_from_csv.py` uses `Path.home()/fls_harvested_videos`.
- The script refuses to pull over a dirty checkout in `/opt/FLS-Training`; clean or move that checkout first.
- Pushing is intentionally token-driven. The repo script does not embed GitHub credentials.
- If the probe reports `0 accessible`, the current target slice is blocked for anonymous `yt-dlp`; provide cookies or use a filtered target subset before retrying.

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
