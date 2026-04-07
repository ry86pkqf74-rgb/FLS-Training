# GPU Training Launch Guide

## Overview

Fine-tune Qwen2.5-VL-7B-Instruct on your FLS scoring data using LoRA via Unsloth.
Expected cost: ~$3-5 on RunPod H200 spot.

## Prerequisites

1. **Scored videos**: At least 15-20 videos with consensus scores in DuckDB
2. **API keys**: Not needed on GPU server (training is local)
3. **RunPod or Vast.ai account** with payment method

## Step-by-Step

### 1. Prepare Training Data (LOCAL machine)

```bash
cd ~/projects/FLS-Training

# Basic (score-only training examples)
python scripts/040_prepare_training_data.py --version 1 --min-confidence 0.5

# With coach feedback (student learns both scoring + coaching)
python scripts/040_prepare_training_data.py --version 1 --min-confidence 0.5 --include-coach
```

This creates `data/training/YYYY-MM-DD_v1/` with `train.jsonl`, `val.jsonl`, `test.jsonl`.

### 2. Launch GPU Instance

**RunPod (recommended for spot pricing):**
- Template: `runpod/pytorch:2.4.0-py3.11-cuda12.4.0-devel-ubuntu22.04`
- GPU: H200 SXM (141GB) — ~$3-4/hr spot, best quality
- Disk: 80GB (model weights ~15GB bf16 + dataset + checkpoint)
- 

**Vast.ai alternative:**
- Search for H200 SXM or H100 SXM
- Use PyTorch 2.4+ CUDA 12.4 template
- H200 ~$3-4/hr, H100 ~$2-3/hr

### 3. Upload Repo + Data to GPU Instance

```bash
# From your local machine:
GPU_IP="your-instance-ip"

# Clone repo on the instance
ssh root@$GPU_IP "cd /workspace && git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git"

# Upload training data (not in git)
rsync -avz data/training/ root@$GPU_IP:/workspace/FLS-Training/data/training/
```

Or if you prefer a single rsync of everything:
```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' \
    ~/projects/FLS-Training/ root@$GPU_IP:/workspace/FLS-Training/
```

### 4. Run Training

```bash
ssh root@$GPU_IP
cd /workspace/FLS-Training

# One-shot: installs deps + validates data + trains
bash deploy/runpod_launch.sh data/training/2026-04-07_v1

# Or step-by-step if you prefer:
pip install -e ".[training]"
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
python -m src.training.finetune_vlm --config src/configs/finetune_task5_v1.yaml
```

Training takes ~15-20 minutes on H200 for 26 videos × 2 epochs.

### 5. Download Checkpoint

```bash
# From local machine:
rsync -avz root@$GPU_IP:/workspace/FLS-Training/memory/model_checkpoints/ \
    ~/projects/FLS-Training/memory/model_checkpoints/
```

### 6. Evaluate & Promote

```bash
# Run student model on held-out videos (requires the checkpoint)
# Then compare against frontier consensus:
python scripts/050_evaluate.py \
    --student-scores data/eval/student_predictions.jsonl \
    --frontier-scores data/eval/frontier_consensus.jsonl \
    --config src/configs/finetune_task5_v1.yaml
```

Promotion criteria (from config):
- Field agreement ≥ 90%
- FLS score MAE ≤ 15 points

### 7. Shut Down Instance

**Don't forget!** Spot instances charge even when idle.

## Cost Estimates

| Phase | Cost |
|-------|------|
| API scoring (26 videos × Claude + GPT-4o) | ~$15-25 |
| Coach feedback (26 videos × Claude) | ~$5-10 |
| GPU training (H200 spot, ~20 min) | ~$1-2 |
| Evaluation inference | ~$2-5 |
| **Total** | **~$25-42** |

## Troubleshooting

**"Unsloth install failed"**: The launch script falls back to `hf_trainer` automatically. Edit `src/configs/finetune_task5_v1.yaml` and set `framework: hf_trainer`.

**OOM (shouldn't happen on H200): If on a smaller GPU, set quantization: "4bit", batch_size: 1, lora_r: 16, gradient_checkpointing: true.

**"No training candidates found"**: Your DuckDB doesn't have scored videos above the confidence threshold. Lower `--min-confidence` or score more videos.

**Training loss not decreasing**: Try `learning_rate: 5.0e-5` (lower) or increase `lora_r` to 32.

## Using Docker (Alternative)

If you prefer containerized training:

```bash
# Build locally
docker build -t fls-trainer -f deploy/Dockerfile.trainer .

# Run on GPU instance
docker run --gpus all -v $(pwd)/data:/workspace/data fls-trainer
```
