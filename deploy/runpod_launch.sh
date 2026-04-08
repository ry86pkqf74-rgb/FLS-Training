#!/usr/bin/env bash
# deploy/runpod_launch.sh — One-shot setup + training on RunPod or Vast.ai
#
# Prerequisites:
#   - GPU instance with CUDA 12.x (RTX 4090 24GB or L40S recommended)
#   - Training data already prepared locally via:
#       python scripts/040_prepare_training_data.py --version 1
#   - JSONL files uploaded to the instance (see LAUNCH_GUIDE.md)
#
# Usage:
#   # On the GPU instance:
#   bash deploy/runpod_launch.sh [dataset_path] [config_path]
#
# Example:
#   bash deploy/runpod_launch.sh data/training/2026-04-07_v1 src/configs/finetune_task5_v1.yaml

set -euo pipefail

DATASET_PATH="${1:-data/training/LATEST}"
CONFIG_PATH="${2:-src/configs/finetune_task5_v1.yaml}"
CONTINUOUS_HOURS="${CONTINUOUS_HOURS:-0}"
SECONDS_PER_STEP_ESTIMATE="${SECONDS_PER_STEP_ESTIMATE:-7}"
BLACKWELL_BATCH_SIZE="${BLACKWELL_BATCH_SIZE:-8}"
BLACKWELL_GRAD_ACCUM="${BLACKWELL_GRAD_ACCUM:-1}"
BLACKWELL_WORKERS="${BLACKWELL_WORKERS:-8}"
BLACKWELL_LORA_DROPOUT="${BLACKWELL_LORA_DROPOUT:-0}"
SKIP_DEP_INSTALL="${SKIP_DEP_INSTALL:-0}"
BASE_MODEL_OVERRIDE="${BASE_MODEL_OVERRIDE:-}"
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

echo "========================================="
echo "FLS-Training: GPU Fine-Tune Launch"
echo "========================================="
echo "Dataset: $DATASET_PATH"
echo "Config:  $CONFIG_PATH"
if [[ "$CONTINUOUS_HOURS" != "0" ]]; then
    echo "Mode:    continuous (${CONTINUOUS_HOURS}h target)"
fi
echo ""

# --- Step 1: System check ---
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || {
    echo "ERROR: No GPU detected. This script requires a CUDA GPU."
    exit 1
}
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# --- Step 2: Install dependencies ---
echo "[2/5] Installing dependencies..."

if [[ "$SKIP_DEP_INSTALL" == "1" ]]; then
    echo "Skipping dependency installation for resume/continuous relaunch."
else

if python -c 'import src.training.finetune_vlm, yaml' >/dev/null 2>&1; then
    echo "Project training package already importable; skipping editable reinstall."
else
    pip install --quiet --break-system-packages -e ".[training]" 2>/dev/null || \
        pip install --quiet -e ".[training]"
fi

# Install Unsloth (try CUDA 12.1 + torch 2.4 first, fall back)
if python -c 'import unsloth, trl, datasets, transformers, peft' >/dev/null 2>&1; then
    echo "Unsloth training stack already present; skipping reinstall."
else
    echo "Installing Unsloth..."
    pip install --quiet --break-system-packages \
        "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null || \
    pip install --quiet --break-system-packages \
        "unsloth @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null || {
        echo "WARNING: Unsloth install failed. Will fall back to hf_trainer backend."
        echo "To use hf_trainer, edit config: framework: hf_trainer"
    }
fi

# Flash attention (optional, speeds up training ~20%)
if python -c 'import flash_attn' >/dev/null 2>&1; then
    echo "flash-attn already installed; skipping reinstall."
else
    echo "WARNING: flash-attn not installed; skipping build during launch (training will still work)"
fi

# The base RunPod Torch stack in this image does not support Blackwell GPUs.
if [[ "$GPU_NAME" == *"Blackwell"* ]]; then
    echo "Checking Blackwell-compatible PyTorch stack..."
    if ! python -c 'import torch, torchvision, sys; sys.exit(0 if torch.__version__.startswith("2.10.0") and torchvision.__version__.startswith("0.25.0") else 1)' >/dev/null 2>&1; then
        echo "Installing Blackwell-compatible PyTorch stack..."
        python -m pip install --break-system-packages --upgrade --force-reinstall \
            torch==2.10.0+cu128 torchvision==0.25.0+cu128 \
            --index-url https://download.pytorch.org/whl/cu128
    else
        echo "Blackwell PyTorch stack already correct; skipping reinstall."
    fi
    python -m pip uninstall -y torchaudio >/dev/null 2>&1 || true
fi

fi

echo ""

# --- Step 3: Validate dataset ---
echo "[3/5] Validating dataset at $DATASET_PATH..."
if [ ! -f "$DATASET_PATH/train.jsonl" ]; then
    echo "ERROR: $DATASET_PATH/train.jsonl not found."
    echo ""
    echo "You need to prepare training data LOCALLY first:"
    echo "  python scripts/040_prepare_training_data.py --version 1"
    echo ""
    echo "Then upload the output directory to this instance:"
    echo "  rsync -avz data/training/ root@\$GPU_IP:/workspace/FLS-Training/data/training/"
    exit 1
fi

TRAIN_COUNT=$(wc -l < "$DATASET_PATH/train.jsonl")
VAL_COUNT=$(wc -l < "$DATASET_PATH/val.jsonl" 2>/dev/null || echo "0")
echo "  Train examples: $TRAIN_COUNT"
echo "  Val examples:   $VAL_COUNT"

if [ "$TRAIN_COUNT" -lt 5 ]; then
    echo "WARNING: Very few training examples ($TRAIN_COUNT). Results may be poor."
    echo "Consider scoring more videos first."
fi
echo ""

# --- Step 4: Update config with actual dataset path ---
echo "[4/5] Updating config with dataset path..."
# Create a runtime config copy with the actual dataset path and pod-specific tuning
RUNTIME_CONFIG="/tmp/finetune_runtime.yaml"
export DATASET_PATH CONFIG_PATH RUNTIME_CONFIG TRAIN_COUNT VAL_COUNT GPU_NAME CONTINUOUS_HOURS \
    SECONDS_PER_STEP_ESTIMATE BLACKWELL_BATCH_SIZE BLACKWELL_GRAD_ACCUM BLACKWELL_WORKERS \
    BLACKWELL_LORA_DROPOUT BASE_MODEL_OVERRIDE OUTPUT_DIR_OVERRIDE RESUME_FROM_CHECKPOINT
python - <<'PY'
import math
import os
from pathlib import Path

import yaml

config_path = Path(os.environ["CONFIG_PATH"])
runtime_path = Path(os.environ["RUNTIME_CONFIG"])

with open(config_path) as f:
    config = yaml.safe_load(f)

config["dataset_path"] = os.environ["DATASET_PATH"]
config.setdefault("max_seq_length", 2048)
base_model_override = os.environ.get("BASE_MODEL_OVERRIDE", "").strip()
if base_model_override:
    config["base_model"] = base_model_override
output_dir_override = os.environ.get("OUTPUT_DIR_OVERRIDE", "").strip()
if output_dir_override:
    config["output_dir"] = output_dir_override
resume_from_checkpoint = os.environ.get("RESUME_FROM_CHECKPOINT", "").strip()
if resume_from_checkpoint:
    config["resume_from_checkpoint"] = resume_from_checkpoint

gpu_name = os.environ.get("GPU_NAME", "")
train_count = max(int(os.environ.get("TRAIN_COUNT", "0")), 1)

if "Blackwell" in gpu_name:
    config["batch_size"] = int(os.environ.get("BLACKWELL_BATCH_SIZE", str(config.get("batch_size", 8))))
    config["gradient_accumulation"] = int(os.environ.get("BLACKWELL_GRAD_ACCUM", str(config.get("gradient_accumulation", 1))))
    config["dataloader_num_workers"] = max(int(config.get("dataloader_num_workers", 0)), int(os.environ.get("BLACKWELL_WORKERS", "8")))
    config["lora_dropout"] = float(os.environ.get("BLACKWELL_LORA_DROPOUT", str(config.get("lora_dropout", 0.0))))
    config["save_strategy"] = "steps"
    config["save_steps"] = int(config.get("save_steps", 200))
    config["save_total_limit"] = int(config.get("save_total_limit", 2))

continuous_hours = float(os.environ.get("CONTINUOUS_HOURS", "0"))
if continuous_hours > 0:
    batch_size = max(int(config.get("batch_size", 1)), 1)
    grad_accum = max(int(config.get("gradient_accumulation", 1)), 1)
    steps_per_epoch = max(1, math.ceil(train_count / (batch_size * grad_accum)))
    step_seconds = max(float(os.environ.get("SECONDS_PER_STEP_ESTIMATE", "7")), 1.0)
    target_steps = max(1, math.ceil((continuous_hours * 3600.0) / step_seconds))
    target_epochs = max(int(config.get("num_epochs", 1)), math.ceil(target_steps / steps_per_epoch))
    config["num_epochs"] = target_epochs
    config["save_strategy"] = "steps"
    config["save_steps"] = min(int(config.get("save_steps", 200)), max(50, target_steps // 4))
    config["save_total_limit"] = int(config.get("save_total_limit", 2))
    config["logging_steps"] = max(int(config.get("logging_steps", 10)), 25)
    config["eval_strategy"] = "no"

with open(runtime_path, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

# If Unsloth imports fail on the host Torch build, fall back to a safer HF config.
if grep -q 'framework: "unsloth"' "$RUNTIME_CONFIG"; then
    if ! python -c 'import unsloth' >/dev/null 2>&1; then
        sed -i 's|framework:.*|framework: "hf_trainer"|' "$RUNTIME_CONFIG"
        sed -i 's|quantization:.*|quantization: "none"|' "$RUNTIME_CONFIG"
        sed -i 's|batch_size:.*|batch_size: 2|' "$RUNTIME_CONFIG"
        sed -i 's|gradient_checkpointing:.*|gradient_checkpointing: true|' "$RUNTIME_CONFIG"
        echo "  WARNING: Unsloth runtime import failed; switching to hf_trainer fallback"
    fi
fi

# Show key settings
echo "  Model:     $(grep base_model $RUNTIME_CONFIG | head -1)"
echo "  Framework: $(grep framework $RUNTIME_CONFIG | head -1)"
echo "  Epochs:    $(grep num_epochs $RUNTIME_CONFIG | head -1)"
echo "  Batch:     $(grep batch_size $RUNTIME_CONFIG | head -1)"
echo "  GradAcc:   $(grep gradient_accumulation $RUNTIME_CONFIG | head -1)"
echo "  SaveMode:  $(grep save_strategy $RUNTIME_CONFIG | head -1)"
echo "  LR:        $(grep learning_rate $RUNTIME_CONFIG | head -1)"
echo ""

# --- Step 5: Launch training ---
echo "[5/5] Starting training..."
echo "========================================="
echo ""

python -m src.training.finetune_vlm --config "$RUNTIME_CONFIG"

echo ""
echo "========================================="
echo "Training complete!"
echo "Checkpoint saved to memory/model_checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Download the checkpoint to your local machine"
echo "  2. Run evaluation: python scripts/050_evaluate.py --student-scores ... --frontier-scores ..."
echo "  3. If promoted, update .env: STUDENT_MODEL=<path_to_checkpoint>"
echo "========================================="
