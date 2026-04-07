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

echo "========================================="
echo "FLS-Training: GPU Fine-Tune Launch"
echo "========================================="
echo "Dataset: $DATASET_PATH"
echo "Config:  $CONFIG_PATH"
echo ""

# --- Step 1: System check ---
echo "[1/5] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || {
    echo "ERROR: No GPU detected. This script requires a CUDA GPU."
    exit 1
}
echo ""

# --- Step 2: Install dependencies ---
echo "[2/5] Installing dependencies..."
pip install --quiet --break-system-packages -e ".[training]" 2>/dev/null || \
    pip install --quiet -e ".[training]"

# Install Unsloth (try CUDA 12.1 + torch 2.4 first, fall back)
echo "Installing Unsloth..."
pip install --quiet --break-system-packages \
    "unsloth[cu124-torch240] @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null || \
pip install --quiet --break-system-packages \
    "unsloth @ git+https://github.com/unslothai/unsloth.git" 2>/dev/null || {
    echo "WARNING: Unsloth install failed. Will fall back to hf_trainer backend."
    echo "To use hf_trainer, edit config: framework: hf_trainer"
}

# Flash attention (optional, speeds up training ~20%)
pip install --quiet --break-system-packages flash-attn --no-build-isolation 2>/dev/null || \
    echo "WARNING: flash-attn not installed (OK, training will still work)"

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
# Create a runtime config copy with the actual dataset path
RUNTIME_CONFIG="/tmp/finetune_runtime.yaml"
sed "s|dataset_path:.*|dataset_path: \"$DATASET_PATH\"|" "$CONFIG_PATH" > "$RUNTIME_CONFIG"

# Show key settings
echo "  Model:     $(grep base_model $RUNTIME_CONFIG | head -1)"
echo "  Framework: $(grep framework $RUNTIME_CONFIG | head -1)"
echo "  Epochs:    $(grep num_epochs $RUNTIME_CONFIG | head -1)"
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
