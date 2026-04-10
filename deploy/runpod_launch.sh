#!/usr/bin/env bash
# deploy/runpod_launch.sh — One-shot setup + training on RunPod or Vast.ai
#
# Prerequisites:
#   - GPU instance with CUDA 12.x (RTX 4090 24GB or L40S recommended)
#   - Training data already prepared locally via:
#       python scripts/040_prepare_training_data.py --ver v2
#       bash scripts/045_prep_v2_training.sh
#   - JSONL files uploaded to the instance (see LAUNCH_GUIDE.md)
#
# Usage:
#   # On the GPU instance:
#   bash deploy/runpod_launch.sh [dataset_path] [config_path]
#
# Example:
#   bash deploy/runpod_launch.sh training/data src/configs/finetune_task5_v2.yaml

set -euo pipefail
DATASET_PATH="${1:-data/training/LATEST}"
CONFIG_PATH="${2:-src/configs/finetune_task5_v2.yaml}"
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

maybe_prepare_named_training_dataset() {
    local config_name target_path prep_script

    config_name="$(basename "$CONFIG_PATH")"
    case "$config_name" in
        finetune_task5_v2.yaml)
            target_path="training/data/v2"
            prep_script="scripts/045_prep_v2_training.sh"
            ;;
        finetune_task5_v3.yaml)
            target_path="training/data/v3"
            prep_script="scripts/045_prep_v3_training.sh"
            ;;
        *)
            return 1
            ;;
    esac

    if [ ! -f "$prep_script" ]; then
        return 1
    fi

    echo "  Preparing canonical dataset alias via $prep_script"
    bash "$prep_script"

    if [ "$DATASET_PATH" = "training/data" ] || [ "$DATASET_PATH" = "./training/data" ]; then
        DATASET_PATH="$target_path"
    fi

    [ -f "$DATASET_PATH/train.jsonl" ] || [ -f "$target_path/train.jsonl" ] || return 1

    if [ ! -f "$DATASET_PATH/train.jsonl" ] && [ -f "$target_path/train.jsonl" ]; then
        DATASET_PATH="$target_path"
    fi

    return 0
}

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

# If the user passed data/training/LATEST-style aliases, resolve them to
# either the symlink target or the newest matching timestamped directory.
if [[ "$DATASET_PATH" == *"LATEST"* ]]; then
    if [ -L "$DATASET_PATH" ]; then
        RESOLVED="$(python3 - <<'PY' "$DATASET_PATH"
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
)"
    elif [[ "$DATASET_PATH" == *"LATEST_LASANA" ]]; then
        RESOLVED=$(ls -1dt data/training/*_lasana_v* 2>/dev/null | head -1 || true)
    else
        RESOLVED=$(ls -1dt data/training/*_v* 2>/dev/null | head -1 || true)
    fi
    if [ -n "${RESOLVED:-}" ] && [ -f "$RESOLVED/train.jsonl" ]; then
        DATASET_PATH="$RESOLVED"
        echo "  LATEST resolved to: $DATASET_PATH"
    fi
fi

if [ ! -f "$DATASET_PATH/train.jsonl" ]; then
    maybe_prepare_named_training_dataset || true
fi

if [ ! -f "$DATASET_PATH/train.jsonl" ]; then
    echo "ERROR: $DATASET_PATH/train.jsonl not found."
    echo ""
    case "$(basename "$CONFIG_PATH")" in
        finetune_task5_v2.yaml)
            echo "Prepare the canonical v2 alias on this host:"
            echo "  bash scripts/045_prep_v2_training.sh"
            echo ""
            echo "Or rebuild the source files locally first:"
            echo "  python scripts/040_prepare_training_data.py --ver v2"
            echo "  rsync -avz training/data/ root@\$GPU_IP:/workspace/FLS-Training/training/data/"
            ;;
        finetune_task5_v3.yaml)
            echo "Prepare the canonical v3 alias on this host:"
            echo "  bash scripts/045_prep_v3_training.sh"
            echo ""
            echo "Or rebuild the source files locally first:"
            echo "  python scripts/040_prepare_training_data.py --ver v3"
            echo "  rsync -avz training/data/ root@\$GPU_IP:/workspace/FLS-Training/training/data/"
            ;;
        *)
            echo "You need to prepare training data LOCALLY first:"
            echo '  python scripts/040_prepare_training_data.py --ver v4 \\'
            echo '      --frames-dir data/frames --max-frames 24 \\'
            echo "      --include-coach-feedback --group-by trainee"
            echo ""
            echo "Then upload the output directory to this instance:"
            echo "  rsync -avz data/training/ root@\$GPU_IP:/workspace/FLS-Training/data/training/"
            ;;
    esac
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

# Vision-mode validation: if the config declares require_vision: true,
# check that the first training example actually contains image blocks
# AND that at least one of those image paths exists on the pod.
REQUIRE_VISION=$(grep -E '^require_vision:' "$CONFIG_PATH" 2>/dev/null | awk '{print $2}' || true)
if [ "$REQUIRE_VISION" = "true" ]; then
    python3 - <<PY
import json, sys
from pathlib import Path
with open("$DATASET_PATH/train.jsonl") as handle:
    first = json.loads(handle.readline())
has_image = False
first_image = None
for message in first.get("messages", []):
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                has_image = True
                first_image = block.get("image")
                break
    if has_image:
        break
if not has_image:
    print("ERROR: config has require_vision: true but train.jsonl has no image blocks.")
    print("Re-run scripts/040_prepare_training_data.py with --frames-dir set.")
    sys.exit(1)
if first_image and not Path(first_image).is_file():
    print(f"ERROR: first image path does not resolve on this pod: {first_image}")
    print("rsync the frames directory onto the pod before launching training.")
    sys.exit(1)
print(f"  Vision check: first image OK ({first_image})")
PY
    if [ $? -ne 0 ]; then
        echo ""
        echo "Vision validation failed. Aborting launch before GPU allocation."
        exit 1
    fi
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
    # Respect the source config's save_strategy. Only default to "steps"
    # if the source config didn't specify one. Fixes F2 bug where epoch-
    # based configs were silently overridden to save_steps=200, producing
    # zero checkpoints on short runs.
    config.setdefault("save_strategy", "steps")
    if config["save_strategy"] == "steps":
        config.setdefault("save_steps", 200)
    config.setdefault("save_total_limit", 2)

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

# --- Write run manifest (scripts/041_write_run_manifest.py) ---
# Captures git state, GPU info, dataset fingerprint, and config hash.
# Patches $RUNTIME_CONFIG with the resolved output_dir when missing so
# finetune_vlm.py and the manifest land in the same directory.
echo "  Writing run manifest..."
TRAINING_OUTPUT_DIR=$(
    python scripts/041_write_run_manifest.py \
        --config "$RUNTIME_CONFIG" \
        --dataset-path "$DATASET_PATH" 2>/tmp/manifest_write.log
) || {
    echo "  WARNING: run manifest write failed — $(cat /tmp/manifest_write.log)"
    TRAINING_OUTPUT_DIR=""
}

echo "========================================="
echo ""

# Capture the training exit code without aborting the script so the
# finalize step always runs regardless of success or failure.
set +e
python -m src.training.finetune_vlm --config "$RUNTIME_CONFIG"
TRAINING_EXIT=$?
set -e

# --- Finalize run manifest with outcome metrics ---
if [ -n "$TRAINING_OUTPUT_DIR" ]; then
    echo "  Finalizing run manifest (exit=$TRAINING_EXIT)..."
    python scripts/042_finalize_run_manifest.py \
        --output-dir "$TRAINING_OUTPUT_DIR" \
        --exit-status "$TRAINING_EXIT" 2>&1 || \
    echo "  WARNING: run manifest finalization failed (non-fatal)"
fi

# --- Post-run cleanup: reclaim failed merged_16bit exports (F3 mitigation) ---
if [ -n "$TRAINING_OUTPUT_DIR" ] && [ -d "$TRAINING_OUTPUT_DIR/merged_16bit" ]; then
    # If the merged dir has no adapter_model.safetensors it means the merge
    # crashed after downloading base shards but before finishing. Reclaim space.
    if [ ! -f "$TRAINING_OUTPUT_DIR/merged_16bit/adapter_model.safetensors" ] && \
       [ ! -f "$TRAINING_OUTPUT_DIR/merged_16bit/model.safetensors" ]; then
        MERGED_SIZE=$(du -sh "$TRAINING_OUTPUT_DIR/merged_16bit" | cut -f1)
        echo "  Cleaning up failed merged_16bit export ($MERGED_SIZE)..."
        rm -rf "$TRAINING_OUTPUT_DIR/merged_16bit"
    fi
fi

# Re-propagate a non-zero training exit so CI/watchdog sees the failure.
if [ "$TRAINING_EXIT" -ne 0 ]; then
    echo "ERROR: finetune_vlm exited with status $TRAINING_EXIT" >&2
    exit "$TRAINING_EXIT"
fi

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

# --- Auto-stop pod after training (saves $1.90/hr idle cost) ---
if [ -n "$RUNPOD_POD_ID" ]; then
  echo ""
  echo "[auto-stop] Training finished (exit=$TRAINING_EXIT). Pushing logs and stopping pod..."

  # Push training logs + manifest to GitHub if git is configured
  if git remote get-url origin &>/dev/null; then
    git add -f memory/model_checkpoints/*/run_manifest.json 2>/dev/null
    git add src/configs/ 2>/dev/null
    git commit -m "training: LASANA pretrain run manifest (exit=$TRAINING_EXIT)

Auto-committed by runpod_launch.sh after training completed." 2>/dev/null && \
    git push origin main 2>/dev/null && \
    echo "[auto-stop] Pushed run manifest to GitHub." || \
    echo "[auto-stop] WARNING: Git push failed (non-fatal)."
  fi

  # Stop the pod via RunPod API
  if [ -n "$RUNPOD_API_KEY" ]; then
    echo "[auto-stop] Stopping pod $RUNPOD_POD_ID via API..."
    curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
      -H "Content-Type: application/json" \
      -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}" && \
    echo "[auto-stop] Pod stop request sent." || \
    echo "[auto-stop] WARNING: Pod stop API call failed."
  else
    echo "[auto-stop] RUNPOD_API_KEY not set  cannot auto-stop. Stop manually!"
  fi
else
  echo "[auto-stop] Not running on RunPod (no RUNPOD_POD_ID). Skipping auto-stop."
fi
