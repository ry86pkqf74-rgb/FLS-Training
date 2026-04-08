#!/usr/bin/env bash
# Prepare v2 training data directory in the format finetune_vlm.py expects.
# The trainer loads {dataset_path}/train.jsonl and {dataset_path}/val.jsonl
# but v2 data is named scoring_train_v2.jsonl etc.
#
# This script creates training/data/v2/ with the right symlinks.
#
# Usage: bash scripts/045_prep_v2_training.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V2_DIR="$REPO_ROOT/training/data/v2"
SRC_DIR="$REPO_ROOT/training/data"

echo "=== Preparing v2 training data ==="

mkdir -p "$V2_DIR"

# Check source files exist
for f in scoring_train_v2.jsonl scoring_val_v2.jsonl; do
    if [ ! -f "$SRC_DIR/$f" ]; then
        echo "ERROR: $SRC_DIR/$f not found"
        echo "Run: python scripts/040_prepare_training_data.py --ver v2"
        exit 1
    fi
done

# Create symlinks (or copies) that finetune_vlm.py expects
ln -sf "$SRC_DIR/scoring_train_v2.jsonl" "$V2_DIR/train.jsonl"
ln -sf "$SRC_DIR/scoring_val_v2.jsonl" "$V2_DIR/val.jsonl"

# Copy metadata
if [ -f "$SRC_DIR/meta_v2.json" ]; then
    cp "$SRC_DIR/meta_v2.json" "$V2_DIR/meta.json"
fi

# Also link coaching data if it exists
if [ -f "$SRC_DIR/coaching_train_v2.jsonl" ]; then
    ln -sf "$SRC_DIR/coaching_train_v2.jsonl" "$V2_DIR/coaching_train.jsonl"
fi
if [ -f "$SRC_DIR/coaching_val_v2.jsonl" ]; then
    ln -sf "$SRC_DIR/coaching_val_v2.jsonl" "$V2_DIR/coaching_val.jsonl"
fi

echo "Created $V2_DIR with:"
ls -la "$V2_DIR/"
echo ""

# Verify counts
echo "Training examples: $(wc -l < "$V2_DIR/train.jsonl")"
echo "Validation examples: $(wc -l < "$V2_DIR/val.jsonl")"
if [ -f "$V2_DIR/coaching_train.jsonl" ]; then
    echo "Coaching train: $(wc -l < "$V2_DIR/coaching_train.jsonl")"
    echo "Coaching val: $(wc -l < "$V2_DIR/coaching_val.jsonl")"
fi
echo ""
echo "=== v2 data ready. Run training with: ==="
echo "python src/training/finetune_vlm.py --config src/configs/finetune_task5_v2.yaml"
