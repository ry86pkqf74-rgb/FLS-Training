#!/usr/bin/env bash
# Prepare v3 training data directory in the format finetune_vlm.py expects.
# The trainer loads {dataset_path}/train.jsonl and {dataset_path}/val.jsonl
# but v3 data is named scoring_train_v3.jsonl etc.
#
# Usage: bash scripts/045_prep_v3_training.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V3_DIR="$REPO_ROOT/training/data/v3"
SRC_DIR="$REPO_ROOT/training/data"

echo "=== Preparing v3 training data ==="

mkdir -p "$V3_DIR"

# Check source files exist
for f in scoring_train_v3.jsonl scoring_val_v3.jsonl; do
    if [ ! -f "$SRC_DIR/$f" ]; then
        echo "ERROR: $SRC_DIR/$f not found"
        echo "Run: python scripts/040_prepare_training_data.py --ver v3"
        exit 1
    fi
done

# Create symlinks that finetune_vlm.py expects
ln -sf "$SRC_DIR/scoring_train_v3.jsonl" "$V3_DIR/train.jsonl"
ln -sf "$SRC_DIR/scoring_val_v3.jsonl" "$V3_DIR/val.jsonl"

# Copy metadata
if [ -f "$SRC_DIR/meta_v3.json" ]; then
    cp "$SRC_DIR/meta_v3.json" "$V3_DIR/meta.json"
fi

# Link coaching data
if [ -f "$SRC_DIR/coaching_train_v3.jsonl" ]; then
    ln -sf "$SRC_DIR/coaching_train_v3.jsonl" "$V3_DIR/coaching_train.jsonl"
fi
if [ -f "$SRC_DIR/coaching_val_v3.jsonl" ]; then
    ln -sf "$SRC_DIR/coaching_val_v3.jsonl" "$V3_DIR/coaching_val.jsonl"
fi

echo "Created $V3_DIR with:"
ls -la "$V3_DIR/"
echo ""

# Verify counts
echo "Training examples: $(wc -l < "$V3_DIR/train.jsonl")"
echo "Validation examples: $(wc -l < "$V3_DIR/val.jsonl")"
if [ -f "$V3_DIR/coaching_train.jsonl" ]; then
    echo "Coaching train: $(wc -l < "$V3_DIR/coaching_train.jsonl")"
    echo "Coaching val: $(wc -l < "$V3_DIR/coaching_val.jsonl")"
fi
echo ""
echo "=== v3 data ready ==="
echo "Run training with:"
echo "  python src/training/finetune_vlm.py --config src/configs/finetune_task5_v3.yaml"
