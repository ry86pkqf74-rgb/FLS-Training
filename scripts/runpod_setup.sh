#!/bin/bash
# === FLS Training — RunPod Pod Setup ===
# Run this FIRST after spinning up a RunPod training pod.
#
# Usage:
#   git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git
#   cd FLS-Training
#   bash scripts/runpod_setup.sh
#

set -e
echo "=== FLS Training — RunPod Setup ==="

DATA_DIR="training/data/v2"

# Install Python deps
echo "Installing dependencies..."
pip install -e '.[training]' --quiet

# Verify GPU
echo ""
echo "GPU check:"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB' if torch.cuda.is_available() else '')"

# Check training data exists
echo ""
if [ -d "$DATA_DIR" ] && [ -n "$(find "$DATA_DIR" -maxdepth 1 -name '*.jsonl' -print -quit 2>/dev/null)" ]; then
    echo "Training data found in $DATA_DIR:"
    find "$DATA_DIR" -maxdepth 1 -name '*.jsonl' -print0 | xargs -0 wc -l
else
    echo "⚠  No training data found. Run locally first:"
    echo "   python scripts/040_prepare_training_data.py --ver v2"
    echo "   bash scripts/045_prep_v2_training.sh"
    echo "   git push"
fi

# Check memory state
echo ""
echo "Memory state:"
echo "  Scores: $(find memory/scores -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ') files"
echo "  Feedback: $(find memory/feedback -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ') files"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  bash scripts/045_prep_v2_training.sh"
echo "  bash deploy/runpod_launch.sh training/data/v2 src/configs/finetune_task5_v2.yaml"
echo ""
echo "After training:"
echo "  python scripts/055_generate_predictions.py --model memory/model_checkpoints/v2_diverse/merged_16bit --data training/data/scoring_val_v2.jsonl --output memory/predictions/v2_on_val"
echo "  python scripts/060_evaluate_student.py --student-scores memory/predictions/v2_on_val"
echo "  git add memory/model_checkpoints/ memory/predictions/ training/runs/"
echo "  git commit -m 'feat: student model trained'"
echo "  git push origin main"
echo "  # Then stop the pod!"
