#!/bin/bash
# === FLS Training — RunPod Pod Setup ===
# Run this FIRST after spinning up a RunPod A100 pod.
#
# Usage:
#   git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git
#   cd FLS-Training
#   bash scripts/runpod_setup.sh
#

set -e
echo "=== FLS Training — RunPod Setup ==="

# Install Python deps
echo "Installing dependencies..."
pip install -e '.[training]' --quiet

# Verify GPU
echo ""
echo "GPU check:"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB' if torch.cuda.is_available() else '')"

# Check training data exists
echo ""
if [ -d "training/data" ] && [ -n "$(ls training/data/*.jsonl 2>/dev/null)" ]; then
    echo "Training data found:"
    wc -l training/data/*.jsonl
else
    echo "⚠  No training data found. Run locally first:"
    echo "   python scripts/040_prepare_training_data.py --ver v1"
    echo "   git push"
fi

# Check memory state
echo ""
echo "Memory state:"
echo "  Scores: $(ls memory/scores/*.json 2>/dev/null | wc -l) files"
echo "  Feedback: $(ls memory/feedback/*.json 2>/dev/null | wc -l) files"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  python scripts/050_runpod_train.py --ver v1"
echo ""
echo "After training:"
echo "  git add models/ training/runs/"
echo "  git commit -m 'feat: student model trained'"
echo "  git push origin main"
echo "  # Then stop the pod!"
