# V2 Training Guide

## What changed from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Videos | 31 (single trainee) | 66 (32 original + 20 YouTube + 14 playlist) |
| Trainees | 1 | ~15+ distinct |
| Epochs | 1075 (massively overfit) | 5 |
| Learning rate | 1e-4 | 5e-5 |
| Base model | Qwen2.5-VL-7B | Qwen2.5-VL-7B (fresh, NOT from v1 ckpt) |
| Eval | eval_loss only | eval every 50 steps + held-out student eval |
| Training data | scoring only | scoring (56) + coaching (27) |

## Steps on RunPod Pod

```bash
# 1. Sync repo
cd /workspace/FLS-Training
git stash          # save v1 artifacts
git pull origin main

# 2. Prep v2 data directory
bash scripts/045_prep_v2_training.sh

# 3. Train (monitor val loss — stop if it starts increasing)
python src/training/finetune_vlm.py --config src/configs/finetune_task5_v2.yaml

# 4. Generate predictions on val set
python scripts/055_generate_predictions.py \
    --model memory/model_checkpoints/v2_diverse/merged_16bit \
    --data training/data/scoring_val_v2.jsonl \
    --output memory/predictions/v2_on_val

# 5. Evaluate
python scripts/060_evaluate_student.py --student-scores memory/predictions/v2_on_val

# 6. Push results
git add memory/model_checkpoints/v2_diverse/eval_results.json \
        memory/model_checkpoints/v2_diverse/training_config.yaml \
        memory/predictions/v2_on_val/ \
        src/configs/finetune_task5_v2.yaml
git commit -m "feat: v2 training results — 66 diverse videos"
git push origin main
```

## Promotion Criteria

The student is ready for production when:
- Avg FLS score error ≤ 12 points (vs teacher consensus)
- Time agreement ≥ 85% (within 10s)
- FLS agreement ≥ 85% (within 20 points)

## If Overfitting

Watch for: val loss increases while train loss keeps dropping.
- Stop training early (ctrl-c or set max_steps)
- The best checkpoint is auto-saved every 50 steps
- Use the checkpoint with lowest val loss, not the final one

## If Underfitting

If val loss plateaus high and metrics are poor:
- The bottleneck is data diversity, NOT compute
- Harvest more YouTube videos with scripts/011_harvest_youtube.py --search
- Score them, rebuild v3 dataset, retrain
