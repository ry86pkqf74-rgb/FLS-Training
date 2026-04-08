# V3 Training Guide

## Dataset Summary

| Metric | V1 | V2 | V3 |
|--------|----|----|-----|
| Total videos | 31 | 66 | 78 |
| Trainees | 1 | ~15 | ~20+ |
| Sources | Personal | Personal + YT + Playlist | + Batch 3 (tutorials) |
| Scoring train | ~25 | 56 | 66 |
| Scoring val | ~6 | 10 | 12 |
| Coaching train | 0 | 27 | 36 |
| Coaching val | 0 | 7 | 10 |

## What's new in V3

- 10 additional videos from batch 3 harvest (expanded criteria)
- Includes tutorial/demonstration videos — these provide coaching signal
- Expert narration in tutorials maps to what the coaching head should learn
- Virtual FLS simulations included (correct procedure, different visual fidelity)

## Steps on RunPod Pod

```bash
# 1. Sync repo
cd /workspace/FLS-Training
git stash
git pull origin main

# 2. Prep v3 data directory
bash scripts/045_prep_v3_training.sh

# 3. Train
python src/training/finetune_vlm.py --config src/configs/finetune_task5_v3.yaml

# 4. Generate predictions on val set
python scripts/055_generate_predictions.py \
    --model memory/model_checkpoints/v3_diverse/merged_16bit \
    --data training/data/scoring_val_v3.jsonl \
    --output memory/predictions/v3_on_val

# 5. Evaluate
python scripts/060_evaluate_student.py --student-scores memory/predictions/v3_on_val

# 6. Push results
git add memory/model_checkpoints/v3_diverse/eval_results.json \
        memory/model_checkpoints/v3_diverse/training_config.yaml \
        memory/predictions/v3_on_val/
git commit -m "feat: v3 training results — 78 diverse videos incl tutorials"
git push origin main
```

## Promotion Criteria

Student is ready for production when:
- Avg FLS score error ≤ 12 points
- Time agreement ≥ 85% (within 10s)
- FLS agreement ≥ 85% (within 20 points)

## Overfitting Watch

Val loss should decrease. If it starts increasing while train loss drops → stop early.
Best checkpoint auto-saved every 50 steps. Use lowest val loss checkpoint, not final.

With 66 training examples at batch_size=4 and gradient_accumulation=2 (effective 8):
- ~8 steps per epoch
- ~40 steps for 5 epochs
- First eval at step 50 may only happen at end of epoch 6+
- Consider reducing eval_steps to 25 if you want mid-training feedback
