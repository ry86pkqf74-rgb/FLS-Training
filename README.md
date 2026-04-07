# FLS-Training

AI-powered FLS surgical skills training with **teacher-critique-student** architecture.

## Architecture

**GitHub = persistent brain.** All memory, models (via LFS), and training logs.
**RunPod = ephemeral gym.** Clone → train → push → shut down.

### Dual Training Objectives

| Head | Input | Output |
|------|-------|--------|
| **Scoring** | Video frames | FLS score, phases, penalties |
| **Coaching** | Frames + history | Feedback with progression-aware drills |

## Quick Start

```bash
pip install -e .
cp .env.example .env  # Add API keys
python scripts/090_status.py
```

## Workflow

1. `python scripts/010_ingest_video.py --video X.mov --task 5`
2. `python scripts/020_score_frontier.py --video-id X --video X.mov`
3. `python scripts/080_generate_feedback_report.py --video-id X`
4. `python scripts/040_prepare_training_data.py --ver v1` → git push
5. On RunPod: `bash scripts/runpod_setup.sh && python scripts/050_runpod_train.py`
6. `python scripts/060_evaluate_student.py` → promote if >85% agreement
7. `python scripts/075_check_drift.py` → retrain when triggered

## Cost

RunPod A100 ~$1.19/hr × 1-2hr = **~$2-3 per training cycle**.
Student inference after: ~$0.001/video.
