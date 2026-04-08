# FLS-Training

AI-powered FLS surgical skills training with **teacher-critique-student** architecture.

Current RunPod deployment docs:
- `deploy/LAUNCH_GUIDE.md` for the concise launch path
- `docs/RUNPOD_RUNBOOK.md` for the proven server setup, continuous-resume flow, verification, and shutdown checklist

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
4. `python scripts/040_prepare_training_data.py --ver <version>` → git push
5. On RunPod: follow `deploy/LAUNCH_GUIDE.md` or `docs/RUNPOD_RUNBOOK.md`
6. `python scripts/060_evaluate_student.py` → promote if >85% agreement
7. `python scripts/075_check_drift.py` → retrain when triggered

## Cost

RunPod A100 ~$1.19/hr × 1-2hr = **~$2-3 per training cycle**.
Student inference after: ~$0.001/video.
