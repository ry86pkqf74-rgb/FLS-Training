# FLS-Training

AI-powered FLS surgical skills training with **teacher-critique-student** architecture.

## Architecture

**GitHub = code, configs, manifests.** No checkpoints, frames, or videos.
**Contabo S8 = durable artifact storage.** Checkpoints, frames, raw videos, large logs.
**RunPod/Vast.ai = ephemeral GPU.** Clone → train → sync → shut down.

See `docs/ARCHITECTURE.md` for the full system diagram and `docs/EXECUTION_PLAN.md`
for the phased roadmap.

### Dual Training Objectives

| Head | Input | Output |
|------|-------|--------|
| **Scoring** | Video frames | FLS score, phases, penalties |
| **Coaching** | Frames + history | Feedback with progression-aware drills |

### Config Families

| Family | Use case | GPU | Time |
|--------|----------|-----|------|
| `config_debug.yaml` | Smoke-test, CI, pipeline validation | Any (24 GB+) | <5 min |
| `config_standard.yaml` | Routine training (<200 examples) | A100 / Blackwell | 1–2 hr |
| `config_full.yaml` | Large corpus (500+ examples, vision) | A100 / H100 | 4–10 hr |

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
5. On RunPod: `bash deploy/runpod_launch.sh data/training/LATEST src/configs/config_standard.yaml`
6. `bash scripts/095_contabo_sync.sh push-checkpoints` (save to durable storage)
7. `python scripts/060_evaluate_student_v2.py <checkpoint_a> [checkpoint_b ...]`
8. `python scripts/075_check_drift.py` → retrain when triggered

## Artifact Sync (Contabo S8)

```bash
bash scripts/095_contabo_sync.sh init              # first-time remote setup
bash scripts/095_contabo_sync.sh push-checkpoints   # after training
bash scripts/095_contabo_sync.sh push-frames         # after frame extraction
bash scripts/095_contabo_sync.sh status              # check disk usage
```

## Run Manifests

Every training launch auto-generates a manifest in `memory/training_runs/`
recording commit SHA, GPU, dataset version, config hash, and vision mode.

## Dataset Licenses

See `data/external/DATASET_LICENSES.yaml` for license status of each
external dataset. Verify before any commercial deployment.

## Cost

RunPod A100 ~$1.19/hr × 1-2hr = **~$2-3 per training cycle**.
Student inference after: ~$0.001/video.

## Docs

- `deploy/LAUNCH_GUIDE.md` — concise launch path
- `docs/RUNPOD_RUNBOOK.md` — proven server setup, resume flow, shutdown
- `docs/EXECUTION_PLAN.md` — phased roadmap with gates
- `docs/DATA_SCALING_PLAN.md` — corpus growth strategy
