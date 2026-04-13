# FLS-Training

AI-powered surgical skills scoring system for Fundamentals of Laparoscopic Surgery (FLS) training videos.

## Architecture

**Teacher-Critique-Student pipeline:**
- **Teacher A:** Claude Sonnet — scores video frames independently
- **Teacher B:** GPT-4o — scores video frames independently  
- **Critique Agent:** Produces consensus score from both teachers
- **Student:** Qwen2.5-VL-7B (LoRA via Unsloth) — fine-tuned to replicate consensus

## Current Status

**Phase: Data Expansion** — Harvesting and scoring YouTube videos to build training dataset.

- 25 validated training examples (ACCEPTED by dual-teacher consensus)
- 573 harvest targets identified, ~431 being scored on Hetzner server
- LASANA dataset (1,270 videos with human GRS labels) downloaded on Contabo
- Training paused until ≥80 ACCEPTED examples

**See [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) for full progress, infrastructure, and decision log.**

## Quick Start

```bash
git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git
cd FLS-Training
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # Add ANTHROPIC_API_KEY and OPENAI_API_KEY
python scripts/090_status.py
```

## Pipeline

```
Video → Ingest (010) → Score (020/021) → Validate (026) → Consensus (030)
     → Prepare Data (040) → Train SFT (050) → Train DPO (051)
     → Evaluate (060) → Deploy → Drift Check (075) → Retrain
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `010_ingest_video.py` | Extract frames + metadata from video |
| `021_batch_score.py` | Task-routed batch scoring with Claude + GPT-4o |
| `026_auto_validate.py` | Auto-validate dual-teacher scores (ACCEPTED/QUARANTINED/REJECTED) |
| `040_prepare_training_data.py` | Export validated scores to training format |
| `050_runpod_train.py` | SFT training on GPU |
| `060_evaluate_student.py` | Student vs teacher evaluation |

## Servers

| Server | Role | Access |
|--------|------|--------|
| Hetzner (77.42.85.109) | Orchestration, scoring pipeline | `ssh -i ~/.ssh/id_ed25519 root@77.42.85.109` |
| Contabo (207.244.235.10) | LASANA data, backups | `ssh -i ~/.ssh/id_ed25519 root@207.244.235.10` |
| RunPod/Vast.ai | GPU training (on-demand) | Not active until data gate passes |

## Docs

- [PROJECT_STATUS.md](docs/PROJECT_STATUS.md) — Current progress, data inventory, decisions
- [EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md) — Phased execution plan
- [FLS-Training-Setup-Guide.md](docs/FLS-Training-Setup-Guide.md) — Setup + Cursor agent prompts
- [V3_TRAINING_GUIDE.md](docs/V3_TRAINING_GUIDE.md) — Training configuration guide
- [DATA_SCALING_PLAN.md](docs/DATA_SCALING_PLAN.md) — Path to 1,000+ examples
