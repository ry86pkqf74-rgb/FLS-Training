# FLS-Training

AI-powered scoring and feedback for FLS (Fundamentals of Laparoscopic Surgery) training videos.

## Architecture

```
Video → Frame Extraction → Teacher A (Claude) + Teacher B (GPT-4o)
    → Critique Agent (consensus) → Memory Store → Feedback Report
    → [accumulate] → Fine-tune Student Model (Qwen2.5-VL-7B) on RunPod
    → Student replaces teachers → Drift detection → Retrain loop
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full system design.

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # Fill in ANTHROPIC_API_KEY and OPENAI_API_KEY

# 1. Ingest a video
python scripts/010_ingest_video.py --video /path/to/video.mov --task 5

# 2. Score with both frontier models + critique
python scripts/020_score_frontier.py --video-id <ID> --video /path/to/video.mov

# 3. Submit corrections (optional)
python scripts/025_submit_correction.py --video-id <ID> --score-id <SCORE_ID> \
    --corrected-fields '{"completion_time_seconds": 147}' --corrector expert

# 4. Generate feedback report
python scripts/080_generate_feedback_report.py --video-id <ID>

# 5. Check if retraining is needed
python scripts/075_check_drift.py

# 6. Prepare training data (after 50+ videos)
python scripts/040_prepare_training_data.py --version 1

# 7. Fine-tune on RunPod (GPU required)
python -m src.training.finetune_vlm --config configs/finetune_task5_v1.yaml
```

## Supported FLS Tasks

| # | Task | Status |
|---|------|--------|
| 5 | Intracorporeal Suture & Knot Tying | ✅ Active |
| 4 | Extracorporeal Suture | 🔜 Planned |
| 3 | Clip Apply (Ligating Loop) | 🔜 Planned |
| 2 | Pattern Cut | 🔜 Planned |
| 1 | Peg Transfer | 🔜 Planned |

## Data & Memory

All scoring artifacts are stored in `memory/` with timestamped paths:
```
memory/scores/YYYY-MM-DD/{video_id}_{model}_{timestamp}.json
memory/comparisons/YYYY-MM-DD/{video_id}_critique_{timestamp}.json
memory/corrections/YYYY-MM-DD/{video_id}_correction_{timestamp}.json
memory/learning_ledger.jsonl   ← append-only event log
```

DuckDB at `data/fls_training.duckdb` provides the queryable index.

## Cost Estimates

| Phase | Compute | Est. Cost |
|-------|---------|-----------|
| Score 100 videos (Claude + GPT-4o) | API calls | ~$50-80 |
| First fine-tune (RunPod A100 80GB) | ~60 GPU-hrs | ~$70 |
| Monthly retrain | ~20 GPU-hrs | ~$24/mo |
