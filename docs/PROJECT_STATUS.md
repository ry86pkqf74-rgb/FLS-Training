# FLS-Training: Project Status & Progress

**Last Updated:** 2026-04-13
**GitHub:** github.com/ry86pkqf74-rgb/FLS-Training

---

## Executive Summary

Building an AI-powered FLS (Fundamentals of Laparoscopic Surgery) scoring system using a Teacher-Critique-Student architecture. Two frontier VLMs (Claude Sonnet + GPT-4o) independently score surgical training videos, a critique agent produces consensus scores, and a fine-tuned student model (Qwen2.5-VL-7B) will replicate teacher consensus at lower cost.

**Current phase:** Data expansion (harvest + score). Training is paused pending sufficient validated examples.

---

## Data Inventory (as of 2026-04-13)

### Validated Scores
| Status | Count | Notes |
|--------|-------|-------|
| ACCEPTED (training-ready) | 25 | Pass all thresholds: score Δ ≤25, time Δ ≤15s, both conf >0.40 |
| Near-misses | 7 | Would pass with slightly relaxed thresholds |
| QUARANTINED | 179 | Includes 138 zero-score task-misclassified videos |
| REJECTED | 11 | Missing teacher scores or extreme divergence |

### Breakdown of Quarantined
- **138 zero-score videos:** Scored with Task 5 prompt but contain Tasks 1-4 content. Need re-scoring with correct task-specific prompts. 39 of these are flagged as wrong-task or demo/instructional content.
- **34 real Task 5 videos:** Fail on score/time delta between teachers
- **7 near-misses:** Barely fail on confidence (0.38-0.40) or time delta (17-24s)

### Harvest Targets
- **harvest_targets.csv:** 573 YouTube URLs across all 5 tasks
- **Already scored:** ~142 (from prior batch runs)
- **Not yet scored:** ~431 (being downloaded and scored on Hetzner — see Active Pipelines below)
- **Task distribution (unscored):** 309 unclassified, 37 task5, 29 task3, 23 task4, 19 task1, 14 task2

### LASANA Dataset (on Contabo)
- **1,270 video trials** with human GRS labels (z-scored)
- **275,715 extracted frames** at /data/fls/lasana_processed/frames/
- **4 tasks:** BalloonResection, CircleCutting, PegTransfer, SutureAndKnot
- Raw video zips downloaded and extracted
- NOT yet integrated into training pipeline (planned for pre-training stage)

### Training Runs
| Run | Date | Status | Notes |
|-----|------|--------|-------|
| 20260407_1619_unsloth | 2026-04-07 | PAUSED (overfit) | 25 examples, too many epochs, eval loss stagnant |
| lasana_pretrain | 2026-04-09 | Complete | Adapter + eval results in repo |

---

## Infrastructure

### Active Servers

| Server | IP | Specs | Role | Status |
|--------|-----|-------|------|--------|
| **Hetzner (FLS)** | 77.42.85.109 | 16C/30GB, 601GB disk | Harvest+Score pipeline, orchestration | ACTIVE — running harvest+score in tmux `fls` |
| **Contabo** | 207.244.235.10 | 12C Ryzen 9 | LASANA dataset storage, backups | IDLE — holds 280GB FLS data |
| RunPod | (not active) | GPU on-demand | SFT/DPO training | NOT DEPLOYED — waiting for ≥80 ACCEPTED |
| Vast.ai | (not active) | GPU on-demand | Alt training compute | NOT DEPLOYED |

### Hetzner Server Notes (77.42.85.109)
- SSH: `ssh -i ~/.ssh/id_ed25519 root@77.42.85.109`
- ResearchFlow services STOPPED and DISABLED (2026-04-13): ollama, researchflow-rotation, crons
- FLS pipeline running in tmux session `fls`
- Monitor: `ssh root@77.42.85.109 'tmux attach -t fls'`

### Contabo Server Notes (207.244.235.10)
- SSH: `ssh -i ~/.ssh/id_ed25519 root@207.244.235.10`
- FLS data at `/data/fls/` (280GB)
- LASANA layout at `/data/fls/lasana_layout/` (1,270 trial directories)
- Processed frames at `/data/fls/lasana_processed/frames/`
- Score backups at `/data/fls/backups/`

### Dead Infrastructure (DO NOT USE)
- S1-S7 (Hetzner ResearchFlow fleet): All timing out as of 2026-04-13. IPs in researchflow-servers skill are stale.
- V1 (Vast.ai H200): Access was via S1 jump host which is unreachable.

---

## Active Pipelines

### Harvest+Score (Hetzner, running)
- **Started:** 2026-04-13 18:15 UTC
- **What:** Downloads ~431 unscored videos from harvest_targets.csv, extracts frames, scores with Claude Sonnet + GPT-4o, pushes results to GitHub
- **Script:** `/opt/fls-training/run_harvest_score.sh` in tmux session `fls`
- **Expected duration:** 4-8 hours
- **Expected cost:** ~$30-50 in API calls
- **Auto-pushes** results to GitHub on completion

---

## Gate Criteria

### Phase 1 Gate (Must pass before training)
- [ ] ≥80 ACCEPTED videos total
- [ ] ≥10 distinct trainees represented
- [ ] ≥3 skill levels: novice (<400), intermediate (400-480), advanced (480+)
- [ ] Teacher agreement rate ≥65%
- [ ] Expert demos score ≥500 FLS

### Training Gate (evaluation thresholds)
- [ ] Student MAE ≤12 FLS points on held-out trainee
- [ ] Phase detection IoU ≥0.70
- [ ] Penalty detection F1 ≥0.60

---

## Process Discipline (ChatGPT Review Items)

### Implemented ✅
1. Config discipline: debug/standard/full configs in `src/configs/`
2. Separated preprocessing, training, evaluation scripts (010→090 numbering)
3. Auto-validation script with defined thresholds (`scripts/026_auto_validate.py`)
4. Recursive score scanning (fixed 2026-04-13 to cover all subdirectories)
5. Task-routing in batch scorer (`scripts/021_batch_score.py`)
6. Content-mismatch detection (wrong task + demo flagging)

### Not Yet Implemented ❌
1. `run_manifest.json` gate — locks dataset/config SHA before training
2. Promoted model registry — `memory/model_registry/` with lineage
3. Promotion gate script — automated eval threshold enforcement
4. Validation gate script — dataset readiness check
5. Automated data validation before training starts
6. Standardized object storage layout for large artifacts

### Planned Next
- Push discipline layer scripts (validation gate, manifest gate, promotion gate)
- Execute after harvest+score pipeline completes and data gate is assessed

---

## Key Files

| File | Purpose |
|------|---------|
| `data/harvest_targets.csv` | 573 YouTube URLs with task labels |
| `scripts/026_auto_validate.py` | Dual-teacher validation (recursive scan) |
| `scripts/021_batch_score.py` | Task-routed batch scoring with audit |
| `memory/validation_results.jsonl` | Latest validation output |
| `memory/validation_reason_summary.json` | Failure category breakdown |
| `prompts/v002_universal_scoring_system.md` | Multi-task scoring prompt |
| `rubrics/task{1-5}_*.yaml` | Task-specific scoring rubrics |
| `src/configs/finetune_task5_*.yaml` | Training configs (debug/standard/full) |
| `data/training/dpo_v1/` | 44 DPO pairs (37 train / 7 val) |
| `docs/FLS-Training-Setup-Guide.md` | Original setup + agent prompts |
| `docs/EXECUTION_PLAN.md` | Full phased execution plan |

---

## Cost Tracking

| Item | Spent | Budget |
|------|-------|--------|
| API scoring (~170 videos, Claude+GPT-4o) | ~$40 | - |
| Hetzner server (free tier) | $0 | $0/mo |
| Contabo (Ryzen 9) | ~$15/mo | ongoing |
| RunPod training (v1 run, overfit) | ~$10 | - |
| Current harvest+score run (est.) | ~$30-50 | - |
| **Total spent** | **~$65-115** | |
| **Remaining to working model** | **~$50-100** | GPU training + eval |

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-07 | Pause v1 training | 25 examples, severe overfitting, eval loss stagnant |
| 2026-04-08 | Merge PR #1 | Coworker's harvest targets, DPO pipeline, rubrics for Tasks 1-4 |
| 2026-04-08 | Adopt v002 universal prompts | Covers all 5 tasks, replaces task5-only v001 |
| 2026-04-09 | Pre-train on LASANA | 1,270 videos with real GRS labels → diversity before fine-tuning |
| 2026-04-13 | Fix validator (recursive scan) | Was only finding 45/215 videos due to glob pattern |
| 2026-04-13 | Kill ResearchFlow on Hetzner | Ollama eating 15 cores, server at load 15.4 |
| 2026-04-13 | Launch harvest+score on Hetzner | 431 unscored targets, runs unattended in tmux |
| 2026-04-13 | Don't spin up GPU yet | Need ≥80 ACCEPTED before training is worthwhile |
