# FLS-Training Execution Plan

> **Last updated:** 2026-04-08 — Repo hardening & deployment review.

## Overview

AI system that scores FLS Task 5 (intracorporeal suture) videos using a
Teacher-Critique-Student architecture. Claude Sonnet and GPT-4o score
independently, a critique agent produces consensus, and a fine-tuned student
model eventually takes over.

## Structural Principles

1. **GitHub is for code, configs, and manifests.** Checkpoints, extracted
   frames, large logs, and videos go to durable storage (Contabo S8 now;
   object storage later at 500+ videos).
2. **Separate scoring from training.** The API-based scoring pipeline
   (Claude + GPT-4o) must never share a pod with GPU fine-tuning.
3. **Run manifest per training run.** Every launch auto-records commit SHA,
   GPU, dataset version, config hash, and vision mode to
   `memory/training_runs/`.
4. **Config families.** `debug` / `standard` / `full` prevent accidental
   expensive launches (see `src/configs/`).
5. **Persistent volume ≥150 GB** for real vision runs — frames +
   checkpoints exceed 30 GB once accumulated.
6. **Dataset license lineage.** Every external dataset carries a license
   tag; the training data prep script logs which datasets contributed to
   each run.

## Storage Tiers (current)

| Tier | Where | What goes there |
|------|-------|-----------------|
| **Code** | GitHub (`FLS-Training`) | Scripts, configs, prompts, rubrics, manifests, small JSONLs |
| **Durable artifacts** | Contabo S8 (`/srv/fls-training/`) | Checkpoints, extracted frames, raw videos, large logs |
| **Ephemeral compute** | RunPod / Vast.ai | GPU training only — clone → train → push results → shut down |

> Graduate to Backblaze B2 / Cloudflare R2 when corpus exceeds 500 videos.
> The `scripts/067_b2_sync.sh` and `docs/BACKBLAZE_SETUP.md` remain as
> reference for that migration.

---

## Phase 0: Pipeline Setup ✅
- Push framework code ✅
- Build auto-validation script (`scripts/026_auto_validate.py`) ✅
- Build YouTube harvesting scripts (`scripts/011_harvest_youtube.py`,
  `scripts/012_harvest_playlist.py`) ✅
- Harden .gitignore for large artifacts ✅
- Add run-manifest generation to launch script ✅
- Add config families (debug/standard/full) ✅
- Run auto-validation on existing 31 videos
- **Gate**: ≥15 of 31 existing videos pass auto-validation

## Phase 1: YouTube Harvest & Calibration (Active)
- Download FLS Task 5 videos from YouTube playlists and search
- Auto-classify as `fls_task5` / `intracorporeal_general` / `non_relevant`
- Score expert demos first as calibration anchors (must score ≥500)
- Mass-score and auto-validate harvested videos
- Sync scored artifacts to Contabo S8 via `scripts/095_contabo_sync.sh`
- **Gate**: ≥80 ACCEPTED videos, ≥10 trainees, calibration anchors stable

## Phase 2: Data Scaling & External Datasets
- Integrate LASANA, PETRAW, SimSurgSkill, JIGSAWS (see `docs/DATA_SCALING_PLAN.md`)
- Teacher-score ambiguous external clips only (leverage native labels)
- Track dataset license lineage per `data/external/DATASET_LICENSES.yaml`
- **Gate**: ≥500 training examples across ≥3 FLS tasks

## Phase 3: Fine-Tune Student
- Use RunPod / Vast.ai single-GPU path (`deploy/LAUNCH_GUIDE.md`,
  `docs/RUNPOD_RUNBOOK.md`)
- Run manifest auto-generated at launch (commit, GPU, config hash, dataset)
- 80/20 split stratified by trainee, one trainee fully held out
- Checkpoint + logs synced to Contabo S8 after each run
- **Gate**: MAE ≤ tiered threshold (15 scoring / 22 coaching, per
  `memory/baselines/2026-04-08_teacher_mae_baseline.md`)

## Phase 4: Deploy & Monitor (Future)
- Student scores in production, teachers spot-check every 10th video
- Drift detection via `scripts/075_check_drift.py`
- Streamlit MVP on Contabo S8 with DuckDB for demo/review UI
- **Gate**: 50 consecutive videos <10 point deviation

## Deprioritized / Deferred

| Item | Reason | Revisit when |
|------|--------|--------------|
| Object storage bucket (B2/R2) | 31 videos, one trainee — Contabo S8 is sufficient | 500+ videos |
| Separate API/DB/Redis/GPU topology | Phase 4+ architecture; Streamlit + DuckDB is the right MVP | Production deployment |
| Promoted model registry | No trained model worth promoting yet | Phase 3 produces a viable checkpoint |
| Multi-node training | Single A100/Blackwell is sufficient at current dataset scale | 10k+ examples |

## Auto-Validation Rules (`scripts/026_auto_validate.py`)
- ACCEPTED: |claude − gpt4o| ≤ 25 FLS pts AND |time diff| ≤ 15s AND both confidence > 0.40
- QUARANTINED: Diverge 25–50 pts OR one confidence < 0.40
- REJECTED: Diverge > 50 pts OR outside time-anchor band
- Time-anchor: score must fall within [600 − time − 20, 600 − time − 0]

## SAGES Opportunity
- SAGES RFP (April 2024) seeks automated FLS scoring system
- Contact: john@sages.org
- They have hundreds of exam videos, working on paired video+score datasets
