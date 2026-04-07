# FLS-Training Execution Plan

## Overview

AI system that scores FLS Task 5 (intracorporeal suture) videos using a Teacher-Critique-Student architecture. Claude Sonnet and GPT-4o score independently, a critique agent produces consensus, and a fine-tuned student model eventually takes over.

## Phase 0: Pipeline Setup (Week 1)
- Push framework code ✅ (already done)
- Build auto-validation script (scripts/026_auto_validate.py) ✅
- Build YouTube harvesting scripts (scripts/011_harvest_youtube.py, scripts/012_harvest_playlist.py) ✅
- Run auto-validation on existing 31 videos
- **Gate**: ≥15 of 31 existing videos pass auto-validation

## Phase 1: YouTube Harvest & Calibration (Weeks 2-3)
- Download FLS Task 5 videos from YouTube playlists and search
- Auto-classify as fls_task5 / intracorporeal_general / non_relevant
- Score expert demos first as calibration anchors (must score ≥500)
- Mass-score and auto-validate harvested videos
- **Gate**: ≥80 ACCEPTED videos, ≥10 trainees, calibration anchors stable

## Phase 2: Instrument Tracking (Weeks 3-4)
- Deploy YOLO instrument detector on S6 (32C/256GB Hetzner)
- Add objective metrics: path length, idle time, phase transitions
- Re-score subset with augmented input
- **Gate**: 90% frame detection, reduced teacher disagreement

## Phase 3: Fine-Tune Student (Weeks 4-5)
- Use V1 (Vast.ai H200, ~$2.07/hr) for training runs
- 80/20 split stratified by trainee, one trainee fully held out
- **Gate**: MAE ≤12 FLS points on held-out trainee

## Phase 4: Deploy & Monitor (Ongoing)
- Student scores in production, teachers spot-check every 10th video
- Drift detection via scripts/075_check_drift.py
- **Gate**: 50 consecutive videos <10 point deviation

## Auto-Validation Rules (scripts/026_auto_validate.py)
- ACCEPTED: |claude - gpt4o| ≤ 25 FLS pts AND |time diff| ≤ 15s AND both confidence > 0.40
- QUARANTINED: Diverge 25-50 pts OR one confidence < 0.40
- REJECTED: Diverge > 50 pts OR outside time-anchor band
- Time-anchor: score must fall within [600 - time - 20, 600 - time - 0]

## SAGES Opportunity
- SAGES RFP (April 2024) seeks automated FLS scoring system
- Contact: john@sages.org
- They have hundreds of exam videos, working on paired video+score datasets
