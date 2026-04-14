# FLS-Training Readiness Assessment — 2026-04-14

Evaluation conducted while Hetzner batch 2 is completing (~378/432 as of writing). Scope: scoring system, prompts, video coverage, training sequence, and readiness for iterative teacher/critique training.

## TL;DR

The scoring pipeline and v002 prompts are solid. **The round 1 adapter is fundamentally broken and should not be warm-started from.** There are two blocking issues:

1. **LASANA SFT targets are format-incompatible with the v002 FLS schema.** The model learned to echo LASANA z-score blobs instead of producing FLS scores. Eval shows the adapter emits only 3 unique `estimated_fls_score` values across 205 test examples (144.6 ×101, 137.0 ×54, 278.1 ×50). This is not "learned priors" — it's mode collapse by memorization.
2. **The base model is text-only `Qwen2.5-7B-Instruct`.** Training ingests image *paths* as text tokens, so no vision signal is actually used. All current "scoring" is text-only pattern-matching on the task prompt.

Round 2 plans must change. Recommendations below.

## 1. Scoring system — OK

- **Pipeline:** dual-teacher (Claude Sonnet 4 + Claude Haiku) scoring each video into two JSONs, then consensus merging via `v002_consensus_system`. Validated by `026_auto_validate.py`, post-hoc patched by `029_normalize_scores.py` (good defensive layer).
- **Batch 2 health:** today's run is producing clean v002-schema JSON with `video_classification`, `task_id`, phase timings, penalties, confidence. Normalizer covers the earlier missing-field bug.
- **Throughput:** ~1.7 videos/min per teacher pair. At 432 videos, one batch is ~4.2 hours. Practical for nightly runs.
- **Scores-per-day inventory:** 04-07: 65, 04-08: 415, 04-13: 1,202 (and climbing). Plus ~200 orphan files written to the root of `memory/scores/` (not date-partitioned) — these leak into SFT prep today. Worth moving to `memory/scores/_legacy/` to avoid contamination.

## 2. Prompts — v002 family is strong, registry is clean

`prompts/prompt_registry.yaml` cleanly separates v001 (task5-only, deprecated) from v002 (universal). The four v002 prompts are well-designed:
- `v002_universal_scoring_system.md` — rubric for all 5 tasks, explicit `video_classification` gate, output schema enforced, good/bad examples included.
- `v002_universal_coach_system.md` — skill-level-aware drill selection.
- `v002_consensus_system.md` — rubric-first merger (good design: doesn't just average).
- `v002_critique_system.md` — phase-by-phase error taxonomy (this is the teacher-of-teachers signal we'd use for iterative improvement).

**One nit:** the registry calls task 3 `task3_ligating_loop` but the harvest CSV / score files call it `task3_endoloop`. The prep script has alias logic for task5 but not task3. Low priority — doesn't break anything today because the harvest CSV wins.

## 3. Video coverage — the big gap

`data/harvest_targets.csv` has 574 rows, task breakdown:

| Task | Count | Notes |
|------|-------|-------|
| unclassified | **309** | 54% of harvest — currently unusable for SFT (030 drops these) |
| task1_peg_transfer | 60 | |
| task5_intracorporeal_suturing | 57 | |
| task2_pattern_cut | 54 | |
| task3_endoloop | 51 | |
| task4_extracorporeal_knot | 42 | Lowest coverage |

So of 432 batch-2 videos, roughly half will be dropped by the new SFT prep (unclassified). Expected training yield after batch 2: **~180–240 usable examples** across 5 tasks, plus the 55 already-prepped. That gets us to ~100 train / 15 val — thin. We need either:
- A classification pass to backfill `task` for the 309 unclassified URLs (cheap: Claude Haiku or GPT-4o with 1–3 thumbnails), OR
- A larger harvest (aim for 150+ per task; task 4 needs most work).

Also note: the prompt already emits `video_classification: "performance" | "expert_demo" | "instructional" | "unusable"`. `instructional` and `unusable` videos shouldn't be in SFT at all — the prep script should filter on `video_classification == "performance"`.

## 4. Training sequence — broken (this is the critical finding)

### What happened in round 1

- `031_train_lora_lasana.py` combined `lasana_{train,val}.jsonl` + `yt_{train,val}.jsonl` and trained Qwen2.5-7B-Instruct (text-only) with 4-bit LoRA.
- **LASANA training rows look like:**
  ```
  messages: [system=<v002 scoring prompt>, user=<image paths + text>, assistant=<entire LASANA label blob>]
  ```
  The "target" the model sees is the full LASANA metadata dict — `grs_z`, `grs_rescaled: 1.0`, `bimanual_dexterity`, `tissue_handling`, etc. — NOT a v002-schema FLS score.
- The training script's `to_conversation(ex)` falls back to `ex.get("target", ex)` — for LASANA rows (which have no `target` key) it JSON-dumps the whole row including the nested `messages` array and metadata. So the assistant target contains its own copy of the `messages` field. This is a self-referential data bug.
- Model learned to: (a) predict the right `task_id` (5-way classification, easy), (b) output a memorized LASANA-style blob per task.

### Evidence of mode collapse

From `memory/model_checkpoints/lasana_pretrain/eval_results.json`:

- 205 test examples, 188 unique expected FLS scores
- **3 unique predicted FLS values**: 144.6 (101×), 137.0 (54×), 278.1 (50×)
- Raw outputs literally contain `"ground_truth": { "grs_zscore": -0.15486, "sub_scores": {...} }` — the model is reciting LASANA ground truth fields. This is memorization, not generalization.
- MAE 57.56 is misleading — it's the expected value of |E[task2_mean] − expected| for whatever single value the model picked.

### Why task1_peg_transfer / task5 eval looked plausible

Per-task MAE in the handoff (task1=40.9, task5=96.3) is explained by the model picking one constant per task. Task5 has max_score=600 so picking any value in [200,500] gets MAE 100ish. This isn't a signal the adapter learned anything about surgical technique.

### Text-only base model

`031_train_lora_lasana.py` loads `Qwen/Qwen2.5-7B-Instruct` — a **text-only** model. The user message contains file paths as text (`/data/fls/lasana_processed/frames/.../frame_0001.jpg`) — those are not images, they're tokens. So we cannot currently score video content. Any meaningful FLS scorer needs a vision model (Qwen2.5-VL-7B, LLaVA, or similar).

## 5. Teacher/critique readiness

**What's in place:**
- Dual-teacher scoring (Sonnet + Haiku) producing consensus labels.
- `v002_critique_system` prompt that does phase-by-phase error taxonomy — this is the right substrate for a critique loop.
- DPO data already prepared at `data/training/dpo_v1/` (127 train / 22 val).
- Gold set at `data/training/gold/` (33 examples hardened).

**What's missing for an iterative teacher/critique loop:**

1. A **vision-capable student**. Without real frames the student can't do the job.
2. A **scoring-diff metric** that drives the critique teacher. Right now scores are consensus-averaged, not triaged by disagreement.
3. A **loop driver** — script that:
   a. Runs student on a pool of videos
   b. Asks Sonnet (teacher) to critique student outputs against its own score
   c. Emits (chosen, rejected) pairs for DPO or a regression target for SFT
   d. Feeds the new batch back to training
4. A **regression/rejection harness** so a bad round 2 adapter doesn't silently replace a working round 1. Held-out YouTube test split is currently missing (all prepped data went to train+val; no test).

## 6. Recommendations — revised round 2 plan

I'd replace the round 2 plan in the handoff with this:

### Short-term (before starting round 2 training)

1. **Drop LASANA from round 2 SFT entirely.** It's format-incompatible with v002 and is the source of mode collapse. LASANA labels are z-score-derived (1–5) on a different rubric; can't safely be blended.
2. **Do not warm-start from the round 1 adapter.** Train fresh from base model. Round 1's weights encode the memorized LASANA blob pattern, which is a regression for our target schema.
3. **Switch base model to a VL variant** — `Qwen/Qwen2.5-VL-7B-Instruct` — so training actually uses frames. Keep LoRA r=16 nf4.
4. **Build YouTube SFT v2 with proper splits.** After batch 2 lands: re-run 030 (expect ~100 train / 15 val usable), then carve off a **held-out test set** of ~20 consensus-scored videos never seen in training.
5. **Fix the prep filter** to require `video_classification == "performance"` AND non-null `task_id` AND `consensus_conf >= 0.5`. Drop instructional / unusable / low-confidence.
6. **Fix `task3_ligating_loop` vs `task3_endoloop` alias** in the registry + prep.

### Medium-term (iterative teacher/critique loop)

Target cadence: one round per night on the RunPod H100 ($2.69/hr × ~4hr = ~$11/night).

Proposed script set (to be written):

- `040_build_sft_v2.py` — v002-schema only, vision-capable, stratified splits
- `041_train_qwen_vl.py` — Qwen2.5-VL LoRA training, proper multimodal formatting
- `042_eval_vl_adapter.py` — FLS MAE + task accuracy + classification rate + score-diversity histogram (to catch collapse early)
- `043_student_score_batch.py` — have the student score a new batch of videos
- `044_teacher_critique.py` — Sonnet-as-critic scoring (student_score, consensus_score) via v002_critique_system → per-phase error taxonomy
- `045_build_dpo_pairs.py` — where critique marks student divergent, use consensus as chosen, student as rejected
- `046_train_dpo.py` — DPO round over round N+1 (already have starter at `051_train_dpo.py`)
- `047_regression_gate.py` — compare adapter N+1 vs N on held-out test set; reject push if MAE regresses > 10 points OR unique-prediction diversity drops.

### Near-term harvest work

- Classify the 309 unclassified harvest_targets. A cheap one-shot Haiku classifier over video titles + description + 1 thumbnail should get >90% on the balanced classes.
- Target 150 per task. Current task 4 (42 videos) is the bottleneck — an extra harvest round biased toward task 4 is probably worth one session.

## 7. What round 2 should actually look like

```
Base:      Qwen2.5-VL-7B-Instruct
Adapter:   Fresh LoRA r=16, nf4, sdpa attention
Data:      YouTube SFT v2 only (~100 train / 15 val / 20 test)
           Filtered: video_classification=performance, consensus_conf >= 0.5
           Stratified by task_id
Epochs:    3 (fresh)  OR  2 (if we later re-introduce LASANA repaired to v002)
LR:        2e-4 cosine, warmup 0.1
Eval:      MAE + task_accuracy + unique_pred_ratio + JSON_validity
Gate:      unique_pred_ratio > 0.5 on held-out test — or we've collapsed
```

## 8. What to ship tonight when Hetzner finishes

Given the time already invested in the existing round-2 scripts, the minimally-changed path is:

A. Let Hetzner batch 2 finish and push scores. ✓ (pipeline OK)
B. Run `030_prep_sft_data.py` to regenerate YouTube SFT v1. ✓
C. **Pause.** Do NOT fire the round 2 script currently on the `round2-scripts` branch — it warm-starts from the broken adapter and stays text-only.
D. Instead, write `041_train_qwen_vl.py` (fresh VL, YouTube-only). This is a 1–2 hour script change I can do next.
E. Launch that on RunPod, eval with a diversity gate, then decide whether to iterate into a critique loop.

Estimated wall time: 1hr scripts + 30min SCP/setup + ~3hr training + 1hr eval = done by 08:00 UTC.

## Appendix — artifacts reviewed

- `prompts/prompt_registry.yaml` and all v002 prompts
- `scripts/020_score_frontier.py`, `026_auto_validate.py`, `029_normalize_scores.py`, `030_prep_sft_data.py`, `031_train_lora_lasana.py`, `032_eval_adapter.py`
- `data/harvest_targets.csv` (574 rows, task distribution above)
- `data/training/2026-04-09_lasana_v1/{train,val,test}.jsonl` (format analysis above)
- `data/training/youtube_sft_v1/{train,val}.jsonl` (format is v002-compatible)
- `memory/model_checkpoints/lasana_pretrain/{run_manifest,eval_results}.json`
- Sample of 2026-04-13 consensus score JSON (schema is clean)
