# Data Scaling Plan

Goal: grow the FLS-Training supervised set from today's **32 examples
(Task 5 only)** to **1,000+ examples across all 5 FLS tasks** over the next
two–three weeks, and fix the overfitting we saw in the first training run.

Companion documents:
- `data/DATA_INVENTORY.md` — current counts and gap analysis.
- `data/external/DATASET_CATALOG.md` — external datasets considered.
- `data/external/LASANA_README.md` — LASANA download instructions.
- `data/harvest_targets.csv` — 248 YouTube candidates (deduped vs
  `harvest_log.jsonl`, task-classified, with search queries in `notes`).

## Guiding principles

1. **Ship the cheapest uplift first.** Re-running prep with a lower confidence
   filter costs nothing and probably doubles the training set.
2. **Break out of Task-5-only.** The model overfits because it has seen one
   task. The very next training run should include at least some Task 1 and
   Task 2 examples to force the vision encoder to generalise.
3. **Prefer datasets with ground-truth skill labels** (LASANA's GRS, SPD-FLS1's
   longitudinal session index) over datasets that require teacher scoring.
   Teacher scoring is our most expensive step.
4. **Track pair yield empirically.** After each phase, recompute the
   pair-per-clip yield and update this file before committing to the next
   phase's budget.

## Phase 1 — Immediate (today, ~1 hour, $0 external cost)

Target: **80–100 training examples, all Task 5.**

| Step | Action |
|---|---|
| 1.1 | Lower `min_confidence` in the dataset prep script from 0.4 → **0.25**. Rerun `scripts/040_prepare_training_data.py`. |
| 1.2 | Include `coach_feedback` rows (`include_coach_feedback: true` in the manifest). These already exist in `memory/feedback/`. |
| 1.3 | Pull in the full 58 teacher scores (Claude + GPT-4o) as first-class examples rather than only the 2 we're shipping now. Each scored clip becomes an absolute-score training example in addition to the consensus pair. |
| 1.4 | Regenerate consensus pairs for the ~15 scored clips that currently fail the confidence gate. |
| 1.5 | Create `data/training/2026-04-08_v2/` with the expanded set. Commit manifest only — no weights yet. |

**Expected yield:** 32 → **80–100 examples**. Still Task 5, still same
videos, but 2–3x the training signal.

**Compute cost:** $0 (local data prep only).

**Risk:** Lowering the confidence gate admits noisier pairs. Mitigate by
tagging low-confidence examples and excluding them from the eval split.

## Phase 2 — This week (2–3 days, ~$40 RunPod)

Target: **200 training examples across Tasks 1, 2, and 5.**

| Step | Action |
|---|---|
| 2.1 | Run the harvest script against `data/harvest_targets.csv`, priority rows `task1_peg_transfer` and `task2_pattern_cut` first (40 each), then `task5` (20 top-up), then `task3`/`task4` (20 each). |
| 2.2 | Teacher-score every newly harvested clip with Claude Sonnet 4 + GPT-4o in parallel (RunPod or local — whichever is cheaper at the moment). |
| 2.3 | Run the consensus pair generator on Tasks 1–5. |
| 2.4 | Retrain the Task-5-only student model first as a control; **then** train a multi-task variant that conditions on the task id. Evaluate both on a held-out Task 5 test split — the multi-task model should not regress by more than ~3% or we stop and debug. |
| 2.5 | Update `DATA_INVENTORY.md` and this file with observed pair-per-clip yield. |

**Expected yield:** 100 → **200+ examples**, 4–5 tasks represented.

**Compute estimate:**
- Teacher scoring: ~150 new clips × ~$0.08 / clip ≈ **$12**.
- Training run: 1x A100 × ~4h × $1.89/h ≈ **$8**.
- Control run: same again ≈ **$8**.
- Overhead + re-runs: ~$12.
- **Phase 2 total: ~$40.**

## Phase 3 — Next week (3–5 days, ~$120 RunPod)

Target: **500+ training examples, all tasks, including external dataset
integration.**

| Step | Action |
|---|---|
| 3.1 | **Download LASANA** manually from `opara.zih.tu-dresden.de` to a personal workstation (Cowork egress is blocked). Unpack into `data/external/lasana/` following the layout in `LASANA_README.md`. |
| 3.2 | Write a small ingestion adapter (`scripts/060_ingest_lasana.py`) that emits LASANA clips into the standard manifest, carrying the GRS rating as a **pre-scored** signal so the teacher pass can skip most of the set. |
| 3.3 | Teacher-score only the LASANA clips whose GRS label is ambiguous (mid-range) to augment the signal there. |
| 3.4 | Regenerate the training set including Task-1/2/4/5 LASANA clips plus Task 3 YouTube clips. |
| 3.5 | Train v3 with the full set. Early-stop aggressively (val loss patience 2 epochs). |
| 3.6 | Record per-task eval accuracy. This is the first training run that should demonstrate real per-task transfer. |

**Expected yield:** 200 → **500–700+ examples** (depending on LASANA's actual
per-task split). 5 tasks represented.

**Compute estimate:**
- LASANA download (manual, off-Cowork): free, ~1 h human time.
- LASANA ingestion + adapter: local, free.
- Teacher scoring for ambiguous LASANA clips: ~300 clips × $0.08 ≈ **$24**.
- Training v3: 1x A100 × ~10h × $1.89/h ≈ **$19**.
- Hyperparameter sweep (3 runs × ~6h each): ≈ **$34**.
- Eval + ablation: ≈ **$15**.
- Buffer + re-runs: ≈ **$28**.
- **Phase 3 total: ~$120.**

## Phase 4 — Stretch (week after next, ~$150 RunPod)

Target: **1,000+ training examples, multi-task model beats single-task
baseline on Task 5 by ≥5%.**

| Step | Action |
|---|---|
| 4.1 | Download **SPD-FLS1** from the Virginia Tech Figshare page. Write an adapter that treats session index + trial time as a longitudinal skill label (novice at session 1, proficient at session 6). ~150–250 additional Task-1 examples. |
| 4.2 | Pull the PhysioNet EEG-Eye-Gaze-for-FLS scored attempts as a **scorer calibration set** — no video, so they don't add training pairs, but they validate that our teacher scorers agree with a human rater on an independent cohort. |
| 4.3 | Exhaust the remaining rows of `data/harvest_targets.csv` for Tasks 3 and 4 (endoloop + extracorporeal knot), which are still the thinnest buckets. |
| 4.4 | Train v4 on the unified corpus with curriculum sampling (start with high-confidence examples, anneal to full set). |
| 4.5 | Publish per-task confusion matrices and decide whether FLS 2.0 Task 4 removal affects our eval protocol. |

**Expected yield:** 500 → **1,000+ examples**.

**Compute estimate:**
- SPD-FLS1 adapter + ingestion: local, free.
- Teacher scoring top-up (~200 clips): ≈ **$16**.
- Training v4 (1x A100 × ~15h): ≈ **$28**.
- Curriculum sweep (4 runs): ≈ **$60**.
- PhysioNet calibration analysis: local, free.
- Buffer: ≈ **$46**.
- **Phase 4 total: ~$150.**

## Aggregate budget

| Phase | Cost  | Cumulative | Examples at end |
|-------|------:|-----------:|----------------:|
| 1     |  $0   |    $0      | 80–100          |
| 2     | $40   |  $40       | 200+            |
| 3     | $120  | $160       | 500–700+        |
| 4     | $150  | $310       | 1,000+          |

Assumptions for compute pricing: RunPod on-demand A100 80GB at ~$1.89/h
(current as of this writing); teacher scoring via Anthropic + OpenAI APIs at
~$0.08 per 24-frame clip (current observed rate in
`memory/scores/2026-04-07/`).

## Success criteria

- **End of Phase 1:** Training run no longer catastrophically overfits Task 5
  (val/train loss ratio < 1.5 at best epoch).
- **End of Phase 2:** Model trained on 4–5 tasks does not regress on Task 5
  eval vs Phase 1 baseline by more than 3%.
- **End of Phase 3:** Per-task eval accuracy ≥ 60% on all 5 FLS tasks.
- **End of Phase 4:** Per-task eval accuracy ≥ 70% on all 5 FLS tasks; model
  agreement with held-out human raters ≥ teacher scorer agreement.

## Open risks

1. **LASANA license.** If it turns out to be CC BY-NC, we can't ship model
   weights commercially. Read the license before downloading.
2. **LASANA task-name mapping.** The four LASANA tasks may or may not line up
   1:1 with FLS Tasks 1/2/4/5 — verify on first download and update
   `DATASET_CATALOG.md` immediately.
3. **Task 4 is being removed in FLS 2.0.** We should still train on it because
   there's a lot of public data, but label it `fls1_only` in the manifest so we
   can strip it later.
4. **YouTube harvest label noise.** The `harvest_targets.csv` uses heuristic
   task classification from video titles. Expect 10–20% mislabelling; the
   harvest script's existing VLM-based classifier should catch most of it.
5. **Teacher scorer drift.** As we scale, we'll hit rate limits. Budget for a
   second scoring pass if early clips get scored with an older prompt version.
