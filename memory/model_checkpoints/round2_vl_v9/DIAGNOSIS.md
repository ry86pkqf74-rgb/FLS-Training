# round2_vl_v9 — Diagnosis

**Run ID:** `fls_round2_vl_v9_20260415_071001`
**Trained:** 2026-04-15, H100 80GB (Runpod pod `g05ctg5w1brtsc`)
**Dataset:** `youtube_sft_v4` (407 train / 49 val / 48 test)
**Base:** Qwen2.5-VL-7B-Instruct, LoRA r=32 α=32, merger unfrozen
**Loss:** train 0.136, eval 0.134 (no overfit), 410 steps in ~47 min

## Verdict
**Mixed.** Pipeline fixes helped task1/task2 as predicted. Task5 remains broken, and
diversity regressed. No further training will move the needle until we collect more
task3/task4/task5 videos with real label diversity.

## What changed from v8 → v9

Three fixes, all upstream of training:

1. **LASANA task5 rescore at 24 frames** (was 8). 080_lasana_rescore_v002.py run on S8
   against 314 suture videos. 261/314 turned out to be unusable "static-view" clips
   (teacher emits `total_fls_score=null`); 37 nonzero labels remain with real FLS
   diversity (min 0, mean 343, max 580). v8 had 82 rows that were 89% zeros.
2. **FLS clamp to [0, max_score]** in 044_expand_sft_pool.py. Removed 27 negative-FLS
   rows that had been poisoning task1/task2.
3. **Pool rebuilt as `youtube_sft_v4/`** with min_conf=0.5. Dropped {v4_schema: 77,
   lasana_schema: 280, low_conf: 273}.

## Gate metrics (v8 → v9)

| metric | v8 (youtube_sft_v2) | v9 (youtube_sft_v4) | target | delta |
|---|---|---|---|---|
| valid_json_rate | 1.000 | 0.917 | > 0.9 | ✓ (slight regression) |
| task_id_accuracy | 1.000 | 0.917 | — | ↓ |
| unique_prediction_ratio | 0.185 | 0.091 | > 0.5 | ✗ regressed |
| unique_fls_values | 10 | 4 | — | ↓ regressed |
| mae_fls_score | 94.5 | 97.9 | lower | ~tied |

## Per-task MAE (v8 → v9)

| task | v8 MAE | v9 MAE | n_test (v9) | read |
|---|---|---|---|---|
| task1 peg_transfer | 127.2 | **96.3** | 28 | Real 25% improvement from clamp fix |
| task2 pattern_cut | 80.9 | **39.9** | 7 | Real 51% improvement — biggest win |
| task3 endoloop | 119.8 | — | 0 | No test data; collection still blocked |
| task5 intracorporeal | **0.0** | 525.0 | 2 | See "Task5 caveat" below |

## Task5 caveat — the v8 zero was garbage-in/garbage-out

v8's "0.0 MAE" on task5 looked like a win but was an artifact: v8's test labels were
all zeros (no real diversity), so predicting near-zero matched trivially. v9's test
set has real labels (508, 542 from the rescored pool) and the model predicts near-zero
anyway. The v9 525 MAE is the **honest** signal: task5 hasn't been learned. 28 train
rows (only ~15 nonzero) is not enough.

## Diversity regression — model is memorizing

`unique_prediction_ratio` fell from 0.185 → 0.091 (only 4 distinct FLS values across
44 predictions). Spot checks of the raw_head output show the model hallucinating
training-set video IDs into its predictions. This is what memorization looks like on
a 407-row dataset. More data — not more epochs — is the fix.

## Recommendation

**Pipeline fixes ship.** Keep the v9 tag as the current best checkpoint: it beats v8
on both tasks where we have real label density (task1, task2), and its task5 number
is honest rather than flattering.

**Do not retrain further** until the blocking data collection is done:

- **task3 (endoloop):** ≥50 user-captured videos with score diversity
- **task4 (extracorporeal knot):** ≥50 user-captured videos
- **task5 (intracorporeal suture):** ≥50 additional user-captured videos — the LASANA
  pool has been mined out (261/314 rejected as static-view).

Until then, additional SFT epochs on `youtube_sft_v4` will continue to regress
diversity without moving MAE.
