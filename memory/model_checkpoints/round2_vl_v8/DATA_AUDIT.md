# Data Audit — Pre-v9 Gate

**Date:** 2026-04-15
**Dataset:** `data/training/youtube_sft_v3/` (449 train / 55 val / 54 test — used by v8)
**Purpose:** Decide whether v9 retrain on current data is worthwhile or blocked on
data collection.

## TL;DR

**Do not deploy v9 on current data.** Three data-quality bugs explain v8's
per-task mode collapse. Two are pipeline fixes; one requires new labels or
new videos.

| Task | Train n | Unique FLS | Zero% | Verdict |
|---|---|---|---|---|
| task1 peg transfer | 232 | 96 | 12.9% | Trainable — has room if negatives clamped |
| task2 pattern cut | 131 | 48 | 28.2% | Trainable — has room if negatives clamped |
| task3 endoloop | 12 | 3 | 83.3% | **Data-starved** — need ≥50 more scored videos |
| task4 extracorporeal knot | 8 | 4 | 50.0% | **Data-starved** — need ≥50 more scored videos |
| task5 intracorporeal suture | 66 | 7 | 89.4% | **Label pipeline bug** — fix before retrain |

Total test-set diversity limitation: task3 has 1 test ex, task4 has 0, task5
has 8 all labeled 0.0. Cannot measure model performance on these regardless
of training.

## Bug 1 — Negative FLS scores (27 rows)

`scripts/044_expand_sft_pool.py` normalizes score_components without clamping
`total_fls_score` to `[0, max_score]`. Rows with time_used + penalties >
max_score produce negatives. Example:

```json
{"max_score": 300, "time_used": 105, "incomplete_penalty": 250,
 "total_fls_score": -55, "formula_applied": "300 - 105 - 0 - 250 = -55.0 (minimum score is 0)"}
```

The author even noted "minimum score is 0" in the formula string, but the code
didn't enforce it. 20 such rows in task1, 7 in task2. These are active noise
that pull peg-transfer predictions toward 120/282 modes.

**Fix applied:** clamp `max(0, min(max_score, total))` in both v4 and
lasana-rescore paths. Commit pending.

## Bug 2 — LASANA suture rescoring collapses to 0 (52 / 58 rows)

`scripts/080_lasana_rescore_v002.py` samples 8 frames from each full-length
LASANA suturing video and asks Claude Sonnet 4 to score. For suturing
(task5, max_time = 600s), 8 frames is ~75s apart — Claude can't see task
completion and defaults to "task not completed" per the rubric, yielding
`time_used=600, penalties=600, total_fls_score=0`.

Distribution of lasana_rescore task5 labels:

```
Counter({0.0: 52, 173.0: 2, 145.0: 1, 92.5: 1, 112.5: 1, 211.0: 1})
```

**Fix options:**
1. Re-run `080_lasana_rescore_v002.py --max-frames 24 --task suture` — gives
   Claude enough temporal coverage to see a completed knot. Cost: ~58 Claude
   API calls, maybe $10.
2. Replace lasana_rescore task5 labels entirely with LASANA's native human GRS
   ratings (already in `data/training/2026-04-09_lasana_v1/`). These are
   real human expert scores, not Claude estimates. Preferred.

## Bug 3 — Consensus task5 also mostly 0 (7 / 8 rows)

```
consensus task5 FLS: [0.0, 0.0, 0.0, 0.0, 358.0, 0.0, 0.0, 0.0]
```

These are teacher-scored YouTube suturing videos, not LASANA. Same root cause
likely: Claude sees partial task in 8 frames and defaults to incomplete = 0.
Fix is the same as Bug 2 option 1 (more frames).

## Data gaps (not bugs — physical scarcity)

- **task3 endoloop**: 12 train / 1 val / 1 test. Only 2 training examples have
  non-zero FLS (145, 138). Test set has 1 example — statistically meaningless.
  Need to collect and score ≥50 endoloop videos before this task is trainable.
- **task4 extracorporeal knot**: 8 train / 1 val / 0 test. 4 non-zero
  (255, 230, 230, 270). Model literally never sees this task at eval. Need
  ≥50 videos.

Both are procedurally-rare on YouTube (unlike peg transfer and pattern cut,
which are FLS-course staples with thousands of tutorial videos). Most likely
source: clinical skills labs willing to share recorded practice sessions, or
direct capture of user performances.

## Endpoint analysis — answer to "more data vs. train harder"

**We are at the endpoint for 3 of 5 tasks on current data.**

- task1 + task2: worth another training round IF pipeline bugs are fixed.
  Expected gain: MAE 127 → ~60 for task1, 80 → ~50 for task2. This alone
  is not a production-ready model.
- task3 + task4 + task5: **no retrain helps** until we have more labels.
  - task5 unblocks with label fix (regenerate rescored labels with 24 frames
    OR swap in LASANA native GRS).
  - task3 + task4 need new videos. Shortest path: capture user performances
    of these two tasks specifically.

## Recommended path before v9

1. Apply negative-FLS clamp to `044_expand_sft_pool.py` (this commit).
2. Regenerate task5 labels: re-run `080` with `--max-frames 24 --task suture`
   and review distribution before re-pooling.
3. Regenerate the dataset (`044` produces `youtube_sft_v4/`).
4. **Collect user videos for task3, task4, and additional task5.** Target
   ≥50 per task with score diversity across novice/intermediate/expert bands.
5. Only then deploy v9. v9 trained on current data will not meaningfully
   improve the gate numbers that matter (diversity, MAE across all 5 tasks).

## Open questions for the user

- Do you have access to LASANA's native human GRS labels for suturing, or
  do we need to regenerate via Claude with more frames?
- Is there an existing pipeline for user-captured task3/task4/task5 videos,
  or do we need to stand one up?
- Budget check: re-scoring task5 at 24 frames costs ~$10; scoring 150 new
  videos (50 per task × 3 tasks) at 8 frames costs ~$30.
