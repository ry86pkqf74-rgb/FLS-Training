# v003 Evaluation Comparison: v17 vs v18

**Date:** 2026-04-27
**Test set:** 23 held-out examples, task-stratified across the 5 official FLS tasks (5/5/4/4/5).
**Eval harness:** `scripts/071_eval_v003_adapter.py`, `--max-new-tokens 1024`, deterministic decoding.

## Top-line numbers

| Metric                      | v17       | v18       | Winner | Notes                                              |
|-----------------------------|-----------|-----------|--------|----------------------------------------------------|
| Score MAE (points)          | **77.0**  | 167.9     | v17    | Lower is better. v18 emits longer outputs that get truncated → JSON parse fails. |
| Score RMSE                  | **101.7** | 237.6     | v17    | Same root cause — failed parses fall back to score=0 inflating tail errors. |
| Normalized score MAE        | **24.2 %**| 41.1 %    | v17    | Score abs error / per-task max score.              |
| Time MAE (seconds)          | 81.2      | 86.4      | tie    | Both very rough, similar.                          |
| Task accuracy               | **96 %**  | 65 %      | v17    | v17 correctly identifies task_id 22/23 times.      |
| Schema compliance           | **96 %**  | 65 %      | v17    | Fraction of outputs with all 8 v003 required fields populated. v18's drop is a token-budget artifact. |
| Critical-error precision    | 0.0       | **0.71**  | v18    | v18 is dramatically better at flagging critical errors. |
| Critical-error recall       | 0.0       | **0.71**  | v18    | v17 emits *zero* critical_errors entries on the test set. |
| Critical-error F1           | 0.0       | **0.71**  | v18    | v18 wins this category outright.                   |

## Per-task score MAE (lower is better)

| Task                               | n | v17 MAE / norm    | v18 MAE / norm    |
|------------------------------------|---|--------------------|--------------------|
| Task 1 (Peg Transfer)              | 5 | 43.5 / 14.5 %     | **35.9 / 12.0 %** |
| Task 2 (Pattern Cutting)           | 5 | 112.5 / 37.5 %    | **105.5 / 35.2 %**|
| Task 3 (Endoloop)                  | 4 | **73.8 / 41.0 %** | 73.8 / 41.0 %     |
| Task 4 (Extracorporeal)            | 4 | **60.6 / 14.4 %** | 241.4 / 57.5 %    |
| Task 5 (Intracorporeal)            | 5 | **90.6 / 15.1 %** | 378.8 / 63.1 %    |

v18 actually beats v17 on the small-denominator tasks (1, 2) but regresses badly on Task 4 and Task 5 — the high-score tasks where a truncated output produces the largest absolute error.

## Training-loss summary

| Run | Resumed from | Epochs | Effective batch | LR     | Train loss | Eval loss | Adapter size |
|-----|-------------|--------|-----------------|--------|------------|-----------|--------------|
| v17 | base Qwen   | 3      | 8               | 1e-4   | 3.7268     | 0.3911    | 381 MB       |
| v18 | v17         | 6      | 4               | 3e-5   | **0.9405** | **0.3267**| 381 MB       |

Cross-entropy / NLL eval loss DID drop (0.39 → 0.33). The training is genuinely producing a better-fit model on the LoRA training distribution. The score MAE regression is driven by **decoding-time output length** — v18's outputs are ~2.5× longer in JSON characters, and at 1024 max-new-tokens they get cut off mid-object.

## Root-cause analysis of the score-MAE regression

Looking at `v18_eval_raw.jsonl`:

- 8/23 examples returned an empty parsed object (`json_parsed: false`). v17 had 1/23.
- Failed-parse rows have `score_pred = 0`, which against typical training-score targets of 200–550 produces individual abs errors of 200–550. Eight of those alone account for ~80% of the v18 MAE.
- Critical_errors **are emitted** (precision = recall = 71 %). The v18 LoRA learned the new behavior cleanly; it just chose an output style that is too verbose for the eval token budget.

A re-eval at `--max-new-tokens 2048` (started, then aborted because each example takes ~70 s on the H200 in 4-bit, vs ~9 s for v17) would likely cut the score MAE substantially. The truncation is the dominant problem, not score regression in the model itself.

## Decision: deploy v17 as production winner

Net-net, v17 is the better production model right now:

- 96 % schema compliance and 96 % task accuracy are non-negotiable for a live-facing demo. 65 % schema compliance from v18 means roughly 1 in 3 reports would render as malformed JSON in the Gradio UI, which is much worse user experience than the v17 issue (no critical-error flagging).
- v17 already runs on the Contabo box (since the previous swap), so deploying v17 = no further deployment work.
- v18 still has value as a research artifact: it clearly demonstrates that the LoRA *can* learn the critical-error contract, the issue is just decoding budget + dataset size.

## Recommended next iteration (v19)

Three concrete changes that should give us "smarter AND more accurate":

1. **Loss reweighting**: train v19 with a higher loss weight on `score_components.total_fls_score` and `estimated_fls_score` token positions. Right now the standard CE loss treats every token equally — a 5-character `397.0` and a 200-character free-text `technique_summary` matter the same. We want the score number to be near-perfect.
2. **Output-length supervision**: clip every assistant message in the training JSONL to ≤ 800 tokens (currently many are 1500+) so the LoRA learns to be concise. v18's verbosity is a learned behavior, not an inherent property of the schema.
3. **Resume from v18 (not v17)**: take v18's critical-error fluency forward, then teach it brevity. This is cheaper than starting over from base Qwen.

Both adapters are saved for the v19 ablation:

- `memory/model_checkpoints/v17_v003/` — current production
- `memory/model_checkpoints/v18_v003/` — research / next-iter starting point

## Artifacts

| File                                    | Description                                              |
|-----------------------------------------|----------------------------------------------------------|
| `v17_eval.json` / `v17_eval_raw.jsonl`  | v17 metrics + per-example raw outputs                    |
| `v18_eval.json` / `v18_eval_raw.jsonl`  | v18 metrics + per-example raw outputs                    |
| `v17_metrics.json` / `v17_train.log`    | v17 training metrics + log                               |
| `v18_metrics.json` / `v18_train.log`    | v18 training metrics + log                               |
| `v003_multimodal_manifest_run2.json`    | Iteration-2 dataset manifest (155 video frames decoded, 138 vision examples) |
| `EVAL_COMPARISON.md`                    | This document                                            |
