# v5 Diagnosis — Mode collapse, not a data-diversity problem

## Gate result

| Metric | v4 | v5 | gate |
|---|---|---|---|
| valid_json_rate | 72.88% | **96.30%** | >0.9 PASS |
| unique_prediction_ratio | 0% | **0%** | >0.5 FAIL |
| task_id_accuracy | 3.4% | 0% | — |
| train_loss (final) | — | 5.79 | — |
| eval_loss | — | 5.51 | — |

**Gates: FAIL (mode collapse).**

## Hypothesis tested

Initial hypothesis was "canonicalization stripped target diversity." **Disconfirmed.**

Audit of `yt_train.jsonl` (449 examples):
- 449/449 unique target strings
- Mean shared prefix ratio: 4% (median 4%)
- FLS scores span [-265, +358], stdev=120, distributed across all 5 tasks
- All 449 targets parseable with non-null `score_components.total_fls_score`

Targets are diverse. The model is not mimicking a centroid *because the targets cluster* — it is mimicking a centroid *despite the targets not clustering*.

## Actual likely causes

1. **Training loss plateau at 5.5.** Dropping from ~12 (step 1) to ~5.5 by step ~40 and flatlining. Cross-entropy of 5.5 per token on JSON is very high — the model is not actually learning the distribution, it is collapsing to a low-entropy output policy that happens to pass syntax validation.
2. **LR too high for LoRA.** 3e-4 on rank-16 over 4-bit Qwen2.5-VL-7B is aggressive; cosine with min 10% keeps it above 3e-5 even at end. Classic symptom of high-LR LoRA instability: early-step blow-up (loss 12→5 in first few steps) then collapse to a mode.
3. **Task imbalance.** task3=12, task4=8 examples (<5% combined). Model optimizes away from those tails.
4. **Frame signal likely weak.** 373/449 training rows are LASANA rescore (synthetic renders, visually near-identical). The 76 v4 YouTube rows have `frames: []` in the raw JSONL and only recover frames via fallback to `/workspace/frames/<video_id>`. Visual conditioning is therefore dominated by a low-diversity visual source.

## v6 knobs

| Knob | v5 | v6 | Rationale |
|---|---|---|---|
| LR | 3e-4 | **1e-4** | reduce LoRA instability |
| LR schedule | cosine_min_lr (min 10%) | cosine (no min) | let LR decay to zero |
| LoRA rank | 16 | **32** | more adapter capacity |
| LoRA alpha | 16 | **32** | match rank |
| Epochs | 5 | **8** | more iterations at lower LR |
| Batch | 1x4 | 1x4 | unchanged (memory) |
| MAX_FRAMES | 8 | 8 | unchanged |
| Frame filter | fallback resolver | **strict** — drop if resolved path list empty or <4 frames | prevent zero/low-signal training |
| Warmup | 3% | 5% | smoother start |
| Logging | every 20 steps | every 10 steps + grad norm | see collapse as it happens |

Total effect: roughly same wall time (~45 min on H100), 3x less LR, 2x adapter capacity, 1.6x steps.

## Gate policy

Unchanged. `042_eval_vl_adapter.py` already gates valid_json >0.9 AND unique_pred_ratio >0.5. v6 is a gate-hunt — if v6 passes syntactically AND has diversity, promote; if it fails on diversity, the problem is data-side (add task3/4 examples, add non-LASANA visual variety) rather than optimization.

