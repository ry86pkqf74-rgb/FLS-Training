# Round 2 VL v1 — Gate Failure Diagnosis (2026-04-14)

**Adapter**: `Qwen2.5-VL-7B-Instruct` + fresh LoRA r=16, 45 YouTube SFT examples, 15 epochs, 180 optimizer steps.
Trained on RunPod pod `v7lexoj3odzen4` (H100 80GB), ~38 min runtime.

## Gate results (on 5-example held-out test set)

| Metric | Value | Gate | Pass |
|---|---|---|---|
| valid_json_rate | 80% | >90% | FAIL (1/5 had no resolvable frames → empty output) |
| task_id_accuracy | 80% (4/4 on examples with frames) | — | Good |
| unique_prediction_ratio | 0.0% | >50% | FAIL |
| mae_n | 0 | — | FAIL — no FLS scores extractable |
| classification_accuracy | 0% | — | Model doesn't emit `video_classification` field |
| train_loss | 6.42 | <1 for SFT | Undertrained |
| eval_loss | 5.69 | — | Matches train loss → not overfit, just not learned |

## Root cause: schema not learned

Training targets have `score_components` as a **dict**:
```json
"score_components": {
  "max_score": 300,
  "time_used": 21.0,
  "total_penalties": 3.5,
  "total_fls_score": 276.2,
  "formula_applied": "300 - 21.0 - 3.5 = 275.5"
}
```
Model emits `score_components` as a **list** of component entries:
```json
"score_components": [
  {"name": "total_fls_score", "value": 19.0, ...},
  {"type": "knot_quality", "score": 8.0, ...}
]
```
This is why `get_fls()` returns None for every example — the extractor (correctly) looks for the dict shape.

The model *did* learn `task_id` (4/4 correct on examples with resolvable frames) and valid-JSON syntax, but
not the nested score-components structure or the `video_classification` field.

## Why undertrained

- 45 train examples × (15 epochs / 4 grad_accum) = **180 optimizer steps**
- Train loss plateaued at ~5.5–6.4 range late in the run, with grad_norm decaying to ~0.05 (cosine LR wind-down
  reached near-zero LR while loss was still high)
- Structured JSON SFT typically needs loss < 1 to reliably reproduce schema

## Recommendations for v2 → v3

1. **More data**: 45 examples is marginal. Pull more YouTube SFT rows from the dual-teacher consensus set;
   target ≥150 train examples if feasible.
2. **More steps**: with the same 45 examples, bump to 30–50 epochs (≈360–600 optimizer steps). Current cosine
   schedule decayed LR too early.
3. **Tighter LR floor**: `lr_scheduler_type="cosine_with_min_lr"` with `min_lr_rate=0.1` so LR doesn't collapse
   before loss converges.
4. **Drop one test sample with no frames** (`yt_IwrNTRVXuJQ`) or fix its `frames` field — currently contributes
   a guaranteed invalid-JSON row to the eval.
5. **Schema-aware training hint**: consider adding a one-shot example of the expected JSON shape in the system
   prompt to anchor the nested structure.

## Artifacts saved here

- `run_manifest.json` — hyperparams, data counts, timings
- `eval_results.json` — per-example predictions + summary metrics
- `train_tail.log` — last 200 lines of training output
- `eval.log` — full eval log

Adapter weights (190MB) were not committed to git. They remain on pod `v7lexoj3odzen4` at
`/workspace/checkpoints_vl/final/` until the pod is stopped; if retention is desired, `scp` them to
a storage node first.
