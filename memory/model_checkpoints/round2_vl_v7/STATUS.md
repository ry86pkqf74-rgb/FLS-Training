# v7 Training Status — 2026-04-14 FINAL

## Outcome: FAILED gate (mode collapse persists)

Training completed successfully (final train_loss 0.150 — the loss-masking fix
worked), but the v7 adapter mode-collapsed at eval. See `DIAGNOSIS.md` for
full analysis and v8 proposal.

## Training (completed)

- **Pod:** `v7lexoj3odzen4` (fls-train-h100-v2), stopped post-handoff
- **Process:** PID 37561, `python3 -u FLS-Training/scripts/050_train_qwen_vl_v7.py`
- **Started:** 2026-04-14 17:22 UTC
- **Completed:** 2026-04-14 20:57 UTC (~3h35m wall)
- **Final checkpoint:** `checkpoints_vl_v7/final/` (scp'd to this directory)

## Loss trajectory (v7 vs v6)

| step | v6 loss (broken) | v7 loss (fixed) |
|-----:|-----------------:|----------------:|
|  10  | 15.14            | 1.065           |
|  30  |  9.46            | 0.348           |
|  60  |  5.63            | 0.205           |
|  80  |  5.58 (plateau)  | n/a             |
| 150  | —                | 0.149           |
| 240  | —                | 0.099           |
| 560 (final) | —         | 0.150           |

Eval loss at step 200 = 0.162. Mask fix confirmed working at the loss level.

## Mask sanity check (printed at startup)
```
total_tokens=2476  prompt_tokens=2223  supervised_tokens=253
```

## Eval results

| Metric | v7 | Gate | Pass? |
|---|---|---|---|
| valid_json_rate | 61.11% | > 90% | NO |
| unique_prediction_ratio | 0.00% | > 50% | NO |
| task_id_accuracy | 0.00% | — | — |
| classification_accuracy | 0.00% | — | — |
| predicted_fls | all null | — | — |

json_valid=33/54, but *none* of them match the expected `{task_id, fls_score}`
schema — they are verbatim echoes of training-set row boilerplate
(`[{"id":"score_consensus_...","video_id":"hLlQOZc7pNk",...}]`). 31 unique
raw_heads across 54 samples, all drawn from training-input JSON. The model's
output does not depend on the test video's frames.

## Root cause (see DIAGNOSIS.md)

LoRA target_modules cover only the LM transformer blocks. The vision→LM
merger is frozen, so no gradient pressure forces the LM to condition on
visual features. Model defaults to memorized training-row continuations.

## v8 proposal

Unfreeze `visual.merger` (either `modules_to_save=["visual.merger"]` or LoRA
its linears). Keep everything else. Add preflight assert that at least one
merger param has `requires_grad=True`. See DIAGNOSIS.md §"v8 proposal".

## SSH gotcha
```
export SSH_AUTH_SOCK=$(launchctl getenv SSH_AUTH_SOCK)
```
