# v7 Diagnosis — Failed Gate (mode collapse persists)

## Summary

v7 fixed the loss-masking bug (prompt tokens now masked out, only assistant response contributes to cross-entropy) and training loss descended cleanly:
- train_loss: 10.0 → 1.065 (step 10) → 0.149 (step 150) → **0.150 final** (step 560)
- eval_loss at step 200: **0.162** (vs v6 plateau of 5.5)

The loss fix worked. Training is now actually learning a response distribution. **But the gate still failed.**

## Eval results (checkpoints_vl_v7/eval_results.json)

| Metric | v7 | Gate | Result |
|---|---|---|---|
| valid_json_rate | 61.11% | > 90% | FAIL |
| unique_prediction_ratio | 0.00% | > 50% | FAIL (mode collapse) |
| task_id_accuracy | 0.00% | — | — |
| classification_accuracy | 0.00% | — | — |
| predicted_fls | all `None` | — | — |

json_valid=33/54, but zero of them contain the expected `task_id`/`fls_score` schema.

## What the model actually emits

Every prediction across 54 test examples is a verbatim echo of training-set JSON records — arrays of `{"id": "score_consensus_...", "video_id": "hLlQOZc7pNk", "source": "consensus", "model_name": "claude-sonnet-4-...", "prompt_version": "v002", ...}` — the *input* side of the dataset. 31 unique raw_heads across 54 samples, all drawn from training row boilerplate. The model is NOT generating a response conditioned on the test video; it is autoregressing from the prompt into memorized training-set rows.

The most-common output (8×) pairs test videos with training video_id `XQznLsU7pWI` regardless of what frames were passed. This is the smoking gun: **the model's output does not depend on the visual input.**

## Root cause

LoRA target_modules in v7 (`scripts/050_train_qwen_vl_v7.py:176`):

```python
target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
```

This adapts **only the language-model transformer blocks**. Nothing in the vision tower or the vision→LM merger is trainable. The visual features reach the LM's input embeddings exactly as the pretrained Qwen2.5-VL base produces them, and the LoRA-adapted LM has no gradient pressure to route information through those tokens — so it doesn't. It learns the highest-likelihood text continuation given the textual prompt alone, which is to regurgitate training-set JSON rows (the prompt template shows the model what a "row" looks like).

v5/v6 had the same target_modules list. v7 corrected the loss-masking bug on top but did not change target_modules. The "model can't see frames" failure mode was masked in v5/v6 by the larger "model never learned the response format" failure; with v7's loss fix that mask is gone and the underlying visual-ignoring behavior is now visible.

Secondary factor: task imbalance is severe. train yt = 445 / val 54 / test 54. Test has task1=29, task2=16, task5=8, task3=1. Training has only 12/8 examples of task3/task4. But this is not the primary failure — even on task1 (29 peg examples, dominant) every prediction is `None`, which means the model is not even picking the majority label. Mode collapse is total, not class-biased. Fixing only task balance won't unblock; fixing LoRA coverage is required.

## v8 proposal

### Primary change: include the vision merger in LoRA

Qwen2.5-VL routes visual features through `visual.merger.mlp.*` (a two-layer MLP) to produce the token sequence that gets concatenated into the LM input. That is the only narrow waist between the vision tower and the LM. It must be trainable if the LM is to learn to use visual information.

Two options, in order of preference:

1. **Full unfreeze of the merger** (`modules_to_save=["visual.merger"]` in LoraConfig, leaves the ViT itself frozen). This is the minimal, clean intervention. The merger is small (~few hundred MB trainable) so memory stays OK.
2. If option 1 is still memory-tight, LoRA the merger linears directly by adding `"visual.merger.mlp.0","visual.merger.mlp.2"` to `target_modules` (verify exact names with `model.named_modules()` — Qwen2.5-VL's merger uses linears, LoRA over them is supported).

Keep current LM target_modules. Keep r=32, lr=1e-4, cosine, 5 epochs, prompt masking, preflight check. Bump preflight to also verify at least one `visual.merger.*` parameter has `requires_grad=True`.

### Secondary: class rebalancing (defer to v9 if v8 gate passes)

If v8 passes valid_json_rate and unique_prediction_ratio gates but task3/task4 classification is weak, upsample those classes 3–5× in the collator and re-run. Not needed for v8 itself — the primary question is whether the model can see frames at all.

### Expected outcome

If the merger-training hypothesis is correct, predictions will (a) differ across test videos, pushing unique_prediction_ratio above 0.5, and (b) match the trained response schema, pushing valid_json_rate above 0.9. Classification accuracy is the downstream metric to watch; if it's non-trivially above chance (1/5 task classes = 20%), the loop is closed.

If v8 *also* mode-collapses with the merger unfrozen, hypotheses to investigate next: (i) frames aren't actually reaching the model (collator bug on the image side), (ii) generation kwargs are forcing greedy with identical seeds so sampling never diverges, (iii) base model requires vision-tower unfreezing too. Verify (i) first with a quick sanity print in the collator showing pixel_values shape and non-zero variance across examples in a batch.
