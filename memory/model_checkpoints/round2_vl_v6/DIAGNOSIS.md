# v6 Diagnosis — killed at step 88/896

## TL;DR
v6 plateaued at the exact same training-loss floor as v5 (~5.5) despite
a lower LR (3e-4 → 1e-4), higher LoRA rank (16 → 32), grad-clip=1.0, and
stricter frame filtering. Killed at step 88 once the flatline was
unambiguous. The knobs were never the bug.

## Loss trajectory (logging_steps=10)

| step | loss  | grad_norm | lr      | notes                          |
|-----:|------:|----------:|--------:|--------------------------------|
|  10  | 15.14 |    56.14  | 2.0e-5  | warmup, pre-descent            |
|  20  | 13.41 |    11.12  | 4.2e-5  | clip engaged                   |
|  30  |  9.46 |     2.96  | 6.4e-5  | real descent                   |
|  40  |  6.77 |     0.54  | 8.7e-5  |                                |
|  50  |  5.90 |     0.27  | 1.0e-4  | LR target reached              |
|  60  |  5.63 |     0.11  | 9.99e-5 |                                |
|  70  |  5.56 |    0.067  | 9.98e-5 | **same floor as v5**           |
|  80  |  5.58 |    0.074  | 9.96e-5 | flat — gradient vanished       |

grad_norm collapsing below 0.1 while loss is flat is the telltale: the
model has reached a local minimum of whatever it's actually being trained
to predict. It's not going to move from here, regardless of remaining
epochs.

## Root cause (identified by reading the collator)

v4, v5, and v6 all share this in `collate(...)`:

```python
labels = input_ids.clone()
labels[labels == processor.tokenizer.pad_token_id] = -100
```

Only pad tokens are masked out of the cross-entropy. That means loss
is computed on **every** non-pad token — the system prompt, the user
message (including hundreds of expanded `<|image_pad|>` tokens), AND the
assistant response. With an identical system prompt across every
example, the overwhelming majority of the sequence is trivial to
predict. The model memorizes the prompt, loss drops to the irreducible
entropy of the response-token distribution, and training ends there.

That's why eval showed 96% valid JSON but **0% unique predictions**:
the "prediction" the model converged to is whatever single JSON minimizes
the response-token entropy across the corpus — effectively the modal
answer for each task. Mode collapse was never caused by LR or rank; it
was baked into the loss function.

## Why v5's data-diversity hypothesis was wrong

The v5 DIAGNOSIS.md conjectured the plateau was LASANA visual homogeneity
+ task imbalance. An audit at that time showed healthy FLS-score variance
across tasks, which disconfirmed part of it. What it didn't check was
**which tokens the loss was computed on**. The loss-mask bug was present
in v4/v5/v6 and masked the real signal.

## v7 fix (see scripts/050_train_qwen_vl_v7.py)

Single targeted change: render the non-assistant messages separately with
`add_generation_prompt=True`, tokenize to get `prompt_len`, and set
`labels[:prompt_len] = -100` in the collator. Now cross-entropy only
penalizes the assistant JSON.

A preflight sanity check prints
`total_tokens / prompt_tokens / supervised_tokens` for the first example
and aborts if supervised_tokens < 20, so the mask can't silently
regress.

Other knobs kept from v6 (they weren't broken):

* LR=1e-4 cosine, warmup 5%, grad_clip=1.0
* LoRA r=32, alpha=32, dropout=0.05
* MIN_FRAMES=4, MAX_FRAMES=8

Epochs reduced 8 → 5: with real gradient signal the model should converge
substantially faster than the prompt-memorization phase did, and 5 full
epochs over 445 examples is already ~560 optimizer steps.

## Gate expectation

If the mask fix is the right story, v7 eval should show:
  * valid_json_rate >= 0.9 (maintained from v5: 96.3%)
  * unique_prediction_ratio > 0.5 (v5/v6 collapsed: 0%)
  * training loss descending PAST 5.5 instead of flattening there

If v7 still plateaus, the remaining suspects are: (a) image tokens not
participating in learning due to the 4-bit quantization of the vision
encoder, (b) the model is not conditioning on video frames at all and
the LoRA adapters on language-only layers can't fix that.
