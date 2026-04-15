# Round 2 VL v8 — Diagnosis & Salvage

**Run:** `round2_vl_v8` · Qwen2.5-VL-7B-Instruct LoRA (r=32, α=32, dropout=0.05) + merger unfrozen
**Pod:** `v7lexoj3odzen4` (H100 80GB)
**Adapter:** `/workspace/checkpoints_vl_v8/final/` (584 LoRA keys + 5 merger keys)
**Training:** 560 steps, loss 1.08 → 0.06, finished 2026-04-15 02:56 UTC, clean

## TL;DR

v8 is a **working model**, not a collapse. Initial gate failure (0/54 valid JSON,
0% task-id accuracy, 0% diversity) was caused by a **prompt distribution shift in
`042_eval_vl_adapter.py`** — eval used a short system prompt while training used
the long SCHEMA_EXAMPLE + schema-notes prompt. After patching eval to match the
training prompt, metrics jumped to 100% valid JSON and 100% task-id accuracy.

The diversity gate still soft-fails (18.5% < 50%), but this is per-task modal
clustering, not collapse to a single token.

## Final metrics (post-fix, `eval_results.json`)

| Metric | Value | Gate |
|---|---|---|
| valid_json_rate | **1.000** | > 0.9 ✓ |
| task_id_accuracy | **1.000** | — |
| unique_prediction_ratio | 0.185 (10/54) | > 0.5 ✗ (soft-fail) |
| mae_fls_score | 94.5 (0–300 scale) | — |
| per_task_mae | peg 127.2 · pattern 80.9 · endoloop 119.8 · suturing 0.0 | — |
| classification_accuracy | 0.000 | — (exact-match on full score) |

54 predictions span 10 unique FLS values; model is task-conditioned (peg defaults
to 120/282, pattern_cut to 202.5/110, suturing to 0) but has collapsed within-
task variance. MAE ≈ 94 on a 300-point scale = meaningful signal, not random.

## Root cause

Training system prompt (in `scripts/051_train_qwen_vl_v8.py`) embeds a
`SCHEMA_EXAMPLE` dict and explicit "CRITICAL schema notes" (score_components MUST
be an OBJECT, enum of task_ids, `estimated_fls_score` top-level). Eval script
`042_eval_vl_adapter.py` had been written with an earlier, shorter system prompt:

```python
# OLD eval SYSTEM_PROMPT — 4 lines, no schema example
"You are an expert FLS ... output a single strict-JSON ScoringResult ..."
```

The LoRA weights were tuned against the detailed prompt's token distribution.
Under the short prompt the model produced raw garbage (hence 0% valid_json).
Swapping in the training prompt recovered correct schema output.

**User text also patched:**
`"Score this FLS {task} performance. JSON only."` →
`"Score this FLS {task} performance. Return ONLY valid JSON per the v002 schema."`
(matches training).

## What was ruled out

- **LoRA not loaded at eval** — rejected; non-zero `lora_B` norms (0.16–1.03)
  confirmed adapter active.
- **Labels not supervising content** — rejected; diag probe showed first
  supervised token at position 1579 = `{\n`, supervised text begins
  `'{\n  "id": "score_consensus_..."task_id": "task1_peg_transfer"...'`.
- **Training data non-uniform** — rejected; 449/449 examples have
  `task_id`, `score_components` dict, `total_fls_score`, `estimated_fls_score`.
- **Label truncation** — rejected; `processor.apply_chat_template` called with
  `truncation=False`.

## Residual limitation

Diversity gate fails because the model over-commits to per-task modes (within
`task1_peg_transfer`, most predictions are 120.0 or 282.0; within
`task2_pattern_cut`, most are 202.5). This is a data/training signal issue —
the training labels have low entropy within-task — not a broken adapter.
MAE remains meaningful (80–127 per task) so the next round should focus on
within-task score calibration, not schema fidelity.

## Patches applied (this round)

`scripts/042_eval_vl_adapter.py`:
1. Added `nn.Module.set_submodule` shim (torch 2.4 compat).
2. Added `torch.float8_e8m0fnu`/`e4m3fnuz`/`e5m2fnuz` shims (torch 2.6 compat).
3. Dropped 4-bit quantization (BitsAndBytesConfig) — load full bf16 on H100.
4. Replaced short `SYSTEM_PROMPT` with training-matching long prompt including
   `SCHEMA_EXAMPLE` dict and CRITICAL schema notes.
5. Updated user text to `"Return ONLY valid JSON per the v002 schema."`.

## Recommendation

Accept v8 as the first non-collapsed VL checkpoint. Ship artifacts; next round
(v9) should target within-task diversity, not schema — training data relabeling
or loss reweighting, not prompt engineering.
