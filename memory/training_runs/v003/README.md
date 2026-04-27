# v003 Training Iteration — 2026-04-27

## v17 Multimodal Scoring LoRA

Run on RunPod H200 (143GB VRAM), pod `9au45uwwizb3oe`.

| Metric | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Resumed from | none (fresh LoRA — v16 weights not accessible) |
| Trainable params | 95,178,752 (1.13%) |
| Train examples | 193 |
| Val examples | 23 |
| Test examples | 23 |
| Epochs | 3 |
| Effective batch | 8 (1 × 16-step grad accum reduced to 8 for H200) |
| Learning rate | 1e-4 (cosine, 5% warmup) |
| LoRA r / alpha | 32 / 32 |
| Train loss | **3.7268** |
| Eval loss | **0.3911** |
| Wall time | 5 min 19 s |

## Dataset (v003_multimodal)

Built by `scripts/030d_build_v003_pod_dataset.py` — task-stratified across the 5 official FLS tasks plus the custom Task 6.

| Task | Count | Max score |
|---|---|---|
| Task 1 (Peg Transfer) | 51 | 300 |
| Task 2 (Pattern Cutting) | 52 | 300 |
| Task 3 (Endoloop) | 41 | 180 |
| Task 4 (Extracorporeal) | 40 | 420 |
| Task 5 (Intracorporeal) | 55 | 600 |
| Task 6 (Rings of Rings) | 0 | 315 |

68 examples carry frames (4 per video, decoded from `memory/frames/<vid>/frames.json`); 171 are text-only fallback.

## Schema verification

Smoke test on a held-out task2 example confirms the v17 LoRA emits clean v003 JSON:
- `max_score = 300` (task-specific, not 600)
- `formula_applied = "300 - 179 - 0 = 121"` (correct math)
- `task_specific_assessments.phases_detected` populated with task-2-relevant phases
- empty arrays for fields it cannot ground (no hallucinated penalties)
- v003 fields all present: `critical_errors`, `cannot_determine`, `confidence_rationale`, `confidence_score`, `score_components`

The score values are still rough (3-epoch LoRA on 193 examples), but the **structure and gating are correct** — the report pipeline (`generate_report_v3`) can now consume model output directly without post-processing repair.

## Artifacts

| File | Description |
|---|---|
| `v17_metrics.json` | Final training metrics |
| `v17_adapter_config.json` | PEFT LoRA config |
| `v17_train.log` | Full training log |
| `v003_multimodal_manifest.json` | Dataset manifest (task distribution, vision count, drops) |

The 381 MB adapter `safetensors` lives on the pod at `/workspace/v17_lora_output/final_adapter/`. Pull with:

```bash
scp -P 14212 -r root@213.181.104.34:/workspace/v17_lora_output/final_adapter ./adapters/v17_v003
```

(or run `python deploy/runpod_v003_launch.py status` to confirm the pod IP/port.)

## Next iterations

1. **More data**: only 239 of 642 unique scored videos survived task-resolution filters. Backfill the harvest CSV / re-classify the unclassified scores to recover the rest (would push training set past 500).
2. **Resume from v17**: subsequent iterations should pass `--base-adapter /workspace/v17_lora_output/final_adapter` to continue rather than restart.
3. **Frame extraction**: only 75 videos had cached frames (~30%). Re-run frame extraction over the full 642-video corpus to get vision coverage on every example.
4. **Report LoRA**: `scripts/063_train_report_lora_v3.py` is wired but not yet executed — needs `scripts/060/061/062` artifacts on the pod.
