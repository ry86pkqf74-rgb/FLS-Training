# FLS Round 2 Readiness — 2026-04-15 (post v2 run)

Supersedes the planning-side of `READINESS_2026-04-14.md`, which was written
before v2 ran. The pre-v2 analysis there (LASANA mode-collapse, data schema
review, harvest gap) is still accurate; only the §7–8 "what to ship tonight"
plan is stale.

## 1. Status snapshot

- v2 adapter (`041_train_qwen_vl.py`, run `fls_round2_vl_20260414_034645`):
  **gate FAIL**. 15 epochs / 180 steps / train_loss 6.42 / eval_loss 5.69.
  - task_id accuracy 80% (4/4 on rows with frames) — model learned which task
  - valid_json_rate 80% — 1/5 test rows had no frames (yt_IwrNTRVXuJQ)
  - unique_pred_ratio 0%, mae_n 0 — `score_components` emitted as LIST, not DICT
  - Full diagnosis: `memory/model_checkpoints/yt_vl_v1/DIAGNOSIS.md`
- RunPod pod `v7lexoj3odzen4` EXITED (billing stopped). Adapter weights are gone
  with the pod; v2 is not warm-startable from retained storage.
- GitHub: branch `round2-vl-pipeline-fixes` @ `03e3def` has every fix + the
  failed run's manifest, eval_results, DIAGNOSIS, and log tails.

## 2. Why v2 failed (one-line)

45 train examples × 180 optimizer steps wasn't enough to teach the base model
the nested `score_components` dict shape; LR decayed to near-zero on a cosine
schedule while loss was still at 5+.

## 3. v3 plan — `scripts/043_train_qwen_vl_v3.py` (ready to launch)

| Knob | v2 | v3 | Why |
|---|---|---|---|
| epochs | 15 | 50 | 180 → ~600 optimizer steps |
| lr_scheduler | cosine | cosine_with_min_lr (min 10%) | don't die out on plateau |
| system prompt | describes schema | **embeds a skeleton JSON** with the exact score_components dict | anchors model to the right shape |
| preflight | none | prints schema shape, aborts if `score_components` isn't a dict | catches data-shape drift before a 40min burn |
| prep filter | skip no-frames | **also skip empty task_id and list-shaped score_components** | don't poison training with unclassified or schema-mismatched rows |

Everything else is unchanged — same base model, LoRA config, quantization,
data layout. Pod launch procedure is identical to v2; just
`python3 scripts/043_train_qwen_vl_v3.py`.

Expected wall time on H100 80GB: ~40min train + ~5min eval, ~$2 of GPU.

## 4. Parallel lever: expand the SFT pool

Training set is the real ceiling. The repo has 389 scored videos but only 101
pass dual-teacher consensus, and only 56 of those had frames on the v2 pod.
Two cheap ways to grow it before the next pod run:

- **Classify the 45 `task_id=""` rows in youtube_sft_v1**: a one-shot Haiku
  call against title + description + 1 thumbnail should hit >90%. This alone
  roughly doubles the usable train set without new teacher spend.
- **Rescore the 228 Claude-FLS=0 performance videos**: if they failed on
  transient API errors (not because they're non-performance), each recovered
  row is free data. Spot-check a handful first; if the FLS=0 failures are
  genuinely non-performance content, skip this.

Both are pre-pod local work; neither needs GPU. Doing them before v3 lands
could push the usable train set to ~130+, which is closer to the floor where
structured SFT tends to converge cleanly.

## 5. Gate policy (unchanged)

`042_eval_vl_adapter.py` gates: valid_json_rate > 0.9 AND unique_pred_ratio > 0.5.
Exit 1 if either fails. These are the right gates — v2 tripped them correctly.
Only adjustment the eval itself might want: drop the no-frames test row
(`yt_IwrNTRVXuJQ`) from `yt_test.jsonl` so we're measuring model behavior, not
data plumbing.

## 6. Decision tree for the next session

```
Before launching v3 pod:
  - (optional, free)  classify the 45 task_id="" rows to grow train set
  - (optional, cheap) rescore Claude=0 videos if they look salvageable
  - (recommended)     fix yt_test.jsonl to drop the no-frames row

Launch v3 (~$2/run on H100):
  - preflight aborts if schema mismatches → no wasted GPU
  - if gates pass: scp adapter to storage, promote to production
  - if valid_json passes but unique_pred still <50%:
      → real mode collapse, need more data (go back to §4)
  - if valid_json still <90%: model is still learning syntax, not schema —
      reduce MAX_FRAMES to 4, or unfreeze vision tower briefly
```

## 7. Deferred items (still open from pre-v2 plan)

- Harvest classifier for 309 unclassified `harvest_targets` URLs
- Scripts 044–047: iterative teacher/critique loop (only worth doing once v3
  produces a usable base adapter)
- Rebalance harvest toward task 4 (currently only 42 videos — bottleneck)

## Appendix — artifacts on `round2-vl-pipeline-fixes`

- `scripts/041_train_qwen_vl.py` — v2 (gate fail, superseded)
- `scripts/042_eval_vl_adapter.py` — eval + diversity gate (works; minor
  test-set cleanup recommended)
- `scripts/043_train_qwen_vl_v3.py` — v3, ready to launch
- `memory/model_checkpoints/yt_vl_v1/` — v2 artifacts + DIAGNOSIS.md
- `READINESS_2026-04-15.md` — this file
