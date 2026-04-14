# v7 Training Status — 2026-04-14

## In flight
- **Pod:** `v7lexoj3odzen4` (fls-train-h100-v2), ssh `root@87.120.211.209 -p 14632`
- **Process:** PID 37561 on pod, `python3 -u FLS-Training/scripts/050_train_qwen_vl_v7.py`
- **Log:** `/workspace/trainv7.log`
- **Started:** 2026-04-14 17:22 UTC
- **Progress @ 2026-04-14 18:54 UTC:** step 246/560 (44%), epoch 2.14, ~1h55m remaining
- **First checkpoint save:** step 400 → `/workspace/checkpoints_vl_v7/checkpoint-400/`
- **Final save:** `/workspace/checkpoints_vl_v7/final/`

## Loss trajectory (v7 vs v6)

| step | v6 loss (broken) | v7 loss (fixed) |
|-----:|-----------------:|----------------:|
|  10  | 15.14            | 1.065           |
|  30  |  9.46            | 0.348           |
|  60  |  5.63            | 0.205           |
|  80  |  5.58 (plateau)  | n/a             |
| 150  | —                | 0.149           |
| 240  | —                | 0.099           |

v7 is 55x below v6's plateau floor. grad_norm stays in 0.2-0.5 range (healthy,
active). Mask fix confirmed working.

## Mask sanity check (printed at startup)
```
total_tokens=2476  prompt_tokens=2223  supervised_tokens=253
```
90% of tokens in v4/v5/v6 training were prompt/image tokens contributing to
loss — model memorized them instead of learning the response. v7 masks them.

## Next steps (for new session to execute)

1. **Monitor training** via the SSH-agent socket (see handoff prompt).
2. **When training completes** (check for `/workspace/checkpoints_vl_v7/final/`
   and the `=== Training complete` line in trainv7.log):
   - scp the final/ adapter to local `memory/model_checkpoints/round2_vl_v7/`
   - copy `run_manifest.json` + last ~200 lines of trainv7.log
3. **Run eval:**
   ```
   ssh root@87.120.211.209 -p 14632 "cd /workspace && python3 FLS-Training/scripts/042_eval_vl_adapter.py --adapter checkpoints_vl_v7/final --out checkpoints_vl_v7/eval_results.json"
   ```
4. **Inspect eval_results.json:**
   - Must pass: `valid_json_rate > 0.9` AND `unique_prediction_ratio > 0.5`
   - v5/v6 failed the unique_prediction_ratio gate (both 0%). v7 should clear it.
5. **Commit v7 artifacts** to `round2-vl-pipeline-fixes`:
   - `memory/model_checkpoints/round2_vl_v7/run_manifest.json`
   - `memory/model_checkpoints/round2_vl_v7/eval_results.json`
   - `memory/training_runs/round2_vl_v7.log` (training tail)
   - Push and tag `round2-vl-v7`
6. **Stop the pod** once eval is committed (~$2.99/hr idle H100).

## If v7 fails the gate
Possible remaining causes in rank order:
- Image tokens not contributing to learning through 4-bit quantization of
  the vision encoder → try `lora_target_modules` that include vision proj
  layers (`merger.*`, `patch_embed`), or unfreeze vision encoder.
- Data too easy (LASANA synthetic dominates) → oversample real YouTube
  task3/task4 which have tiny N (12 and 8).
- `eval_results.json` only measures JSON validity / uniqueness — add a
  score-accuracy metric (MAE against target.score_components.total_fls_score).

## SSH gotcha (important for new session)
Desktop Commander's default shell env doesn't see the system ssh-agent.
Before any SSH command, export:
```
export SSH_AUTH_SOCK=/var/run/com.apple.launchd.NWjpYCNETm/Listeners
```
(Exact socket path may differ across reboots; find with
`launchctl getenv SSH_AUTH_SOCK`.)
