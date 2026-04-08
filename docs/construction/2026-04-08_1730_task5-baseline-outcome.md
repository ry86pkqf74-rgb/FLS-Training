---
name: task5-baseline-runpod-outcome
description: Post-mortem for the first Task5 baseline launch on fls-training-hq (RunPod RTX PRO 6000 Blackwell)
date: 2026-04-08
time: "17:30-17:49 UTC"
status: shipped
authors: [claude-cowork, logan]
related_commits: [b83161a]
related_files:
  - src/configs/finetune_task5_standard.yaml
  - deploy/runpod_launch.sh
  - src/training/finetune_vlm.py
  - data/training/2026-04-08_v4/
supersedes: 2026-04-08_1305_task5-baseline-runpod-launch.md
superseded_by: null
---

# Task5 Baseline — RunPod Launch Outcome

## Overview

First end-to-end Task5 baseline fine-tune launched on the existing `fls-training-hq` RunPod pod (cr03n8zxq9g2dr, RTX PRO 6000 Blackwell Server Edition, 97 GB VRAM). Run completed in 5.7 minutes wall-clock. Final adapter is saved. **The run is pipeline-valid but scientifically void** — see findings.

## Context

The 13:05 launch plan assumed a new pod would be created with RunPod credentials. In fact a pod was already running from earlier in the day and (per the `ps` snapshot at session start) had been auto-restarting stale `v2_diverse` training via a watchdog loop that was doing no real work — the GPU read **0% / 0 MiB / 91 W** the moment SSH came up. Pivoted: use the existing pod, clean house, launch the correct Task5 standard config on the v4 dataset, then report.

## Timeline (UTC)

| t | event |
|---|---|
| 17:15 | RunPod API key verified; `currentSpendPerHr` = $1.514/hr from pre-existing pod |
| 17:16 | SSH via `researchflow_cluster` key direct-to-IP; found stale `finetune_task5_v2` ghost procs, GPU idle |
| 17:20 | Confirmed `FLS-Training-main` clone is clean at `origin/main` b673b13 with b83161a fix present |
| 17:22 | Smoke-test log `/tmp/fls_smoke.log` reviewed: 13 passed, 2 failed (B2 upload, HF_TOKEN) — both non-blocking for baseline |
| 17:25 | First launch attempt; nohup detach didn't survive; pod transitioned to `EXITED` (spot preemption) |
| 17:29 | Resumed pod via `podBidResume` at $1.50 bid; new SSH port `12731` |
| 17:32 | Disk alarm: pod-volume 50 GB cap hit. `du -sh` showed 32 G + 16 G across two repo clones, most in `memory/model_checkpoints/v2_diverse/merged_16bit` exports |
| 17:33 | Cleanup: removed 30 GB of `merged_16bit` dirs + 3.3 GB of stale intermediate checkpoints. Total pod-volume usage 48 G → 16 G |
| 17:38 | Launched `bash deploy/runpod_launch.sh data/training/2026-04-08_v4 src/configs/finetune_task5_standard.yaml` with `nohup ... </dev/null >log 2>&1 & disown` |
| 17:40 | `Num examples = 144 / Num Epochs = 5 / Total steps = 90` — b83161a fix confirmed |
| 17:42 | Step 1/90; GPU 100% / 13.8 GB / 464 W |
| 17:45 | Step 43/90, 3.5 s/step; ETA ~3 min |
| 17:48 | Training complete, train_runtime 342.2 s |
| 17:49 | Adapter saved, `merged_16bit` export failed (known unsloth LoRA-count bug), broken merged dir reclaimed (16.6 GB) |

## Results

### Eval loss trajectory
```
epoch 1: eval_loss 1.605
epoch 2: eval_loss 0.7841
epoch 3: eval_loss 0.0541
epoch 4: eval_loss 0.003652
epoch 5: eval_loss 0.002268   <- final
train_loss (step avg): 1.111
```

### Artifacts on pod
```
/workspace/FLS-Training-main/memory/model_checkpoints/task5_standard/
├── adapter_config.json
├── adapter_model.safetensors     (727 MB; canonical LoRA adapter)
├── checkpoint-90/                (final HF-style checkpoint)
├── chat_template.jinja
├── processor_config.json
├── run_manifest.json
├── tokenizer.json
├── tokenizer_config.json
└── README.md
```
Total: 1.8 GB (after `merged_16bit` reclaim).

Log: `/workspace/logs/task5_baseline_20260408_1738.log`

## Findings

### F1 — Run is degenerate (eval_loss 0.002 is not a real result)

The trainer log reports `Dataset vision mode: False` and `Tokenizing [""] (num_proc=64)` with 144 empty-string entries. A `train.jsonl` sample inspection shows each training example has a ~2 KB deterministic system prompt containing the full FLS rubric, a user message (presumably with image content blocks being dropped because vision is off), and an assistant message containing a highly stereotyped scoring JSON. With the image channel removed, the trainer is learning to map `(rubric → assistant JSON)` with zero surgical-video signal. The eval set uses the same template, so loss collapses.

**Why this matters:** `eval_loss` of 0.002 on a VLM surgical-skill task after 5 minutes of training on 144 examples is not plausible under any reading of the literature. The saved adapter is **not a useful baseline** and must not be used for evaluation, promotion, or as a warm-start for the LASANA pretrain. The run's value is purely that it proved the pipeline walks end-to-end.

**Root cause (hypothesis, needs confirmation):** `src/training/finetune_vlm.py` and/or the data-collation layer has a branch that falls back to text-only when the image content blocks can't be resolved (path missing, unsupported format, or the processor dropping them). VSC agent should trace why `vision_mode` is False when `train.jsonl` clearly includes image content blocks.

**Block until fixed:** do not run another fine-tune until `vision_mode: True` can be confirmed at the top of the trainer log on a dry-run.

### F2 — Blackwell branch in `deploy/runpod_launch.sh` silently overrides `save_strategy`

`deploy/runpod_launch.sh` lines 291–297 (embedded python block):

```python
if "Blackwell" in gpu_name:
    config["batch_size"] = int(os.environ.get("BLACKWELL_BATCH_SIZE", ...))
    config["gradient_accumulation"] = int(os.environ.get("BLACKWELL_GRAD_ACCUM", ...))
    config["dataloader_num_workers"] = max(...)
    config["lora_dropout"] = float(...)
    config["save_strategy"] = "steps"                                  # unconditional
    config["save_steps"] = int(config.get("save_steps", 200))          # default 200
    config["save_total_limit"] = int(config.get("save_total_limit", 2))
```

Because the pod GPU name contains "Blackwell", this branch fires and rewrites the yaml's `save_strategy: "epoch"` to `"steps"`, with `save_steps = 200`. With `num_epochs = 5` × 18 steps/epoch = 90 total steps, **zero intermediate step-checkpoints are ever written**. HF Trainer's terminal `save_model()` still fires, so a final `checkpoint-90/` dir landed, but the run produced no best-of-eval checkpoint.

**Proposed fix:**
```python
if "Blackwell" in gpu_name:
    config["batch_size"] = int(os.environ.get("BLACKWELL_BATCH_SIZE", ...))
    config["gradient_accumulation"] = int(os.environ.get("BLACKWELL_GRAD_ACCUM", ...))
    config["dataloader_num_workers"] = max(...)
    config["lora_dropout"] = float(...)
    # Respect the source config's save_strategy. Only force "steps" if the
    # source didn't specify. If the caller asked for "epoch", leave it alone.
    config.setdefault("save_strategy", "steps")
    if config["save_strategy"] == "steps":
        config.setdefault("save_steps", 200)
    config.setdefault("save_total_limit", 2)
```

### F3 — Unsloth `merged_16bit` export fails every run with LoRA module-count mismatch

Both the v2_diverse run (4:41 AM) and this Task5 run hit:

```
WARNING: Merged 16-bit export failed; keeping adapter checkpoint only:
  Unsloth: Saving LoRA finetune failed since # of LoRAs = 292 does not match # of saved modules = 0.
  Please file a bug report!
```

The export reaches the point of downloading all four base safetensor shards to `merged_16bit/` (~16.6 GB) before crashing on the LoRA merge step. The downloaded shards are *unmerged base weights*, not a merged model — they are useless and consume the pod's 50 GB quota. Action items:

1. Catch the exception earlier and skip the base-weight download when merging is known-broken.
2. Either upstream a bug report to unsloth with a minimal repro from our Qwen2.5-VL-7B config, or switch to HF Trainer's native PEFT merge path.

Until then, **runs must cleanup `merged_16bit/` in a `trap` handler** so the 16 GB blob doesn't linger.

### F4 — Pod is spot-priced and can preempt mid-run

`podResume` with the basic mutation returned `Cannot resume a spot pod as an on demand pod`. Required `podBidResume` with an explicit `bidPerGpu: 1.5`. This is a spot/interruptable pod — which is why it exited mid-session without warning. For a 5-minute Task5 baseline this is a non-issue; for the LASANA Stage-1 pretrain (multi-hour) it will bite us. Options: (a) create an on-demand pod for the pretrain, (b) stay on spot and wire `resume_from_checkpoint` + a supervising loop, (c) switch to a reserved-price lock.

### F5 — Pod volume is capped at 50 GB and approaches the limit quickly

- v2_diverse run left 32 GB across duplicated checkpoint dirs + `merged_16bit` exports
- Every run the `merged_16bit` export re-downloads 16 GB of base weights before crashing
- HuggingFace cache in `/root/.cache` is currently small but will grow

Mitigation: add a post-run cleanup step in `deploy/runpod_launch.sh` that removes `merged_16bit/` when the merge fails, prunes intermediate checkpoints beyond `save_total_limit`, and vacuums stale HF cache shards.

## Decisions

1. **Do not use this checkpoint.** Do not load `memory/model_checkpoints/task5_standard/adapter_model.safetensors` for evaluation, promotion, or as the Stage-2 resume target. It is degenerate.
2. **Block further training until `vision_mode: True` is confirmed** on a dry-run.
3. **File the Blackwell save_strategy override as a repo bug** — VSC agent to patch `deploy/runpod_launch.sh` per the diff above.
4. **Keep the current pod for the LASANA pretrain window** but plan to create an on-demand pod before Stage-1 so a preempt doesn't eat multi-hour work.
5. **Keep the pod cleanup sequence as a standing habit** until `merged_16bit` cleanup is automated.

## Follow-ups

- [ ] VSC agent: debug why `Dataset vision mode: False` on a jsonl that contains image content blocks
- [ ] VSC agent: apply `setdefault` fix to the Blackwell branch in `deploy/runpod_launch.sh`
- [ ] VSC agent: add trap-based cleanup for failed `merged_16bit` exports
- [ ] Logan: rotate the RunPod API key that was exposed in chat
- [ ] Logan: decide on-demand vs. spot pricing for LASANA Stage-1 pretrain pod
- [ ] After vision-mode fix lands: rerun Task5 baseline to get an actual first data point; keep this run's log alongside as the negative-control

## Outcome

- Task5 baseline pipeline proved to walk end-to-end: config → launch → dataset load → LoRA init → 90 training steps → per-epoch eval → final adapter save. All stages green.
- The saved adapter is degenerate and should not be used.
- Three repo bugs surfaced (vision_mode, Blackwell save_strategy override, unsloth merge failure) — each with a concrete fix recommended.
- Cost: ~5 min of RTX PRO 6000 Blackwell at $1.50/hr ≈ **$0.15** training time + overhead for the preempt/resume cycle.
