---
date: 2026-04-08
time: 13:05 EDT
status: proposed
authors: [logan, claude]
related_commits: [b83161a]
related_files:
  - src/configs/finetune_task5_standard.yaml
  - deploy/runpod_launch.sh
  - smoke_test.sh
supersedes: null
superseded_by: null
---

# Task5 standalone baseline — RunPod launch

## Overview

Launch a standalone Stage-2 fine-tune of Qwen2.5-VL-7B + LoRA on the v4
Task5 corpus (144 train / 27 val / 9 test) on RunPod, **in parallel with**
the LASANA pretrain pipeline. This is the baseline number we will compare
against the LASANA-pretrained Stage 2 once that lands. Running it now is
cheap (~$15), independent of LASANA, and fits exactly in the ~3-hour
window while the LASANA streaming pipeline runs unattended.


## Context

- Config `src/configs/finetune_task5_standard.yaml` was fixed in commit
  `b83161a` earlier today. Per-epoch eval and save strategies are now in
  place. 5 epochs on 144 train samples = ~90 optimizer steps. See
  `2026-04-08_1015_task5-config-fix.md` for the full rationale.
- Dataset `data/training/2026-04-08_v4` is ready: 144 train / 27 val / 9
  test, manifest verified, sources = critique_consensus + teacher_claude +
  teacher_gpt4o.
- The LASANA streaming pipeline (`2026-04-08_1040_path-b-lasana-pretrain.md`
  outcome update) is fully automated and expected to complete at
  ~T+1.76h. During this window no human attention is needed on the LASANA
  path. This is the window where the baseline run should fit.
- RunPod template is the same proven A100 80GB / RTX PRO 6000 path from
  the April run. No new infrastructure work.

## Why now, specifically

Three non-obvious reasons the *timing* matters, not just the run itself:

1. **Timing dominates.** Running this baseline later costs the same $15
   but arrives later. Running it now while the LASANA pipeline is
   self-driving has zero opportunity cost on attention.
2. **Pipeline validation at scale.** The CPU smoke test proves the
   pipeline doesn't crash; it does not prove eval metrics are sane or
   checkpoint files are valid on a real GPU. A real run catches pipeline
   bugs at the earliest possible moment, before Stage 1 launches.
3. **Required for comparison.** Stage 2 with a LASANA checkpoint is only
   interesting relative to Stage 2 without one. We need this number
   either way. Running it before the LASANA pretrain lands means we have
   the comparison ready the moment the pretrain finishes.

## Decision

Launch the standalone Task5 baseline on RunPod in the window starting
now, **only after** the pre-launch checklist below is fully green.
Do **not** launch without completing every item in order.

## Pre-launch checklist — do not skip

All items are required. Any failure aborts the launch.

### 1. CPU smoke test on the Mac (~30 min, $0)

```bash
cd ~/Downloads/FLS-Training
# The smoke harness lives in the Cowork workspace folder
bash /path/to/cowork/FLS/smoke_test.sh data/training/2026-04-08_v4
```

Expected output: `[smoke] PASSED — pipeline is wired end-to-end.` plus a
list of artifacts under `/tmp/smoke_run/`. If this prints anything else,
**stop and debug locally before spending a cent on the pod**.

### 2. RunPod spending cap

In RunPod billing dashboard, set a **hard spending cap of $150** before
booting the pod. Not after. Screenshot the cap for the construction
record.

### 3. Calendar reminders

Before booting the pod, set two reminders on your phone or calendar:

- **T+5h** — "Check FLS baseline run — eval curve healthy?"
- **T+8h** — **"STOP THE FLS POD NOW"** (in caps, intentional)

The T+8h reminder is a hard commitment. Idle pods are the most expensive
thing in AI. A 4-hour run should not be left running for 10.

### 4. Pod template sanity

Confirm:
- A100 80GB or RTX PRO 6000 Blackwell 97GB
- `outputs/` persists to a volume, not pod-local disk
- At least 150 GB disk allocated for the base model + checkpoints

### 5. Verify `main` is current on the pod after clone

Commit hash on pod must include `b83161a` (config fix) at minimum. The
simple test: `git log --oneline | head -5` on the pod should match
`git log --oneline origin/main | head -5` on the Mac.

## Launch commands

Only run these after every checklist item is green.

```bash
# On the RunPod pod, after SSH
cd /workspace
git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git
cd FLS-Training
bash scripts/runpod_setup.sh

# Verify dataset is present (rsync it in from Contabo if not)
ls data/training/2026-04-08_v4/

# Launch
bash deploy/runpod_launch.sh \
    data/training/2026-04-08_v4 \
    src/configs/finetune_task5_standard.yaml \
    2>&1 | tee /workspace/logs/task5_baseline_$(date +%Y%m%d_%H%M).log
```

## Monitoring plan

Watch for these at each epoch boundary (logs will print them, since
`eval_strategy: "epoch"` is now set correctly):

- **mae_fls_score** — should decrease over epochs. If flat or rising
  after epoch 3, the run is overfitting and stopping early is fine.
- **train_loss** — should decrease monotonically or near-monotonically.
  Sudden spikes mean a bad batch or LR instability.
- **GPU memory utilization** — should be stable. If it creeps up each
  step, there's a memory leak somewhere and the run will OOM later.
- **Checkpoint writes** — should see one per epoch under `outputs/`.
  If none appear, the config patch did not actually take effect and
  you need to abort.

## Kill criteria (stop the run if any of these hit)

- Any epoch produces `mae_fls_score > 25` (well above the teacher-vs-
  teacher noise floor of 21.6 — it means the model is worse than random
  guessing within the teacher distribution).
- GPU OOM at any epoch.
- `train_loss` rises by >50% between consecutive log windows.
- Wall clock exceeds 6h (the run is supposed to finish in ~3-4h; 6h
  means something is wrong).
- T+8h calendar alarm regardless of state. Hard stop.

## Success criteria

A reviewer should be able to say "yes, this run produced usable signal"
if all of the following are true:

- All 5 epochs completed without kill-criteria triggers.
- At least one checkpoint was saved to the persistent volume.
- Final `mae_fls_score` is below 22 (at or under the teacher noise
  floor — anything better is real signal; equal means the baseline is
  matching inter-rater variance which is already meaningful).
- Final `f1_penalty_detection` is above 0.3 (sanity check that the
  model learned *something* about penalties, not just averaged scores).

## Outcome

_Filled in after the run finishes. Record: pod ID, actual wall time,
actual cost, final eval metrics per epoch, checkpoint path, any
deviations from the plan above._
