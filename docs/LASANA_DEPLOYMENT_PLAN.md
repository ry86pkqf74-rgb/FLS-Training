# LASANA + Multi-Dataset GPU Deployment Plan

> ## 🟡 PARTIALLY GATED 2026-04-08 — hardening sprint code-complete
>
> Code gates CLEARED (2026-04-08):
>   - `src/training/schema_adapter.py` is the single source of truth for
>     v001/v002 score records; `eval_v2.py` reads scores through it.
>   - `src/training/data_prep.py` + `runpod_trainer.py` +
>     `scripts/050_runpod_train.py` are quarantined under
>     `archive/deprecated/` with stubs that raise `RuntimeError`. The
>     live training path is `src/training/prepare_dataset.py` →
>     `src/training/finetune_vlm.py` → `deploy/runpod_launch.sh`.
>   - `prepare_dataset.py` now emits real image content blocks
>     (`[{"type":"image","image":path}, {"type":"text",...}]`) from
>     `--frames-dir`, attaches `trainee_id` + `source_domain` metadata,
>     and supports resident-aware splits via `--group-by trainee`.
>   - `finetune_vlm.py` detects vision mode from the first example,
>     switches to `FastVisionModel.for_training` +
>     `UnslothVisionDataCollator`, and aborts before GPU allocation when
>     `require_vision: true` and no image blocks are present.
>   - `deploy/runpod_launch.sh` resolves `LATEST` to the newest
>     `data/training/*_v*` dir and runs an inline vision-content +
>     image-path-exists check before touching the GPU.
>   - `scripts/runpod_smoke_test.sh` asserts `schema_adapter` is
>     functional and the quarantined `data_prep` stub raises, so a
>     silent revert can't sneak onto a pod.
>
> Data gate STILL OPEN: the locked gold set built by
> `scripts/047_build_gold_set.py` currently contains only 5 videos
> (41 no_consensus, 16 single_teacher, 11 teacher_disagreement,
> 5 low_confidence out of 78 candidates). Before launching Phase 3,
> either (a) run `scripts/025_rescore_missing_gpt.py --execute`
> (~$1.45 for the 29 Claude-only videos) and rebuild the gold set, or
> (b) gate-train on the LASANA/PETRAW/SimSurg corpus where labels come
> from the source datasets rather than reconciled teacher scores.
>
> Baseline teacher-vs-teacher MAE is **≈21.6** (see
> `memory/baselines/2026-04-08_teacher_mae_baseline.md`). The "MAE > 12"
> abort threshold below was set blind and sits below the teacher noise
> floor. Revised thresholds are at the bottom of §11.

> **Status:** plan, not yet executed.
> **Budget ceiling:** **$200 total**, target spend **$70–110**.
> **Companion docs:** `docs/RUNPOD_RUNBOOK.md` (proven April 2026 path),
> `docs/DATA_SCALING_PLAN.md` (corpus strategy),
> `data/external/LASANA_README.md`, `data/external/PETRAW_README.md`,
> `data/external/JIGSAWS_README.md`, `data/external/SIMSURGSKILL_README.md`.
> **Date written:** 2026-04-08.

---

## 1. Why this document exists

`docs/RUNPOD_RUNBOOK.md` already covers the proven Blackwell + A100 RunPod
fine-tune path that worked in April 2026. Since then, the corpus has
grown from 31 single-trainee videos to four research datasets:

| Dataset       | Status                                  | Bytes  | Has labels? |
|---------------|-----------------------------------------|--------|-------------|
| LASANA        | download in flight on Contabo + Hetzner | 185 GB | yes (z-GRS) |
| PETRAW        | downloading on Mac (Synapse)            | 24 GB  | yes (workflow + segmentation) |
| SimSurgSkill  | downloaded + unzipped on Mac            |  5 GB  | yes (bbox + orientation) |
| JIGSAWS       | metadata-only (no video archive exists) | <5 MB  | yes (GRS, kinematics) |

The runbook does not address (a) where to ingest 185 GB of HEVC video,
(b) whether to do feature extraction on the same GPU pod that does
training, or (c) Vast.ai vs RunPod cost-efficiency under the new
$200 ceiling. This doc fills those gaps and is **additive** to the
runbook — when it comes time to launch the actual fine-tune, you still
follow `RUNPOD_RUNBOOK.md` step-by-step.

---

## 2. TL;DR

1. **Two-phase architecture.** Cheap CPU/storage pod for LASANA download
   + frame decode + feature extraction. Separate single-GPU pod for
   training only. Never let a GPU clock tick while bytes are moving over
   the network or while ffmpeg is decoding HEVC on the CPU.
2. **A100 80GB on Vast.ai** for training, not H100 SXM on RunPod.
   At April 2026 marketplace rates the A100 is roughly half the price
   per training hour, and the workload (LoRA fine-tune of a 7B-class
   VLM on ~30k frame-text pairs) does not saturate H100 throughput.
3. **RunPod stays the canary.** The proven Blackwell launcher
   (`deploy/runpod_launch.sh`) already handles dependency hell on
   RunPod. We keep RunPod as the **fallback** path in case Vast.ai
   marketplace listings dry up or a host is unreliable.
4. **Persist artifacts to a network volume**, not the pod local disk,
   so a forgotten shutdown doesn't lose work.
5. **Hard budget alarms.** Provider spending limit set to $150, calendar
   reminders every 4h, and an explicit abort criterion: if held-out
   MAE > 12 after two training runs, stop spending and harvest more
   data instead of throwing more GPU at the problem.

---

## 3. Workload inventory — what actually needs a GPU

| Workload                                | Where           | GPU needed? | Why |
|-----------------------------------------|-----------------|-------------|-----|
| LASANA bitstream download (185 GB)      | CPU pod / Mac   | no          | network-bound |
| PETRAW + SimSurgSkill ingest            | Mac (in flight) | no          | already running |
| HEVC stereo decode → frames             | CPU pod         | no          | ffmpeg+libx265 is CPU-bound |
| DINOv2 / CLIP feature extraction        | small GPU       | yes (T4/A10)| 30k frames, embarrassingly parallel |
| Tool tracking (YOLO) on extracted frames| S6 (Hetzner)    | no          | proven CPU path |
| Regression-head training on cached features | CPU or T4   | barely      | <1h on CPU is plausible |
| LoRA fine-tune of VLM (Qwen2-VL-7B)     | A100 80GB       | yes         | the only real GPU job |
| Eval / inference sweep                  | same A100       | yes         | reuses same pod |
| Final model export                      | same A100       | no          | bookkeeping |

The **only workload that actually justifies GPU time** is the VLM LoRA
fine-tune. Everything upstream is CPU-bound or trivially small. The
common money-burn pattern is to do all of this on a single H100 pod
because "we already paid for it" — that pattern is exactly how
$200 budgets become $400 budgets.

---

## 4. Provider comparison (April 2026 rates)

Rates pulled 2026-04-08 from RunPod, Vast.ai, ComputePrices, Thunder
Compute, Northflank, and Jarvislabs blogs. Marketplace rates fluctuate
hour-to-hour; treat these as "what to expect," not guarantees.

| GPU                | RunPod community | RunPod secure | Vast.ai marketplace floor |
|--------------------|------------------|----------------|---------------------------|
| RTX 4090 24GB      | $0.34–0.44/hr    | $0.69/hr       | $0.18–0.30/hr             |
| L40S 48GB          | $0.79/hr         | $1.19/hr       | $0.50–0.80/hr             |
| A100 80GB PCIe     | $1.64/hr         | $1.89/hr       | **$0.74–0.90/hr**         |
| A100 80GB SXM4     | $1.79/hr         | $2.09/hr       | **$0.78/hr**              |
| H100 80GB PCIe     | $1.99/hr         | $2.39/hr       | **$1.49–2.00/hr**         |
| H100 80GB SXM      | $2.69/hr         | $2.99/hr       | $1.99–2.50/hr             |
| H200 / GB200       | $3.39+/hr        | $3.99+/hr      | $2.50+/hr                 |
| RTX PRO 6000 Blackwell | $1.89/hr     | varies         | $1.50–2.00/hr             |

**Storage and transfer:**

- RunPod network volumes: $0.07/GB/month, must be in same datacenter as pod.
- RunPod egress: free up to typical thresholds.
- Vast.ai: storage is per-host; budget for "bring your own" workflow
  with `rsync` between hosts. Offload artifacts to S6 or Backblaze B2
  ($0.005/GB/month) when the pod is done.

### Honest pros/cons

**RunPod**

- ✅ Proven path: `deploy/runpod_launch.sh` already handles Blackwell
  PyTorch quirks, Unsloth → hf_trainer fallback, Flash-Attention
  skip-on-resume, watchdog/resume.
- ✅ Network volumes survive pod shutdown → no `rsync` dance.
- ✅ Cleaner UI, better abort safety, per-second billing.
- ❌ ~30–60% more expensive than Vast.ai for the same silicon.
- ❌ Datacenter-locked volumes constrain pod placement.

**Vast.ai**

- ✅ Cheapest A100 80GB on the open market ($0.74/hr vs $1.64).
- ✅ The user already has V1 running there → SSH pattern is established.
- ✅ The launcher script is provider-agnostic; same `bash deploy/runpod_launch.sh`
  call works.
- ❌ Host quality varies. Some listings have slow disks, throttled CPU,
  or unreliable network. Read host reviews before launching.
- ❌ No first-class network volumes — checkpoint persistence has to
  be solved with `rsync` to S6 or to a Backblaze bucket on shutdown.
- ❌ Spot instances exist but are off the table — preemption mid-train
  loses the run.

### Decision

**Primary path: Vast.ai A100 80GB on-demand.** Save ~$10/training hour
vs the equivalent RunPod box. Cost savings funded directly from picking
the cheaper marketplace.

**Fallback path: RunPod A100 80GB on-demand** if Vast.ai listings are
poor quality or no usable hosts are available within 30 minutes of
search. The proven launcher works on either provider; only the SSH
endpoint and storage layout change.

**Rejected paths:**

- ❌ H100 SXM on RunPod ($2.69/hr) — only worth it if we were doing full
  backbone fine-tuning of a 30B+ model, which we are not.
- ❌ H200 / GB200 — overkill, eats 30%+ of the entire budget per hour.
- ❌ Multi-GPU — 30k samples won't benefit from distributed training and
  the synchronization overhead would dominate.
- ❌ Spot / interruptible instances — preemption mid-train wastes the
  whole run; a single preemption costs more than the savings on a
  4–6 hour job.

---

## 5. The "go big to go cheap" question, answered honestly

The naive intuition is "rent the biggest box, finish faster, save
money." It's only true when:

1. The workload is **GPU-throughput-bound**, not data-loader-bound
   or memory-bandwidth-bound. (LoRA fine-tunes on small batches are
   often dataloader-bound.)
2. The **bigger box is more than proportionally faster** at your
   specific workload. (H100 is ~2x A100 on FP16 matmul, but VLM
   fine-tuning rarely sees the full 2x because of attention overhead
   and Python-side bottlenecks.)
3. **Setup time amortizes**. If you spend 30 minutes installing
   dependencies on a $2.69/hr pod, that's $1.35 of pure dependency tax
   regardless of how fast training is.

Concrete back-of-envelope for a Qwen2-VL-7B LoRA fine-tune on
~30,000 frame-text pairs, 2 epochs:

| Box                 | Throughput est | Train hours | $/hr  | Train cost | Setup tax | Total |
|---------------------|----------------|-------------|-------|------------|-----------|-------|
| Vast.ai A100 80GB   | 1.0× (baseline)| 5.0         | $0.78 | $3.90      | $0.39     | **$4.29** |
| RunPod A100 80GB    | 1.0×           | 5.0         | $1.64 | $8.20      | $0.82     | $9.02 |
| Vast.ai H100 PCIe   | 1.6×           | 3.1         | $1.79 | $5.55      | $0.90     | $6.45 |
| RunPod H100 PCIe    | 1.6×           | 3.1         | $1.99 | $6.17      | $1.00     | $7.17 |
| RunPod H100 SXM     | 1.7×           | 2.9         | $2.69 | $7.80      | $1.34     | $9.14 |
| RunPod RTX 4090 24GB| 0.6×           | 8.3         | $0.34 | $2.82      | $0.17     | **$2.99** |

The 4090 row is interesting on paper — it's the cheapest by raw
training cost — **but** Qwen2-VL-7B with LoRA + gradient checkpointing
needs 30+ GB of VRAM in practice. The 4090's 24 GB ceiling means we'd
have to use 4-bit quantization and a tiny batch size, which usually
wipes out the cost win and risks accuracy loss. Use the 4090 only
for a smaller backbone (Qwen2-VL-2B or LLaVA-1.5-7B in 4-bit).

**Verdict.** Vast.ai A100 80GB is the cheapest box that still has
enough VRAM headroom to do the training without contortion. Going
larger doesn't save money on this workload; going smaller risks
correctness.

---

## 6. Two-phase architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 1 — INGEST (cheap, no GPU)                                   │
│                                                                    │
│  Vast.ai or RunPod CPU-only pod   ($0.10–0.20/hr)                  │
│  ├─ Pull LASANA bitstreams from opara.zih.tu-dresden.de            │
│  ├─ Pull PETRAW (already done on Mac, sync up)                     │
│  ├─ Pull SimSurgSkill (already done on Mac, sync up)               │
│  ├─ ffmpeg HEVC decode → 1 fps left-channel JPEGs                  │
│  ├─ Persist to network volume (RunPod) or Backblaze B2 (Vast.ai)   │
│  └─ Output: ~8 GB of JPEG frames + a manifest CSV                  │
│                                                                    │
│  Estimated wall-clock: 6–10 hours.                                 │
│  Estimated cost:       $1–2.                                       │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 2 — FEATURE EXTRACT (small GPU, brief)                       │
│                                                                    │
│  Vast.ai L40S 48GB or A10 24GB   ($0.30–0.80/hr)                   │
│  ├─ Mount the same network volume / sync from Backblaze            │
│  ├─ Run DINOv2-base over all extracted frames in batch             │
│  ├─ Write one .npy per video_id (768-d vectors × N frames)         │
│  └─ Output: ~1.5 GB of cached features                             │
│                                                                    │
│  Estimated wall-clock: 2–3 hours.                                  │
│  Estimated cost:       $1–2.                                       │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 3 — TRAIN (single A100, deliberate)                          │
│                                                                    │
│  Vast.ai A100 80GB on-demand   ($0.74–0.90/hr)                     │
│  ├─ Mount features volume (or rsync the 1.5 GB up)                 │
│  ├─ Run deploy/runpod_launch.sh against new dataset version        │
│  ├─ Save adapter checkpoints to memory/model_checkpoints/          │
│  ├─ Push checkpoints to S6 or Backblaze BEFORE shutdown            │
│  └─ Output: LoRA adapter + eval metrics                            │
│                                                                    │
│  Estimated wall-clock: 4–8 hours per run, 2 runs planned.          │
│  Estimated cost:       $7–14.                                      │
└────────────────────────────────────────────────────────────────────┘
```

The point of the split is that **GPU time only ticks during phase 3**.
Phases 1 and 2 each cost roughly the price of a coffee.

---

## 7. Storage strategy

### On RunPod
Create a **persistent network volume** in the same datacenter as the
intended training pod. Recommended: 250 GB. Cost: 250 × $0.07 =
**$17.50/month** if left in place. After training, snapshot the
features + checkpoint to Backblaze B2 ($0.005/GB/mo) and **delete
the volume** to stop the meter.

### On Vast.ai
Vast hosts have local SSD; you bring your own persistence. Pattern:

1. Phase 1 pod writes to a Backblaze B2 bucket (`b2 sync`).
2. Phase 2 pod pulls from B2, writes features back to B2.
3. Phase 3 pod pulls features from B2, pushes checkpoints back to B2
   on `trap EXIT`.

B2 cost for ~10 GB across three phases ≈ $0.05/month + ~$1 in egress.
Negligible compared to GPU time.

### Local fallback
S6 (Hetzner 32C/256GB) has plenty of disk and is already in the
fleet. Phase 1 and 2 outputs can also live on S6 indefinitely
($0 marginal cost). Phase 3 then `rsync`s features from S6 to the
GPU pod over the open internet — slower than a network volume but
free.

---

## 8. Dependency audit (do this BEFORE launching any pod)

The proven `deploy/runpod_launch.sh` already auto-handles most of these.
This section catalogs **everything that can go wrong on a fresh box**
so you don't pay $1.64/hr to debug it.

### Already handled by `deploy/runpod_launch.sh`
- ✅ GPU detection via `nvidia-smi`
- ✅ Editable install of `.[training]` extras
- ✅ Unsloth install with `cu124-torch240` first attempt
- ✅ Fallback to plain `unsloth @ git+...` if the cu124 wheel fails
- ✅ Final fallback to `framework: hf_trainer` config rewrite if Unsloth
  refuses to import on the host
- ✅ Skip flash-attn source build (it just warns and continues)
- ✅ Blackwell-specific torch upgrade to `torch==2.10.0+cu128`
- ✅ Runtime config generation at `/tmp/finetune_runtime.yaml`
- ✅ Per-GPU batch size / grad accum tuning when GPU is Blackwell
- ✅ `SKIP_DEP_INSTALL=1` for resume launches

### NOT handled — you must verify before launch
- ⚠ **CUDA driver version on the host.** Vast.ai listings vary; some
  hosts ship with driver 525 (CUDA 12.0) which is too old for
  `torch==2.10.0+cu128`. Filter listings to driver ≥ 535.
- ⚠ **ffmpeg with libx265.** Phase 1 needs HEVC decode. Stock Ubuntu
  22.04 has it, minimal images don't. Test:
  `ffmpeg -decoders 2>/dev/null | grep hevc` should print at least
  `hevc`. If not: `apt-get install -y ffmpeg libavcodec-extra`.
- ⚠ **PyAV vs decord for stereo video reading.** LASANA is HEVC stereo.
  decord 0.6.x has known issues with HEVC on some CPUs; PyAV is more
  reliable. `pip install av==12.*`.
- ⚠ **HuggingFace Hub auth for gated models.** Qwen2-VL is gated.
  Set `HF_TOKEN` in the pod's env before pulling weights.
- ⚠ **Disk space on root volume.** Default RunPod pod has 30 GB root.
  LASANA download → 185 GB does NOT fit. Either use a network volume
  (recommended) or mount `/workspace` to a fresh container disk
  ≥ 250 GB at launch.
- ⚠ **wandb login.** Set `WANDB_API_KEY` env var before launch, or
  pass `report_to: none` in the runtime config.
- ⚠ **Python version.** Some Vast.ai images ship Python 3.13, which
  PyTorch 2.10 wheels don't always cover. Filter to images with
  Python 3.10 or 3.11.
- ⚠ **bitsandbytes ↔ CUDA mismatch.** `bitsandbytes>=0.43` needs CUDA
  ≥ 11.8 at runtime. Check `python -c 'import bitsandbytes; print(bitsandbytes.__version__)'`
  and confirm `python -c 'import torch; print(torch.version.cuda)'`
  matches before hitting Go.
- ⚠ **Git LFS.** If we ever store checkpoints in the repo, the pod
  needs `git lfs install`. Currently we don't, but worth knowing.

### Pre-launch smoke test
Before authorizing payment on a new pod, run this from the pod the
moment SSH lands. The whole script should complete in under 60 seconds
and cost less than 3 cents:

```bash
set -e
echo "=== smoke test ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c "import sys; print('python', sys.version)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
python -c "import transformers, peft, accelerate; print('hf stack ok')"
python -c "import bitsandbytes; print('bnb', bitsandbytes.__version__)"
ffmpeg -decoders 2>/dev/null | grep -E '^V.....hevc' | head -1
df -h /workspace 2>/dev/null || df -h /
echo "=== smoke test ok ==="
```

If any line fails, **destroy the pod and pick a different host**. Do
not try to fix a broken host — it will cost more than picking a good
one on the second try.

---

## 9. Budget plan

### Aggressive ("everything goes right") — $40 spent

| Item                                    | Hours | Rate    | Cost   |
|-----------------------------------------|-------|---------|--------|
| Phase 1 CPU pod (LASANA download + decode) | 8h | $0.15  | $1.20  |
| Phase 2 small GPU (DINOv2 features)     | 2.5h  | $0.60   | $1.50  |
| Phase 3 A100 80GB train run 1           | 5h    | $0.78   | $3.90  |
| Phase 3 A100 80GB train run 2 (sweep)   | 5h    | $0.78   | $3.90  |
| Eval + export pass                      | 1h    | $0.78   | $0.78  |
| Backblaze B2 storage + egress           | —     | —       | $1.50  |
| **Subtotal GPU + storage**              |       |         | **$13** |
| API spend (~200 YouTube dual-teacher scoring) | — | —     | $25    |
| **Total**                               |       |         | **$38** |

### Conservative ("two reruns + a wrong host") — $108 spent

| Item                                    | Hours | Rate    | Cost   |
|-----------------------------------------|-------|---------|--------|
| Phase 1 CPU pod                         | 12h   | $0.20   | $2.40  |
| Phase 2 small GPU                       | 4h    | $0.80   | $3.20  |
| Phase 3 A100 train run 1                | 8h    | $0.90   | $7.20  |
| Phase 3 A100 train run 2                | 8h    | $0.90   | $7.20  |
| Phase 3 A100 train run 3 (one wrong run) | 4h   | $0.90   | $3.60  |
| Wasted host (smoke test fail, destroyed) | 0.2h | $1.50   | $0.30  |
| Eval / debugging on A100                | 2h    | $0.90   | $1.80  |
| Network volume on RunPod fallback (1 mo)| —     | —       | $17.50 |
| Backblaze B2                            | —     | —       | $3.00  |
| **Subtotal GPU + storage**              |       |         | **$46** |
| API spend (~400 YouTube videos, dual teacher) | — | —    | $60    |
| **Total**                               |       |         | **$106** |

Both scenarios fit comfortably under the $200 ceiling. The aggressive
plan leaves $160+ in headroom for follow-on work.

---

## 10. Execution sequence (in order, no skipping)

This is the operational checklist. Items marked **GATE** must pass
before moving on; items marked **$** start the meter.

### Stage 0 — local prep (no money, do today)

1. ✅ Push 4-dataset README + ingestion scripts (already done in PR #1).
2. ☐ Finish PETRAW Test.zip download on Mac (in flight, ~10 min).
3. ☐ Verify PETRAW + SimSurgSkill checksums on Mac.
4. ✅ Write `scripts/068_lasana_extract_features.py` and teach it the
   task-qualified W6 layout (`<video_id>/video.hevc`), with regression
   coverage for the new path.
5. ✅ Write `scripts/071_lasana_unzip_and_layout.py` so completed
   task-level zip archives are unpacked into per-trial video folders as
   downloads finish.
6. ☐ Write `scripts/069_train_head.py` (frozen-backbone regression
   head, runnable on CPU with 2 samples).
7. ☐ Decide which Hugging Face base model we're fine-tuning; record
   in `src/configs/finetune_lasana_v1.yaml`.
8. ☐ Set `HF_TOKEN`, `WANDB_API_KEY`, `BACKBLAZE_KEY` in `.env.example`
   (placeholders, not real values).
9. **GATE:** the new archive-layout + frame-extraction scripts run
   end-to-end on a 2-trial sample before any paid pod work starts.
   If they don't, do not pay for any GPU.

### Stage 1 — phase 1 ingest pod ($, ~$2)

1. ✅ Contabo is the primary manifest downloader for the straggler task:
   `python scripts/070_lasana_download.py --manifest-path data/external/lasana/_meta/bitstreams.json --out-dir /data/fls/raw-videos/lasana --resume --task SutureAndKnot`
2. ✅ Before adding a second box, run a 60-second per-IP-vs-per-account
   overlap check from Hetzner S5 while Contabo is already downloading.
   If both `.part` files grow, parallelize across hosts.
3. ✅ Hetzner S5 can carry the other three tasks in parallel. Launch one
   resumable worker each for `PegTransfer`, `CircleCutting`, and
   `BalloonResection`, writing task archives into
   `/data/fls/raw-videos/lasana` on the Hetzner host.
4. ☐ On each storage host, run
   `python scripts/071_lasana_unzip_and_layout.py --raw-dir <raw-zips> --out-dir <laid-out-videos> --watch`
   so completed `*.zip` archives are unpacked into
   `<video_id>/video.hevc` as soon as the final rename lands.
5. ☐ Run `python scripts/068_lasana_extract_features.py --frames-only`
   against the laid-out W6 tree, not the raw archive directory. Example:
   `python scripts/068_lasana_extract_features.py --frames-only --lasana-dir <laid-out-videos> --out-dir <processed-root> --fps 1`
6. ☐ Verify frame counts and the extractor manifest before destroying any
   paid pod. Archive byte counts alone are no longer the only gate.
7. **GATE:** the processed root contains the expected JPEG frames plus a
   manifest proving `068` consumed the task-qualified layout.

### Stage 2 — phase 2 small GPU ($, ~$2)

1. ☐ Pick Vast.ai L40S 48GB or A10 24GB listing.
2. ☐ SSH in, run smoke test.
3. ☐ Pull frames from B2.
4. ☐ Run `python scripts/068_lasana_extract_features.py --features-only`
   to compute DINOv2 features per frame.
5. ☐ Push `.npy` features back to B2.
6. ☐ Destroy the pod.
7. **GATE:** B2 bucket contains ~1.5 GB of `.npy` features.

### Stage 3 — local training data prep (free)

1. ☐ Pull features from B2 to S6 (or laptop).
2. ☐ Run `python scripts/040_prepare_training_data.py --version v4`
   to merge LASANA + PETRAW + SimSurgSkill + JIGSAWS metadata into
   `data/training/2026-04-XX_v4/`.
3. ☐ Run `python scripts/050_evaluate.py --dry-run` against the new
   dataset version on CPU to confirm the data loader works.
4. ☐ Commit + push the dataset version to GitHub.
5. **GATE:** `train.jsonl` + `val.jsonl` + `test.jsonl` all present
   and `python scripts/090_status.py` shows the new corpus.

### Stage 4 — phase 3 training pod ($, ~$8 per run)

1. ☐ Pick Vast.ai A100 80GB on-demand listing. Filter: driver ≥ 535,
   Ubuntu 22.04, Python 3.10/3.11, ≥250 GB disk, ≥4 host reviews,
   ≥99% uptime. Target rate ≤ $0.90/hr.
2. ☐ Set provider spending limit at $50 in the dashboard.
3. ☐ Add a calendar reminder to check pod status every 4 hours.
4. ☐ SSH in, run the smoke test from §8. **If anything fails, destroy
   and pick a different host.** Do not try to fix it.
5. ☐ `git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git && cd FLS-Training`
6. ☐ Pull features + dataset version from B2 (or rsync from S6).
7. ☐ Follow `docs/RUNPOD_RUNBOOK.md` from "Standard One-Shot Launch"
   onwards. The same launcher works on Vast.ai.
8. ☐ Monitor `nvidia-smi`. If utilization < 50% for > 5 minutes, kill
   and investigate dataloader bottlenecks.
9. ☐ When training finishes, **immediately** push checkpoints to B2:
   `b2 sync memory/model_checkpoints/ b2://fls-checkpoints/`
10. ☐ **STOP THE POD.** Verify in the dashboard. Confirm the meter
    is off.
11. **GATE:** adapter checkpoint in B2, eval metrics committed to
    GitHub, pod confirmed stopped.

### Stage 5 — eval & decide (free)

1. ☐ Pull checkpoint from B2 to S6 (or laptop).
2. ☐ Run `python scripts/050_evaluate.py --student <ckpt>` against
   the held-out test split.
3. ☐ Compute MAE on the held-out trainee split.
4. ☐ **Abort criterion:** if held-out MAE > 12 FLS points after this
   run AND a second run, **stop GPU spending**. The answer is more
   data, not more compute. Go harvest more YouTube + score it
   through the dual-teacher pipeline.
5. ☐ If MAE ≤ 12, declare v1 done. Update `STUDENT_MODEL` in `.env`.

---

## 11. Kill switches & abort criteria

These are the conditions under which **we stop spending immediately**:

1. **Held-out MAE thresholds (revised 2026-04-08 from baseline report):**
   - MAE > **22** → abort. This is above teacher-vs-teacher noise; it is
     a data problem, not a training problem. Stop pods, harvest + rescore.
   - MAE 15–22 → training is learning but inside teacher noise. At most
     one additional run; do NOT scale, do NOT keep buying GPU hours.
   - MAE < 15 → genuine signal. Further GPU spend is justified.
   The original "MAE > 12" threshold in this doc was set before the
   baseline was computed and is below the teacher noise floor; ignore it
   wherever it still appears in this file.
2. **Smoke test fails on a fresh pod.** Destroy and pick a different
   host. Do not debug a broken host.
3. **GPU utilization < 50% for > 5 min during training.** Almost
   certainly a dataloader or HEVC decode issue. Stop, fix the input
   pipeline locally, restart.
4. **A pod has been running > 12 hours since last checked.** Even if
   training is going well, log in and verify it's still doing real
   work. Idle pods are the #1 budget killer.
5. **Provider spending limit hit.** This should not happen if §10
   is followed, but the limit is the safety net.
6. **Forgotten pod overnight.** If you wake up and a pod has been
   running 8+ hours and you didn't intend it, that's a $13 mistake on
   A100 — survivable. The same on H100 SXM is $22. The same on
   H200 is $30+. Hence the A100 recommendation: even the worst
   forgotten-pod scenario fits in the contingency budget.

---

## 12. What this plan does NOT cover (intentional)

- **YouTube harvesting + dual-teacher scoring**: this is the API spend
  side of the budget; covered by `docs/EXECUTION_PLAN.md` and
  `scripts/011_harvest_youtube.py`. Unrelated to GPU sizing.
- **The ResearchFlow fleet (S1–S8 except S6)**: those servers belong
  to a different project. **Do not deploy FLS-Training code to S1–S8
  except S6.**
- **Long-term continuous training (multi-day watchdog mode)**: covered
  in `docs/RUNPOD_RUNBOOK.md` §"Continuous / Resume Mode". For the
  v1 LASANA fine-tune we don't need it; two ~5h runs is enough.
- **Multi-task / multi-head architectures**: a v2 concern. v1 is
  single-task: predict GRS z-score from a video.

---

## 13. References

- `docs/RUNPOD_RUNBOOK.md` — proven Blackwell + A100 path
- `docs/DATA_SCALING_PLAN.md` — corpus design
- `data/external/CITATIONS.md` — required citations for all 4 datasets
- `deploy/runpod_launch.sh` — provider-agnostic one-shot launcher
- `deploy/runpod_watchdog.sh` — continuous mode + checkpoint resume
- `scripts/070_lasana_download.py` — manifest-aware downloader for task archives
- `scripts/071_lasana_unzip_and_layout.py` — task-archive watcher that emits `<video_id>/video.hevc`
- `scripts/068_lasana_extract_features.py` — frame extraction + feature caching over the W6 layout
- RunPod pricing: https://www.runpod.io/pricing
- Vast.ai pricing: https://vast.ai/pricing
- Cloud GPU price tracker: https://getdeploying.com/gpus/nvidia-h100
