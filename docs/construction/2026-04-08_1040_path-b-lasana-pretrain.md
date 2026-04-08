---
date: 2026-04-08
time: 10:40 EDT
status: proposed
authors: [logan, claude]
related_commits: []
related_files:
  - scripts/070_lasana_download.py
  - scripts/069_ingest_lasana_to_store.py
  - scripts/068_lasana_extract_features.py
  - scripts/071_lasana_unzip_and_layout.py
  - scripts/040_prepare_training_data.py
  - src/configs/pretrain_lasana.yaml
  - src/configs/finetune_task5_standard.yaml
supersedes: null
superseded_by: null
---

# Path B — LASANA pretrain → Task5 fine-tune

## Overview

Two-stage training: (1) pre-train Qwen2.5-VL-7B + LoRA on all four LASANA
tasks using the real human GRS labels, then (2) fine-tune the resulting
adapter on the FLS Task5 v4 corpus. The point is to escape single-trainee
bias by borrowing LASANA's diversity, *without* polluting the clean Task5
corpus. This entry is the design handed to a VSC coding agent — implement
against it; do not re-litigate the architectural choices below without
flagging them in the PR description.

## Context (verified 2026-04-08)

- `data/external/lasana/` is **annotations only** (340K). Zero HEVC files.
- Annotations: 4 tasks × ~315 trials × 4 raters + consolidated CSVs +
  pre-built `*_split.csv` train/val/test splits. Schema:
  `id;duration;frame_count;GRS;bimanual_dexterity;depth_perception;efficiency;tissue_handling;<task-specific binary errors>`.
  GRS is **z-scored** (mean 0, ~unit scale), not raw 1–5.
- `bitstreams.json` is a HAL-format download manifest from a research data
  portal. It is the source of truth for where to fetch the HEVC files.
- `MemoryStore` currently contains only the FLS YouTube corpus
  (`source ∈ {critique_consensus, teacher_claude, teacher_gpt4o}`). LASANA
  is not in the store.
- `scripts/040_prepare_training_data.py` reads from `MemoryStore` and has
  **no source filter**. Without one, any LASANA ingest will silently
  contaminate Task5 builds.
- `scripts/068_lasana_extract_features.py --frames-only` extracts JPEG
  frames at 1 fps from HEVC into
  `data/external/lasana_processed/frames/<video_id>/*.jpg`, where
  `video_id` is task-qualified (for example `lasana_suture_kiourf`).
- `scripts/071_lasana_unzip_and_layout.py` is the bridge between HAL's
  task-level zip archives and the frame extractor: it watches for
  completed `*.zip` downloads and emits `<video_id>/video.hevc`.
- The trainer (`src/training/finetune_vlm.py`) reads only this set of flat
  keys: `num_epochs`, `warmup_ratio`, `logging_steps`, `save_strategy`,
  `eval_strategy`, `save_total_limit`, `dataset_path`,
  `resume_from_checkpoint`, `require_vision`. Configs MUST stick to those.

## Infrastructure constraint

There is one Contabo box for this project plus S5 (Hetzner) available as a
parallel download/unzip worker. The safe operating pattern is:

- Contabo remains the primary manifest-driven downloader.
- Hetzner is allowed to run additional task-scoped archive downloads only
  after a short overlap test proves both hosts advance bytes concurrently.
- Frame extraction still consumes the laid-out `<video_id>/video.hevc`
  tree produced by W6; do not point `068` at raw task zips.

## Non-goals

- Do not VLM-rescore the LASANA labels. The human GRS values are the asset.
- Do not modify `finetune_lasana_v4.yaml` — that's an aspirational
  multi-dataset config and is out of scope here.
- Do not change `src/training/finetune_vlm.py`'s training-args contract. If
  you find yourself wanting to add `load_best_model_at_end` or
  `metric_for_best_model`, that is a separate PR.

## Workstreams

### W1 — `scripts/070_lasana_download.py` (NEW)

Parse `data/external/lasana/_meta/bitstreams.json` (HAL format with
`_embedded`, `page`, `_links`) and download the HEVC bitstreams to
`<--out-dir>/<trial_id>.hevc`.

- **Run target:** Contabo first, with Hetzner S5 available for parallel
  task sharding once the per-IP-vs-per-account overlap check passes.
  Verify free disk on both hosts before kicking off (`df -h`); keep at
  least ~500 GB free on whichever box is receiving a large task archive.
- **CLI:** `--out-dir`, `--max-trials` (testing), `--resume` (skip existing),
  `--task` (optional filter on procedure name).
- **Resilience:** retry on transient HTTP errors, write a `manifest.csv`
  recording (trial_id, url, bytes, sha256, status). Don't fail the whole run
  on a single bad URL.
- **Auth:** check `bitstreams.json` for whether links require a token. If
  they do, read it from `LASANA_API_TOKEN` env var; fail loudly if missing.

### W2 — `scripts/069_ingest_lasana_to_store.py` (NEW)

> Note: an untracked file at this path already exists in Logan's working
> tree. Reconcile with that work before starting from scratch.

Walk the annotation CSVs and insert one record per trial into `MemoryStore`.

- **Per trial, average across the 4 raters** (`*_rater0.csv` ...
  `*_rater3.csv`) for GRS, bimanual_dexterity, depth_perception, efficiency,
  tissue_handling. Use mean. Compute and store inter-rater std as
  `score_std` so downstream loss weighting can de-emphasize noisy trials.
- **GRS is z-scored.** Store both the raw z-score (`grs_z`) and a rescaled
  1–5 value (`grs_rescaled = clip(3 + grs_z, 1, 5)`) so the same field name
  the FLS pipeline expects (`fls_score`) is populated. Document the rescale
  in the inserted record's `metadata.notes`.
- **`source` tag:** `"lasana"`. **`task` tag:** `"lasana_balloon"`,
  `"lasana_circle"`, `"lasana_peg"`, `"lasana_suture"`. Do **not** collide
  with FLS `task1..task5` — the namespace must be disjoint so 040's filter
  works cleanly.
- **Splits:** honor `*_split.csv` from LASANA. Do not re-split. Add a
  `split` field (`train`/`val`/`test`) to each record so 040 can pass it
  through instead of re-shuffling.
- **Frame paths:** point at
  `data/external/lasana_processed/frames/<trial_id>/`. Do not require frames
  to exist at ingest time; that decouples ingest from W1. Validation happens
  at 040 build time.
- **Confidence:** set `confidence = 1.0 - normalized_rater_std` (clipped to
  [0.5, 1.0]). This lets `--min-confidence` in 040 still work as a filter.
- **Idempotence:** re-running the script must update existing records, not
  duplicate them. Key on `(source, trial_id)`.

### W3 — `scripts/040_prepare_training_data.py` (EDIT)

Add source filtering and honor pre-existing splits.

- New CLI: `--include-sources` (comma-separated list, default = all),
  `--exclude-sources` (comma-separated, default = none). Both filter on
  `MemoryStore` records' `source` field before any other processing.
- New CLI: `--respect-existing-splits` (bool, default `False`). When set,
  records with a non-null `split` field skip the random splitter and land
  in their declared split. This is what makes LASANA's published splits
  work.
- The `manifest.yaml` `sources:` block must reflect the post-filter counts
  so it's obvious from the manifest which build is which.
- **Non-regression:** running `040 --ver v5` with no new flags must produce
  byte-identical output to the existing v4 build path. Add a unit test for
  this if there isn't one.

### W4 — `src/configs/pretrain_lasana.yaml` (NEW)

Drop in next to `finetune_task5_standard.yaml`. Use the **flat-key schema**
that `src/training/finetune_vlm.py` actually reads. Reference fields:

```yaml
purpose: "Stage-1 LASANA pretrain — diversity layer for Task5 transfer"
base_model: "Qwen/Qwen2.5-VL-7B-Instruct"
framework: "unsloth"
fallback_framework: "hf_trainer"
config_family: "pretrain"

vision_mode: true
require_vision: true            # abort pre-GPU if frames missing
max_frames_per_sample: 10
dataset_path: "data/training/LATEST_LASANA"  # symlink set by 040 build
split_strategy: "lasana_published"

quantization: "4bit"
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_quant_type: "nf4"

lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_modules: [q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj]

# Sized for ~1,000 train samples, effective batch 8 → ~125 steps/epoch
num_epochs: 3                    # ~375 optimizer steps
batch_size: 2
gradient_accumulation: 4
learning_rate: 1.0e-5            # lower than Stage 2; broader data, want less drift
warmup_ratio: 0.05
bf16: true
flash_attention_2: true
gradient_checkpointing: true
logging_steps: 5
save_strategy: "epoch"
eval_strategy: "epoch"
save_total_limit: 3
max_seq_length: 2048
packing: false
export_merged_16bit: false       # adapter only — Stage 2 will resume from it
dataloader_num_workers: 4

output_dir: "memory/model_checkpoints/lasana_pretrain"
resume_from_checkpoint: null
```

### W5 — `src/configs/finetune_task5_standard.yaml` (EDIT)

Single line change:

```yaml
resume_from_checkpoint: "memory/model_checkpoints/lasana_pretrain"
```

This must be **commented out by default** with a short note explaining how
to flip it on after the LASANA pretrain finishes. Reason: we don't want a
fresh clone to attempt to resume from a checkpoint that doesn't exist on a
new pod.

### W6 — `scripts/072_lasana_stream_watch.py` (NEW)

Add a thin host-role watcher that turns the batch primitives into the live
streaming path:

`070 -> 071 -> 068 -> rsync -> 040`

- **No core rewrites.** Reuse `070` as the downloader, `071` as the
  archive-to-layout bridge, `068` for frame extraction, and `040` for
  dataset materialization.
- **Hetzner role:** run the watcher against the three non-suture tasks.
  As each archive finishes, `071` emits `<video_id>/video.hevc`, `068`
  extracts the new frame directory, and the watcher `rsync`s that single
  `<video_id>/` frame tree to Contabo.
- **Contabo role:** run the watcher against `SutureAndKnot`, let it
  extract local frames, and gate `040` until the local
  `data/external/lasana_processed/frames/` tree contains every LASANA
  `video_id` backed by `memory/scores/lasana/*.json` across all four tasks.
- **Idempotence:** persist synced frame dirs + the one-shot `040` trigger
  under `data/external/lasana_processed/.stream_state/` so re-runs resume
  cleanly instead of re-rsyncing or rebuilding the LASANA dataset.

## Acceptance criteria

A reviewer accepting this PR should be able to verify, in order:

1. `python scripts/070_lasana_download.py --max-trials 3 --out-dir /tmp/lasana_test`
   produces 3 HEVC files and a `manifest.csv` with `status=ok` for each.
2. `python scripts/069_ingest_lasana_to_store.py --dry-run` reports
   ~1,260 records that would be inserted, with non-zero `score_std` and
   `confidence` populated, and a balanced split count from the
   `*_split.csv` files.
3. `python scripts/040_prepare_training_data.py --ver lasana_v1 \
       --include-sources lasana --respect-existing-splits \
       --frames-dir data/external/lasana_processed/frames`
   produces `data/training/<date>_lasana_v1/{train,val,test}.jsonl` whose
   `metadata.source` is uniformly `"lasana"` and whose `metadata.task`
   values are in the `lasana_*` namespace only. Zero `task1..task5` records
   leak in.
4. `python scripts/040_prepare_training_data.py --ver v5_check` (no source
   filter) produces output byte-identical to a fresh `--ver v4` rebuild.
5. CPU smoke test: `bash smoke_test.sh data/training/<date>_lasana_v1` runs
   to completion (4 samples, 1 epoch, no quantization).
6. Host streaming orchestration can be started with:
  ```bash
  python scripts/072_lasana_stream_watch.py \
     --host-role hetzner \
     --run-downloader --resume-downloads \
     --download-task PegTransfer \
     --download-task CircleCutting \
     --download-task BalloonResection \
     --raw-dir /data/fls/raw-videos/lasana \
     --layout-dir /data/fls/lasana_layout \
     --processed-dir /data/fls/lasana_processed \
     --rsync-dest root@<CONTABO_HOST>:/data/fls/lasana_processed/frames \
     --watch

  python scripts/072_lasana_stream_watch.py \
     --host-role contabo \
     --run-downloader --resume-downloads \
     --download-task SutureAndKnot \
     --raw-dir /data/fls/raw-videos/lasana \
     --layout-dir /data/fls/lasana_layout \
     --processed-dir /data/fls/lasana_processed \
     --prepare-when-ready \
     --prepare-ver lasana_v1 \
     --watch
  ```
7. Pod launch sequence (do not actually run on GPU as part of PR review):
   ```bash
   bash deploy/runpod_launch.sh data/training/LATEST_LASANA src/configs/pretrain_lasana.yaml
   # ...wait for completion, then flip resume_from_checkpoint in finetune_task5_standard.yaml
   bash deploy/runpod_launch.sh data/training/2026-04-08_v4 src/configs/finetune_task5_standard.yaml
   ```

## Open questions for the implementer

Flag in PR description, do not block on:

1. **Rater averaging vs concatenation.** I chose mean across 4 raters
   because it's simple and gives one record per trial. Alternative: emit 4
   records per trial (one per rater) for 4× data and natural label noise.
   Mean is the right call for a small pretrain budget; flag if you disagree.
2. **Z-score → 1–5 rescale.** The `clip(3 + grs_z, 1, 5)` rescale is a
   convenience for keeping the FLS pipeline's `fls_score` field populated.
   The *real* training target should be `grs_z` directly to preserve the
   calibrated spread. If the loss head expects 1–5, fix the head, not the
   data.
3. **Task naming.** I used `lasana_balloon / lasana_circle / lasana_peg /
   lasana_suture`. If the eval/metrics layer hard-codes `task1..task5`,
   you'll need a small adapter. Don't rename to `task1..task5` — the
   namespace collision will burn you in 040's filter.

## What lives outside this PR

- The actual LASANA HEVC download (W1's *execution*, not the script). Run
  operationally via the new watcher on Contabo first, then shard the other
  three task archives onto Hetzner only after the short overlap test shows
  both hosts can advance.
- The downstream Stage-2 launch. That's a config flip + a `bash` command,
  not code.
- Any change to `src/training/finetune_vlm.py`.

## Outcome

_To be filled in as workstreams land. Append commit hashes, dates, and any
deviations from the plan above._

---

## Outcome (appended 2026-04-08 14:20 EDT)

Status advanced: `in-progress` now includes the streaming watcher architecture required for the
host-split LASANA ingest path.

- Canonical data path is now `070 -> 071 -> 068 -> rsync -> 040`.
- `scripts/072_lasana_stream_watch.py` is the main host-role wrapper around that path. It keeps the
  existing downloader, layout, frame-extraction, rsync, and dataset-build primitives intact while
  making the end-to-end stream restart-safe through `.stream_state/` receipts.
- `scripts/072_lasana_frame_watch.py`, `scripts/073_lasana_rsync_watch.py`, and
  `scripts/074_lasana_prepare_watch.py` remain as lower-level stage helpers for debugging or
  one-off execution, but the intended operations entrypoint is the single stream watcher above.
- `040` remains Contabo-only because `memory/scores/lasana` is authoritative there. Hetzner is now
  explicitly an approved parallel worker for download, unzip, frame extraction, and frame rsync,
  but not for label storage or dataset assembly.
- Contabo also remains responsible for the local `SutureAndKnot` branch of the stream so the full
  four-task frame corpus converges on the same host before dataset preparation starts.


---

## Outcome (appended 2026-04-08 11:05 EDT)

Status promoted: `proposed` → `in-progress`. All five code workstreams have
shipped to `main` and a non-trivial design bug was caught and fixed during
implementation. Vision-dataset build is blocked on frame availability, not
on code.

### Commits that landed against this design

| Commit | Workstream | Notes |
|---|---|---|
| `2c1b2a4` | W1 — `070_lasana_download.py` | HAL manifest downloader, 368 LOC |
| `4046428` | W2 — `069_ingest_lasana_to_store.py` initial | 375 LOC |
| `3cc3bbb` | W3+W4+W5 bundle | `040` source filters + `LATEST_LASANA` symlink, `pretrain_lasana.yaml` (matches design exactly), `finetune_task5_standard.yaml` `resume_from_checkpoint` wired and commented out by default, `prepare_dataset.py` + `schema_adapter.py` updated to honor `include_sources` / `exclude_sources` / `respect_existing_splits` / pre-existing splits, `runpod_launch.sh` updated, regression tests in `tests/test_prepare_dataset.py` (non-regression byte-identical check + filter behavior check) |
| `d04b4f6` | W2 fix | Cross-task trial-id collision fix; added `tests/test_lasana_ingest.py` |

### Design deviations and discoveries

**Trial-ID collision (real bug, design assumed unique IDs).** During
implementation, the agent discovered that LASANA trial IDs are unique only
*within* a task, not across all tasks. Concrete example: `kiourf` appears
in both `BalloonResection.csv` and `SutureAndKnot.csv` as different trials.
The original ingest script keyed records on `(source, trial_id)` per the
design — this design is **wrong** for LASANA and would have silently
overwritten ~half the records on the second-task pass.

Fix: task-qualify the score_id, video_id, and frame-dir name in
`069_ingest_lasana_to_store.py`. Regression coverage in
`tests/test_lasana_ingest.py`. **Update the design key to
`(source, task, trial_id)` if this doc is ever cloned for another dataset.**

### Operational state on Contabo

- Durable LASANA labels ingested into `/data/fls/memory/scores/lasana`:
  **1,270 score files**.
- Published splits honored: **train=944, val=121, test=205**. Note this is
  ~74/10/16, not the conventional 80/10/10. This is what LASANA published;
  do not re-balance.
- Raw HEVC download still in flight: `/data/fls/raw-videos/lasana/BalloonResection_left.zip.part`
  at ~2.5 GB. Downloader is one-archive-at-a-time, no parallel fetches.
- LASANA frame directory `/data/fls/data/external/lasana_processed/frames/`
  is **empty**. Frame extraction has not been attempted yet — it's blocked
  on the unzip step which is itself blocked on the download finishing.

### Outcome (appended 2026-04-08 16:30 UTC)

W6 shipped and the operator path changed accordingly.

- `scripts/071_lasana_unzip_and_layout.py` now watches completed
  task-level archives and materializes the task-qualified
  `<video_id>/video.hevc` layout required by `068`.
- `scripts/068_lasana_extract_features.py` now accepts that W6 layout in
  addition to the legacy tree, with regression coverage in
  `tests/test_lasana_extract_features.py`.
- A 60-second overlap test between Contabo and Hetzner showed both hosts
  could grow their `.part` files concurrently, so Hetzner was promoted
  from "out of scope" to an approved parallel downloader for
  `PegTransfer`, `CircleCutting`, and `BalloonResection` while Contabo
  continues `SutureAndKnot`.

### Next operational step

Keep the live download workers running, then start `071` in `--watch`
mode on each host's raw archive directory so frame extraction can begin as
soon as each zip finishes.


---

## Status update — streaming pipeline live (appended 2026-04-08 13:05 EDT)

ETA collapsed from ~24h sequential to ~1.75h parallel. Hetzner cpx62 (16
vCPU / 32 GB / 640 GB, Helsinki, ASN Hetzner Online GmbH) brought online as
a second downloader on a distinct IP, which empirically defeated whatever
rate-limiting the LASANA portal was applying to Contabo alone. The full
streaming pipeline is now live on both hosts.

### What shipped

- `5f28e80` — `scripts/071_lasana_unzip_and_layout.py` (W6) with
  task-qualified archive layout. `--watch` mode polls the raw download
  directory and extracts + lays out each archive the moment it completes.
- Hetzner: repo cloned to `/root/FLS-Training`, 071 watcher running as
  PID 456503 against `/data/fls/data/external/lasana_layout`.
- Contabo: 070 downloader active on SutureAndKnot, 071 watcher active,
  plus a named background resume loop `lasana_suture_resume_watch`
  (PID 4114118) that auto-retries `SutureAndKnot_left.zip.part` the
  moment `SutureAndKnot_right.zip` finishes. No human babysitting needed.
- Hetzner: three concurrent downloaders active on BalloonResection,
  CircleCutting, PegTransfer.

### Live state (observed ~13:00 EDT)

| Host | Task | Partial size | ETA |
|---|---|---|---|
| Contabo | SutureAndKnot_right | 21.6 GB | ~23 min |
| Contabo | SutureAndKnot_left (retry armed) | 8.1 GB | ~68 min after right finishes |
| Hetzner | BalloonResection_left | 5.5 GB | ~44 min |
| Hetzner | PegTransfer_left | 5.0 GB | ~51 min |
| Hetzner | CircleCutting_left | 3.8 GB | ~1.76 hr |

**Critical path: Hetzner CircleCutting at ~1.76 hr.** Contabo's total
workload (~1.52 hr) is comfortably under the Hetzner bound.

### Architectural note — per-IP not per-account

The empirical result (Hetzner + Contabo downloading in parallel without
contention) confirms the LASANA portal's rate limit is per-IP, not per
account/token. This means future downloads of comparable volume can
legitimately parallelize across any distinct-IP hosts we have access to.
Do **not** try to scale this to 4+ hosts without first checking the
portal's terms of service — polite parallelism at 2 is different from
aggressive parallelism at 8.

### Still-outstanding work before Stage 1 can launch

The streaming pipeline handles `070 → 071` automatically. Everything
downstream is still manual-trigger:

1. **Auto-trigger `068 --frames-only`** on each layout directory as it
   appears. Currently the watcher produces the layout, but nothing picks
   up from there. Agent action: wrap 068 in a second watcher that polls
   the layout directory the same way 071 polls the raw directory.
2. **Hetzner → Contabo frame rsync.** 040 must run on Contabo because
   MemoryStore (with the 1,270 LASANA labels) lives there. The ~28 GB of
   frames extracted on Hetzner need to land on Contabo before 040 can
   build the unified corpus. Estimated rsync time over datacenter link:
   ~10 min per task.
3. **040 build on Contabo** — invoked as:
   ```bash
   python scripts/040_prepare_training_data.py --ver lasana_v1 \
       --include-sources lasana --respect-existing-splits \
       --frames-dir /data/fls/data/external/lasana_processed/frames \
       --max-frames 10
   ```
4. **CPU smoke test** on the resulting corpus before pod launch.
5. **Stage 1 RunPod launch** with `pretrain_lasana.yaml`.

### Full streaming pipeline (canonical)

```
Hetzner: 070 → 071 → 068 → rsync → Contabo
Contabo: 070 → 071 → 068 → (stay local)
Contabo: wait for all 4 task frame dirs → 040 → smoke_test → ready for Stage 1
```
