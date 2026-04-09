---
date: 2026-04-09
time: 21:30 UTC
status: resolved
authors: [logan, claude]
related_commits:
  - a9efac6f
related_files:
  - src/training/prepare_dataset.py
  - scripts/072_lasana_stream_watch.py
  - memory/scores/lasana/
  - data/training/2026-04-09_lasana_v1/
supersedes: null
superseded_by: null
---

# Hetzner / Contabo reconciliation — LASANA pipeline recovered

## TL;DR

Hetzner CPX62 (77.42.85.109) was sitting idle — its watcher had died at
17:00 UTC mid-run with 4 of 6 archives downloaded and nothing laid out.
Contabo (207.244.235.10) was the node actually doing the work, but its
`prepare_dataset` step had been silently producing empty outputs all day
because of two cascading bugs. Both fixed. We now have a real
training-ready LASANA dataset on disk: **1270 videos, 944 / 121 / 205
train/val/test split, vision mode True, all frames present.**

## Why Hetzner looked idle

- VSC had flagged the Hetzner box as inactive. Suspicion was that the
  stream watcher had been silently pushing to the wrong GitHub repo
  (ROS_FLOW_2_1 vs FLS-Training). Checked all nine training branches of
  ROS_FLOW_2_1: no FLS/LASANA content. Theory disproven.
- Actual cause on Hetzner: `.archive_state/` contains only
  `BalloonResection_left.zip.done` (17:00 UTC) and
  `PegTransfer_left.zip.done` (16:56 UTC). The watcher died before
  completing the remaining four archives and never laid anything out.
  Layout dir had 316 BalloonResection + 329 PegTransfer + 0
  CircleCutting directories — partial, stale, unusable.
- Hetzner was never the primary. Contabo is. The confusion came from
  running the same `072_lasana_stream_watch.py` orchestrator on both
  hosts without a clear owner designation.

## Why Contabo looked successful but produced nothing

Contabo's 072 watcher ran the full pipeline end-to-end (070 download →
071 unzip → 068 frame extract → 040 prepare) and marked
`prepare_completed: true` in `watch_state.json`. But the actual
`data/training/` directory had no new dated folder. Investigation
uncovered two independent bugs:

### Bug 1 — `_normalize_source` had no `lasana` case

In `src/training/prepare_dataset.py`, every score file is run through
`_normalize_source(source, filename)` to bucket it into one of the
known source categories. The function recognized
`critique_consensus`, `teacher_gpt4o`, `student_model`,
`expert_correction` — and fell through to `teacher_claude` for
everything else. When `069_ingest_lasana_to_store.py` started writing
score files with `source: "lasana"`, they were all silently relabeled
`teacher_claude` and then dropped by `--include-sources lasana`.

Fix (commit `a9efac6f` on `main`):

```python
if raw == "lasana" or "lasana" in raw or "lasana" in name:
    return "lasana"
```

### Bug 2 — `memory/scores/lasana/` was a symlink

The LASANA scores were being written to
`/data/fls/memory/scores/lasana/` (fast NVMe) with a symlink from
`/opt/FLS-Training/memory/scores/lasana → /data/fls/...`.
`prepare_dataset._iter_json_files` uses a top-level glob + one-level
subdirectory glob off `SCORES_DIR`. Python's `pathlib` does not
traverse symlinked directories with `rglob` / `glob` unless told to.
Result: 1270 LASANA score JSONs were invisible to the prepare step.

Fix (applied in place on Contabo):

```
rm /opt/FLS-Training/memory/scores/lasana
mkdir -p /opt/FLS-Training/memory/scores/lasana
cp /data/fls/memory/scores/lasana/*.json \
   /opt/FLS-Training/memory/scores/lasana/
rm -rf /data/fls/memory/scores/lasana   # remove the divergent source
```

Both sides were verified to hold 1270 files before the source-side
deletion.

## The dataset we actually produced

After both fixes, re-running prepare with
`--include-sources lasana --version lasana_v1` produced:

```
/opt/FLS-Training/data/training/2026-04-09_lasana_v1/
├── train.jsonl    13M   944 examples
├── val.jsonl     1.7M   121 examples
├── test.jsonl    2.8M   205 examples
├── manifest.yaml
└── *.meta.json    (3 lineage sidecars)
```

Vision mode: **True**. Every example carries real frame paths.
`source_domains`: `{'lasana': 1270}`. The `LATEST_LASANA` symlink now
points at this directory. This is the dataset the Path B pretrain was
supposed to get this morning but didn't.

## Watcher state reset

`/data/fls/lasana_processed/.stream_state/watch_state.json` had
`prepare_completed: true` + `prepare_completed_at: 2026-04-09T00:34:52Z`
from the no-op run. Reset to `false` so future watcher iterations will
re-run prepare if new scores land. The live 072 process (PID 4153096)
is still running in idle-poll mode and does not need restart — it
re-reads state on each tick.

## Cleanup

Contabo:
- Deleted `/data/fls/data/external/lasana_layout` (27 GB stale cruft
  from an aborted earlier attempt).
- Deleted `/data/fls/memory/scores/lasana/` (5.1 MB duplicate; the
  authoritative copy is now at `/opt/FLS-Training/memory/scores/lasana/`).
- Disk after cleanup: 324 GB used / 558 GB free.

Hetzner:
- Deleted `/root/FLS-Training` stale clone (48 MB). `/root/FLS-Training-main`
  retained.
- Disk after cleanup: 313 GB used / 265 GB free.
- **Pending user decision**: 208 GB of cruft remains:
  - 136 GB of `.zip` archives under `/root/lasana_raw/`
  - 39 GB partial layout dir (`/root/lasana_layout/`)
  - 33 GB orphaned `.part` file from the interrupted download
  Contabo now has the authoritative copy of everything this cruft
  represents.

## What this unblocks

Stage-1 LASANA pretrain (Path B) can now launch against
`data/training/2026-04-09_lasana_v1/` on a GPU pod. The degenerate
baseline we saw earlier today (F1 vision_mode=False, F2 Blackwell
save_strategy) had nothing to do with dataset content; those are
separate training-loop bugs still to be fixed before the pretrain run.

## Lessons

1. **Never rely on symlinks for data directories that will be scanned
   by `pathlib.rglob`.** It silently returns nothing. Either use real
   directories or pass `follow_symlinks=True` explicitly on each call.
2. **`_normalize_source` should whitelist, not fallthrough.** The
   default-to-`teacher_claude` behavior meant every new source type
   gets silently mislabeled and then filtered out. Next time a new
   source is added, it should either match an explicit case or raise.
3. **A watcher that writes `prepare_completed: true` on zero output is
   lying.** `maybe_run_prepare()` should verify at least one row
   landed in the output JSONL before marking completion.
4. **Host ownership should be explicit.** Running the same orchestrator
   on two hosts without designated roles wasted the whole day of
   Hetzner compute. The stream watcher now takes `--host-role`; we
   should treat any node without a role assignment as not-owned.

## Open items (tracked separately)

- Rotate exposed RunPod API key.
- Fix F1 (vision_mode=False default) and F2 (Blackwell save_strategy
  override) in the training loop.
- Decide spot vs on-demand pricing for the Stage-1 pretrain pod.
- Add the Hetzner cleanup decision (208 GB).
