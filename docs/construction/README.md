# Construction log

Dated, append-only record of design decisions, plans, and structural changes
to FLS-Training. Each entry is a single markdown file with a frontmatter block
(date, time, status, related commits) and a short overview.

## Why this folder exists

`docs/*.md` at the top level holds long-form reference documents (architecture,
runbooks, guides). Those drift as the codebase changes. This folder holds
**point-in-time records** that do not drift — they capture what was decided,
when, why, and what shipped as a result. If you want to know "what did we do
on 2026-04-08?", you read this folder. If you want to know "how do I launch
RunPod?", you read `docs/RUNPOD_RUNBOOK.md`.

## File naming

```
YYYY-MM-DD_HHMM_short-slug.md
```

Slug is lower-case, hyphen-separated, ≤6 words. Timestamps are local to the
machine the entry was authored on (note the timezone in the frontmatter).
Files sort chronologically by name.

## Frontmatter template

```yaml
---
date: 2026-04-08
time: 10:40 EDT
status: proposed | in-progress | shipped | abandoned | superseded
authors: [logan, claude]
related_commits: [b83161a]
related_files: [src/configs/finetune_task5_standard.yaml]
supersedes: null
superseded_by: null
---
```

After the frontmatter, every entry should have at minimum:
- **Overview** — 2-4 sentences a stranger could read in 30 seconds.
- **Context** — what was true before this entry was written.
- **Decision / Plan** — what we're doing and why.
- **Outcome** — filled in after the work ships (commit hashes, what worked,
  what didn't). Leave blank initially.

## Index

Newest first. Update this when adding an entry.

| Date | Time | Slug | Status | Overview |
|---|---|---|---|---|
| 2026-04-08 | 13:05 EDT | [task5-baseline-runpod-launch](2026-04-08_1305_task5-baseline-runpod-launch.md) | proposed | Standalone Task5 baseline on RunPod, in parallel with the LASANA pipeline. ~$15, ~3-4h, fits exactly in the window while LASANA streaming pipeline is self-driving. Includes full pre-launch checklist, kill criteria, success criteria. Required for comparison against the LASANA-pretrained run either way. |
| 2026-04-08 | 10:40 EDT | [path-b-lasana-pretrain](2026-04-08_1040_path-b-lasana-pretrain.md) | in-progress | Two-stage training: pre-train on LASANA's 4 tasks (real human GRS labels), then fine-tune on FLS Task5 v4. W1-W6 shipped (`2c1b2a4`, `4046428`, `3cc3bbb`, `d04b4f6`, `5f28e80`); streaming pipeline live on Contabo + Hetzner in parallel; ETA collapsed from ~24h to ~1.76h bounded by Hetzner CircleCutting. |
| 2026-04-08 | 10:15 EDT | [task5-config-fix](2026-04-08_1015_task5-config-fix.md) | shipped | Patched `finetune_task5_standard.yaml` so the v4 144-sample run actually evals and checkpoints. The previous config would have produced zero evals over the entire run. Commit `b83161a`. |

## How to add an entry

1. Pick a slug, get the current local time:
   ```bash
   date '+%Y-%m-%d_%H%M'
   ```
2. Create `docs/construction/<date>_<time>_<slug>.md` with the frontmatter
   block above.
3. Write Overview / Context / Decision sections immediately. Outcome stays
   blank until the work lands.
4. Add a row to the index table in this README.
5. Commit with message `docs(construction): add <slug>` so the construction
   log is greppable in git history too.

## How to update an entry

Construction entries are **append-only** in spirit. If a plan changes:

- For small additions (new context, links to PRs as they land): append a
  dated section at the bottom of the existing file. Do not rewrite earlier
  text — that's the historical record.
- For a fundamental rethink: write a NEW entry with a new timestamp, set the
  new entry's `supersedes:` field to the old slug, and set the old entry's
  `superseded_by:` and `status: superseded`. Both entries stay in the folder.

The point of this folder is forensics, not aesthetics. A messy chronological
trail of what we actually thought is more valuable than a clean revisionist
snapshot.
