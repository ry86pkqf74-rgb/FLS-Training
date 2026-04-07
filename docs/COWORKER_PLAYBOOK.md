# Coworker Playbook — Pushing Claude Scores to GitHub

This doc exists so any future Claude (Cowork mode or otherwise) working on
this repo knows the exact sequence for getting newly-scored videos out of
project knowledge and into the `FLS-Training` GitHub repo.

**Repo:** https://github.com/ry86pkqf74-rgb/FLS-Training
**Local checkout on Logan's Mac:** `~/Downloads/FLS-Training`
**Project knowledge path (in Cowork sandbox):**
`/sessions/<session>/mnt/.projects/019d6673-00e9-71d3-9a4f-75c0ad02f3b5/docs/`

## How the score pipeline works

1. Logan records a training video (e.g. `V21.mov`) and drops it in
   `~/Downloads/FLS training Videos/` on his Mac. Raw videos are **gitignored**
   (`*.mov`, `*.mp4`, `videos/`) — do **not** try to push them to main. If
   Logan ever asks to "push the videos", ask whether he means the raw files
   (needs Git LFS or GitHub Releases) or the scoring data (this playbook).
2. A scoring run produces two artifacts saved into the Claude project
   knowledge (visible to the next Cowork session under `.projects/.../docs/`):
   - `<video_id>_claude-sonnet-4_<timestamp>.json` — full score JSON
   - `learning_ledger_v<N>_append.jsonl` — two JSONL lines (`video_ingested`
     + `frontier_scored`) to append to the main ledger
3. Those artifacts need to land in the GitHub repo under:
   - `memory/scores/<YYYY-MM-DD>/<video_id>_claude-sonnet-4_*.json`
   - appended to `memory/learning_ledger.jsonl`

## The sequence (copy-paste friendly)

When Logan says something like "push all training to github that hasn't been
pushed yet", do this:

### 1. Figure out what's missing

Compare `.projects/.../docs/` (source of truth for new scores) against
`memory/scores/<date>/` and the `video_id`s already present in
`memory/learning_ledger.jsonl`. Only Claude-sonnet-4 scores belong in the
"Claude scoring" batch — gpt-4o scores are a separate workstream that
someone else (or a separate commit) owns, so don't bundle them.

```bash
# videos already in the ledger
grep -o '"video_id":\s*"[^"]*"' memory/learning_ledger.jsonl | sort -u

# Claude score JSONs in project knowledge
ls /sessions/*/mnt/.projects/019d6673-*/docs/V*_claude-sonnet-4_*.json
```

### 2. Copy the JSONs and append the ledger

Operate on the mounted repo path (`/sessions/.../mnt/FLS-Training/`) — the
sandbox cannot `git push`, but file edits there are reflected on Logan's Mac
checkout instantly.

```bash
PROJ=/sessions/<session>/mnt/.projects/019d6673-00e9-71d3-9a4f-75c0ad02f3b5/docs
REPO=/sessions/<session>/mnt/FLS-Training
DEST=$REPO/memory/scores/2026-04-07   # adjust date

for v in 13 14 15 16 17 18 19 20; do
  src=$(ls "$PROJ"/V${v}_video_claude-sonnet-4_*.json | head -1)
  cp "$src" "$DEST/"
done

# ensure trailing newline on the ledger, then append
tail -c1 "$REPO/memory/learning_ledger.jsonl" | od -An -c | grep -q '\\n' \
  || echo "" >> "$REPO/memory/learning_ledger.jsonl"
for v in 13 14 15 16 17 18 19 20; do
  cat "$PROJ/learning_ledger_v${v}_append.jsonl" \
    >> "$REPO/memory/learning_ledger.jsonl"
done
```

### 3. Commit + push from the Mac (not the sandbox)

The sandbox has no outbound network / no credentials, so commit and push
from the Mac via Desktop Commander's `start_process` → `interact_with_process`
against a `bash` or `zsh` shell. There's no credential prompt — the repo
already has HTTPS creds cached in the Mac keychain.

```bash
cd ~/Downloads/FLS-Training
git add memory/scores/<date>/V*.json memory/learning_ledger.jsonl
# also stage any stray untracked docs Logan mentions
git commit -m "data: add Claude scores V<lo>-V<hi>"
git pull --rebase origin main   # someone else may have pushed gpt-4o scores
git push origin main
```

Use the commit message pattern from existing history:
`data: add Claude scores V<lo>-V<hi>` with a bullet list of
`V<n>: <fls_score> (conf <n.nn>)` in the body. Match the existing per-video
style — look at recent `git log --oneline` for the exact wording Logan likes.

### 4. Verify

```bash
git log --oneline -3
# then check on GitHub that the commit sha is on main
```

## Gotchas

- **Raw videos are gitignored.** Do not add them. 7+ GB of `.mov` files
  would blow past GitHub's 100 MB/file limit and free LFS quota.
- **Rebase over the remote.** Logan (or another session) occasionally pushes
  gpt-4o scores to main between runs, so always `git pull --rebase` before
  `git push`. No conflicts expected — the two workstreams touch different
  files.
- **Sandbox can't push.** Run `git` commands via Desktop Commander on the
  Mac. File edits can happen in the sandbox (they propagate to
  `~/Downloads/FLS-Training` via the mount).
- **Ledger dedup.** Before appending, grep the ledger for the `video_id`s
  you're about to add. If any already exist, skip them — the append files
  are single-shot artifacts.
- **Don't confuse the two "FLS" repos.** `ROS_FLOW_2_1` (ResearchFlow fleet)
  is a completely different project with its own deploy skill
  (`researchflow-servers`). This playbook only covers `FLS-Training`.

## Related

- `docs/ARCHITECTURE.md` — how scoring/training fit together
- `docs/CONTINUAL_LEARNING.md` — the learning ledger format
- `memory/learning_ledger.jsonl` — single source of truth for scored runs
