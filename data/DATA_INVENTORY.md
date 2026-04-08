# FLS-Training Data Inventory

Snapshot: **2026-04-08** (derived from `memory/` and `data/training/2026-04-07_v1/`
on branch `data/scaling-sources`). Regenerate any time by re-running the queries
shown in each section against the repo.

## TL;DR

- **Videos ingested (all sources):** 180 (per `learning_ledger.jsonl`)
- **Videos ingested from YouTube harvest specifically:** 43 unique IDs in
  `harvest_log.jsonl`, plus internal practice clips (`V6`–`V31`, `postV*`,
  `lap_pre`, etc.) bringing the scored set to 32 unique videos.
- **Teacher scores saved:** 194 events in the learning ledger (reflects retries
  and regenerations); on disk that resolves to **58 score files across 29
  unique videos per teacher** (Claude Sonnet 4 + GPT-4o).
- **Consensus pairs generated:** 31 `*_video_consensus_*.json` files in
  `memory/comparisons/` (31 `consensus_generated` events + 1
  `consensus_regenerated`).
- **Training set in `data/training/2026-04-07_v1/`:** **32 total examples**
  (25 train / 3 val / 4 test) — entirely Task 5, `min_confidence=0.4`.
- **Gap to 500 examples:** **-468**.

## Per-task breakdown

All current ingestion, scoring and training is **Task 5 (intracorporeal
suturing)**. The other four FLS tasks have effectively **zero** scored clips
and zero training examples.

| Task | Scored clips | Consensus pairs | Training examples | Harvested raw (YT) |
|------|-------------:|----------------:|------------------:|-------------------:|
| 1 — Peg Transfer                | 0 | 0 | 0 | ~0 (any stragglers in 43-video bucket are not bucketed) |
| 2 — Pattern Cut                 | 0 | 0 | 0 | ~0 |
| 3 — Endoloop / Ligating Loop    | 0 | 0 | 0 | ~0 |
| 4 — Extracorporeal Knot Tying   | 0 | 0 | 0 | ~0 |
| 5 — Intracorporeal Suturing     | 32 | 31 | 32 | 43 (9 high-conf, 20 medium-conf, rest unknown) |

> The 43 harvested YouTube videos tag `task5_confidence` in `harvest_log.jsonl`
> but do not assign the clip to any other task, so non-Task-5 content that
> slipped through the classifier is currently uncategorised.

## Scoring by teacher

From `memory/scores/2026-04-07/` (65 files total, filename parse):

| Teacher          | Unique videos scored |
|------------------|---------------------:|
| claude-sonnet-4  | 29 |
| gpt-4o           | 29 |
| (consensus files, separate) | 31 |

Teacher overlap is effectively complete — every video scored by Claude has also
been scored by GPT-4o, which is why the consensus-pair count (31) tracks the
per-teacher count (29) so closely.

## Training manifest (`data/training/2026-04-07_v1/manifest.yaml`)

```yaml
total_examples: 32
n_train: 25
n_val: 3
n_test: 4
min_confidence: 0.4
include_coach_feedback: false
sources:
  critique_consensus: 30
  teacher_claude: 1
  teacher_gpt4o: 1
```

**All 25 training examples are Task 5.** The single Claude-teacher and single
GPT-4o-teacher examples are kept as anchor/calibration points. The confidence
threshold of 0.4 is already quite loose and produced only 32 useful examples
out of 31 consensus pairs plus 29 × 2 = 58 solo teacher scores.

## Ledger event counts (`memory/learning_ledger.jsonl`)

| Event type            | Count |
|-----------------------|------:|
| video_ingested        | 180 |
| score_saved           | 194 |
| frontier_scored       | 53 |
| feedback_saved        | 86 |
| teacher_comparison    | 23 |
| consensus_generated   | 31 |
| consensus_regenerated | 1 |
| dataset_prepared      | 1 |
| score_corrected       | 1 |
| score_relabeled       | 1 |
| score_metadata_fixed  | 1 |

The 180 `video_ingested` events vs. 32 unique *scored* videos means roughly
~148 ingested clips **have not been scored yet** — most of those are internal
practice/debug frames and YouTube clips from the 43-video harvest bucket that
failed task classification or fell below the ingestion confidence threshold.

## Gap analysis: what it takes to reach 500 training examples

The pipeline's historical yield (observed in the current manifest) is roughly
**1 training example per scored clip**, because consensus pairs are 1:1 with
videos and each one counts as a single supervised example.

To hit **500 total training examples** we need ~**500 scored clips that survive
the confidence filter**, across all 5 tasks. Current gap: **-468**.

### Required uplift by source

| Source | Current | Target delta | Realistic? | Notes |
|---|---:|---:|---|---|
| YouTube harvest (Task 5)    | 32 | +50  | Yes | Already harvested ~43; just finish scoring + rerun prep. |
| YouTube harvest (Tasks 1–4) |  0 | +100 | Yes | See `data/harvest_targets.csv`. Requires 40–60 new ingested clips per task. |
| LASANA (all 4 tasks)        |  0 | +300 | Yes, once downloaded | See `data/external/LASANA_README.md`. Biggest single lever. |
| SPD-FLS1 (Task 1)           |  0 | +100 | Yes | Longitudinal peg-transfer dataset; see `DATASET_CATALOG.md`. |
| Internal practice captures  | 32 | +50  | Yes | More `postV*` / `V*` clips as practice continues. |

Totals to 500+: **32 current + 600 potential = ample headroom**, assuming
LASANA is downloadable and the pair yield matches the estimate in
`LASANA_README.md`.

### Required uplift by task (to 100 examples/task)

| Task | Current | Target | Gap |
|------|--------:|-------:|----:|
| 1 — Peg Transfer            |  0 | 100 | -100 |
| 2 — Pattern Cut             |  0 | 100 | -100 |
| 3 — Endoloop                |  0 | 100 | -100 |
| 4 — Extracorporeal Knot     |  0 | 100 | -100 |
| 5 — Intracorporeal Suturing | 32 | 100 |  -68 |
| **Total**                   | 32 | 500 | **-468** |

## Known limitations of this snapshot

- Score files are not self-describing about which FLS task they cover; task
  attribution is currently inferred from the prompt embedded in the training
  JSONL and from filename conventions (`V*`, `postV*` = Task 5 practice).
- `video_ingested` events in the learning ledger include internal debug and
  frame-extraction artefacts, so 180 should not be interpreted as "180 unique
  training-quality videos".
- Task 1–4 counts are effectively zero; any filtering bug that hides Task 1–4
  clips would not change the picture meaningfully.

## How to regenerate this file

```bash
# Ingestion / score / consensus counts
awk -F'"event_type":' '{print $2}' memory/learning_ledger.jsonl \
  | awk -F'"' '{print $2}' | sort | uniq -c

# Unique scored videos per teacher
ls memory/scores/2026-04-07/ | ...  # see dedup script in PR

# Training manifest
cat data/training/2026-04-07_v1/manifest.yaml
wc -l data/training/2026-04-07_v1/*.jsonl

# YouTube harvest
grep -c '"event": "downloaded"' harvest_log.jsonl
```
