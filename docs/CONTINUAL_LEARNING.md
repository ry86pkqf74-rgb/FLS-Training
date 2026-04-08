# Continual Learning System

## How the Learning Loop Works

### Phase 1-2: Bootstrap (You Are Here)

1. Score each video with **both** Claude and GPT-4o via `020_score_frontier.py`
2. The **Critique Agent** automatically resolves disagreements and produces a consensus score
3. Review consensus scores — submit corrections where needed via `025_submit_correction.py`
4. Every score, critique, and correction is timestamped and logged to:
   - `memory/scores/` — individual score JSONs
   - `memory/comparisons/` — critique/consensus JSONs
   - `memory/corrections/` — human correction JSONs
   - `memory/learning_ledger.jsonl` — append-only event stream
   - `data/fls_training.duckdb` — queryable index

### Phase 3: First Fine-Tune

When you have ~50-100 scored videos (ideally with some corrections):

```bash
python scripts/040_prepare_training_data.py --ver <version>
# Then on RunPod, follow deploy/LAUNCH_GUIDE.md or docs/RUNPOD_RUNBOOK.md
bash deploy/runpod_launch.sh <dataset_path> <config_path>
```

### Phase 4: Student Takes Over

Once the student model meets quality thresholds (evaluated against held-out consensus scores), it becomes the primary scorer. Frontier models become the fallback for low-confidence scores.

### Phase 5: Ongoing Learning

```
New video → Student scores it
  → If confidence < 0.7: also score with frontier (fallback)
  → Resident reviews feedback
  → Expert optionally corrects
  → Corrections accumulate
  → Drift detector runs (scripts/075_check_drift.py)
  → When triggered: prepare new dataset, retrain, evaluate, promote
```

## Retraining Triggers

The drift detector checks these conditions:

| Trigger | Threshold | Rationale |
|---------|-----------|-----------|
| New corrections | ≥ 20 | Enough new signal to improve |
| Confidence drop | > 10% decrease | Model uncertain on new data |
| Time since training | > 30 days | Periodic refresh |
| Frontier agreement | < 85% | Prompts or scoring criteria may need updating |

## Data Lineage

Every training example traces back to:
- Which video it came from
- Which model(s) scored it
- Which prompt version was used
- Whether it was corrected by an expert
- When each step happened

This is tracked in `manifest.yaml` in each training dataset directory and in the learning ledger.
