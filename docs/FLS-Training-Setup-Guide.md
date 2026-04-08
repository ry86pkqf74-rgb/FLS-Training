# FLS-Training: Setup & Cursor Agent Prompts

## Step 1 — Push the Repo to GitHub

```bash
# Extract the tarball (or copy the FLS-Training directory)
cd ~/projects  # or wherever you work
tar xzf FLS-Training-repo.tar.gz
cd FLS-Training

# Initialize and push
git init
git remote add origin https://github.com/ry86pkqf74-rgb/FLS-Training.git
git add -A
git commit -m "feat: initial FLS-Training framework with teacher-critique-student architecture"
git branch -M main
git push -u origin main
```

## Step 2 — Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY and OPENAI_API_KEY
```

## Step 3 — Verify Setup

```bash
python scripts/090_status.py
# Should show all zeros — empty database, ready to go
```

## Step 4 — Score Your First Video

This is where you start feeding videos through Claude + ChatGPT side-by-side.

```bash
# Ingest
python scripts/010_ingest_video.py --video ~/videos/PostV7.mov --task 5

# Score (this calls both Claude and GPT-4o APIs)
python scripts/020_score_frontier.py --video-id <ID_FROM_ABOVE> --video ~/videos/PostV7.mov

# The script will:
# 1. Extract 20 uniform + 3 final-state frames
# 2. Send to Claude → get ScoringResult A
# 3. Send to GPT-4o → get ScoringResult B
# 4. Run Critique Agent → produce consensus score
# 5. Save everything to memory/ and DuckDB
# 6. Print rich-formatted comparison table
```

## Step 5 — Review & Correct (Repeat for Each Video)

After each scoring run, review the output. If you disagree with any field:

```bash
python scripts/025_submit_correction.py \
    --video-id <ID> \
    --score-id <CONSENSUS_SCORE_ID> \
    --corrected-fields '{"completion_time_seconds": 147, "estimated_fls_score": 453}' \
    --corrector expert \
    --notes "Verified time from video, knot was secure"
```

## Step 6 — Batch Processing

Once you have the workflow down, you can score videos in a loop:

```bash
for video in ~/videos/*.mov; do
    ID=$(python scripts/010_ingest_video.py --video "$video" --task 5 2>/dev/null | grep "ID" | awk '{print $2}')
    python scripts/020_score_frontier.py --video-id "$ID" --video "$video"
    echo "---"
done
```

---

## Cursor / Claude Code Agent Prompts

Use these prompts in Cursor to extend the system as needed.

### Prompt: Add a New FLS Task

```
In the FLS-Training repo, add support for FLS Task {N} ({task_name}).

1. Create rubrics/task{N}_{snake_name}.yaml with the official scoring criteria:
   - Max time, equipment, scoring formula, penalties, phases
   (Reference: https://www.flsprogram.org FLS Manual Skills Guidelines)

2. Create prompts/v001_task{N}_system.md with a scoring prompt that:
   - Describes the task procedure
   - Lists all penalty criteria
   - Specifies the JSON output format (reuse ScoringResult schema)
   - Includes task-specific assessment fields

3. Create prompts/v001_task{N}_critique.md for the critique agent

4. Update prompts/prompt_registry.yaml with the new entry

5. Update the FLSTask enum in src/scoring/schema.py if not already present

6. Test by scoring a sample video:
   python scripts/020_score_frontier.py --video-id test --video sample.mov --task {N}
```

### Prompt: Improve a Scoring Prompt

```
In the FLS-Training repo, I've been scoring Task 5 videos and the model
consistently gets {specific_issue} wrong (e.g., "misidentifies hand switches",
"overestimates suture deviation", "misses the second throw phase").

Review prompts/v001_task5_system.md and create v002 that:
1. Addresses this specific issue with clearer instructions
2. Adds examples if helpful
3. Keeps all other scoring criteria unchanged
4. Updates prompt_registry.yaml with v002 entry (status: testing)

Also create a test plan: which existing scored videos should be re-scored
with v002 to measure improvement? Reference memory/scores/ for available videos.
```

### Prompt: Build the Streamlit Review UI

```
In the FLS-Training repo, create a Streamlit app at src/app.py that provides:

1. Dashboard tab:
   - Total videos scored, corrections, training runs
   - Score distribution histogram
   - Confidence trend over time
   - API cost tracker

2. Score Review tab:
   - Select a video from DuckDB
   - Show side-by-side Teacher A vs Teacher B vs Consensus scores
   - Highlight divergences in red
   - Button to submit corrections inline

3. Feedback tab:
   - Select a video
   - Show the generated feedback report (from feedback_generator)
   - Export as PDF or markdown

Use src/memory/memory_store.py for all data access.
Run with: streamlit run src/app.py
```

### Prompt: Add Instrument Tracking

```
In the FLS-Training repo, add instrument tracking to the frame analysis pipeline.

1. Create src/ingest/instrument_tracker.py that:
   - Uses a pretrained surgical instrument detection model
     (research: CholecT50 dataset, SurgToolLoc, or YOLO fine-tuned on surgical tools)
   - Detects: left_needle_driver, right_needle_driver, scissors, needle, suture
   - Returns bounding boxes + labels per frame
   - Tracks instrument positions across frames for motion analysis

2. Add to src/ingest/phase_detector.py:
   - Use instrument positions to automatically detect phase transitions
   - Needle in driver = needle_load phase
   - Needle through tissue = suture_placement
   - Suture wrapping = throw phases
   - Scissors in frame = suture_cut

3. Add motion metrics to ScoringResult:
   - total_instrument_path_length_px
   - economy_of_motion_score
   - idle_time_seconds (frames with no instrument movement)

This feeds into more objective scoring that doesn't rely on VLM interpretation.
```

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│  YOUR WORKFLOW (repeat for each video)                   │
│                                                          │
│  1. python scripts/010_ingest_video.py --video X --task 5│
│  2. python scripts/020_score_frontier.py --video-id Y    │
│  3. Review output → optionally correct with 025_*        │
│  4. python scripts/080_generate_feedback_report.py       │
│                                                          │
│  After ~50 videos:                                       │
│  5. python scripts/040_prepare_training_data.py --ver X  │
│  6. Fine-tune on RunPod via LAUNCH_GUIDE / RUNBOOK       │
│  7. Evaluate → promote → student takes over              │
│                                                          │
│  Ongoing:                                                │
│  8. python scripts/075_check_drift.py                    │
│  9. Retrain when triggered                               │
└─────────────────────────────────────────────────────────┘
```
