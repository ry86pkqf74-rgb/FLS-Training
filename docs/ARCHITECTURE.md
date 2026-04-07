# FLS-Training: System Architecture

## Agent / Model Roles

```
┌─────────────────────────────────────────────────────────────┐
│                    SCORING PIPELINE                          │
│                                                              │
│  ┌──────────────┐   ┌──────────────┐                        │
│  │  TEACHER A    │   │  TEACHER B    │    Frontier VLMs      │
│  │  Claude       │   │  GPT-4o      │    score independently │
│  │  Sonnet 4     │   │              │                        │
│  └──────┬───────┘   └──────┬───────┘                        │
│         │                   │                                │
│         └─────────┬─────────┘                                │
│                   │                                          │
│         ┌─────────▼─────────┐                                │
│         │    CRITIQUE AGENT  │    Compares teacher scores,   │
│         │    Claude Opus 4   │    resolves disagreements,    │
│         │    (or Sonnet 4)   │    produces consensus score   │
│         └─────────┬─────────┘                                │
│                   │                                          │
│         ┌─────────▼─────────┐                                │
│         │  CONSENSUS SCORE   │    Final silver-label used    │
│         │  + FEEDBACK        │    for training & feedback    │
│         └─────────┬─────────┘                                │
│                   │                                          │
│    ┌──────────────▼──────────────┐                           │
│    │      MEMORY STORE           │   DuckDB + JSON files    │
│    │  scores, comparisons,       │   timestamped, versioned │
│    │  corrections, lineage       │                           │
│    └──────────────┬──────────────┘                           │
│                   │                                          │
│    ┌──────────────▼──────────────┐                           │
│    │      STUDENT MODEL          │   Qwen2.5-VL-7B-Instruct│
│    │  (fine-tuned on RunPod)     │   LoRA fine-tuned        │
│    │  Replaces frontier models   │   on consensus + corr.   │
│    │  once good enough           │                           │
│    └──────────────┬──────────────┘                           │
│                   │                                          │
│    ┌──────────────▼──────────────┐                           │
│    │      DRIFT MONITOR          │   Detects degradation,   │
│    │                             │   triggers retraining     │
│    └─────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## Model Specifications

| Role | Model | Purpose | When Used |
|------|-------|---------|-----------|
| **Teacher A** | `claude-sonnet-4-20250514` | Score video frames against FLS rubric | Phase 1-2: every video |
| **Teacher B** | `gpt-4o` (latest) | Independent second scoring | Phase 1-2: every video |
| **Critique Agent** | `claude-sonnet-4-20250514` | Resolve teacher disagreements, produce consensus | Phase 1-2: every video |
| **Student** | `Qwen/Qwen2.5-VL-7B-Instruct` | Local inference after fine-tuning | Phase 3+: replaces teachers |
| **Drift Monitor** | Rule-based + statistics | Detect when student degrades | Phase 4+: continuous |

## Why These Specific Models

- **Claude Sonnet 4 as Teacher A**: Best vision understanding for surgical video, structured JSON output, cost-effective at ~$3/1M input tokens for images
- **GPT-4o as Teacher B**: Independent architecture prevents shared blind spots; strong at spatial reasoning
- **Claude as Critique Agent**: Needs to reason about *why* two scores differ and which is more defensible — requires strong reasoning, same model family as Teacher A is fine since it's evaluating structured data not raw frames
- **Qwen2.5-VL-7B as Student**: Open-weight, strong vision-language performance at 7B scale, LoRA-friendly, runs on single A100, Apache 2.0 license

## Scoring Pipeline Flow (per video)

```
1. INGEST
   video.mov → frame_extractor → 20 uniform frames + 3 final-state frames
                                → video_metadata (duration, resolution, fps)

2. TEACHER SCORING (parallel)
   frames + rubric + prompt → Teacher A (Claude) → ScoringResult A
   frames + rubric + prompt → Teacher B (GPT-4o) → ScoringResult B

3. CRITIQUE & CONSENSUS
   ScoringResult A + ScoringResult B + rubric → Critique Agent →
   {
     agreement_score: 0.0-1.0,
     divergences: [{field, teacher_a_value, teacher_b_value, resolution, reasoning}],
     consensus_score: ScoringResult (final),
     confidence: 0.0-1.0
   }

4. STORE
   All three results → DuckDB + memory/scores/YYYY-MM-DD/
   Consensus → memory/comparisons/YYYY-MM-DD/
   Event → memory/learning_ledger.jsonl

5. FEEDBACK
   Consensus score → feedback_generator → structured feedback report
   (strengths, weaknesses, specific improvement actions)

6. (OPTIONAL) HUMAN CORRECTION
   Expert reviews consensus → submits correction → stored in memory/corrections/
```

## Continual Learning Cycle

```
Score videos (Phase 1-2)
    → Accumulate consensus scores + corrections
    → Prepare training dataset (Phase 3)
    → Fine-tune Student on RunPod
    → Evaluate Student vs held-out frontier consensus
    → If Student meets threshold: promote to production
    → New videos scored by Student (with frontier fallback for low confidence)
    → Corrections continue accumulating
    → Drift detector triggers retrain when needed
    → Repeat
```

## File Naming & Versioning

All artifacts in `memory/` follow:
```
memory/{category}/YYYY-MM-DD/{video_id}_{context}_{YYYYMMDD_HHMMSS}.json
```

Prompt versions: `prompts/v{NNN}_task{N}_{model}.md`
Training datasets: `data/training/YYYY-MM-DD_v{N}/`
Model checkpoints: `memory/model_checkpoints/YYYY-MM-DD_{run_id}/`
