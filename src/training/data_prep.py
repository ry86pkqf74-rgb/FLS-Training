"""Prepare training data for fine-tuning the student model.

Converts scored video data into two training datasets:
1. Scoring head: frames → ScoringResult JSON
2. Coaching head: frames + history → FeedbackReport JSON

Output format: JSONL files compatible with HuggingFace datasets.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.memory.memory_store import MemoryStore
from src.scoring.schema import ScoringResult
from src.feedback.schema import FeedbackReport


def prepare_training_data(
    base_dir: str | Path = ".",
    output_dir: str | Path = "training/data",
    version: str = "v1",
    val_split: float = 0.15,
    min_confidence: float = 0.3,
) -> dict:
    """Prepare training datasets from scored memory.

    Returns:
        dict with paths to scoring_train, scoring_val, coaching_train, coaching_val
    """
    base = Path(base_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(base)
    all_scores = store.get_all_scores()
    all_feedback = store.get_all_feedback()

    # Filter by confidence
    valid_scores = [s for s in all_scores if s.confidence_score >= min_confidence]
    valid_scores.sort(key=lambda s: s.scored_at)

    print(f"Total scores: {len(all_scores)}, above confidence threshold: {len(valid_scores)}")

    # Split train/val
    split_idx = max(1, int(len(valid_scores) * (1 - val_split)))
    train_scores = valid_scores[:split_idx]
    val_scores = valid_scores[split_idx:]

    # === Scoring dataset ===
    scoring_train = _build_scoring_dataset(train_scores)
    scoring_val = _build_scoring_dataset(val_scores)

    scoring_train_path = out / f"scoring_train_{version}.jsonl"
    scoring_val_path = out / f"scoring_val_{version}.jsonl"
    _write_jsonl(scoring_train, scoring_train_path)
    _write_jsonl(scoring_val, scoring_val_path)

    # === Coaching dataset ===
    coaching_train = _build_coaching_dataset(train_scores, all_feedback, all_scores)
    coaching_val = _build_coaching_dataset(val_scores, all_feedback, all_scores)

    coaching_train_path = out / f"coaching_train_{version}.jsonl"
    coaching_val_path = out / f"coaching_val_{version}.jsonl"
    _write_jsonl(coaching_train, coaching_train_path)
    _write_jsonl(coaching_val, coaching_val_path)

    # === Metadata ===
    meta = {
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "min_confidence": min_confidence,
        "val_split": val_split,
        "scoring_train_count": len(scoring_train),
        "scoring_val_count": len(scoring_val),
        "coaching_train_count": len(coaching_train),
        "coaching_val_count": len(coaching_val),
        "files": {
            "scoring_train": str(scoring_train_path),
            "scoring_val": str(scoring_val_path),
            "coaching_train": str(coaching_train_path),
            "coaching_val": str(coaching_val_path),
        }
    }
    meta_path = out / f"meta_{version}.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nDataset prepared:")
    print(f"  Scoring:  {len(scoring_train)} train / {len(scoring_val)} val")
    print(f"  Coaching: {len(coaching_train)} train / {len(coaching_val)} val")
    print(f"  Output:   {out}")

    return meta


def _build_scoring_dataset(scores: list[ScoringResult]) -> list[dict]:
    """Build scoring training examples.

    Each example: {instruction, input (frame descriptions), output (scoring JSON)}
    """
    examples = []
    for score in scores:
        # Build input from frame analyses (simulates what the model sees)
        frame_descriptions = []
        for fa in score.frame_analyses:
            frame_descriptions.append(
                f"Frame {fa.frame_number} ({fa.phase.value if hasattr(fa.phase, 'value') else fa.phase}): {fa.description}"
            )

        input_text = "\n".join(frame_descriptions)

        # Build output (the scoring result without frame analyses)
        output_data = {
            "completion_time_seconds": score.completion_time_seconds,
            "phase_timings": [pt.model_dump() for pt in score.phase_timings],
            "knot_assessments": [ka.model_dump() for ka in score.knot_assessments],
            "suture_placement": score.suture_placement.model_dump() if score.suture_placement else None,
            "drain_assessment": score.drain_assessment.model_dump() if score.drain_assessment else None,
            "estimated_penalties": score.estimated_penalties,
            "estimated_fls_score": score.estimated_fls_score,
            "technique_summary": score.technique_summary,
        }

        examples.append({
            "instruction": "Score this FLS Task 5 intracorporeal suture attempt. Analyze the frames and produce a structured scoring result.",
            "input": input_text,
            "output": json.dumps(output_data),
            "video_id": score.video_id,
            "confidence": score.confidence_score,
        })

    return examples


def _build_coaching_dataset(
    current_scores: list[ScoringResult],
    all_feedback: list[FeedbackReport],
    all_scores: list[ScoringResult],
) -> list[dict]:
    """Build coaching training examples.

    Each example: {instruction, input (frame descriptions + history), output (feedback JSON)}
    """
    # Build lookup for feedback
    feedback_by_video = {f.video_id: f for f in all_feedback}
    scores_by_time = sorted(all_scores, key=lambda s: s.scored_at)

    examples = []
    for score in current_scores:
        # Get matching feedback report
        feedback = feedback_by_video.get(score.video_id)
        if not feedback:
            continue

        # Build frame input
        frame_descriptions = []
        for fa in score.frame_analyses:
            frame_descriptions.append(
                f"Frame {fa.frame_number} ({fa.phase.value if hasattr(fa.phase, 'value') else fa.phase}): {fa.description}"
            )

        # Build history summary (all scores before this one)
        history_lines = []
        for prev in scores_by_time:
            if prev.scored_at >= score.scored_at:
                break
            history_lines.append(
                f"  {prev.video_id}: {prev.completion_time_seconds:.0f}s / "
                f"{prev.estimated_fls_score:.0f} FLS (confidence: {prev.confidence_score:.2f})"
            )

        history_text = "Previous attempts:\n" + "\n".join(history_lines[-10:]) if history_lines else "First attempt."

        input_text = "\n".join(frame_descriptions) + "\n\n" + history_text

        # Output is the full feedback report
        output_data = feedback.model_dump(mode="json")
        # Remove metadata fields the model shouldn't generate
        for key in ["feedback_id", "generated_at", "generator"]:
            output_data.pop(key, None)

        examples.append({
            "instruction": "Analyze this FLS Task 5 attempt and generate detailed coaching feedback. Consider the trainee's history to provide progression-aware recommendations.",
            "input": input_text,
            "output": json.dumps(output_data),
            "video_id": score.video_id,
            "confidence": score.confidence_score,
        })

    return examples


def _write_jsonl(data: list[dict], path: Path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def _serialize_phase(phase) -> str:
    return phase.value if hasattr(phase, 'value') else str(phase)
