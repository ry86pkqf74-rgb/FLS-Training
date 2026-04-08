"""Prepare training data for fine-tuning the student model.

Converts scored video data into two training datasets:
1. Scoring head: frames → ScoringResult JSON
2. Coaching head: frames + history → FeedbackReport JSON

Output format: JSONL files compatible with HuggingFace datasets.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _ts(s) -> datetime:
    """Normalize ScoringResult.scored_at to UTC-aware for safe comparison.

    Older score files were written with naive datetimes (datetime.utcnow);
    newer YouTube-harvest files include explicit UTC offsets. Mixing them
    in a sort raises TypeError, so we coerce naive values to UTC.
    """
    dt = s.scored_at if hasattr(s, "scored_at") else s
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

from src.memory.memory_store import MemoryStore
from src.scoring.schema import ScoringResult
from src.feedback.schema import FeedbackReport


_TASK_LABELS = {
    "task1": "FLS Task 1 peg transfer",
    "task2": "FLS Task 2 pattern cut",
    "task3": "FLS Task 3 ligating loop",
    "task4": "FLS Task 4 extracorporeal suture",
    "task5": "FLS Task 5 intracorporeal suture",
}


def _canonical_task_id(task_id: str | None) -> str:
    raw = str(task_id or "task5").strip().lower()
    aliases = {
        "task1_peg_transfer": "task1",
        "task2_pattern_cut": "task2",
        "task3_endoloop": "task3",
        "task3_ligating_loop": "task3",
        "task4_extracorporeal_knot": "task4",
        "task4_extracorporeal_suture": "task4",
        "task5_intracorporeal_suturing": "task5",
        "task5_intracorporeal_suture": "task5",
    }
    if raw in aliases:
        return aliases[raw]
    if raw.startswith("task"):
        return raw.split("_", 1)[0]
    if raw.isdigit():
        return f"task{raw}"
    return "task5"


def _task_instruction(task_id: str | None, action: str) -> str:
    canonical = _canonical_task_id(task_id)
    label = _TASK_LABELS.get(canonical, _TASK_LABELS["task5"])
    if action == "score":
        return f"Score this {label} attempt. Analyze the frames and produce a structured scoring result."
    return f"Analyze this {label} attempt and generate detailed coaching feedback. Consider the trainee's history to provide progression-aware recommendations."


# Source preference order when the same video has multiple scores.
# Consensus is the highest-quality teacher signal because it has been
# reconciled across Claude + GPT; raw teacher scores are fallback.
_SOURCE_PRIORITY = {
    "consensus": 0,
    "teacher_claude_corrected": 1,
    "teacher_claude": 2,
    "teacher_gpt": 3,
    "student": 99,
}


def _dedupe_by_video(scores: list[ScoringResult]) -> list[ScoringResult]:
    """Collapse multiple scores for the same video to a single record.

    Selection rule:
        1. Lowest source-priority value (consensus > corrected > claude > gpt).
        2. Tiebreak: latest scored_at.

    Drops zero records (other than dropping duplicates) and preserves
    deterministic ordering by video_id.
    """
    by_video: dict[str, ScoringResult] = {}
    for s in scores:
        key = s.video_id
        prev = by_video.get(key)
        if prev is None:
            by_video[key] = s
            continue
        prio_new = _SOURCE_PRIORITY.get(s.source, 50)
        prio_prev = _SOURCE_PRIORITY.get(prev.source, 50)
        if prio_new < prio_prev or (prio_new == prio_prev and _ts(s) > _ts(prev)):
            by_video[key] = s
    return [by_video[k] for k in sorted(by_video.keys())]


def _stratified_video_split(
    scores: list[ScoringResult],
    val_split: float,
    seed: int,
) -> tuple[list[ScoringResult], list[ScoringResult]]:
    """Split by video_id with a fixed seed.

    Holds out whole videos for validation rather than slicing the
    chronological tail. This produces an honest generalization signal
    for the Phase-3 gate (MAE on held-out trainee).
    """
    rng = random.Random(seed)
    video_ids = sorted({s.video_id for s in scores})
    rng.shuffle(video_ids)
    n_val = max(1, int(round(len(video_ids) * val_split)))
    val_ids = set(video_ids[:n_val])
    train = [s for s in scores if s.video_id not in val_ids]
    val = [s for s in scores if s.video_id in val_ids]
    return train, val


def prepare_training_data(
    base_dir: str | Path = ".",
    output_dir: str | Path = "training/data",
    version: str = "v1",
    val_split: float = 0.15,
    min_confidence: float = 0.3,
    seed: int = 42,
) -> dict:
    """Prepare training datasets from scored memory.

    Returns:
        dict with paths to scoring_train, scoring_val, coaching_train, coaching_val
    """
    base = Path(base_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    store = MemoryStore(base)
    all_scores = store.get_all_scores()  # already filters superseded
    all_feedback = store.get_all_feedback()

    # Filter by confidence
    valid_scores = [s for s in all_scores if s.confidence_score >= min_confidence]

    # Dedupe by video_id (consensus > corrected > raw teacher; latest wins on ties)
    pre_dedupe = len(valid_scores)
    valid_scores = _dedupe_by_video(valid_scores)
    post_dedupe = len(valid_scores)

    print(
        f"Total scores: {len(all_scores)} | "
        f"above conf {min_confidence}: {pre_dedupe} | "
        f"after video dedupe: {post_dedupe}"
    )

    # Stratified train/val split by video_id with fixed seed
    train_scores, val_scores = _stratified_video_split(valid_scores, val_split, seed)
    print(
        f"Split (seed={seed}): {len(train_scores)} train videos / "
        f"{len(val_scores)} val videos"
    )

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
        "seed": seed,
        "split_strategy": "stratified_by_video_id",
        "dedup_strategy": "source_priority_then_latest",
        "raw_score_count": len(all_scores),
        "after_confidence_filter": pre_dedupe,
        "after_video_dedupe": post_dedupe,
        "train_video_ids": sorted({s.video_id for s in train_scores}),
        "val_video_ids": sorted({s.video_id for s in val_scores}),
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
            "instruction": _task_instruction(score.task_id, "score"),
            "input": input_text,
            "output": json.dumps(output_data),
            "video_id": score.video_id,
            "confidence": score.confidence_score,
            "task_id": _canonical_task_id(score.task_id),
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
    scores_by_time = sorted(all_scores, key=_ts)

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
        score_ts = _ts(score)
        for prev in scores_by_time:
            if _ts(prev) >= score_ts:
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
            "instruction": _task_instruction(score.task_id, "coach"),
            "input": input_text,
            "output": json.dumps(output_data),
            "video_id": score.video_id,
            "confidence": score.confidence_score,
            "task_id": _canonical_task_id(score.task_id),
        })

    return examples


def _write_jsonl(data: list[dict], path: Path):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def _serialize_phase(phase) -> str:
    return phase.value if hasattr(phase, 'value') else str(phase)
