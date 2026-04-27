from __future__ import annotations

from typing import Any

from src.scoring.schema import CriticalError, PenaltyItem, ScoreComponents, ScoringResult


def make_score(
    task_id: str = "task5",
    time: float = 142,
    penalties: float | list[dict[str, Any]] = 61,
    estimated_fls_score: float | None = None,
    critical_errors: list[str] | None = None,
    confidence: float = 0.9,
    video_classification: str = "performance",
) -> ScoringResult:
    penalty_items: list[PenaltyItem]
    if isinstance(penalties, list):
        penalty_items = [PenaltyItem(**penalty) for penalty in penalties]
        penalty_total = sum(p.points_deducted for p in penalty_items)
    else:
        penalty_items = [
            PenaltyItem(
                type="aggregate_penalty",
                description="Aggregate rubric penalty burden.",
                points_deducted=float(penalties),
                severity="moderate" if penalties else "minor",
            )
        ] if penalties else []
        penalty_total = float(penalties)

    max_scores = {"task1": 300, "task2": 300, "task3": 180, "task4": 420, "task5": 600, "task6": 315}
    max_score = max_scores[task_id]
    score_value = (
        float(estimated_fls_score)
        if estimated_fls_score is not None
        else max(0.0, float(max_score) - float(time) - float(penalty_total))
    )

    errors = [
        CriticalError(type=error_type, present=True, reason=f"{error_type} observed")
        for error_type in (critical_errors or [])
    ]

    return ScoringResult(
        id=f"score_{task_id}",
        video_id=f"video_{task_id}",
        video_filename=f"{task_id}.mp4",
        source="test",
        model_name="test_model",
        model_version="test",
        prompt_version="v003",
        task_id=task_id,
        completion_time_seconds=float(time),
        estimated_penalties=penalty_total,
        estimated_fls_score=score_value,
        confidence_score=confidence,
        penalties=penalty_items,
        critical_errors=errors,
        score_components=ScoreComponents(
            max_score=float(max_score),
            time_used=float(time),
            total_penalties=penalty_total,
            total_fls_score=score_value,
            formula_applied=f"{max_score:g} - {time:g} - {penalty_total:g} = {score_value:g}",
        ),
        task_specific_assessments={"video_classification": video_classification},
    )
