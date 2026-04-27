"""Readiness and proficiency-claim gating for v003 reports."""
from __future__ import annotations

from src.scoring.schema import CriticalError, ScoringResult


BLOCKING_ERROR_TYPES = {
    "task1": {
        "incomplete_task",
        "lost_object",
        "lost_object_outside_field",
        "wrong_transfer_sequence",
    },
    "task2": {
        "gauze_detachment",
        "incomplete_cut",
        "incomplete_circle",
        "large_off_line_deviation",
        "cannot_assess_final_cut",
    },
    "task3": {
        "loop_not_cinched",
        "unsecured_loop",
        "loop_grossly_off_mark",
        "appendage_transection",
        "incomplete_ligation",
        "incomplete_task",
    },
    "task4": {
        "drain_avulsion",
        "knot_failure",
        "knot_slippage",
        "visible_gap_after_knot",
        "slit_not_closed",
        "gross_mark_miss",
        "knot_pusher_failure",
    },
    "task5": {
        "drain_avulsion",
        "knot_failure",
        "visible_gap_after_final_knot",
        "gap_visible",
        "failure_to_complete_required_throws",
        "failure_to_switch_hands",
        "hand_switch_failure",
        "gross_mark_miss",
    },
    "task6": {
        "needle_out_of_view",
        "needle_leaves_field_of_view",
        "block_dislodged",
        "failure_to_complete_required_ring_pairs",
        "incomplete_ring_sequence",
    },
}

AUTO_FAIL_TYPES = {
    "drain_avulsion",
    "needle_out_of_view",
    "needle_leaves_field_of_view",
    "block_dislodged",
    "incomplete_task",
}


def _normalize_error_type(error_type: str) -> str:
    return error_type.strip().lower().replace("-", "_").replace(" ", "_")


def derive_blocking_errors(score: ScoringResult) -> list[CriticalError]:
    """Build blocking critical errors from explicit critical errors and penalties."""
    task_id = score.task_id or "task5"
    blocking_names = BLOCKING_ERROR_TYPES.get(task_id, set())
    errors: list[CriticalError] = [
        error for error in score.critical_errors if error.present and error.blocks_proficiency_claim
    ]
    seen = {_normalize_error_type(error.type) for error in errors}

    for penalty in score.penalties:
        penalty_type = _normalize_error_type(penalty.type)
        if penalty_type not in blocking_names or penalty_type in seen:
            continue
        severity_blocks = penalty.severity in {"major", "critical", "auto_fail"}
        if not severity_blocks and penalty_type not in AUTO_FAIL_TYPES:
            continue
        errors.append(
            CriticalError(
                type=penalty_type,
                present=True,
                reason=penalty.description or f"{penalty_type.replace('_', ' ')} observed.",
                frame_evidence=penalty.frame_evidence,
                forces_zero_score=penalty.severity == "auto_fail" or penalty_type in AUTO_FAIL_TYPES,
                blocks_proficiency_claim=True,
            )
        )
        seen.add(penalty_type)
    return errors


def determine_readiness(score: ScoringResult, rubric: dict) -> dict:
    """Return deterministic readiness label and rationale for a scored attempt."""
    video_classification = str(
        score.task_specific_assessments.get("video_classification", "performance")
    ).lower()
    errors = derive_blocking_errors(score)
    rationale: list[str] = []

    if video_classification in {"instructional", "unusable"}:
        return {
            "label": "unscorable",
            "proficiency_claim_allowed": False,
            "rationale": ["Video appears instructional or unusable rather than a scorable performance."],
        }

    if any(error.forces_zero_score for error in errors):
        return {
            "label": "automatic_fail",
            "proficiency_claim_allowed": False,
            "rationale": [f"Automatic zero condition: {error.type.replace('_', ' ')}." for error in errors if error.forces_zero_score],
        }

    if score.confidence_score < 0.60:
        return {
            "label": "needs_human_review",
            "proficiency_claim_allowed": False,
            "rationale": ["Model confidence below 0.60; human review is needed."],
        }

    if errors:
        rationale.extend(
            f"Major technical error: {error.type.replace('_', ' ')}." for error in errors
        )
        return {
            "label": "needs_focused_remediation",
            "proficiency_claim_allowed": False,
            "rationale": rationale,
        }

    local_target = float(rubric.get("passing_score") or 0)
    if not local_target:
        local_target = max(0.0, float(rubric["max_score"]) - float(rubric.get("proficiency_time_seconds", 0)))

    if score.estimated_fls_score < local_target:
        return {
            "label": "needs_focused_remediation",
            "proficiency_claim_allowed": False,
            "rationale": [f"Score below configured {rubric['task_id'].title()} local training target."],
        }

    if abs(score.estimated_fls_score - local_target) <= local_target * 0.05:
        return {
            "label": "borderline",
            "proficiency_claim_allowed": False,
            "rationale": ["Score is within 5% of the local training target; repeat performance and review penalties."],
        }

    return {
        "label": "meets_local_training_target",
        "proficiency_claim_allowed": True,
        "rationale": ["Score exceeds the configured local training target with no blocking critical errors."],
    }
