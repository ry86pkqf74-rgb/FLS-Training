"""v003 task-aware, rubric-faithful report generation."""
from __future__ import annotations

from typing import Any, Optional

from src.feedback.schema import TraineeProfile
from src.reporting.readiness import derive_blocking_errors, determine_readiness
from src.reporting.render_markdown_v3 import render_markdown_v3
from src.reporting.task_templates import TASK_REPORT_TEMPLATES
from src.rubrics.loader import canonical_task_id, load_rubric
from src.scoring.schema import ScoringResult


DISCLAIMER = "AI-assisted training feedback; not an official FLS certification score."
WARNING_BANNER = (
    "Major technical issue detected. Interpret score with caution; focus on correctness before speed."
)


def _reconciled_score(score: ScoringResult, rubric: dict[str, Any]) -> tuple[float, float, str]:
    max_score = float(rubric["max_score"])
    time_used = float(score.completion_time_seconds or 0)
    penalties = float(score.estimated_penalties or 0)
    if score.score_components:
        penalties = float(score.score_components.total_penalties)
        total_score = float(score.score_components.total_fls_score)
        formula = score.score_components.formula_applied
    else:
        total_score = max(0.0, max_score - time_used - penalties)
        formula = f"{max_score:g} - {time_used:g} - {penalties:g} = {total_score:g}"
    return total_score, penalties, formula


def _score_interpretation(score_value: float | None, penalties: float, rubric: dict[str, Any]) -> str:
    if score_value is None:
        return "Attempt is not scorable from the supplied video evidence."
    local_target = float(rubric.get("passing_score") or 0)
    if not local_target:
        local_target = max(0.0, float(rubric["max_score"]) - float(rubric.get("proficiency_time_seconds", 0)))
    if score_value < local_target and penalties > 0:
        return f"Below local {rubric['task_id'].title()} training target; technical penalties dominate."
    if score_value < local_target:
        return f"Below local {rubric['task_id'].title()} training target."
    return f"Meets the configured local {rubric['task_id'].title()} training target."


def _critical_findings(score: ScoringResult) -> list[dict[str, str]]:
    findings = []
    for error in derive_blocking_errors(score):
        readable = error.type.replace("_", " ")
        if "knot" in error.type:
            focus = "Practice square-knot construction and equal tension across all throws."
            impact = "This is a major technical issue because the knot must maintain closure under tension."
        elif "gap" in error.type or "slit" in error.type:
            focus = "Verify closure before cutting tails."
            impact = "Persistent gap suggests insufficient approximation or tension control."
        elif "needle" in error.type and "view" in error.type:
            focus = "Keep the needle visible through the full custom Task 6 sequence."
            impact = "Needle visibility is required to judge safe ring traversal."
        else:
            focus = "Repeat the task slowly with video review before optimizing time."
            impact = "This blocks a cautious readiness claim until corrected."
        findings.append(
            {
                "type": error.type,
                "observation": error.reason or f"{readable.title()} was observed.",
                "impact": impact,
                "coaching_focus": focus,
            }
        )
    return findings


def _strengths(score: ScoringResult, task_id: str) -> list[str]:
    strengths: list[str] = []
    if score.strengths:
        strengths.extend(strength for strength in score.strengths if strength)
    if score.completion_time_seconds:
        strengths.append(
            f"Efficient task flow: completion time was {score.completion_time_seconds:g} seconds; keep this flow while correcting technical quality issues."
        )
    penalty_types = {penalty.type for penalty in score.penalties}
    if task_id in {"task4", "task5"} and "drain_avulsion" not in penalty_types:
        strengths.append("No drain avulsion was observed in the structured score record.")
    if task_id == "task6" and "needle_out_of_view" not in penalty_types:
        strengths.append("Needle visibility was not flagged as lost in the structured score record.")
    return strengths[:5]


def _priority_for_topic(topic: str, rank: int, template: dict[str, Any]) -> dict[str, str | int]:
    drill_lookup = {
        "knot_security": "Alternating-hand square-knot drill: first throw, switch hands, second throw, switch hands, third throw.",
        "slit_closure": "Closure-before-cut drill: pause before cutting tails and verify no visible slit gap.",
        "needle_visibility": "No-exit field discipline drill: keep needle tip visible continuously for the entire sequence.",
        "ring_accuracy": "Two-ring repeat drill: repeat one ring pair until central and peripheral passes are clean.",
        "penalty_reduction": "Slow deliberate repetitions with post-repetition video review.",
    }
    titles = {
        "knot_security": "Knot security",
        "slit_closure": "Slit closure",
        "needle_visibility": "Needle visibility discipline",
        "ring_accuracy": "Ring-pair targeting accuracy",
        "penalty_reduction": "Penalty reduction before speed optimization",
    }
    observations = {
        "knot_security": "The final knot appeared insecure or was flagged as failed.",
        "slit_closure": "A visible gap or incomplete slit closure was flagged.",
        "needle_visibility": "Needle visibility was flagged as a critical custom Task 6 issue.",
        "ring_accuracy": "Ring traversal accuracy needs focused review.",
        "penalty_reduction": "Penalty burden is limiting the training score more than speed.",
    }
    return {
        "rank": rank,
        "topic": topic,
        "title": titles.get(topic, "Task-specific accuracy"),
        "observation": observations.get(topic, "The structured score record identified a task-specific technique issue."),
        "why_it_matters": "Correct task completion and secure final state are higher priority than additional speed gains.",
        "practice_target": "Complete the relevant phase correctly on three consecutive deliberate repetitions.",
        "drill": drill_lookup.get(topic, template["recommended_drills"][0]),
        "success_metric": "Video review shows the error absent while completion remains within the task time limit.",
    }


def _improvement_priorities(score: ScoringResult, task_id: str, template: dict[str, Any]) -> list[dict[str, Any]]:
    error_types = {error.type for error in derive_blocking_errors(score)}
    penalty_types = {penalty.type for penalty in score.penalties}
    topics: list[str] = []

    if "knot_failure" in error_types or "knot_failure" in penalty_types or "knot_slippage" in penalty_types:
        topics.append("knot_security")
    if {"gap_visible", "visible_gap_after_final_knot", "slit_not_closed"} & (error_types | penalty_types):
        topics.append("slit_closure")
    if {"needle_out_of_view", "needle_leaves_field_of_view"} & (error_types | penalty_types):
        topics.append("needle_visibility")
    if task_id == "task6" and not topics:
        topics.append("ring_accuracy")
    if score.estimated_penalties > 0 and "penalty_reduction" not in topics:
        topics.append("penalty_reduction")
    if not topics:
        topics.append("penalty_reduction")

    return [_priority_for_topic(topic, rank, template) for rank, topic in enumerate(topics[:3], start=1)]


def _phase_breakdown(score: ScoringResult, rubric: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark_map = {
        phase.get("name"): phase.get("intermediate_range", ["", ""])[-1]
        for phase in rubric.get("phases", [])
        if phase.get("name")
    }
    breakdown = []
    for timing in score.phase_timings:
        phase_name = timing.phase.value if hasattr(timing.phase, "value") else str(timing.phase)
        benchmark = benchmark_map.get(phase_name, "")
        interpretation = "Within task-specific benchmark" if benchmark and timing.duration_seconds <= float(benchmark) else "Review for efficiency after technique is correct"
        breakdown.append(
            {
                "phase": phase_name,
                "duration_seconds": timing.duration_seconds,
                "benchmark_seconds": benchmark,
                "interpretation": interpretation,
            }
        )
    return breakdown


def _experimental_metrics(score: ScoringResult, include: bool) -> dict[str, Any]:
    if not include:
        return {"displayed": False}
    raw = score.task_specific_assessments.get("experimental_metrics", {})
    metrics = []
    for name, value in raw.items():
        numeric = float(value)
        interpretation = (
            "higher relative model-derived signal"
            if numeric >= 0
            else "lower relative model-derived signal; may support coaching focus"
        )
        metrics.append(
            {
                "name": name,
                "value": value,
                "scale": "z_score",
                "interpretation": interpretation,
                "not_official_score": True,
            }
        )
    return {
        "displayed": True,
        "title": "AI-derived process metrics",
        "disclaimer": "These are internal model-derived coaching features and are not part of official FLS scoring.",
        "reference_cohort": {"name": "", "n": None, "date_range": "", "validated": False},
        "metrics": metrics,
    }


def generate_report_v3(
    score: ScoringResult,
    previous_scores: list[ScoringResult] | None = None,
    profile: Optional[TraineeProfile] = None,
    include_experimental_metrics: bool = True,
) -> dict:
    """Generate a structured v003 FLS training report."""
    del previous_scores, profile
    task_id = canonical_task_id(score.task_id or "task5")
    rubric = load_rubric(task_id)
    template = TASK_REPORT_TEMPLATES[task_id]
    readiness = determine_readiness(score, rubric)
    unscorable = readiness["label"] == "unscorable"
    training_score, total_penalties, formula = _reconciled_score(score, rubric)

    if readiness["label"] == "automatic_fail":
        training_score = 0.0
    if unscorable:
        training_score_for_report: float | None = None
    else:
        training_score_for_report = training_score

    report = {
        "report_version": "v003",
        "disclaimer": DISCLAIMER,
        "task": {
            "task_id": task_id,
            "task_name": rubric["name"],
            "official_fls_task": bool(rubric["official_fls_task"]),
            "custom_training_task": bool(rubric["custom_training_task"]),
            "certification_eligible": bool(rubric["certification_eligible"]),
            "max_score": float(rubric["max_score"]),
            "max_time_seconds": float(rubric["max_time_seconds"]),
        },
        "score_summary": {
            "training_score": training_score_for_report,
            "max_score": float(rubric["max_score"]),
            "completion_time_seconds": float(score.completion_time_seconds or 0),
            "total_penalties": total_penalties,
            "formula_applied": formula,
            "score_interpretation": _score_interpretation(training_score_for_report, total_penalties, rubric),
        },
        "readiness_status": readiness,
        "critical_findings": _critical_findings(score),
        "strengths": _strengths(score, task_id),
        "improvement_priorities": _improvement_priorities(score, task_id, template),
        "phase_breakdown": _phase_breakdown(score, rubric),
        "task_specific_feedback": {
            "focus": template["phase_focus_rules"],
            "recommended_drills": template["recommended_drills"],
            "assessments": score.task_specific_assessments,
        },
        "next_practice_plan": {
            "steps": [
                "Begin with the top priority drill for 10 minutes.",
                "Record one deliberate full-task repetition focused on correctness.",
                "Review the final state before increasing speed.",
            ]
        },
        "experimental_metrics": _experimental_metrics(score, include_experimental_metrics),
        "cannot_determine": list(score.cannot_determine),
        "confidence": {
            "score": float(score.confidence_score),
            "rationale": score.confidence_rationale,
        },
    }
    if derive_blocking_errors(score):
        report["warning_banner"] = WARNING_BANNER
    report["markdown"] = render_markdown_v3(report)
    return report
