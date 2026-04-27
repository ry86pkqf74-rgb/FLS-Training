"""Validation checks for v003 structured reports."""
from __future__ import annotations

from src.reporting.readiness import derive_blocking_errors
from src.scoring.schema import ScoringResult


def validate_report_v3(report: dict, score: ScoringResult, rubric: dict) -> list[str]:
    errors: list[str] = []
    markdown = str(report.get("markdown", ""))
    markdown_lower = markdown.lower()

    if report.get("score_summary", {}).get("max_score") != float(rubric["max_score"]):
        errors.append("wrong denominator")

    readiness_label = report.get("readiness_status", {}).get("label")
    if readiness_label != "unscorable":
        expected_score = score.estimated_fls_score
        if score.score_components:
            expected_score = score.score_components.total_fls_score
        if report.get("score_summary", {}).get("training_score") != expected_score:
            errors.append("training score does not match scoring result")

    if score.estimated_penalties > 0 and "no significant penalties" in markdown_lower:
        errors.append("contradictory no-penalties language")

    critical_errors = derive_blocking_errors(score)
    if critical_errors:
        if "proficient" in markdown_lower:
            errors.append("proficiency language with critical error")
        if report.get("readiness_status", {}).get("proficiency_claim_allowed") is not False:
            errors.append("readiness gating allows claim despite critical error")

    overall_assessment = str(report.get("overall_assessment", ""))
    if "global z-score" in overall_assessment.lower():
        errors.append("z-score appears in primary assessment")

    priorities = report.get("improvement_priorities", [])
    if len(priorities) < 1:
        errors.append("missing task-specific priority")
    for priority in priorities:
        for field in ("observation", "practice_target", "drill", "success_metric"):
            if not priority.get(field):
                errors.append(f"priority missing {field}")

    return errors
