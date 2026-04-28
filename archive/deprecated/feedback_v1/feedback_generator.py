"""Generate structured feedback reports from scoring results.

Converts raw ScoringResult data into human-readable, actionable feedback
for surgical residents.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.scoring.schema import Phase, ScoringResult


def generate_feedback(score: ScoringResult) -> dict:
    """Generate a structured feedback report from a ScoringResult.

    Returns a dict suitable for rendering as a report or JSON.
    """
    report = {
        "video_id": score.video_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_score": score.estimated_fls_score,
        "completion_time_seconds": score.completion_time_seconds,
        "max_time_seconds": 600,
        "time_efficiency_pct": round(
            (1 - score.completion_time_seconds / 600) * 100, 1
        ) if score.completion_time_seconds < 600 else 0,
        "pass_likely": score.estimated_fls_score > 0 and score.completion_time_seconds < 600,
        "confidence": score.confidence_score,
    }

    # Phase breakdown
    phase_analysis = []
    if score.phase_timings:
        total_time = score.completion_time_seconds or sum(
            pt.duration_seconds for pt in score.phase_timings
        )
        for pt in score.phase_timings:
            pct = round(pt.duration_seconds / total_time * 100, 1) if total_time > 0 else 0
            phase_analysis.append({
                "phase": pt.phase.value,
                "duration_seconds": pt.duration_seconds,
                "percentage_of_total": pct,
                "note": _phase_time_note(pt.phase, pt.duration_seconds),
            })
    report["phase_breakdown"] = phase_analysis

    # Knot quality
    knot_report = []
    for ka in score.knot_assessments:
        knot_report.append({
            "throw": ka.throw_number,
            "type": "surgeon's knot" if ka.is_surgeon_knot else "single throw",
            "secure": ka.appears_secure,
            "hand_switched": ka.hand_switched,
            "hand_used": ka.hand_used.value,
            "feedback": _knot_feedback(ka),
        })
    report["knot_quality"] = knot_report

    # Suture placement
    if score.suture_placement:
        sp = score.suture_placement
        report["suture_placement"] = {
            "deviation_mark1_mm": sp.deviation_from_mark1_mm,
            "deviation_mark2_mm": sp.deviation_from_mark2_mm,
            "total_penalty": sp.total_deviation_penalty,
            "confidence": sp.confidence.value,
            "feedback": _placement_feedback(sp.total_deviation_penalty),
        }

    # Drain assessment
    if score.drain_assessment:
        da = score.drain_assessment
        report["drain_status"] = {
            "gap_visible": da.gap_visible,
            "avulsed": da.drain_avulsed,
            "closure_quality": da.slit_closure_quality,
            "feedback": _drain_feedback(da),
        }

    # Compiled feedback
    report["strengths"] = score.strengths
    report["improvements"] = score.improvement_suggestions
    report["technique_summary"] = score.technique_summary

    # Priority action items (top 3 most impactful)
    report["priority_actions"] = _prioritize_actions(score)

    return report


def _phase_time_note(phase: Phase, duration: float) -> str:
    """Contextual note about phase duration."""
    benchmarks = {
        Phase.NEEDLE_LOAD: (10, 30, "Efficient needle loading saves significant time"),
        Phase.SUTURE_PLACEMENT: (20, 60, "Smooth needle driving through both marks is key"),
        Phase.FIRST_THROW: (15, 45, "The surgeon's knot requires practice for speed"),
        Phase.SECOND_THROW: (10, 30, "Hand switching should become automatic"),
        Phase.THIRD_THROW: (10, 30, "Final throw — maintain focus on square knot"),
        Phase.SUTURE_CUT: (5, 15, "Clean cuts close to the knot"),
    }
    if phase in benchmarks:
        fast, slow, tip = benchmarks[phase]
        if duration <= fast:
            return f"Excellent speed. {tip}"
        elif duration >= slow:
            return f"Slower than typical. {tip}"
        else:
            return f"Reasonable pace. {tip}"
    return ""


def _knot_feedback(ka) -> str:
    issues = []
    if ka.throw_number == 1 and not ka.is_surgeon_knot:
        issues.append("First throw must be a surgeon's knot (double throw)")
    if ka.throw_number > 1 and not ka.hand_switched:
        issues.append(f"Must switch hands before throw {ka.throw_number} for a square knot")
    if not ka.appears_secure:
        issues.append(f"Throw {ka.throw_number} does not appear secure")
    return "; ".join(issues) if issues else "Good technique"


def _placement_feedback(deviation: float) -> str:
    if deviation <= 1.0:
        return "Excellent precision — minimal deviation from marks"
    elif deviation <= 3.0:
        return "Acceptable placement — minor deviation"
    else:
        return f"Significant deviation ({deviation}mm total) — focus on needle angle and mark alignment"


def _drain_feedback(da) -> str:
    if da.drain_avulsed:
        return "CRITICAL: Drain avulsed from block — automatic score of zero. Use gentler tissue handling."
    if da.gap_visible:
        return "Gap visible in drain slit — tighten knot more before locking throws"
    if da.slit_closure_quality == "complete":
        return "Excellent closure of drain slit"
    return "Partial closure — ensure adequate tension on each throw"


def _prioritize_actions(score: ScoringResult) -> list[str]:
    """Identify the 3 most impactful improvement actions."""
    actions = []

    # Check for automatic failures first
    if score.drain_assessment and score.drain_assessment.drain_avulsed:
        actions.append("CRITICAL: Practice gentler tissue handling to prevent drain avulsion")

    # Knot issues
    for ka in score.knot_assessments:
        if ka.throw_number == 1 and not ka.is_surgeon_knot:
            actions.append("Practice the surgeon's knot (double throw) as your first throw")
        if ka.throw_number > 1 and ka.hand_switched is False:
            actions.append(f"Practice hand exchange before throw {ka.throw_number}")
        if not ka.appears_secure:
            actions.append(f"Work on knot security for throw {ka.throw_number}")

    # Time efficiency
    if score.completion_time_seconds > 400:
        actions.append("Focus on overall speed — current time leaves little margin for penalties")
    elif score.completion_time_seconds > 300:
        actions.append("Good pace but room for improvement — identify your slowest phase")

    # Placement
    if score.suture_placement and score.suture_placement.total_deviation_penalty > 3:
        actions.append("Improve needle placement precision — practice hitting both marks consistently")

    return actions[:3]


def feedback_to_markdown(report: dict) -> str:
    """Render a feedback report as markdown."""
    lines = [
        f"# FLS Task 5 Feedback Report",
        f"**Video:** {report['video_id']}  ",
        f"**Generated:** {report['generated_at']}  ",
        f"**FLS Score:** {report['overall_score']:.1f} / 600  ",
        f"**Time:** {report['completion_time_seconds']:.1f}s  ",
        f"**Pass Likely:** {'Yes' if report['pass_likely'] else 'No'}  ",
        "",
        "## Priority Actions",
    ]

    for i, action in enumerate(report.get("priority_actions", []), 1):
        lines.append(f"{i}. {action}")

    lines.extend(["", "## Strengths"])
    for s in report.get("strengths", []):
        lines.append(f"- {s}")

    lines.extend(["", "## Areas for Improvement"])
    for s in report.get("improvements", []):
        lines.append(f"- {s}")

    if report.get("phase_breakdown"):
        lines.extend(["", "## Phase Breakdown"])
        lines.append("| Phase | Duration | % of Total | Note |")
        lines.append("|-------|----------|------------|------|")
        for p in report["phase_breakdown"]:
            lines.append(f"| {p['phase']} | {p['duration_seconds']:.1f}s | {p['percentage_of_total']}% | {p['note']} |")

    lines.extend(["", f"## Technique Summary", "", report.get("technique_summary", "")])

    return "\n".join(lines)
