"""Markdown renderer for v003 FLS training reports."""
from __future__ import annotations


def render_markdown_v3(report: dict) -> str:
    """Render a structured v003 report to markdown."""
    task = report["task"]
    summary = report["score_summary"]
    readiness = report["readiness_status"]
    confidence = report.get("confidence", {})

    lines = ["# FLS Training Feedback Report", "", "## 1. Task and Score Summary"]
    lines.append(f"- Task: {task['task_name']}")
    if task.get("custom_training_task"):
        lines.append("- Note: Task 6 is a custom training task and is not part of official FLS certification scoring.")
        lines.append("- Label: Custom training task, not official FLS manual skills task.")
    if summary.get("training_score") is not None:
        lines.append(f"- Training score: {summary['training_score']:g} / {summary['max_score']:g}")
        lines.append(f"- Formula: {summary['formula_applied']}")
    lines.append(f"- Completion time: {summary.get('completion_time_seconds', 0):g} s")
    lines.append(f"- Penalty total: {summary.get('total_penalties', 0):g}")
    lines.append(f"- Confidence: {confidence.get('score', '')}")
    lines.append(f"- Training-readiness status: {readiness['label']}")

    lines.extend([
        "",
        "## 2. Important Interpretation",
        report["disclaimer"],
    ])
    if report.get("warning_banner"):
        lines.append("")
        lines.append(f"**{report['warning_banner']}**")

    lines.extend(["", "## 3. Critical Findings"])
    if report.get("critical_findings"):
        for finding in report["critical_findings"]:
            lines.append(f"- {finding['observation']} Impact: {finding['impact']} Coaching focus: {finding['coaching_focus']}")
    else:
        lines.append("- No blocking critical error was identified from the available evidence.")

    lines.extend(["", "## 4. Strengths"])
    for strength in report.get("strengths", []):
        lines.append(f"- {strength}")
    if not report.get("strengths"):
        lines.append("- No specific strength could be determined from the structured score record.")

    lines.extend(["", "## 5. Priority Improvements"])
    for priority in report.get("improvement_priorities", [])[:3]:
        lines.append(f"### {priority['rank']}. {priority['title']}")
        lines.append(f"- Observation: {priority['observation']}")
        lines.append(f"- Why it matters: {priority['why_it_matters']}")
        lines.append(f"- Practice target: {priority['practice_target']}")
        lines.append(f"- Recommended drill: {priority['drill']}")
        lines.append(f"- How to know it improved: {priority['success_metric']}")
    if not report.get("improvement_priorities"):
        lines.append("- Continue structured repetitions while keeping the full task in view for review.")

    lines.extend(["", "## 6. Phase Breakdown"])
    lines.append("| Phase | Duration | Benchmark | Interpretation |")
    lines.append("|-------|----------|-----------|----------------|")
    for phase in report.get("phase_breakdown", []):
        lines.append(
            f"| {phase['phase']} | {phase['duration_seconds']:g}s | "
            f"{phase.get('benchmark_seconds', '')}s | {phase.get('interpretation', '')} |"
        )
    if not report.get("phase_breakdown"):
        lines.append("| Not available |  |  | Phase timing was not available. |")

    lines.extend(["", "## 7. Task-Specific Coaching"])
    task_feedback = report.get("task_specific_feedback", {})
    for item in task_feedback.get("focus", []):
        lines.append(f"- {item}")
    if task.get("task_id") == "task6":
        lines.append("- Task-specific findings should track rings completed, rings missed, needle visibility, block stability, and central/peripheral alternation.")

    lines.extend(["", "## 8. Next Practice Plan"])
    plan = report.get("next_practice_plan", {})
    for step in plan.get("steps", []):
        lines.append(f"- {step}")

    if report.get("experimental_metrics", {}).get("displayed"):
        metrics = report["experimental_metrics"]
        lines.extend(["", "## 9. Experimental AI Metrics", metrics["disclaimer"]])
        for metric in metrics.get("metrics", []):
            lines.append(f"- {metric['name']}: {metric['value']} ({metric['interpretation']})")

    lines.extend(["", "## 10. Cannot Determine / Review Flags"])
    for item in report.get("cannot_determine", []):
        lines.append(f"- {item}")
    if readiness["label"] == "needs_human_review":
        lines.append("- Needs human review")
    if not report.get("cannot_determine") and readiness["label"] != "needs_human_review":
        lines.append("- No additional ambiguity flags were supplied.")

    return "\n".join(lines).strip() + "\n"
