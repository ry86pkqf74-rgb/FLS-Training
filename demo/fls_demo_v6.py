#!/usr/bin/env python3
"""FLS Video Scoring Platform v6.

This v6 demo keeps the v5 resident/report workflow direction but moves display
language to v003 scoring/reporting rules: task-required upload, rubric-loaded
denominators, training-score framing, readiness labels, and experimental metric
isolation.
"""
from __future__ import annotations

import json
from typing import Any

import gradio as gr

from src.reporting.readiness import determine_readiness
from src.rubrics.loader import (
    canonical_task_id,
    get_task_max_score,
    get_task_max_time,
    get_task_name,
    is_official_fls_task,
    load_rubric,
)
from src.scoring.schema import ScoreComponents, ScoringResult


TASK_METADATA = {
    task_id: {
        "task_name": get_task_name(task_id),
        "max_score": get_task_max_score(task_id),
        "max_time_seconds": get_task_max_time(task_id),
        "official_fls_task": is_official_fls_task(task_id),
    }
    for task_id in ["task1", "task2", "task3", "task4", "task5", "task6"]
}
TASK_METADATA["task6"]["display_note"] = "Custom training task, not official FLS manual skills task"


def _format_formula(max_score: float, completion_time: float, penalties: float) -> str:
    score = max(0.0, max_score - completion_time - penalties)
    return f"{max_score:g} - {completion_time:g} - {penalties:g} = {score:g}"


def _build_display_score(task_id: str, completion_time: float, penalties: float) -> tuple[str, dict[str, Any]]:
    canonical = canonical_task_id(task_id)
    rubric = load_rubric(canonical)
    max_score = float(rubric["max_score"])
    training_score = max(0.0, max_score - float(completion_time or 0) - float(penalties or 0))
    formula = _format_formula(max_score, float(completion_time or 0), float(penalties or 0))

    score = ScoringResult(
        id="demo_v6_preview",
        video_id="demo_v6_preview",
        video_filename="preview.mp4",
        source="demo",
        model_name="demo",
        model_version="v6",
        task_id=canonical,
        task_name=str(rubric["name"]),
        max_time_seconds=float(rubric["max_time_seconds"]),
        max_score=max_score,
        completion_time_seconds=float(completion_time or 0),
        estimated_penalties=float(penalties or 0),
        estimated_fls_score=training_score,
        confidence_score=0.9,
        score_components=ScoreComponents(
            max_score=max_score,
            time_used=float(completion_time or 0),
            total_penalties=float(penalties or 0),
            total_fls_score=training_score,
            formula_applied=formula,
        ),
    )
    readiness = determine_readiness(score, rubric)
    return (
        f"Training score: {training_score:g} / {max_score:g}\n"
        f"Score = max score - completion time - penalties\n"
        f"Score = {formula}\n"
        f"Training-readiness status: {readiness['label']}",
        readiness,
    )


def evaluate_fls_v6(
    email: str,
    task_id: str | None,
    completion_time: float,
    penalty_total: float,
    confidence_score: float,
    has_critical_errors: bool,
    video_file: str | None,
) -> tuple[str, str, str]:
    """Validate upload inputs and render a v003-compliant preview report."""
    if not email or "@" not in email:
        return "Please log in first.", "", ""
    if not task_id:
        return "Please select a task before upload or scoring.", "", ""
    if video_file is None:
        return "Please upload a video for scoring.", "", ""

    canonical = canonical_task_id(task_id)
    display_score, readiness = _build_display_score(canonical, completion_time, penalty_total)
    metadata = TASK_METADATA[canonical]

    warnings = []
    if has_critical_errors:
        warnings.append(
            "Major technical issue detected. Interpret score with caution; focus on correctness before speed."
        )
    if confidence_score < 0.60:
        warnings.append("Needs human review")
    if canonical == "task6":
        warnings.append("Custom training task, not official FLS manual skills task")

    report = [
        "# FLS Training Feedback Report",
        "",
        f"- Task: {metadata['task_name']}",
        f"- Training score, not official certification score: {display_score.splitlines()[0].replace('Training score: ', '')}",
        f"- Denominator from rubric: /{metadata['max_score']:g}",
        f"- {display_score.splitlines()[2]}",
        f"- Status: {readiness['label']}",
        "",
        "This is AI-assisted training feedback, not an official FLS certification result.",
    ]
    if warnings:
        report.extend(["", "## Review Flags", *[f"- {warning}" for warning in warnings]])
    report.extend([
        "",
        "<details><summary>Experimental AI-derived metrics</summary>",
        "",
        "These are internal model-derived coaching features and are not part of official FLS scoring.",
        "",
        "</details>",
    ])

    raw = {
        "task_id": canonical,
        "max_score": metadata["max_score"],
        "readiness": readiness,
        "formula": display_score.splitlines()[2].replace("Score = ", ""),
    }
    return "\n".join(report), display_score, json.dumps(raw, indent=2)


with gr.Blocks(title="FLS Scoring Platform v6") as demo:
    gr.Markdown("# FLS Surgical Skills Scoring Platform v6")
    gr.Markdown("Training score, not official certification score.")
    current_email = gr.Textbox(label="Email", placeholder="resident@emory.edu")
    video_input = gr.Video(label="Upload FLS Video", interactive=True)
    task_dropdown = gr.Dropdown(
        choices=list(TASK_METADATA.keys()),
        value=None,
        label="FLS or custom training task",
        interactive=True,
    )
    time_input = gr.Number(label="Completion Time (seconds)", value=0)
    penalty_input = gr.Number(label="Penalty total", value=0)
    confidence_input = gr.Number(label="Model confidence", value=0.9)
    critical_input = gr.Checkbox(label="Critical error present", value=False)
    evaluate_btn = gr.Button("Evaluate", variant="primary")
    report_output = gr.Markdown()
    score_output = gr.Textbox(label="Score", interactive=False)
    json_output = gr.Code(label="Structured JSON", language="json")

    evaluate_btn.click(
        fn=evaluate_fls_v6,
        inputs=[
            current_email,
            task_dropdown,
            time_input,
            penalty_input,
            confidence_input,
            critical_input,
            video_input,
        ],
        outputs=[report_output, score_output, json_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
