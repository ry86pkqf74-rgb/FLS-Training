"""Frontier scorer: dual-model scoring with Claude + GPT-4o + critique consensus."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from src.scoring.schema import ScoringResult


REPO_ROOT = Path(__file__).resolve().parents[2]
RUBRICS_ROOT = REPO_ROOT / "rubrics"

TASK_RUBRIC_FILES = {
    "task1": "task1_peg_transfer.yaml",
    "task2": "task2_pattern_cut.yaml",
    "task3": "task3_ligating_loop.yaml",
    "task4": "task4_extracorporeal_suture.yaml",
    "task5": "task5_intracorporeal_suture.yaml",
}
TASK_ID_ALIASES = {
    "task1": "task1",
    "task1_peg_transfer": "task1",
    "task2": "task2",
    "task2_pattern_cut": "task2",
    "task3": "task3",
    "task3_ligating_loop": "task3",
    "task4": "task4",
    "task4_extracorporeal": "task4",
    "task4_extracorporeal_suture": "task4",
    "task5": "task5",
    "task5_intracorporeal": "task5",
    "task5_intracorporeal_suture": "task5",
}
VALID_PHASES = {
    "idle",
    "needle_load",
    "suture_placement",
    "first_throw",
    "second_throw",
    "third_throw",
    "suture_cut",
    "completion",
}
PHASE_ALIASES = {
    "setup": "idle",
    "instructional_content": "idle",
    "educational_demonstration": "idle",
    "demonstration": "idle",
    "idle_or_setup": "idle",
    "loop_deploy": "needle_load",
    "needle_positioning": "needle_load",
    "needle_pickup": "needle_load",
    "pickup_nondominant": "needle_load",
    "reverse_pickup": "needle_load",
    "transfer_midair": "suture_placement",
    "place_dominant": "suture_placement",
    "reverse_transfer": "suture_placement",
    "reverse_place": "suture_placement",
    "peg_transfer": "suture_placement",
    "pattern_cut": "suture_placement",
    "ligating_loop": "suture_placement",
    "extracorporeal_suture_demo": "suture_placement",
    "intracorporeal_suture_demo": "suture_placement",
    "knot_formation": "first_throw",
    "first_knot": "first_throw",
    "second_knot": "second_throw",
    "third_knot": "third_throw",
    "knot_tightening": "third_throw",
    "final_knot_tightening": "third_throw",
    "cutting": "suture_cut",
    "suture_trim": "suture_cut",
    "done": "completion",
    "finished": "completion",
}


def _extract_anthropic_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


def _extract_openai_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _parse_json_payload(raw_text: str) -> dict:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
    if raw_text.endswith("```"):
        raw_text = raw_text.rsplit("```", 1)[0]

    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw_text[start:end + 1])
        raise


def _strip_managed_scoring_fields(data: dict) -> dict:
    cleaned = dict(data)
    for field in [
        "id",
        "source",
        "model_name",
        "model_version",
        "prompt_version",
        "video_id",
        "video_filename",
        "video_hash",
        "scored_at",
    ]:
        cleaned.pop(field, None)
    return cleaned


def _canonical_task_id(task: int | str) -> str:
    task_name = str(task).strip().lower()
    if task_name.isdigit():
        return f"task{task_name}"
    if task_name in TASK_ID_ALIASES:
        return TASK_ID_ALIASES[task_name]
    if task_name.startswith("task"):
        return task_name.split("_", 1)[0]
    return f"task{task_name}"


def _normalize_task_name(task: int | str) -> str:
    return _canonical_task_id(task)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return default
    return default


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return default
    return default


def _normalize_phase_value(value: Any) -> str:
    if not value:
        return "idle"
    if isinstance(value, str):
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
        if normalized in VALID_PHASES:
            return normalized
        if normalized in PHASE_ALIASES:
            return PHASE_ALIASES[normalized]
        if "throw" in normalized:
            if "third" in normalized or "final" in normalized:
                return "third_throw"
            if "second" in normalized:
                return "second_throw"
            return "first_throw"
        if any(token in normalized for token in ("cut", "trim", "snip")):
            return "suture_cut"
        if any(token in normalized for token in ("complete", "finish", "end")):
            return "completion"
        if any(token in normalized for token in ("load", "pickup", "deploy", "grasp")):
            return "needle_load"
        if any(token in normalized for token in ("transfer", "place", "pass", "position", "pattern", "loop")):
            return "suture_placement"
    return "idle"


def _normalize_phase_collections(cleaned: dict) -> None:
    for field_name in ("frame_analyses", "phase_timings"):
        items = cleaned.get(field_name) or []
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                item["phase"] = _normalize_phase_value(item.get("phase"))


def _load_rubric(task: int | str) -> dict:
    task_id = _canonical_task_id(task)
    rubric_name = TASK_RUBRIC_FILES.get(task_id)
    if not rubric_name:
        raise FileNotFoundError(f"No rubric configured for task: {task}")
    rubric_path = RUBRICS_ROOT / rubric_name
    return yaml.safe_load(rubric_path.read_text()) or {}


def _build_task_context(task: int | str) -> str:
    rubric = _load_rubric(task)
    phases = ", ".join(phase.get("name", "") for phase in rubric.get("phases", []))
    penalties = ", ".join(penalty.get("name", "") for penalty in rubric.get("penalties", []))
    return (
        f"Task ID: {rubric.get('task_id', _canonical_task_id(task))}\n"
        f"Task Name: {rubric.get('name', '')}\n"
        f"Maximum Time: {rubric.get('max_time_seconds', 'unknown')} seconds\n"
        f"Proficiency Target: {rubric.get('proficiency_time_seconds', 'unknown')} seconds\n"
        f"Score Formula: {rubric.get('score_formula', '')}\n"
        f"Observed Penalty Categories: {penalties}\n"
        f"Task Phases: {phases}"
    )


def _prepare_scoring_payload(data: dict, task: int | str) -> dict:
    cleaned = dict(data)
    cleaned.setdefault("task_id", _canonical_task_id(task))

    for numeric_field in [
        "completion_time_seconds",
        "estimated_penalties",
        "estimated_fls_score",
        "confidence_score",
    ]:
        if cleaned.get(numeric_field) is None:
            cleaned[numeric_field] = 0.0

    score_components = cleaned.get("score_components") or {}
    if score_components:
        cleaned.setdefault(
            "estimated_penalties",
            _coerce_float(score_components.get("penalty_deductions", 0.0)),
        )
        cleaned.setdefault(
            "estimated_fls_score",
            _coerce_float(score_components.get("total_fls_score", 0.0)),
        )

        score_components["time_score"] = _coerce_float(score_components.get("time_score"))
        score_components["penalty_deductions"] = _coerce_float(score_components.get("penalty_deductions"))
        score_components["total_fls_score"] = _coerce_float(score_components.get("total_fls_score"))
        cleaned["score_components"] = score_components

    if "confidence" in cleaned and "confidence_score" not in cleaned:
        cleaned["confidence_score"] = _coerce_float(cleaned.get("confidence"))

    penalties = cleaned.get("penalties") or []
    for penalty in penalties:
        if isinstance(penalty, dict):
            penalty["count"] = _coerce_int(penalty.get("count"), default=1)

    cleaned["completion_time_seconds"] = _coerce_float(cleaned.get("completion_time_seconds"))
    cleaned["estimated_penalties"] = _coerce_float(cleaned.get("estimated_penalties"))
    cleaned["estimated_fls_score"] = _coerce_float(cleaned.get("estimated_fls_score"))
    cleaned["confidence_score"] = _coerce_float(cleaned.get("confidence_score"))
    _normalize_phase_collections(cleaned)

    if cleaned.get("reasoning") and not cleaned.get("technique_summary"):
        cleaned["technique_summary"] = str(cleaned["reasoning"])

    return cleaned


def _prepare_consensus_payload(data: dict, task: int | str) -> dict:
    if "consensus_score" not in data:
        return _prepare_scoring_payload(data, task)

    cleaned = _prepare_scoring_payload(data.get("consensus_score") or {}, task)
    cleaned["comparison_to_previous"] = {
        "disagreements": data.get("disagreements", []),
        "overall_confidence": data.get("overall_confidence"),
    }

    overall_confidence = data.get("overall_confidence")
    if overall_confidence is not None:
        cleaned["confidence_score"] = float(overall_confidence)

    if not cleaned.get("reasoning") and cleaned["comparison_to_previous"]["disagreements"]:
        cleaned["reasoning"] = "Consensus resolved field-level disagreements; see comparison_to_previous.disagreements."
        cleaned["technique_summary"] = cleaned["reasoning"]

    return cleaned


def _load_prompt(version: str = "v001", task: int | str = 5) -> str:
    if version.startswith("v002"):
        prompt_path = REPO_ROOT / "prompts" / "v002_universal_scoring_system.md"
        if prompt_path.exists():
            return prompt_path.read_text()
    task_name = _normalize_task_name(task)
    prompt_path = REPO_ROOT / "prompts" / f"{version}_{task_name}_system.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"Prompt not found: {prompt_path}")


def _load_critique_prompt(version: str = "v001", task: int | str = 5) -> str:
    if version.startswith("v002"):
        prompt_path = REPO_ROOT / "prompts" / "v002_consensus_system.md"
        if prompt_path.exists():
            return prompt_path.read_text()
    task_name = _normalize_task_name(task)
    prompt_path = REPO_ROOT / "prompts" / f"{version}_{task_name}_critique.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"Prompt not found: {prompt_path}")


def score_with_claude(
    frames_b64: list[str],
    video_id: str,
    video_filename: str,
    video_hash: str = "",
    prompt_version: str = "v001",
    task: int | str = 5,
    previous_scores_summary: str = "",
) -> ScoringResult:
    """Score frames using Claude Sonnet."""
    import anthropic

    client = anthropic.Anthropic()
    system_prompt = _load_prompt(prompt_version, task)
    task_context = _build_task_context(task)

    if previous_scores_summary:
        system_prompt += f"\n\n## Trainee History\n{previous_scores_summary}"

    content = []
    for i, b64 in enumerate(frames_b64):
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
        content.append({"type": "text", "text": f"Frame {i+1} of {len(frames_b64)}"})

    content.append({
        "type": "text",
        "text": f"{task_context}\nAnalyze all frames and produce the scoring JSON.",
    })

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )

    raw_text = _extract_anthropic_text(response)
    data = _prepare_scoring_payload(
        _strip_managed_scoring_fields(_parse_json_payload(raw_text)),
        task,
    )

    score_id = f"score_claude_{video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    return ScoringResult(
        id=score_id,
        video_id=video_id,
        video_filename=video_filename,
        video_hash=video_hash,
        source="teacher_claude",
        model_name="claude-sonnet-4-20250514",
        model_version="claude-sonnet-4-20250514",
        prompt_version=prompt_version,
        **data,
    )


def score_with_gpt(
    frames_b64: list[str],
    video_id: str,
    video_filename: str,
    video_hash: str = "",
    prompt_version: str = "v001",
    task: int | str = 5,
    previous_scores_summary: str = "",
) -> ScoringResult:
    """Score frames using GPT-4o."""
    from openai import OpenAI

    client = OpenAI()
    system_prompt = _load_prompt(prompt_version, task)
    task_context = _build_task_context(task)

    if previous_scores_summary:
        system_prompt += f"\n\n## Trainee History\n{previous_scores_summary}"

    content = []
    for i, b64 in enumerate(frames_b64):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
        })
        content.append({"type": "text", "text": f"Frame {i+1} of {len(frames_b64)}"})

    content.append({
        "type": "text",
        "text": f"{task_context}\nAnalyze all frames and produce the scoring JSON.",
    })

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    )

    raw_text = _extract_openai_text(response.choices[0].message.content)
    data = _prepare_scoring_payload(
        _strip_managed_scoring_fields(_parse_json_payload(raw_text)),
        task,
    )

    score_id = f"score_gpt_{video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    return ScoringResult(
        id=score_id,
        video_id=video_id,
        video_filename=video_filename,
        video_hash=video_hash,
        source="teacher_gpt",
        model_name="gpt-4o",
        model_version="gpt-4o-2024-08-06",
        prompt_version=prompt_version,
        **data,
    )


def run_critique(
    score_a: ScoringResult,
    score_b: ScoringResult,
    video_id: str,
    prompt_version: str = "v001",
    task: int | str = 5,
) -> ScoringResult:
    """Run critique agent to produce consensus from two teacher scores."""
    import anthropic

    client = anthropic.Anthropic()
    critique_prompt = _load_critique_prompt(prompt_version, task)
    task_context = _build_task_context(task)

    teacher_a = score_a.model_dump(mode="json", exclude={"frame_analyses"})
    teacher_b = score_b.model_dump(mode="json", exclude={"frame_analyses"})

    message = f"""{task_context}

## Teacher A (Claude) Score:
{json.dumps(teacher_a, indent=2)}

## Teacher B (GPT-4o) Score:
{json.dumps(teacher_b, indent=2)}

Produce a consensus score resolving any disagreements."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=critique_prompt,
        messages=[{"role": "user", "content": message}],
    )

    raw_text = _extract_anthropic_text(response)
    data = _prepare_consensus_payload(
        _strip_managed_scoring_fields(_parse_json_payload(raw_text)),
        task,
    )

    score_id = f"score_consensus_{video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    return ScoringResult(
        id=score_id,
        video_id=video_id,
        video_filename=score_a.video_filename,
        video_hash=score_a.video_hash,
        source="consensus",
        model_name="critique_agent",
        model_version="claude-sonnet-4-20250514",
        prompt_version=prompt_version,
        **data,
    )
