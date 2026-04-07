"""Frontier scorer: dual-model scoring with Claude + GPT-4o + critique consensus."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from src.scoring.schema import ScoringResult
from src.ingest.frame_extractor import frames_to_base64


REPO_ROOT = Path(__file__).resolve().parents[2]


def _extract_anthropic_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


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


def _normalize_task_name(task: int | str) -> str:
    task_name = str(task)
    if task_name.isdigit():
        return f"task{task_name}"
    if task_name.startswith("task"):
        return task_name
    return f"task{task_name}"


def _load_prompt(version: str = "v001", task: int | str = 5) -> str:
    task_name = _normalize_task_name(task)
    prompt_path = REPO_ROOT / "prompts" / f"{version}_{task_name}_system.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    raise FileNotFoundError(f"Prompt not found: {prompt_path}")


def _load_critique_prompt(version: str = "v001", task: int | str = 5) -> str:
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

    if previous_scores_summary:
        system_prompt += f"\n\n## Trainee History\n{previous_scores_summary}"

    content = []
    for i, b64 in enumerate(frames_b64):
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
        content.append({"type": "text", "text": f"Frame {i+1} of {len(frames_b64)}"})

    content.append({"type": "text", "text": "Analyze all frames and produce the scoring JSON."})

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )

    raw_text = _extract_anthropic_text(response)
    data = _strip_managed_scoring_fields(_parse_json_payload(raw_text))

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

    if previous_scores_summary:
        system_prompt += f"\n\n## Trainee History\n{previous_scores_summary}"

    content = []
    for i, b64 in enumerate(frames_b64):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
        })
        content.append({"type": "text", "text": f"Frame {i+1} of {len(frames_b64)}"})

    content.append({"type": "text", "text": "Analyze all frames and produce the scoring JSON."})

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    )

    raw_content = response.choices[0].message.content or ""
    raw_text = raw_content.strip()
    data = _strip_managed_scoring_fields(_parse_json_payload(raw_text))

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

    teacher_a = score_a.model_dump(mode="json", exclude={"frame_analyses"})
    teacher_b = score_b.model_dump(mode="json", exclude={"frame_analyses"})

    message = f"""## Teacher A (Claude) Score:
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
    data = _strip_managed_scoring_fields(_parse_json_payload(raw_text))

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
