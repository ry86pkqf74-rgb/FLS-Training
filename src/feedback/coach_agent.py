"""Coach Agent — generates rich, technique-level feedback using a frontier VLM.

Separate from the scoring pipeline by design: the score is rubric-strict,
the coach is pedagogically rich. This separation prevents score contamination
and allows independent iteration on coaching quality.

Usage:
    from src.feedback.coach_agent import generate_coach_feedback
    result = generate_coach_feedback(consensus_json, teacher_outputs, frame_b64s, ...)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import anthropic

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"


def _load_coach_prompt(version: str = "v001") -> str:
    path = PROMPT_DIR / f"{version}_task5_coach.md"
    return path.read_text()


def _build_coach_user_message(
    consensus_json: dict,
    teacher_a_json: dict | None,
    teacher_b_json: dict | None,
    frame_timestamps: list[float],
    trainee_history: list[dict] | None = None,
) -> str:
    """Build the text portion of the coach's user message."""
    parts = []

    parts.append("## Consensus Score (from rubric evaluation pipeline)")
    parts.append("```json")
    parts.append(json.dumps(consensus_json, indent=2))
    parts.append("```\n")

    if teacher_a_json:
        parts.append("## Teacher A Frame Analyses (Claude Sonnet 4)")
        fa = teacher_a_json.get("frame_analyses", [])
        for f in fa:
            parts.append(
                f"- Frame {f.get('frame_number', '?')} ({f.get('phase', '?')}): "
                f"{f.get('description', '')} | {f.get('technique_notes', '')}"
            )
        parts.append("")

    if teacher_b_json:
        parts.append("## Teacher B Frame Analyses (GPT-4o)")
        fa = teacher_b_json.get("frame_analyses", [])
        for f in fa:
            parts.append(
                f"- Frame {f.get('frame_number', '?')} ({f.get('phase', '?')}): "
                f"{f.get('description', '')} | {f.get('technique_notes', '')}"
            )
        parts.append("")

    ts_str = ", ".join(f"{t:.1f}s" for t in frame_timestamps)
    parts.append(f"## Frame Timestamps\n{ts_str}\n")

    if trainee_history:
        parts.append("## Trainee History (previous sessions, chronological)")
        for h in trainee_history[-10:]:  # last 10 sessions max
            parts.append(
                f"- {h.get('video_id', '?')}: {h.get('fls_score', '?')} FLS, "
                f"{h.get('completion_time_seconds', '?')}s, "
                f"confidence {h.get('confidence', '?')}"
            )
        parts.append("")

    parts.append(
        "Analyze the attached frames and the scoring data above. "
        "Provide your coaching feedback as the JSON structure specified in your system prompt."
    )

    return "\n".join(parts)


def generate_coach_feedback(
    consensus_json: dict,
    teacher_a_json: dict | None = None,
    teacher_b_json: dict | None = None,
    frame_b64s: list[str] | None = None,
    frame_timestamps: list[float] | None = None,
    trainee_history: list[dict] | None = None,
    model: str | None = None,
    prompt_version: str = "v001",
) -> dict:
    """Call frontier model to generate rich coaching feedback.

    Args:
        consensus_json: The rubric consensus score (from critique agent)
        teacher_a_json: Full Teacher A output (Claude) — used for frame analyses
        teacher_b_json: Full Teacher B output (GPT-4o) — used for frame analyses
        frame_b64s: Base64-encoded JPEG frames from the video
        frame_timestamps: Timestamp in seconds for each frame
        trainee_history: List of prior session summaries for progress tracking
        model: Override model name (default from env or claude-sonnet-4-20250514)
        prompt_version: Which coach prompt version to use

    Returns:
        Parsed coach feedback dict, or error dict on failure.
    """
    model = model or os.getenv("COACH_MODEL", "claude-sonnet-4-20250514")
    frame_timestamps = frame_timestamps or []
    frame_b64s = frame_b64s or []

    system_prompt = _load_coach_prompt(prompt_version)
    user_text = _build_coach_user_message(
        consensus_json, teacher_a_json, teacher_b_json,
        frame_timestamps, trainee_history,
    )

    # Build message content: frames (as images) + text
    content = []
    for i, b64 in enumerate(frame_b64s):
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })
    content.append({"type": "text", "text": user_text})

    client = anthropic.Anthropic()
    logger.info(f"Calling coach agent ({model}) with {len(frame_b64s)} frames...")

    t0 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0.4,  # slightly creative for coaching language
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    elapsed = time.time() - t0

    raw_text = response.content[0].text.strip()

    # Parse JSON
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        ).strip()

    try:
        feedback = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"Coach returned invalid JSON: {e}")
        return {
            "error": "json_parse_failure",
            "raw_response": raw_text[:2000],
            "elapsed_seconds": round(elapsed, 2),
        }

    # Attach metadata
    feedback["_meta"] = {
        "model": model,
        "prompt_version": prompt_version,
        "elapsed_seconds": round(elapsed, 2),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd_approx": round(
            response.usage.input_tokens * 3 / 1_000_000
            + response.usage.output_tokens * 15 / 1_000_000, 4
        ),
    }

    logger.info(
        f"Coach feedback generated in {elapsed:.1f}s "
        f"({response.usage.input_tokens}+{response.usage.output_tokens} tokens, "
        f"~${feedback['_meta']['cost_usd_approx']})"
    )

    return feedback
