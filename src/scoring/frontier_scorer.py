"""Score FLS videos using frontier VLMs (Claude Sonnet 4, GPT-4o).

Each teacher independently analyzes extracted frames and returns a ScoringResult.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import anthropic
import openai

from src.scoring.schema import (
    Confidence,
    DrainAssessment,
    FrameAnalysis,
    Hand,
    Phase,
    PhaseTiming,
    ScoreSource,
    ScoringResult,
    SuturePlacement,
    ThrowAssessment,
    VideoMetadata,
)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"


def _load_system_prompt(version: str = "v001") -> str:
    """Load the system scoring prompt."""
    path = PROMPT_DIR / f"{version}_task5_system.md"
    return path.read_text()


def _build_user_message(
    frame_b64s: list[str],
    timestamps: list[float],
    duration_seconds: float,
    n_final: int = 3,
) -> str:
    """Build the user message text (without images — those are added per-provider)."""
    ts_str = ", ".join(f"{t:.1f}" for t in timestamps)
    return (
        f"Here are {len(frame_b64s)} frames extracted from a {duration_seconds:.1f}-second "
        f"video of a surgical trainee performing FLS Task 5 (Intracorporeal Suture with "
        f"Knot Tying). Frames are in chronological order. The last {n_final} frames show "
        f"the completed result.\n\n"
        f"Frame timestamps (seconds): {ts_str}\n\n"
        f"Please analyze these frames and score the performance according to FLS Task 5 "
        f"criteria. Respond with ONLY the JSON scoring object."
    )


def _parse_scoring_json(raw: str, video_id: str, source: ScoreSource, model_name: str,
                         prompt_version: str) -> ScoringResult:
    """Parse raw JSON from a VLM response into a ScoringResult."""
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    text = text.strip()

    data = json.loads(text)

    # Parse frame analyses
    frame_analyses = []
    for fa in data.get("frame_analyses", []):
        frame_analyses.append(FrameAnalysis(
            frame_number=fa["frame_number"],
            phase=Phase(fa.get("phase", "unknown")),
            description=fa.get("description", ""),
            technique_notes=fa.get("technique_notes", ""),
        ))

    # Parse phase timings
    phase_timings = []
    for pt in data.get("phase_timings", []):
        phase_timings.append(PhaseTiming(
            phase=Phase(pt["phase"]),
            start_seconds=pt["start_seconds"],
            end_seconds=pt["end_seconds"],
            duration_seconds=pt["duration_seconds"],
        ))

    # Parse knot assessments
    knot_assessments = []
    for ka in data.get("knot_assessments", []):
        knot_assessments.append(ThrowAssessment(
            throw_number=ka["throw_number"],
            is_surgeon_knot=ka.get("is_surgeon_knot"),
            is_single_throw=ka.get("is_single_throw"),
            hand_used=Hand(ka.get("hand_used", "unclear")),
            hand_switched=ka.get("hand_switched"),
            appears_secure=ka.get("appears_secure", False),
            notes=ka.get("notes", ""),
        ))

    # Parse suture placement
    sp_data = data.get("suture_placement")
    suture_placement = None
    if sp_data:
        suture_placement = SuturePlacement(
            deviation_from_mark1_mm=sp_data.get("deviation_from_mark1_mm", 0),
            deviation_from_mark2_mm=sp_data.get("deviation_from_mark2_mm", 0),
            total_deviation_penalty=sp_data.get("total_deviation_penalty", 0),
            confidence=Confidence(sp_data.get("confidence", "low")),
        )

    # Parse drain assessment
    da_data = data.get("drain_assessment")
    drain_assessment = None
    if da_data:
        drain_assessment = DrainAssessment(
            gap_visible=da_data.get("gap_visible", False),
            drain_avulsed=da_data.get("drain_avulsed", False),
            slit_closure_quality=da_data.get("slit_closure_quality", "unknown"),
        )

    return ScoringResult(
        video_id=video_id,
        source=source,
        model_name=model_name,
        prompt_version=prompt_version,
        frame_analyses=frame_analyses,
        completion_time_seconds=data.get("completion_time_seconds", 0),
        phase_timings=phase_timings,
        knot_assessments=knot_assessments,
        suture_placement=suture_placement,
        drain_assessment=drain_assessment,
        estimated_penalties=data.get("estimated_penalties", 0),
        estimated_fls_score=data.get("estimated_fls_score", 0),
        confidence_score=data.get("confidence_score", 0),
        technique_summary=data.get("technique_summary", ""),
        improvement_suggestions=data.get("improvement_suggestions", []),
        strengths=data.get("strengths", []),
    )


# ---------------------------------------------------------------------------
# Teacher A: Claude
# ---------------------------------------------------------------------------

def score_with_claude(
    frame_b64s: list[str],
    timestamps: list[float],
    video_meta: VideoMetadata,
    prompt_version: str = "v001",
    model: str = "claude-sonnet-4-20250514",
) -> ScoringResult:
    """Score video frames using Claude as Teacher A."""
    client = anthropic.Anthropic()
    system_prompt = _load_system_prompt(prompt_version)
    user_text = _build_user_message(
        frame_b64s, timestamps, video_meta.duration_seconds
    )

    # Build content blocks: images interleaved with frame labels, then user text
    content: list[dict] = []
    for i, b64 in enumerate(frame_b64s):
        content.append({
            "type": "text",
            "text": f"[Frame {i+1} — {timestamps[i]:.1f}s]",
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64,
            },
        })
    content.append({"type": "text", "text": user_text})

    logger.info(f"Scoring {video_meta.id} with Claude ({model}), {len(frame_b64s)} frames")
    t0 = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )

    latency = time.time() - t0
    raw_text = response.content[0].text

    # Estimate cost (Sonnet: $3/1M input, $15/1M output)
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000

    result = _parse_scoring_json(
        raw_text, video_meta.id, ScoreSource.TEACHER_CLAUDE, model, prompt_version
    )
    result.latency_seconds = round(latency, 2)
    result.api_cost_usd = round(cost, 4)

    logger.info(
        f"Claude scored {video_meta.id}: FLS={result.estimated_fls_score}, "
        f"confidence={result.confidence_score}, cost=${cost:.4f}, {latency:.1f}s"
    )
    return result


# ---------------------------------------------------------------------------
# Teacher B: GPT-4o
# ---------------------------------------------------------------------------

def score_with_gpt4o(
    frame_b64s: list[str],
    timestamps: list[float],
    video_meta: VideoMetadata,
    prompt_version: str = "v001",
    model: str = "gpt-4o",
) -> ScoringResult:
    """Score video frames using GPT-4o as Teacher B."""
    client = openai.OpenAI()
    system_prompt = _load_system_prompt(prompt_version)
    user_text = _build_user_message(
        frame_b64s, timestamps, video_meta.duration_seconds
    )

    # Build content blocks for OpenAI format
    content: list[dict] = []
    for i, b64 in enumerate(frame_b64s):
        content.append({
            "type": "text",
            "text": f"[Frame {i+1} — {timestamps[i]:.1f}s]",
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "high",
            },
        })
    content.append({"type": "text", "text": user_text})

    logger.info(f"Scoring {video_meta.id} with GPT-4o ({model}), {len(frame_b64s)} frames")
    t0 = time.time()

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    )

    latency = time.time() - t0
    raw_text = response.choices[0].message.content

    # Estimate cost (GPT-4o: ~$2.50/1M input, $10/1M output)
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost = (input_tokens * 2.50 + output_tokens * 10.0) / 1_000_000

    result = _parse_scoring_json(
        raw_text, video_meta.id, ScoreSource.TEACHER_GPT4O, model, prompt_version
    )
    result.latency_seconds = round(latency, 2)
    result.api_cost_usd = round(cost, 4)

    logger.info(
        f"GPT-4o scored {video_meta.id}: FLS={result.estimated_fls_score}, "
        f"confidence={result.confidence_score}, cost=${cost:.4f}, {latency:.1f}s"
    )
    return result
