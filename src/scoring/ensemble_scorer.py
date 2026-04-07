"""Critique agent: compares two teacher scores and produces a consensus.

Uses Claude to reason about disagreements between Teacher A and Teacher B.
For high-agreement cases, uses rule-based averaging to save API costs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import anthropic

from src.scoring.schema import (
    CritiqueResult,
    Divergence,
    ScoreSource,
    ScoringResult,
)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent.parent.parent / "prompts"

# If teachers agree within these thresholds, skip LLM critique and average
FAST_AGREE_THRESHOLDS = {
    "completion_time_seconds": 10.0,  # within 10 seconds
    "estimated_fls_score": 20.0,      # within 20 points
    "estimated_penalties": 5.0,       # within 5 penalty points
}


def _compute_agreement(score_a: ScoringResult, score_b: ScoringResult) -> float:
    """Compute 0-1 agreement score between two teacher results."""
    agreements = []

    # Time agreement (within 10%)
    if score_a.completion_time_seconds > 0 and score_b.completion_time_seconds > 0:
        diff = abs(score_a.completion_time_seconds - score_b.completion_time_seconds)
        avg = (score_a.completion_time_seconds + score_b.completion_time_seconds) / 2
        agreements.append(max(0, 1.0 - diff / max(avg, 1)))

    # FLS score agreement
    if score_a.estimated_fls_score > 0 or score_b.estimated_fls_score > 0:
        diff = abs(score_a.estimated_fls_score - score_b.estimated_fls_score)
        agreements.append(max(0, 1.0 - diff / 600))

    # Knot assessment agreement
    for i in range(min(len(score_a.knot_assessments), len(score_b.knot_assessments))):
        ka_a = score_a.knot_assessments[i]
        ka_b = score_b.knot_assessments[i]
        matches = 0
        total = 0

        if ka_a.appears_secure is not None and ka_b.appears_secure is not None:
            matches += int(ka_a.appears_secure == ka_b.appears_secure)
            total += 1
        if ka_a.hand_switched is not None and ka_b.hand_switched is not None:
            matches += int(ka_a.hand_switched == ka_b.hand_switched)
            total += 1

        if total > 0:
            agreements.append(matches / total)

    # Drain assessment agreement
    if score_a.drain_assessment and score_b.drain_assessment:
        da_matches = 0
        da_matches += int(score_a.drain_assessment.gap_visible == score_b.drain_assessment.gap_visible)
        da_matches += int(score_a.drain_assessment.drain_avulsed == score_b.drain_assessment.drain_avulsed)
        agreements.append(da_matches / 2)

    return round(sum(agreements) / max(len(agreements), 1), 3)


def _fast_consensus(score_a: ScoringResult, score_b: ScoringResult, video_id: str) -> ScoringResult:
    """When teachers highly agree, average without LLM call."""
    return ScoringResult(
        video_id=video_id,
        source=ScoreSource.CRITIQUE_CONSENSUS,
        model_name="rule_based_average",
        prompt_version=score_a.prompt_version,
        completion_time_seconds=round(
            (score_a.completion_time_seconds + score_b.completion_time_seconds) / 2, 1
        ),
        phase_timings=score_a.phase_timings or score_b.phase_timings,
        knot_assessments=score_a.knot_assessments or score_b.knot_assessments,
        suture_placement=score_a.suture_placement or score_b.suture_placement,
        drain_assessment=score_a.drain_assessment or score_b.drain_assessment,
        estimated_penalties=round(
            (score_a.estimated_penalties + score_b.estimated_penalties) / 2, 1
        ),
        estimated_fls_score=round(
            (score_a.estimated_fls_score + score_b.estimated_fls_score) / 2, 1
        ),
        confidence_score=round(
            max(score_a.confidence_score, score_b.confidence_score), 2
        ),
        technique_summary=score_a.technique_summary or score_b.technique_summary,
        improvement_suggestions=list(set(
            score_a.improvement_suggestions + score_b.improvement_suggestions
        )),
        strengths=list(set(score_a.strengths + score_b.strengths)),
    )


def _identify_divergences(score_a: ScoringResult, score_b: ScoringResult) -> list[dict]:
    """Identify specific fields where teachers disagree."""
    divs = []

    # Time
    time_diff = abs(score_a.completion_time_seconds - score_b.completion_time_seconds)
    if time_diff > FAST_AGREE_THRESHOLDS["completion_time_seconds"]:
        divs.append({
            "field": "completion_time_seconds",
            "teacher_a": score_a.completion_time_seconds,
            "teacher_b": score_b.completion_time_seconds,
        })

    # Knot assessments
    for i in range(min(len(score_a.knot_assessments), len(score_b.knot_assessments))):
        ka_a = score_a.knot_assessments[i]
        ka_b = score_b.knot_assessments[i]

        if ka_a.appears_secure != ka_b.appears_secure:
            divs.append({
                "field": f"knot_assessments[{i}].appears_secure",
                "teacher_a": ka_a.appears_secure,
                "teacher_b": ka_b.appears_secure,
            })
        if ka_a.hand_switched != ka_b.hand_switched:
            divs.append({
                "field": f"knot_assessments[{i}].hand_switched",
                "teacher_a": ka_a.hand_switched,
                "teacher_b": ka_b.hand_switched,
            })
        if ka_a.is_surgeon_knot != ka_b.is_surgeon_knot and i == 0:
            divs.append({
                "field": "knot_assessments[0].is_surgeon_knot",
                "teacher_a": ka_a.is_surgeon_knot,
                "teacher_b": ka_b.is_surgeon_knot,
            })

    # Drain
    if score_a.drain_assessment and score_b.drain_assessment:
        if score_a.drain_assessment.gap_visible != score_b.drain_assessment.gap_visible:
            divs.append({
                "field": "drain_assessment.gap_visible",
                "teacher_a": score_a.drain_assessment.gap_visible,
                "teacher_b": score_b.drain_assessment.gap_visible,
            })

    return divs


def critique_and_resolve(
    score_a: ScoringResult,
    score_b: ScoringResult,
    video_id: str,
    model: str = "claude-sonnet-4-20250514",
) -> CritiqueResult:
    """Compare two teacher scores. Use LLM critique for disagreements, fast average for agreement."""
    agreement = _compute_agreement(score_a, score_b)
    divergences_raw = _identify_divergences(score_a, score_b)

    logger.info(
        f"Agreement for {video_id}: {agreement:.2f} with {len(divergences_raw)} divergences"
    )

    # Fast path: high agreement → rule-based average
    if agreement >= 0.9 and len(divergences_raw) == 0:
        consensus = _fast_consensus(score_a, score_b, video_id)
        return CritiqueResult(
            video_id=video_id,
            teacher_a_score_id=score_a.id,
            teacher_b_score_id=score_b.id,
            agreement_score=agreement,
            divergences=[],
            consensus_score=consensus,
            critique_reasoning="High agreement between teachers. Used rule-based averaging.",
            confidence=consensus.confidence_score,
        )

    # Slow path: LLM critique for disagreements
    critique_prompt_path = PROMPT_DIR / "v001_task5_critique.md"
    critique_system = critique_prompt_path.read_text()

    user_content = (
        f"## Teacher A Score (Claude)\n\n"
        f"```json\n{score_a.model_dump_json(indent=2)}\n```\n\n"
        f"## Teacher B Score (GPT-4o)\n\n"
        f"```json\n{score_b.model_dump_json(indent=2)}\n```\n\n"
        f"## Identified Divergences\n\n"
        f"```json\n{json.dumps(divergences_raw, indent=2, default=str)}\n```\n\n"
        f"Video duration: {score_a.completion_time_seconds}s (Teacher A) vs "
        f"{score_b.completion_time_seconds}s (Teacher B)\n\n"
        f"Please produce the consensus JSON."
    )

    client = anthropic.Anthropic()
    t0 = time.time()

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=critique_system,
        messages=[{"role": "user", "content": user_content}],
    )

    latency = time.time() - t0
    raw_text = response.content[0].text
    cost = (response.usage.input_tokens * 3.0 + response.usage.output_tokens * 15.0) / 1_000_000

    # Parse critique response
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        critique_data = json.loads(text)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse critique response for {video_id}")
        # Fallback to averaging
        consensus = _fast_consensus(score_a, score_b, video_id)
        return CritiqueResult(
            video_id=video_id,
            teacher_a_score_id=score_a.id,
            teacher_b_score_id=score_b.id,
            agreement_score=agreement,
            divergences=[],
            consensus_score=consensus,
            critique_reasoning="Critique LLM parse failed. Fell back to rule-based average.",
            confidence=min(score_a.confidence_score, score_b.confidence_score),
            api_cost_usd=round(cost, 4),
            latency_seconds=round(latency, 2),
        )

    # Parse divergences from critique
    parsed_divergences = []
    for div in critique_data.get("divergences", []):
        parsed_divergences.append(Divergence(
            field=div["field"],
            teacher_a_value=str(div["teacher_a_value"]),
            teacher_b_value=str(div["teacher_b_value"]),
            resolution=str(div["resolution"]),
            reasoning=div.get("reasoning", ""),
        ))

    # Parse consensus score from critique
    cs_data = critique_data.get("consensus_score", {})
    consensus = _fast_consensus(score_a, score_b, video_id)  # start from average
    # Override with critique's specific resolutions
    if "completion_time_seconds" in cs_data:
        consensus.completion_time_seconds = cs_data["completion_time_seconds"]
    if "estimated_penalties" in cs_data:
        consensus.estimated_penalties = cs_data["estimated_penalties"]
    if "estimated_fls_score" in cs_data:
        consensus.estimated_fls_score = cs_data["estimated_fls_score"]
    if "confidence_score" in cs_data:
        consensus.confidence_score = cs_data["confidence_score"]
    if "technique_summary" in cs_data:
        consensus.technique_summary = cs_data["technique_summary"]
    if "improvement_suggestions" in cs_data:
        consensus.improvement_suggestions = cs_data["improvement_suggestions"]
    if "strengths" in cs_data:
        consensus.strengths = cs_data["strengths"]

    consensus.model_name = f"critique_{model}"
    consensus.source = ScoreSource.CRITIQUE_CONSENSUS

    return CritiqueResult(
        video_id=video_id,
        teacher_a_score_id=score_a.id,
        teacher_b_score_id=score_b.id,
        agreement_score=critique_data.get("agreement_score", agreement),
        divergences=parsed_divergences,
        consensus_score=consensus,
        critique_reasoning=critique_data.get("critique_reasoning", ""),
        confidence=critique_data.get("confidence", consensus.confidence_score),
        api_cost_usd=round(cost, 4),
        latency_seconds=round(latency, 2),
    )
