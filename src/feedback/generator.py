"""Generate coaching feedback reports from scores and trainee history.

This module produces FeedbackReport objects that serve as training targets
for the coaching head of the student model.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.scoring.schema import ScoringResult
from src.feedback.schema import (
    FeedbackReport, PhaseCoaching, ImprovementPriority, ProgressionInsight,
    EquipmentNote, FeedbackPriority, DrillType, TraineeProfile,
)


# Expert benchmarks for Task 5 phases (seconds)
EXPERT_BENCHMARKS = {
    "needle_load": 5,
    "suture_placement": 15,
    "first_throw": 15,
    "second_throw": 12,
    "third_throw": 12,
    "suture_cut": 10,
}

# Intermediate benchmarks (what a good trainee targets)
INTERMEDIATE_BENCHMARKS = {
    "needle_load": 8,
    "suture_placement": 22,
    "first_throw": 20,
    "second_throw": 18,
    "third_throw": 18,
    "suture_cut": 20,
}


def generate_feedback(
    current_score: ScoringResult,
    all_previous_scores: list[ScoringResult],
    profile: Optional[TraineeProfile] = None,
) -> FeedbackReport:
    """Generate a comprehensive coaching feedback report."""

    attempt_number = len(all_previous_scores) + 1

    # Build phase coaching
    phase_coaching = _build_phase_coaching(current_score, all_previous_scores)

    # Identify top priorities
    priorities = _identify_priorities(current_score, phase_coaching)

    # Progression insights
    insights = _build_progression_insights(current_score, all_previous_scores, profile)

    # Strengths
    strengths = _identify_strengths(current_score, all_previous_scores)

    # Fatigue assessment
    fatigue_risk, fatigue_evidence = _assess_fatigue(current_score, all_previous_scores)

    # Headline
    headline = _generate_headline(current_score, all_previous_scores, priorities)

    # Distance to proficiency
    # FLS passing score is typically ~475+ (600 - 125 max time penalty)
    distance = ""
    if current_score.estimated_fls_score < 475:
        gap = 475 - current_score.estimated_fls_score
        distance = f"~{gap:.0f} points below typical proficiency cutoff (475)"
    else:
        distance = "At or above proficiency level"

    # Next session plan
    plan = _build_session_plan(priorities, fatigue_risk, attempt_number)

    feedback_id = f"feedback_{current_score.video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    return FeedbackReport(
        feedback_id=feedback_id,
        video_id=current_score.video_id,
        score_id=current_score.id,
        headline=headline,
        fls_score=current_score.estimated_fls_score,
        completion_time=current_score.completion_time_seconds,
        attempt_number=attempt_number,
        phase_coaching=phase_coaching,
        top_priorities=priorities[:3],
        progression_insights=insights,
        strengths=strengths,
        fatigue_risk=fatigue_risk,
        fatigue_evidence=fatigue_evidence,
        distance_to_proficiency=distance,
        next_session_plan=plan,
    )


def _build_phase_coaching(
    current: ScoringResult,
    previous: list[ScoringResult],
) -> list[PhaseCoaching]:
    """Build per-phase coaching with trends."""
    coaching = []
    for pt in current.phase_timings:
        phase_name = pt.phase.value if hasattr(pt.phase, 'value') else pt.phase

        # Get trend from previous videos
        trend = []
        for prev in previous[-5:]:
            for prev_pt in prev.phase_timings:
                prev_phase = prev_pt.phase.value if hasattr(prev_pt.phase, 'value') else prev_pt.phase
                if prev_phase == phase_name:
                    trend.append(prev_pt.duration_seconds)

        # Determine status
        status = "on_track"
        if len(trend) >= 3:
            recent_avg = sum(trend[-3:]) / 3
            if pt.duration_seconds < recent_avg * 0.9:
                status = "improving"
            elif pt.duration_seconds > recent_avg * 1.15:
                status = "regressing"
            elif abs(pt.duration_seconds - recent_avg) < recent_avg * 0.05:
                status = "plateau"

        expert = EXPERT_BENCHMARKS.get(phase_name, 15)
        intermediate = INTERMEDIATE_BENCHMARKS.get(phase_name, 25)

        # Coaching note
        note = ""
        if status == "regressing":
            note = f"Regressed from recent average of {sum(trend[-3:])/3:.0f}s. Check for fatigue or equipment change."
        elif status == "plateau":
            note = f"Stable at ~{pt.duration_seconds:.0f}s for {len(trend)} attempts. Consider targeted drills."
        elif status == "improving":
            note = f"Trending down — keep current approach."

        # Drill recommendation
        drill = _recommend_drill(phase_name, pt.duration_seconds, intermediate)

        coaching.append(PhaseCoaching(
            phase=phase_name,
            duration_seconds=pt.duration_seconds,
            benchmark_seconds=intermediate,
            expert_seconds=expert,
            status=status,
            trend_last_5=trend[-5:],
            coaching_note=note,
            recommended_drill=drill,
        ))

    return coaching


def _recommend_drill(phase: str, duration: float, benchmark: float) -> Optional[DrillType]:
    """Recommend a specific drill based on phase performance."""
    if duration <= benchmark:
        return None

    drill_map = {
        "needle_load": DrillType.NEEDLE_LOADING,
        "suture_placement": DrillType.NEEDLE_DRIVING_ANGLE,
        "first_throw": DrillType.WRAPPING_SPEED,
        "second_throw": DrillType.HAND_SWITCHING,
        "third_throw": DrillType.HAND_SWITCHING,
        "suture_cut": DrillType.SCISSORS_TECHNIQUE,
    }
    return drill_map.get(phase)


def _identify_priorities(
    current: ScoringResult,
    phase_coaching: list[PhaseCoaching],
) -> list[ImprovementPriority]:
    """Rank improvement priorities by potential time savings."""
    priorities = []
    for pc in phase_coaching:
        gap = pc.duration_seconds - pc.benchmark_seconds
        if gap <= 0:
            continue

        priority_level = FeedbackPriority.LOW
        if gap > 20:
            priority_level = FeedbackPriority.HIGH
        elif gap > 10:
            priority_level = FeedbackPriority.MEDIUM

        priorities.append(ImprovementPriority(
            rank=1,  # temporary placeholder; reassigned after sorting
            priority=priority_level,
            phase=pc.phase,
            current_value=f"{pc.duration_seconds:.0f}s",
            target_value=f"{pc.benchmark_seconds:.0f}s",
            description=f"{pc.phase}: {pc.duration_seconds:.0f}s → target {pc.benchmark_seconds:.0f}s ({gap:.0f}s potential savings)",
            drill=pc.recommended_drill,
            expected_time_savings=f"~{gap:.0f}s",
        ))

    # Sort by gap (largest first) and assign ranks
    priorities.sort(key=lambda p: float(p.expected_time_savings.strip("~s")), reverse=True)
    for i, p in enumerate(priorities[:3]):
        p.rank = i + 1

    return priorities


def _build_progression_insights(
    current: ScoringResult,
    previous: list[ScoringResult],
    profile: Optional[TraineeProfile] = None,
) -> list[ProgressionInsight]:
    """Generate longitudinal progression insights."""
    insights = []

    if not previous:
        return [ProgressionInsight(
            insight_type="baseline",
            description="First scored attempt — this establishes the baseline.",
            evidence=f"Time: {current.completion_time_seconds}s, FLS: {current.estimated_fls_score}",
            recommendation="Focus on completing all phases correctly before optimizing speed.",
        )]

    times = [s.completion_time_seconds for s in previous] + [current.completion_time_seconds]
    baseline = times[0]
    current_time = times[-1]
    best_time = min(times)

    # Overall improvement
    if baseline > 0:
        improvement_pct = (baseline - current_time) / baseline * 100
        insights.append(ProgressionInsight(
            insight_type="overall_progress",
            description=f"{improvement_pct:.0f}% improvement from baseline ({baseline:.0f}s → {current_time:.0f}s)",
            evidence=f"Baseline: {baseline:.0f}s, Current: {current_time:.0f}s, Best: {best_time:.0f}s",
            recommendation="" if improvement_pct > 0 else "Performance below baseline — check equipment or fatigue.",
        ))
    else:
        insights.append(ProgressionInsight(
            insight_type="overall_progress",
            description="Baseline timing unavailable because earlier scored attempts had zero or missing duration.",
            evidence=f"Baseline: {baseline:.0f}s, Current: {current_time:.0f}s, Best: {best_time:.0f}s",
            recommendation="Interpret progression cautiously until scored attempts include measurable task completion times.",
        ))

    # Plateau detection (last 5 within 10% of each other)
    if len(times) >= 5:
        last5 = times[-5:]
        avg = sum(last5) / 5
        spread = (max(last5) - min(last5)) / avg if avg > 0 else 0
        if avg > 0 and spread < 0.10:
            insights.append(ProgressionInsight(
                insight_type="plateau_detection",
                description=f"Performance plateau at ~{avg:.0f}s for last 5 attempts",
                evidence=f"Last 5 times: {[f'{t:.0f}s' for t in last5]}",
                recommendation="Consider changing practice focus — targeted drills on weakest phase may break through.",
            ))

    # Fatigue pattern
    if len(times) >= 8:
        session_sizes = [3, 4, 5]
        for size in session_sizes:
            chunks = [times[i:i+size] for i in range(0, len(times)-size+1, size)]
            crashes = sum(1 for chunk in chunks if chunk[-1] > chunk[0] * 1.3)
            if crashes >= 2:
                insights.append(ProgressionInsight(
                    insight_type="fatigue_pattern",
                    description=f"Fatigue crashes detected after ~{size} consecutive attempts",
                    evidence=f"{crashes} instances of >30% slowdown within {size}-attempt blocks",
                    recommendation=f"Limit sessions to {size-1} scored attempts, then rest.",
                ))
                break

    return insights


def _identify_strengths(
    current: ScoringResult,
    previous: list[ScoringResult],
) -> list[str]:
    """Identify specific things the trainee did well."""
    strengths = []

    # From the score itself
    if current.strengths:
        strengths.extend(current.strengths[:3])

    # Drain not avulsed
    if current.drain_assessment and not current.drain_assessment.drain_avulsed:
        strengths.append("Drain not avulsed — safe tissue handling maintained")

    # Improvement from previous
    if previous:
        prev_time = previous[-1].completion_time_seconds
        if current.completion_time_seconds < prev_time:
            delta = prev_time - current.completion_time_seconds
            strengths.append(f"{delta:.0f}s faster than previous attempt")

    return strengths[:5]


def _assess_fatigue(
    current: ScoringResult,
    previous: list[ScoringResult],
) -> tuple[str, str]:
    """Assess fatigue risk from recent performance trajectory."""
    if len(previous) < 2:
        return "none", ""

    last3_times = [s.completion_time_seconds for s in previous[-2:]] + [current.completion_time_seconds]

    if len(last3_times) >= 3 and last3_times[-1] > last3_times[-3] * 1.2:
        return "moderate", f"Time increased 20%+ over last 3 attempts: {[f'{t:.0f}s' for t in last3_times]}"

    if len(last3_times) >= 2 and last3_times[-1] > last3_times[-2] * 1.3:
        return "high", f"30%+ slowdown from previous attempt: {last3_times[-2]:.0f}s → {last3_times[-1]:.0f}s"

    return "none", ""


def _generate_headline(
    current: ScoringResult,
    previous: list[ScoringResult],
    priorities: list[ImprovementPriority],
) -> str:
    """Generate a one-sentence coaching headline."""
    if not previous:
        return f"Baseline established: {current.completion_time_seconds:.0f}s / {current.estimated_fls_score:.0f} FLS"

    prev_time = previous[-1].completion_time_seconds
    delta = prev_time - current.completion_time_seconds

    if delta > 10:
        return f"Strong improvement: {delta:.0f}s faster ({current.completion_time_seconds:.0f}s / {current.estimated_fls_score:.0f} FLS)"
    elif delta < -15:
        return f"Regression: {abs(delta):.0f}s slower — check fatigue or equipment change"
    elif priorities:
        return f"Steady at {current.completion_time_seconds:.0f}s — focus on {priorities[0].phase} for next breakthrough"
    else:
        return f"Consistent performance: {current.completion_time_seconds:.0f}s / {current.estimated_fls_score:.0f} FLS"


def _build_session_plan(
    priorities: list[ImprovementPriority],
    fatigue_risk: str,
    attempt_number: int,
) -> str:
    """Build a concrete plan for the next practice session."""
    parts = []

    if fatigue_risk in ("moderate", "high"):
        parts.append("REST before next session — fatigue detected.")
        parts.append("When fresh, limit to 2-3 scored attempts.")
    else:
        parts.append("Good to continue — aim for 3-4 scored attempts next session.")

    if priorities:
        top = priorities[0]
        if top.drill:
            parts.append(f"Warm up with 5 minutes of {top.drill.value} drills before scoring.")
        parts.append(f"Primary focus: {top.phase} (currently {top.current_value}, target {top.target_value}).")

    if attempt_number >= 20:
        parts.append("Consider recording with direct camera for highest scoring confidence.")

    return " ".join(parts)
