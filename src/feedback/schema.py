"""Feedback schema for coaching output — the second training head.

This defines what the student model must learn to produce:
not just scores, but actionable, progression-aware coaching feedback
that helps trainees actually improve their surgical technique.
"""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class FeedbackPriority(str, enum.Enum):
    CRITICAL = "critical"      # Safety issue or major penalty
    HIGH = "high"              # Biggest time-saver available
    MEDIUM = "medium"          # Meaningful but not urgent
    LOW = "low"                # Polish / optimization


class DrillType(str, enum.Enum):
    """Specific practice drills the coach can recommend."""
    NEEDLE_LOADING = "needle_loading"
    NEEDLE_DRIVING_ANGLE = "needle_driving_angle"
    SUTURE_PLACEMENT_ACCURACY = "suture_placement_accuracy"
    WRAPPING_SPEED = "wrapping_speed"
    HAND_SWITCHING = "hand_switching"
    KNOT_TENSION = "knot_tension"
    SCISSORS_TECHNIQUE = "scissors_technique"
    TAIL_LENGTH_CONTROL = "tail_length_control"
    ECONOMY_OF_MOTION = "economy_of_motion"
    TRANSITION_EFFICIENCY = "transition_efficiency"


class PhaseCoaching(BaseModel):
    """Coaching feedback for a specific phase."""
    phase: str
    duration_seconds: float
    benchmark_seconds: float = Field(
        description="Target time for this phase at trainee's level"
    )
    expert_seconds: float = Field(
        description="Expert-level time for reference"
    )
    status: str = "on_track"  # improving, plateau, regressing, on_track
    trend_last_5: list[float] = Field(
        default_factory=list,
        description="Duration of this phase in last 5 videos"
    )
    coaching_note: str = ""
    recommended_drill: Optional[DrillType] = None


class ProgressionInsight(BaseModel):
    """Longitudinal observation about the trainee's development."""
    insight_type: str  # equipment_adaptation, fatigue_pattern, plateau_detection,
                       # breakthrough, regression, consistency_improvement
    description: str
    evidence: str = Field(description="Specific data points supporting this insight")
    actionable: bool = True
    recommendation: str = ""


class EquipmentNote(BaseModel):
    """How equipment changes affect performance."""
    equipment_id: str  # rca_station, vizio_monitor, blue_trainer
    attempts_on_equipment: int
    adaptation_curve: str = ""  # e.g. "365→295→... (improving)"
    comparison_to_best_equipment: str = ""


class FeedbackReport(BaseModel):
    """Complete coaching feedback for a single video attempt.

    This is the PRIMARY training target for the coaching head.
    The student model must learn to produce this given:
    - Current video frames
    - Trainee history (previous scores summary)
    """
    feedback_id: str
    video_id: str
    score_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generator: str = "teacher"  # teacher, student, consensus

    # === Overall assessment ===
    headline: str = Field(
        description="One-sentence summary: what happened and what it means. "
        "e.g. 'Strong adaptation to new equipment — 13% faster than first attempt'"
    )
    fls_score: float
    completion_time: float
    attempt_number: int = Field(description="Which attempt overall (1-indexed)")

    # === Phase-by-phase coaching ===
    phase_coaching: list[PhaseCoaching] = []

    # === Top priorities (ordered) ===
    top_priorities: list[ImprovementPriority] = Field(
        default_factory=list,
        description="Max 3 ranked priorities for next practice session"
    )

    # === Progression insights ===
    progression_insights: list[ProgressionInsight] = Field(
        default_factory=list,
        description="Longitudinal observations about development trajectory"
    )

    # === Equipment context ===
    equipment_note: Optional[EquipmentNote] = None

    # === Strengths (positive reinforcement) ===
    strengths: list[str] = Field(
        default_factory=list,
        description="Specific things done well — be concrete, not generic"
    )

    # === Fatigue assessment ===
    fatigue_risk: str = "none"  # none, low, moderate, high
    fatigue_evidence: str = ""
    session_recommendation: str = Field(
        default="",
        description="e.g. 'Take a break' or 'Good to continue' or 'End session'"
    )

    # === Benchmarks ===
    percentile_estimate: Optional[float] = Field(
        default=None,
        description="Estimated percentile among FLS trainees (0-100)"
    )
    distance_to_proficiency: str = Field(
        default="",
        description="How far from passing FLS cutoff score"
    )

    # === Next session plan ===
    next_session_plan: str = Field(
        default="",
        description="Concrete plan for next practice: what to focus on, how many attempts, "
        "what to watch for"
    )


class ImprovementPriority(BaseModel):
    """A single ranked improvement priority."""
    rank: int = Field(ge=1, le=3)
    priority: FeedbackPriority
    phase: str
    current_value: str = ""
    target_value: str = ""
    description: str
    drill: Optional[DrillType] = None
    expected_time_savings: str = ""


class TraineeProfile(BaseModel):
    """Accumulated profile of a trainee built from all their videos.

    Stored in GitHub memory and loaded as context for each new scoring.
    """
    trainee_id: str = "default"
    total_attempts: int = 0
    first_attempt_date: Optional[datetime] = None
    last_attempt_date: Optional[datetime] = None

    # Performance summary
    best_time_seconds: float = 999.0
    best_fls_score: float = 0.0
    current_plateau_time: float = 0.0
    baseline_time: float = 0.0

    # Phase averages (last 5)
    avg_placement_last5: float = 0.0
    avg_throws_last5: float = 0.0
    avg_cutting_last5: float = 0.0

    # Equipment history
    equipment_used: list[str] = Field(default_factory=list)
    equipment_best_times: dict = Field(default_factory=dict)

    # Identified patterns
    fatigue_crash_count: int = 0
    typical_crash_threshold: int = 0  # attempts per session before fatigue
    identified_plateaus: list[str] = Field(default_factory=list)

    # Strengths and weaknesses
    consistent_strengths: list[str] = Field(default_factory=list)
    persistent_weaknesses: list[str] = Field(default_factory=list)
    bottleneck_phase: str = ""  # which phase is the biggest time sink


# Fix forward reference
FeedbackReport.model_rebuild()
