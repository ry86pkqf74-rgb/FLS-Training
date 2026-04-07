"""Pydantic models for FLS scoring pipeline.

Every data structure in the system flows through these schemas.
All timestamps are ISO 8601. All IDs are generated via ulid-style or uuid4.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FLSTask(str, Enum):
    PEG_TRANSFER = "task1_peg_transfer"
    PATTERN_CUT = "task2_pattern_cut"
    CLIP_APPLY = "task3_clip_apply"
    EXTRACORPOREAL = "task4_extracorporeal_suture"
    INTRACORPOREAL = "task5_intracorporeal_suture"


class Phase(str, Enum):
    NEEDLE_LOAD = "needle_load"
    SUTURE_PLACEMENT = "suture_placement"
    FIRST_THROW = "first_throw"
    SECOND_THROW = "second_throw"
    THIRD_THROW = "third_throw"
    SUTURE_CUT = "suture_cut"
    COMPLETION = "completion"
    IDLE = "idle"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Hand(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UNCLEAR = "unclear"


class CorrectorRole(str, Enum):
    EXPERT = "expert"
    RESIDENT = "resident"
    SELF = "self"


class ScoreSource(str, Enum):
    TEACHER_CLAUDE = "teacher_claude"
    TEACHER_GPT4O = "teacher_gpt4o"
    CRITIQUE_CONSENSUS = "critique_consensus"
    EXPERT_CORRECTION = "expert_correction"
    STUDENT_MODEL = "student_model"


# ---------------------------------------------------------------------------
# Video & Frame Models
# ---------------------------------------------------------------------------

class VideoMetadata(BaseModel):
    id: str = Field(default_factory=_new_id)
    filename: str
    task: FLSTask
    duration_seconds: float
    resolution: str  # "1280x720"
    fps: float
    file_hash: str = ""
    ingested_at: datetime = Field(default_factory=_now)


class Frame(BaseModel):
    frame_number: int
    timestamp_seconds: float
    image_b64: str = Field(exclude=True, default="")  # excluded from JSON serialization by default


class FrameAnalysis(BaseModel):
    frame_number: int
    phase: Phase
    description: str
    technique_notes: str = ""


# ---------------------------------------------------------------------------
# Scoring Components
# ---------------------------------------------------------------------------

class PhaseTiming(BaseModel):
    phase: Phase
    start_seconds: float
    end_seconds: float
    duration_seconds: float


class ThrowAssessment(BaseModel):
    throw_number: int  # 1, 2, or 3
    is_surgeon_knot: Optional[bool] = None  # only relevant for throw 1
    is_single_throw: Optional[bool] = None  # relevant for throws 2, 3
    hand_used: Hand = Hand.UNCLEAR
    hand_switched: Optional[bool] = None  # relative to previous throw
    appears_secure: bool = False
    notes: str = ""


class SuturePlacement(BaseModel):
    deviation_from_mark1_mm: float = 0.0
    deviation_from_mark2_mm: float = 0.0
    total_deviation_penalty: float = 0.0
    confidence: Confidence = Confidence.LOW


class DrainAssessment(BaseModel):
    gap_visible: bool = False
    drain_avulsed: bool = False
    slit_closure_quality: str = "unknown"  # complete | partial | poor


# ---------------------------------------------------------------------------
# Core Scoring Result
# ---------------------------------------------------------------------------

class ScoringResult(BaseModel):
    """The central output of any scoring operation — teacher, critique, or student."""

    id: str = Field(default_factory=_new_id)
    video_id: str
    source: ScoreSource
    model_name: str
    model_version: str = ""
    prompt_version: str = "v001"
    scored_at: datetime = Field(default_factory=_now)

    # Frame-level analysis
    frame_analyses: list[FrameAnalysis] = []

    # Timing
    completion_time_seconds: float = 0.0
    phase_timings: list[PhaseTiming] = []

    # Task-specific assessments
    knot_assessments: list[ThrowAssessment] = []
    suture_placement: Optional[SuturePlacement] = None
    drain_assessment: Optional[DrainAssessment] = None

    # Computed scores
    estimated_penalties: float = 0.0
    estimated_fls_score: float = 0.0

    # Meta
    confidence_score: float = 0.0  # 0.0 to 1.0
    technique_summary: str = ""
    improvement_suggestions: list[str] = []
    strengths: list[str] = []

    # Cost tracking
    api_cost_usd: float = 0.0
    latency_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Critique / Consensus
# ---------------------------------------------------------------------------

class Divergence(BaseModel):
    field: str
    teacher_a_value: str
    teacher_b_value: str
    resolution: str  # which value was chosen or a new value
    reasoning: str


class CritiqueResult(BaseModel):
    """Output of the critique agent comparing two teacher scores."""

    id: str = Field(default_factory=_new_id)
    video_id: str
    teacher_a_score_id: str
    teacher_b_score_id: str
    critiqued_at: datetime = Field(default_factory=_now)

    agreement_score: float = 0.0  # 0.0 to 1.0
    divergences: list[Divergence] = []
    consensus_score: Optional[ScoringResult] = None
    critique_reasoning: str = ""
    confidence: float = 0.0

    # Cost
    api_cost_usd: float = 0.0
    latency_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Corrections (human-in-the-loop)
# ---------------------------------------------------------------------------

class Correction(BaseModel):
    id: str = Field(default_factory=_new_id)
    video_id: str
    original_score_id: str
    corrected_at: datetime = Field(default_factory=_now)
    corrector_role: CorrectorRole
    corrected_fields: dict  # field_name -> corrected_value
    notes: str = ""


# ---------------------------------------------------------------------------
# Training & Learning
# ---------------------------------------------------------------------------

class TrainingRun(BaseModel):
    id: str = Field(default_factory=_new_id)
    started_at: datetime = Field(default_factory=_now)
    completed_at: Optional[datetime] = None
    model_base: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    dataset_version: str = ""
    n_train_samples: int = 0
    n_val_samples: int = 0
    n_test_samples: int = 0
    config: dict = {}
    eval_metrics: dict = {}
    checkpoint_path: str = ""
    status: str = "pending"  # pending | running | completed | failed


class LearningEvent(BaseModel):
    timestamp: datetime = Field(default_factory=_now)
    event_type: str
    data: dict = {}
