"""Core data models for FLS scoring and feedback."""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class FLSTask(str, enum.Enum):
    TASK1_PEG_TRANSFER = "task1_peg_transfer"
    TASK2_PATTERN_CUT = "task2_pattern_cut"
    TASK3_LIGATING_LOOP = "task3_ligating_loop"
    TASK4_EXTRACORPOREAL = "task4_extracorporeal_suture"
    TASK5_INTRACORPOREAL = "task5_intracorporeal_suture"


class Phase(str, enum.Enum):
    IDLE = "idle"
    NEEDLE_LOAD = "needle_load"
    SUTURE_PLACEMENT = "suture_placement"
    FIRST_THROW = "first_throw"
    SECOND_THROW = "second_throw"
    THIRD_THROW = "third_throw"
    SUTURE_CUT = "suture_cut"
    COMPLETION = "completion"
    RING_TRAVERSAL = "ring_traversal"
    SETUP = "setup"


class FrameAnalysis(BaseModel):
    frame_number: int
    phase: Phase
    description: str
    technique_notes: str = ""


class PhaseTiming(BaseModel):
    phase: Phase
    start_seconds: float
    end_seconds: float
    duration_seconds: float


class KnotAssessment(BaseModel):
    throw_number: int = Field(ge=1, le=3)
    is_surgeon_knot: Optional[bool] = None
    is_single_throw: Optional[bool] = None
    hand_used: str = "unclear"  # left, right, unclear
    hand_switched: Optional[bool] = None
    appears_secure: bool = True
    notes: str = ""


class SuturePlacement(BaseModel):
    deviation_from_mark1_mm: float = 0.0
    deviation_from_mark2_mm: float = 0.0
    total_deviation_penalty: float = 0.0
    confidence: str = "low"  # low, medium, high


class DrainAssessment(BaseModel):
    gap_visible: bool = False
    drain_avulsed: bool = False
    slit_closure_quality: str = "unknown"  # complete, well_closed, partial, poor, unknown
    assessment_note: str = ""


class PenaltyItem(BaseModel):
    type: str
    count: int = 1
    description: str = ""


class ScoreComponents(BaseModel):
    time_score: float = 0.0
    penalty_deductions: float = 0.0
    total_fls_score: float = 0.0


class ScoringResult(BaseModel):
    """Complete scoring output from a teacher model."""
    id: str
    video_id: str
    video_filename: str
    video_hash: str = ""
    source: str  # teacher_claude, teacher_gpt, consensus, student
    model_name: str
    model_version: str
    prompt_version: str = "v001"
    scored_at: datetime = Field(default_factory=datetime.utcnow)

    task_id: str = ""
    penalties: list[PenaltyItem] = []
    score_components: Optional[ScoreComponents] = None
    phases_detected: list[str] = []
    reasoning: str = ""

    frame_analyses: list[FrameAnalysis] = []
    completion_time_seconds: float
    phase_timings: list[PhaseTiming] = []
    knot_assessments: list[KnotAssessment] = []
    suture_placement: Optional[SuturePlacement] = None
    drain_assessment: Optional[DrainAssessment] = None

    estimated_penalties: float = 0.0
    estimated_fls_score: float = 0.0
    confidence_score: float = 0.5

    technique_summary: str = ""
    improvement_suggestions: list[str] = []
    strengths: list[str] = []

    comparison_to_previous: dict = {}

    # Supersession metadata: when a score is forensically corrected,
    # the stale record is retained on disk for audit but flagged here
    # so that downstream pipelines (data_prep, drift, profile rebuild)
    # can skip it. See memory_store.get_all_scores(skip_superseded=True).
    superseded: bool = False
    superseded_by: Optional[str] = None
    superseded_at: Optional[datetime] = None
    superseded_reason: str = ""


class VideoRecord(BaseModel):
    """Metadata for an ingested video."""
    video_id: str
    filename: str
    task: FLSTask
    duration_seconds: float
    resolution: str = ""
    fps: int = 30
    file_hash: str = ""
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    recording_note: str = ""
    frame_count_extracted: int = 0


class CorrectionRecord(BaseModel):
    """Expert correction applied to a score."""
    correction_id: str
    video_id: str
    score_id: str
    corrected_fields: dict
    corrector: str = "expert"
    notes: str = ""
    corrected_at: datetime = Field(default_factory=datetime.utcnow)
