"""Evaluate student model against teacher scores."""
from __future__ import annotations

import json
from pathlib import Path

from src.memory.memory_store import MemoryStore
from src.scoring.schema import ScoringResult


def evaluate_student(
    student_scores_dir: str | Path,
    base_dir: str | Path = ".",
) -> dict:
    """Compare student scores against teacher consensus.

    Returns agreement metrics for scoring accuracy and feedback quality.
    """
    store = MemoryStore(base_dir)
    teacher_scores = {s.video_id: s for s in store.get_all_scores() if s.source == "consensus"}

    student_dir = Path(student_scores_dir)
    student_scores = {}
    for f in student_dir.glob("*.json"):
        s = ScoringResult.model_validate_json(f.read_text())
        student_scores[s.video_id] = s

    common_videos = set(teacher_scores) & set(student_scores)
    if not common_videos:
        return {"error": "No overlapping videos between student and teacher"}

    time_errors = []
    fls_errors = []
    phase_agreements = []

    for vid in sorted(common_videos):
        teacher = teacher_scores[vid]
        student = student_scores[vid]

        time_err = abs(teacher.completion_time_seconds - student.completion_time_seconds)
        fls_err = abs(teacher.estimated_fls_score - student.estimated_fls_score)
        time_errors.append(time_err)
        fls_errors.append(fls_err)

        # Phase timing agreement
        t_phases = {p.phase: p.duration_seconds for p in teacher.phase_timings}
        s_phases = {p.phase: p.duration_seconds for p in student.phase_timings}
        common_phases = set(t_phases) & set(s_phases)
        if common_phases:
            phase_err = sum(abs(t_phases[p] - s_phases[p]) for p in common_phases) / len(common_phases)
            phase_agreements.append(phase_err)

    avg_time_err = sum(time_errors) / len(time_errors)
    avg_fls_err = sum(fls_errors) / len(fls_errors)
    avg_phase_err = sum(phase_agreements) / len(phase_agreements) if phase_agreements else float('inf')

    # Agreement threshold: within 10s on time, 20 on FLS
    time_agree_pct = sum(1 for e in time_errors if e <= 10) / len(time_errors) * 100
    fls_agree_pct = sum(1 for e in fls_errors if e <= 20) / len(fls_errors) * 100

    result = {
        "videos_evaluated": len(common_videos),
        "avg_time_error_seconds": round(avg_time_err, 1),
        "avg_fls_score_error": round(avg_fls_err, 1),
        "avg_phase_error_seconds": round(avg_phase_err, 1),
        "time_agreement_pct": round(time_agree_pct, 1),
        "fls_agreement_pct": round(fls_agree_pct, 1),
        "ready_for_promotion": time_agree_pct >= 85 and fls_agree_pct >= 85,
        "per_video": {
            vid: {
                "time_error": round(abs(teacher_scores[vid].completion_time_seconds - student_scores[vid].completion_time_seconds), 1),
                "fls_error": round(abs(teacher_scores[vid].estimated_fls_score - student_scores[vid].estimated_fls_score), 1),
            }
            for vid in sorted(common_videos)
        }
    }

    return result
