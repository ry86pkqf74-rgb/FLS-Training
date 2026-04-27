"""Normalize a raw scoring JSON into a v003-shape training target.

The v003 reporting pipeline expects every score to carry the new fields
(critical_errors, points_deducted/severity, cannot_determine, formula_applied,
task_specific_assessments, confidence_rationale). Old teacher scores in
``memory/scores/`` and the existing ``youtube_sft_v1`` JSONL only carry a
subset. This module enriches them deterministically so the LoRA learns the
v003 contract without re-scoring videos.

Usage::

    from src.training.v003_target import enrich_to_v003_target
    target = enrich_to_v003_target(score_dict, task_id="task5")

The function never raises on malformed input — it fills sensible defaults so
the prep pipeline can run end-to-end across the whole dataset.
"""
from __future__ import annotations

from typing import Any

from src.rubrics.loader import (
    canonical_task_id,
    get_task_max_score,
    get_task_max_time,
    get_task_name,
    is_official_fls_task,
)
from src.scoring.frontier_scorer import recompute_score_from_components


# Penalty types the rubric flags as critical (forces zero or blocks proficiency).
# Keep the keys lowercase, snake/kebab-tolerant — match by substring.
CRITICAL_PENALTY_TYPES: dict[str, dict[str, bool]] = {
    "drain_avulsion":       {"forces_zero_score": True,  "blocks_proficiency_claim": True},
    "drain_avulsed":        {"forces_zero_score": True,  "blocks_proficiency_claim": True},
    "gauze_detachment":     {"forces_zero_score": True,  "blocks_proficiency_claim": True},
    "gauze_detached":       {"forces_zero_score": True,  "blocks_proficiency_claim": True},
    "appendage_transection":{"forces_zero_score": True,  "blocks_proficiency_claim": True},
    "block_dislodged":      {"forces_zero_score": True,  "blocks_proficiency_claim": True},
    "needle_left_view":     {"forces_zero_score": True,  "blocks_proficiency_claim": True},
    "needle_exits_field":   {"forces_zero_score": True,  "blocks_proficiency_claim": True},
    "incomplete_task":      {"forces_zero_score": True,  "blocks_proficiency_claim": True},
    "knot_failure":         {"forces_zero_score": False, "blocks_proficiency_claim": True},
    "gap_visible":          {"forces_zero_score": False, "blocks_proficiency_claim": True},
    "loop_off_mark":        {"forces_zero_score": False, "blocks_proficiency_claim": True},
    "hand_switch_failure":  {"forces_zero_score": False, "blocks_proficiency_claim": True},
    "wrong_transfer":       {"forces_zero_score": False, "blocks_proficiency_claim": True},
    "lost_object":          {"forces_zero_score": False, "blocks_proficiency_claim": True},
}


def _severity_from_points(points: float) -> str:
    if points >= 50:
        return "critical"
    if points >= 20:
        return "major"
    if points >= 5:
        return "moderate"
    return "minor"


def _match_critical(penalty_type: str) -> dict[str, bool] | None:
    norm = (penalty_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not norm:
        return None
    if norm in CRITICAL_PENALTY_TYPES:
        return CRITICAL_PENALTY_TYPES[norm]
    for key, flags in CRITICAL_PENALTY_TYPES.items():
        if key in norm or norm in key:
            return flags
    return None


def _enrich_penalty(raw: Any) -> dict[str, Any]:
    """Coerce a raw penalty into the v003 PenaltyItem shape."""
    if not isinstance(raw, dict):
        return {
            "type": str(raw),
            "description": "",
            "points_deducted": 0.0,
            "count": 1,
            "severity": "minor",
            "frame_evidence": [],
            "confidence": 0.5,
            "rubric_reference": "",
        }
    points = float(
        raw.get("points_deducted")
        or raw.get("value")
        or raw.get("deduction")
        or 0.0
    )
    severity = raw.get("severity") or _severity_from_points(points)
    if _match_critical(raw.get("type", "")) and severity == "minor":
        severity = "major"
    return {
        "type": raw.get("type", "unspecified"),
        "description": raw.get("description", ""),
        "points_deducted": points,
        "count": int(raw.get("count", 1) or 1),
        "severity": severity,
        "frame_evidence": list(raw.get("frame_evidence", [])),
        "confidence": float(raw.get("confidence", 0.5) or 0.5),
        "rubric_reference": raw.get("rubric_reference", ""),
    }


def _derive_critical_errors(penalties: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Promote any penalty matching CRITICAL_PENALTY_TYPES to a CriticalError."""
    errors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for p in penalties:
        flags = _match_critical(p.get("type", ""))
        if not flags:
            continue
        key = p.get("type", "").lower()
        if key in seen:
            continue
        seen.add(key)
        errors.append({
            "type": p["type"],
            "present": True,
            "reason": p.get("description", ""),
            "frame_evidence": list(p.get("frame_evidence", [])),
            "forces_zero_score": flags["forces_zero_score"],
            "blocks_proficiency_claim": flags["blocks_proficiency_claim"],
        })
    return errors


def _confidence_rationale(score: dict[str, Any]) -> str:
    confidence = float(score.get("confidence_score") or score.get("confidence") or 0.5)
    if confidence >= 0.85:
        return "High confidence: clear visual evidence across frames."
    if confidence >= 0.6:
        return "Moderate confidence: most key events were observable, minor uncertainty remains."
    if confidence >= 0.4:
        return "Lower confidence: ambiguity in camera angle or partial visibility."
    return "Low confidence: significant portions of the task were not clearly observable."


def _build_task_specific_assessments(score: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "knot_assessments",
        "suture_placement",
        "drain_assessment",
        "phase_timings",
        "phases_detected",
        "frame_analyses",
    ):
        val = score.get(key)
        if val:
            out[key] = val
    return out


def enrich_to_v003_target(
    score: dict[str, Any],
    task_id: str | int | None = None,
) -> dict[str, Any]:
    """Return a v003-shape target dict suitable for use as assistant_content.

    The function is deterministic and side-effect-free relative to the input
    (input is shallow-copied; the score_components block is rewritten via
    ``recompute_score_from_components``).
    """
    payload = dict(score)  # shallow copy
    canonical = canonical_task_id(task_id or payload.get("task_id") or "task5")
    payload.setdefault("task_id", canonical)

    # Coerce penalties to v003 shape first (recompute uses points_deducted).
    raw_penalties = payload.get("penalties") or []
    enriched_penalties = [_enrich_penalty(p) for p in raw_penalties]
    payload["penalties"] = enriched_penalties

    # Recompute score math from rubric.
    recompute_score_from_components(payload, canonical)

    # Critical errors: derive from major/critical penalties.
    critical_errors = list(payload.get("critical_errors") or [])
    if not critical_errors:
        critical_errors = _derive_critical_errors(enriched_penalties)

    # Cannot determine: keep what the model said; fall back to empty list.
    cannot_determine = list(payload.get("cannot_determine") or [])

    target = {
        "task_id": canonical,
        "task_name": payload.get("task_name") or get_task_name(canonical),
        "official_fls_task": is_official_fls_task(canonical),
        "max_score": get_task_max_score(canonical),
        "max_time_seconds": get_task_max_time(canonical),
        "completion_time_seconds": float(payload.get("completion_time_seconds") or 0.0),
        "score_components": payload["score_components"],
        "estimated_fls_score": payload["estimated_fls_score"],
        "estimated_penalties": payload["estimated_penalties"],
        "confidence_score": float(payload.get("confidence_score") or payload.get("confidence") or 0.5),
        "confidence_rationale": payload.get("confidence_rationale") or _confidence_rationale(payload),
        "video_classification": payload.get("video_classification", "scorable"),
        "penalties": enriched_penalties,
        "critical_errors": critical_errors,
        "cannot_determine": cannot_determine,
        "task_specific_assessments": _build_task_specific_assessments(payload),
        "technique_summary": payload.get("technique_summary", ""),
        "strengths": list(payload.get("strengths") or []),
        "improvement_suggestions": list(payload.get("improvement_suggestions") or []),
    }
    return target


def is_v003_target(target: dict[str, Any]) -> bool:
    """Cheap shape check used by validators."""
    if not isinstance(target, dict):
        return False
    sc = target.get("score_components")
    if not isinstance(sc, dict):
        return False
    required = {"max_score", "time_used", "total_penalties", "total_fls_score", "formula_applied"}
    return required.issubset(sc.keys()) and "critical_errors" in target
