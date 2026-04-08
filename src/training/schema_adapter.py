"""Schema adapter: normalize v001 and v002 scoring records to one canonical shape.

Why this exists
---------------
The ScoringResult Pydantic model in src/scoring/schema.py grew a new nested
``score_components`` block (time_score, penalty_deductions, total_fls_score)
that is intended to become authoritative in v002. Historical records on disk
(all 256 files surveyed on 2026-04-08) still carry v001 fields: the
authoritative number lives in top-level ``estimated_fls_score`` and
``score_components`` is ``None``.

If training or evaluation code hardcodes one shape, we either

  (a) silently read ``estimated_fls_score=0.0`` from a fresh v002 record and
      train / grade the model against the wrong truth, or
  (b) silently read ``score_components=None`` from a legacy record and
      drop every data point we actually have.

Both failure modes look like "the loss went down but the model is bad" — the
worst possible bug to debug on a rented A100.

This module is the single source of truth for reading scoring records. Every
training-data builder, evaluator, or analysis script should pull fields
through ``normalize_score`` or the helper accessors below, NEVER by reaching
directly into the dict.

Canonical shape (v002 target)
-----------------------------
After ``normalize_score`` the returned dict is guaranteed to contain:

    video_id            : str
    task_id             : str          ("task1" ... "task5"; defaults to "task5" only if source record is also unset)
    source              : str          ("consensus" / "teacher_claude" / "teacher_gpt" / "student" / ...)
    prompt_version      : str          ("v001" or "v002")
    total_fls_score     : float        authoritative score
    penalty_deductions  : float        sum of penalty points
    time_score          : float        component score
    completion_time_seconds : float
    suture_placement    : dict | None  (v001 shape preserved)
    drain_assessment    : dict | None
    knot_assessments    : list[dict]
    phase_timings       : list[dict]
    phases_detected     : list[str]
    frame_analyses      : list[dict]
    confidence_score    : float
    trainee_id          : str | None   (pulled from metadata if present)
    source_domain       : str | None   (lasana / petraw / simsurg / jigsaws / production / None)
    raw                 : dict         original record, untouched

The accessors ``get_total_score``, ``get_penalty_labels``, and
``get_task_id`` are deliberately cheap so hot loops can call them directly
without constructing the full normalized dict.
"""

from __future__ import annotations

from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# Canonical enumerations (must match src/scoring/schema.py)
# --------------------------------------------------------------------------- #

CANONICAL_TASK_IDS = {"task1", "task2", "task3", "task4", "task5"}
SPECIAL_TASK_PREFIXES = ("lasana_",)

_TASK_ALIASES = {
    "task1_peg_transfer": "task1",
    "task2_pattern_cut": "task2",
    "task3_endoloop": "task3",
    "task3_ligating_loop": "task3",
    "task4_extracorporeal_knot": "task4",
    "task4_extracorporeal_suture": "task4",
    "task5_intracorporeal_suturing": "task5",
    "task5_intracorporeal_suture": "task5",
}

CANONICAL_PENALTY_KEYS = [
    "suture_deviation",
    "slit_not_closed",
    "drain_avulsion",
    "hand_switch_failure",
    "throw_sequence_error",
    "knot_security_issue",
]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _as_dict(obj: Any) -> dict[str, Any]:
    """Accept either a Pydantic model or a plain dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def _float_or_zero(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def canonical_task_id(task_id: Any, *, default: str | None = None) -> str:
    """Normalize a task id string.

    Unlike the legacy helper in data_prep.py this one does NOT silently
    default to 'task5' unless the caller explicitly passes a default. That
    made it easy for a misrouted Task 1 score to be trained as Task 5.
    """
    raw = str(task_id or "").strip().lower()
    if not raw:
        return default or ""
    if raw in _TASK_ALIASES:
        return _TASK_ALIASES[raw]
    if raw in CANONICAL_TASK_IDS:
        return raw
    if any(raw.startswith(prefix) for prefix in SPECIAL_TASK_PREFIXES):
        return raw
    if raw.startswith("task"):
        prefix = raw.split("_", 1)[0]
        if prefix in CANONICAL_TASK_IDS:
            return prefix
    if raw.isdigit() and f"task{raw}" in CANONICAL_TASK_IDS:
        return f"task{raw}"
    return default or ""


# --------------------------------------------------------------------------- #
# Core accessors
# --------------------------------------------------------------------------- #

def get_total_score(record: Any) -> float:
    """Read the authoritative FLS score from a v001 OR v002 record.

    Precedence:
      1. v002 ``score_components.total_fls_score`` if present and > 0
      2. v001 ``estimated_fls_score``
      3. 0.0
    """
    payload = _as_dict(record)
    components = payload.get("score_components")
    if components:
        comp = _as_dict(components)
        total = _float_or_zero(comp.get("total_fls_score"))
        if total != 0.0:
            return total
    return _float_or_zero(payload.get("estimated_fls_score"))


def get_penalty_deductions(record: Any) -> float:
    """Read penalty deductions from either schema version."""
    payload = _as_dict(record)
    components = payload.get("score_components")
    if components:
        comp = _as_dict(components)
        penalty = _float_or_zero(comp.get("penalty_deductions"))
        if penalty != 0.0:
            return penalty
    return _float_or_zero(payload.get("estimated_penalties"))


def get_time_score(record: Any) -> float:
    payload = _as_dict(record)
    components = payload.get("score_components")
    if components:
        comp = _as_dict(components)
        time_score = _float_or_zero(comp.get("time_score"))
        if time_score != 0.0:
            return time_score
    # v001 never recorded time_score separately, derive from 600 - penalties
    # only if we have both totals.
    total = _float_or_zero(payload.get("estimated_fls_score"))
    penalties = _float_or_zero(payload.get("estimated_penalties"))
    if total or penalties:
        return total + penalties
    return 0.0


def get_task_id(record: Any, *, default: str | None = None) -> str:
    """Return canonical task_id, preferring task_id then metadata.task_id."""
    payload = _as_dict(record)
    tid = canonical_task_id(payload.get("task_id"), default="")
    if tid:
        return tid
    metadata = payload.get("metadata") or {}
    tid = canonical_task_id(metadata.get("task_id") if isinstance(metadata, dict) else None, default="")
    if tid:
        return tid
    return default or ""


def get_trainee_id(record: Any) -> str | None:
    """Resident-aware field. Reads metadata.trainee_id if present, else None."""
    payload = _as_dict(record)
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, dict):
        tid = metadata.get("trainee_id") or metadata.get("resident_id")
        if tid:
            return str(tid)
    tid = payload.get("trainee_id") or payload.get("resident_id")
    return str(tid) if tid else None


def get_source_domain(record: Any) -> str | None:
    """Which dataset did this come from? lasana / petraw / simsurg / jigsaws / production."""
    payload = _as_dict(record)
    metadata = payload.get("metadata") or {}
    if isinstance(metadata, dict):
        domain = metadata.get("source_domain") or metadata.get("dataset")
        if domain:
            return str(domain).lower()
    # Heuristic fallback from video_id prefix.
    video_id = str(payload.get("video_id") or "")
    for prefix, domain in (
        ("lasana_", "lasana"),
        ("petraw_", "petraw"),
        ("simsurg_", "simsurg"),
        ("jigsaws_", "jigsaws"),
        ("yt_", "production"),
    ):
        if video_id.startswith(prefix):
            return domain
    return None


# --------------------------------------------------------------------------- #
# Penalty label derivation (moved from eval_v2 so eval and training agree)
# --------------------------------------------------------------------------- #

def get_penalty_labels(record: Any) -> set[str]:
    """Derive the canonical penalty-present set for an FLS scoring record.

    The label set is the same for v001 and v002 because the sub-structures
    (suture_placement, drain_assessment, knot_assessments) did not change
    between versions — only the top-level score location did.
    """
    payload = _as_dict(record)
    labels: set[str] = set()

    placement = _as_dict(payload.get("suture_placement"))
    if _float_or_zero(placement.get("total_deviation_penalty")) > 0:
        labels.add("suture_deviation")

    drain = _as_dict(payload.get("drain_assessment"))
    closure = str(drain.get("slit_closure_quality") or "").lower()
    if bool(drain.get("gap_visible")) or closure in {"partial", "poor", "incomplete"}:
        labels.add("slit_not_closed")
    if bool(drain.get("drain_avulsed")):
        labels.add("drain_avulsion")

    knot_assessments = payload.get("knot_assessments") or []
    knots = [_as_dict(item) for item in knot_assessments if item]

    if any(item.get("appears_secure") is False for item in knots):
        labels.add("knot_security_issue")
    if any(
        item.get("throw_number") in {2, 3} and item.get("hand_switched") is False
        for item in knots
    ):
        labels.add("hand_switch_failure")

    throw_map = {
        throw_number: item
        for item in knots
        for throw_number in [_int_or_none(item.get("throw_number"))]
        if throw_number is not None
    }
    if throw_map:
        first = throw_map.get(1) or {}
        second = throw_map.get(2) or {}
        third = throw_map.get(3) or {}
        sequence_error = (
            first.get("is_surgeon_knot") is False
            or second.get("is_single_throw") is False
            or third.get("is_single_throw") is False
        )
        if sequence_error:
            labels.add("throw_sequence_error")

    return labels


def get_phase_presence(record: Any) -> set[str]:
    payload = _as_dict(record)
    phases: set[str] = set()
    for phase in payload.get("phases_detected", []) or []:
        phases.add(str(phase))
    for phase in payload.get("phase_timings", []) or []:
        if isinstance(phase, dict) and phase.get("phase"):
            phases.add(str(phase["phase"]))
    return phases


# --------------------------------------------------------------------------- #
# Full normalization
# --------------------------------------------------------------------------- #

def normalize_score(record: Any, *, default_task: str | None = None) -> dict[str, Any]:
    """Normalize a v001 or v002 scoring record to a stable dict.

    See module docstring for the canonical shape.
    """
    payload = _as_dict(record)
    return {
        "video_id": str(payload.get("video_id") or ""),
        "task_id": get_task_id(payload, default=default_task),
        "source": str(payload.get("source") or ""),
        "prompt_version": str(payload.get("prompt_version") or "v001"),
        "total_fls_score": get_total_score(payload),
        "penalty_deductions": get_penalty_deductions(payload),
        "time_score": get_time_score(payload),
        "completion_time_seconds": _float_or_zero(payload.get("completion_time_seconds")),
        "suture_placement": payload.get("suture_placement") or None,
        "drain_assessment": payload.get("drain_assessment") or None,
        "knot_assessments": payload.get("knot_assessments") or [],
        "phase_timings": payload.get("phase_timings") or [],
        "phases_detected": list(get_phase_presence(payload)),
        "frame_analyses": payload.get("frame_analyses") or [],
        "confidence_score": _float_or_zero(payload.get("confidence_score")),
        "trainee_id": get_trainee_id(payload),
        "source_domain": get_source_domain(payload),
        "penalty_labels": sorted(get_penalty_labels(payload)),
        "raw": payload,
    }


def normalize_scores(records: Iterable[Any], *, default_task: str | None = None) -> list[dict[str, Any]]:
    return [normalize_score(r, default_task=default_task) for r in records]


__all__ = [
    "CANONICAL_PENALTY_KEYS",
    "CANONICAL_TASK_IDS",
    "canonical_task_id",
    "get_penalty_deductions",
    "get_penalty_labels",
    "get_phase_presence",
    "get_source_domain",
    "get_task_id",
    "get_time_score",
    "get_total_score",
    "get_trainee_id",
    "normalize_score",
    "normalize_scores",
]
