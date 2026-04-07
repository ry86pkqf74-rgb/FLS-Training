"""Detect scoring drift and determine when retraining is needed."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from src.memory.memory_store import MemoryStore
from src.memory.learning_log import LearningLog

logger = logging.getLogger(__name__)

RETRAIN_TRIGGERS = {
    "min_new_corrections": 20,
    "confidence_drop_threshold": 0.1,
    "max_days_since_training": 30,
    "min_frontier_agreement": 0.85,
}


def check_retrain_needed(
    store: MemoryStore,
    log: LearningLog,
) -> dict:
    """Check if retraining is recommended.

    Returns dict with should_retrain, reasons, and stats.
    """
    now = datetime.now(timezone.utc)
    reasons: list[str] = []

    # Find last training event
    training_events = log.read_events(event_type="training_completed")
    if training_events:
        last_train = datetime.fromisoformat(training_events[-1]["timestamp"])
        days_since = (now - last_train).days
    else:
        last_train = None
        days_since = 999

    # Count corrections since last training
    corrections_events = log.read_events(
        event_type="correction_submitted",
        after=last_train,
    )
    n_corrections = len(corrections_events)

    # Average confidence of recent scores
    recent_scores = log.read_events(event_type="frontier_scored")
    if len(recent_scores) >= 10:
        recent_conf = [e["data"].get("confidence", 0) for e in recent_scores[-20:]]
        older_conf = [e["data"].get("confidence", 0) for e in recent_scores[-40:-20]]
        avg_recent = sum(recent_conf) / len(recent_conf) if recent_conf else 0
        avg_older = sum(older_conf) / len(older_conf) if older_conf else 0
        confidence_drop = avg_older - avg_recent
    else:
        avg_recent = 0
        avg_older = 0
        confidence_drop = 0

    # Average agreement rate
    comparison_events = log.read_events(event_type="comparison_generated")
    if comparison_events:
        agreements = [e["data"].get("agreement", 1.0) for e in comparison_events[-20:]]
        avg_agreement = sum(agreements) / len(agreements)
    else:
        avg_agreement = 1.0

    # Check triggers
    if n_corrections >= RETRAIN_TRIGGERS["min_new_corrections"]:
        reasons.append(
            f"{n_corrections} corrections since last training "
            f"(threshold: {RETRAIN_TRIGGERS['min_new_corrections']})"
        )

    if confidence_drop > RETRAIN_TRIGGERS["confidence_drop_threshold"]:
        reasons.append(
            f"Confidence dropped by {confidence_drop:.2f} "
            f"(threshold: {RETRAIN_TRIGGERS['confidence_drop_threshold']})"
        )

    if days_since > RETRAIN_TRIGGERS["max_days_since_training"]:
        reasons.append(
            f"{days_since} days since last training "
            f"(threshold: {RETRAIN_TRIGGERS['max_days_since_training']})"
        )

    if avg_agreement < RETRAIN_TRIGGERS["min_frontier_agreement"]:
        reasons.append(
            f"Frontier agreement at {avg_agreement:.2f} "
            f"(threshold: {RETRAIN_TRIGGERS['min_frontier_agreement']})"
        )

    stats = {
        "corrections_since_last_train": n_corrections,
        "avg_confidence_recent": round(avg_recent, 3),
        "avg_confidence_older": round(avg_older, 3),
        "confidence_drop": round(confidence_drop, 3),
        "days_since_last_training": days_since,
        "frontier_agreement_rate": round(avg_agreement, 3),
        "total_videos": store.get_stats()["total_videos"],
        "total_corrections": store.get_stats()["total_corrections"],
    }

    should_retrain = len(reasons) > 0

    log.append_event("drift_check", {
        "should_retrain": should_retrain,
        "reasons": reasons,
        **stats,
    })

    return {
        "should_retrain": should_retrain,
        "reasons": reasons,
        "stats": stats,
    }
