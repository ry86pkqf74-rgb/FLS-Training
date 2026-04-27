from __future__ import annotations

from src.reporting.report_v3 import generate_report_v3
from src.scoring.frontier_scorer import recompute_score_from_components

from tests.report_v3_helpers import make_score


def test_report_formula_recomputes_task5_sample() -> None:
    score = make_score(task_id="task5", time=142, penalties=61)

    report = generate_report_v3(score)

    assert report["score_summary"]["training_score"] == 397
    assert report["score_summary"]["formula_applied"] == "600 - 142 - 61 = 397"
    assert "600 - 142 - 61 = 397" in report["markdown"]


def test_frontier_recompute_overwrites_model_math() -> None:
    payload = {
        "completion_time_seconds": 142,
        "estimated_fls_score": 599,
        "estimated_penalties": 0,
        "penalties": [{"type": "gap_visible", "points_deducted": 61, "severity": "major"}],
        "score_components": {"formula_applied": "incorrect"},
    }

    recomputed = recompute_score_from_components(payload, "task5")

    assert recomputed["estimated_fls_score"] == 397
    assert recomputed["estimated_penalties"] == 61
    assert recomputed["score_components"]["total_fls_score"] == 397
    assert recomputed["score_components"]["formula_applied"] == "600 - 142 - 61 = 397"


def test_auto_fail_recompute_forces_zero() -> None:
    payload = {
        "completion_time_seconds": 80,
        "penalties": [{"type": "drain_avulsion", "points_deducted": 0, "severity": "auto_fail"}],
    }

    recomputed = recompute_score_from_components(payload, "task5")

    assert recomputed["estimated_fls_score"] == 0
    assert recomputed["score_components"]["formula_applied"] == "automatic zero due to auto-fail penalty"
