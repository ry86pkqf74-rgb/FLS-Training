from __future__ import annotations

from src.reporting.report_v3 import generate_report_v3

from tests.report_v3_helpers import make_score


def test_no_proficiency_claim_with_knot_failure() -> None:
    score = make_score(
        task_id="task5",
        time=142,
        penalties=[{"type": "knot_failure", "points_deducted": 50, "severity": "major"}],
        estimated_fls_score=408,
    )

    report = generate_report_v3(score)

    assert report["readiness_status"]["proficiency_claim_allowed"] is False
    assert "proficient" not in report["markdown"].lower()


def test_critical_errors_outrank_speed_in_priorities() -> None:
    score = make_score(
        task_id="task5",
        time=120,
        penalties=0,
        critical_errors=["gap_visible", "knot_failure"],
    )

    report = generate_report_v3(score)

    assert report["improvement_priorities"][0]["topic"] in ["knot_security", "slit_closure"]
    assert "correctness before speed" in report["markdown"].lower()


def test_unusable_video_is_unscorable_and_hides_score() -> None:
    score = make_score(
        task_id="task5",
        time=0,
        penalties=0,
        video_classification="unusable",
    )

    report = generate_report_v3(score)

    assert report["readiness_status"]["label"] == "unscorable"
    assert report["score_summary"]["training_score"] is None
    assert "training score:" not in report["markdown"].lower()
