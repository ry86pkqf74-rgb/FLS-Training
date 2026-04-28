from __future__ import annotations

from src.reporting.report_v3 import generate_report_v3

from tests.report_v3_helpers import make_score


def test_z_scores_are_experimental_and_not_primary_praise() -> None:
    score = make_score(task_id="task5")
    score.task_specific_assessments["experimental_metrics"] = {
        "bimanual_dexterity": 0.71932,
        "efficiency": -0.2,
    }

    report = generate_report_v3(score, include_experimental_metrics=True)
    markdown = report["markdown"].lower()

    assert report["experimental_metrics"]["displayed"] is True
    assert "not part of official fls scoring" in markdown
    assert "above average" not in markdown
    assert "global z-score" not in markdown
    assert "higher relative model-derived signal" in report["markdown"]
