from __future__ import annotations

from src.reporting.report_v3 import generate_report_v3
from src.reporting.validator import validate_report_v3
from src.rubrics.loader import load_rubric

from tests.report_v3_helpers import make_score


def test_sample_task5_report_prevents_contradictions() -> None:
    score = make_score(
        task_id="task5",
        time=142,
        penalties=[
            {"type": "gap_visible", "points_deducted": 11, "severity": "major"},
            {"type": "knot_failure", "points_deducted": 50, "severity": "major"},
        ],
        estimated_fls_score=397,
        critical_errors=["gap_visible", "knot_failure"],
    )

    report = generate_report_v3(score)
    markdown = report["markdown"].lower()

    assert "no significant penalties" not in markdown
    assert "demonstrated proficiency" not in markdown
    assert "knot" in markdown
    assert "slit" in markdown or "gap" in markdown
    assert "600 - 142 - 61 = 397" in report["markdown"]
    assert validate_report_v3(report, score, load_rubric("task5")) == []
