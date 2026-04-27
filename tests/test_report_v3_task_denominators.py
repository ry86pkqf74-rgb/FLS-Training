from __future__ import annotations

from src.reporting.report_v3 import generate_report_v3
from src.rubrics.loader import get_task_max_score, is_official_fls_task

from tests.report_v3_helpers import make_score


def test_rubric_loader_uses_task_specific_denominators() -> None:
    assert get_task_max_score("task1") == 300
    assert get_task_max_score("task2") == 300
    assert get_task_max_score("task3") == 180
    assert get_task_max_score("task4") == 420
    assert get_task_max_score("task5") == 600
    assert get_task_max_score("task6") == 315


def test_reports_use_task_specific_denominators_and_task6_label() -> None:
    assert generate_report_v3(make_score(task_id="task1"))["score_summary"]["max_score"] == 300
    assert generate_report_v3(make_score(task_id="task5"))["score_summary"]["max_score"] == 600

    task6_report = generate_report_v3(make_score(task_id="task6", time=100, penalties=20))
    assert task6_report["score_summary"]["max_score"] == 315
    assert task6_report["task"]["official_fls_task"] is False
    assert task6_report["task"]["custom_training_task"] is True
    assert "not part of official FLS certification scoring" in task6_report["markdown"]
    assert is_official_fls_task("task6") is False
