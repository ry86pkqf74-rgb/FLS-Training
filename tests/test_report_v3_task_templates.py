from __future__ import annotations

from src.reporting.task_templates import TASK_REPORT_TEMPLATES


def test_all_tasks_have_required_template_sections() -> None:
    expected_tasks = {f"task{i}" for i in range(1, 7)}
    assert set(TASK_REPORT_TEMPLATES) == expected_tasks

    for template in TASK_REPORT_TEMPLATES.values():
        assert template["strength_signals"]
        assert template["weakness_signals"]
        assert template["critical_errors"]
        assert template["recommended_drills"]
        assert template["phase_focus_rules"]
        assert template["plain_language_summary_rules"]


def test_task6_template_preserves_custom_rings_task() -> None:
    task6 = TASK_REPORT_TEMPLATES["task6"]

    assert any("needle visibility" in drill.lower() for drill in task6["recommended_drills"])
    assert any("ring" in signal.lower() for signal in task6["weakness_signals"])
    assert any("block dislodged" in error.lower() for error in task6["critical_errors"])
