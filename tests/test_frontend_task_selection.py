from __future__ import annotations

from pathlib import Path


DEMO_V5 = Path("demo/fls_demo_v5.py")
DEMO_V6 = Path("demo/fls_demo_v6.py")


def test_v5_task_max_scores_are_corrected() -> None:
    source = DEMO_V5.read_text()

    assert '"task3_endoloop": 180' in source
    assert '"task4_extracorporeal_knot": 420' in source


def test_v6_requires_explicit_task_selection_and_uses_training_score_language() -> None:
    source = DEMO_V6.read_text()

    assert "TASK_METADATA" in source
    assert "task_dropdown = gr.Dropdown" in source
    assert "value=None" in source
    assert "if not task_id" in source
    assert "Training score" in source
    assert "official certification score" in source
    assert "Experimental AI-derived metrics" in source
    assert "Major technical issue detected" in source
    assert "Needs human review" in source
    assert "Custom training task, not official FLS manual skills task" in source
    assert "determine_readiness" in source
    assert "pct >= 75" not in source
    assert "proficient" not in source.lower()
