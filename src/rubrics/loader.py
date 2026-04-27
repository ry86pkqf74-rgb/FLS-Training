"""Task-aware rubric loading utilities for FLS scoring and reporting."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
RUBRICS_ROOT = REPO_ROOT / "rubrics"

TASK_RUBRIC_FILES = {
    "task1": "task1_peg_transfer.yaml",
    "task2": "task2_pattern_cut.yaml",
    "task3": "task3_ligating_loop.yaml",
    "task4": "task4_extracorporeal_suture.yaml",
    "task5": "task5_intracorporeal_suture.yaml",
    "task6": "task6_rings_needle_manipulation.yaml",
}

TASK_ID_ALIASES = {
    "1": "task1",
    "task1": "task1",
    "task1_peg_transfer": "task1",
    "peg_transfer": "task1",
    "2": "task2",
    "task2": "task2",
    "task2_pattern_cut": "task2",
    "pattern_cut": "task2",
    "precision_cutting": "task2",
    "3": "task3",
    "task3": "task3",
    "task3_ligating_loop": "task3",
    "task3_endoloop": "task3",
    "endoloop": "task3",
    "ligating_loop": "task3",
    "4": "task4",
    "task4": "task4",
    "task4_extracorporeal": "task4",
    "task4_extracorporeal_suture": "task4",
    "task4_extracorporeal_knot": "task4",
    "extracorporeal_suture": "task4",
    "extracorporeal_knot": "task4",
    "5": "task5",
    "task5": "task5",
    "task5_intracorporeal": "task5",
    "task5_intracorporeal_suture": "task5",
    "task5_intracorporeal_suturing": "task5",
    "intracorporeal_suture": "task5",
    "intracorporeal_suturing": "task5",
    "6": "task6",
    "task6": "task6",
    "task6_rings": "task6",
    "task6_rings_of_rings": "task6",
    "task6_rings_needle_manipulation": "task6",
    "rings_of_rings": "task6",
    "rings_needle_manipulation": "task6",
}

OFFICIAL_TASK_IDS = {"task1", "task2", "task3", "task4", "task5"}
TASK6_NOTE = (
    "Custom FLS-adjacent training task: Rings of Rings Needle Manipulation. "
    "Task 6 is a custom training task and is not part of official FLS certification scoring."
)


def canonical_task_id(task_id: str | int) -> str:
    """Return the short canonical task id, e.g. ``task5``."""
    normalized = str(task_id).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in TASK_ID_ALIASES:
        return TASK_ID_ALIASES[normalized]
    if normalized.startswith("task") and normalized[4:].isdigit():
        return normalized
    if normalized.isdigit():
        return f"task{normalized}"
    raise ValueError(f"Unknown FLS task id: {task_id}")


@lru_cache(maxsize=None)
def load_rubric(task_id: str) -> dict[str, Any]:
    """Load a rubric YAML by task id and attach v003 metadata."""
    canonical = canonical_task_id(task_id)
    rubric_file = TASK_RUBRIC_FILES.get(canonical)
    if not rubric_file:
        raise FileNotFoundError(f"No rubric configured for task: {task_id}")

    rubric_path = RUBRICS_ROOT / rubric_file
    rubric = yaml.safe_load(rubric_path.read_text()) or {}
    rubric["task_id"] = canonical
    rubric.setdefault("task", rubric.get("task"))
    rubric["task_name"] = rubric.get("name", canonical)
    rubric["official_fls_task"] = canonical in OFFICIAL_TASK_IDS
    rubric["custom_training_task"] = canonical == "task6"
    rubric["certification_eligible"] = canonical in OFFICIAL_TASK_IDS

    if canonical == "task6":
        rubric["canonical_task_id"] = "task6_rings_needle_manipulation"
        rubric["score_units"] = "training points"
        rubric["task6_note"] = TASK6_NOTE
        rubric["score_formula"] = (
            "315 - completion_time_seconds - 20 * failed_or_incomplete_ring_pairs, "
            "unless needle exits field of view or block is dislodged; those auto-fail to zero."
        )
        rubric["task6_scoring_rule"] = {
            "auto_fail": ["needle_out_of_view", "block_dislodged"],
            "ring_pair_penalty": 20,
            "minimum_attempt_rule": (
                "If the task is abandoned or too few rings are attempted to judge the attempt, "
                "mark incomplete and require human review."
            ),
        }

    return rubric


def get_task_max_score(task_id: str) -> float:
    return float(load_rubric(task_id)["max_score"])


def get_task_max_time(task_id: str) -> float:
    return float(load_rubric(task_id)["max_time_seconds"])


def get_task_name(task_id: str) -> str:
    return str(load_rubric(task_id).get("name") or load_rubric(task_id)["task_name"])


def get_task_phase_benchmarks(task_id: str) -> dict[str, dict[str, Any]]:
    rubric = load_rubric(task_id)
    return {phase["name"]: phase for phase in rubric.get("phases", []) if "name" in phase}


def get_task_penalty_definitions(task_id: str) -> dict[str, dict[str, Any]]:
    rubric = load_rubric(task_id)
    return {penalty["name"]: penalty for penalty in rubric.get("penalties", []) if "name" in penalty}


def is_official_fls_task(task_id: str) -> bool:
    return canonical_task_id(task_id) in OFFICIAL_TASK_IDS
