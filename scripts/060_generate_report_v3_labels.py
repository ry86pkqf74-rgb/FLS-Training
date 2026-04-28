#!/usr/bin/env python3
"""Generate v003 structured report labels from existing score JSON records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.reporting.report_v3 import generate_report_v3
from src.rubrics.loader import canonical_task_id, load_rubric
from src.scoring.frontier_scorer import recompute_score_from_components
from src.scoring.schema import ScoringResult


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _normalize_score(payload: dict[str, Any], fallback_task_id: str | None = None) -> ScoringResult:
    task_id = canonical_task_id(payload.get("task_id") or fallback_task_id or "task5")
    recompute_score_from_components(payload, task_id)
    payload.setdefault("id", f"label_{task_id}")
    payload.setdefault("video_id", payload["id"])
    payload.setdefault("video_filename", f"{payload['video_id']}.mp4")
    payload.setdefault("source", "label_generation")
    payload.setdefault("model_name", "existing_score_record")
    payload.setdefault("model_version", "v003_label")
    payload.setdefault("prompt_version", "v003")
    payload.setdefault("completion_time_seconds", 0.0)
    payload.setdefault("confidence_score", 0.5)
    return ScoringResult(**payload)


def generate_labels(input_dir: Path, output_dir: Path, fallback_task_id: str | None = None) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for score_path in sorted(input_dir.glob("*.json")):
        payload = _load_json(score_path)
        score = _normalize_score(payload, fallback_task_id)
        rubric = load_rubric(score.task_id)
        report = generate_report_v3(score)
        label = {
            "score_json": score.model_dump(mode="json"),
            "rubric_json": rubric,
            "report_json": report,
        }
        (output_dir / f"{score_path.stem}.report_v3.json").write_text(json.dumps(label, indent=2))
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task-id", default=None)
    args = parser.parse_args()
    count = generate_labels(args.input_dir, args.output_dir, args.task_id)
    print(f"Generated {count} v003 report label(s).")


if __name__ == "__main__":
    main()
