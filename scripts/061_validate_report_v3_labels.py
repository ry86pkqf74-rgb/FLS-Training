#!/usr/bin/env python3
"""Validate generated v003 report labels before training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.reporting.validator import validate_report_v3
from src.rubrics.loader import load_rubric
from src.scoring.schema import ScoringResult


def validate_labels(label_dir: Path) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    for label_path in sorted(label_dir.glob("*.json")):
        label = json.loads(label_path.read_text())
        score = ScoringResult(**label["score_json"])
        rubric = label.get("rubric_json") or load_rubric(score.task_id)
        report = label["report_json"]
        errors = validate_report_v3(report, score, rubric)
        markdown = report.get("markdown", "").lower()
        if score.estimated_penalties > 0 and "no significant penalties" in markdown:
            errors.append("says no penalties despite penalties > 0")
        if "above average" in markdown or "global z-score" in markdown:
            errors.append("z-score appears in primary assessment")
        if errors:
            failures[str(label_path)] = errors
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label_dir", type=Path)
    args = parser.parse_args()
    failures = validate_labels(args.label_dir)
    if failures:
        print(json.dumps(failures, indent=2))
        raise SystemExit(1)
    print("All v003 labels passed validation.")


if __name__ == "__main__":
    main()
