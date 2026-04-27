#!/usr/bin/env python3
"""Prepare chat-format LoRA examples from validated v003 report labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_MESSAGE = (
    "You generate rubric-faithful FLS training reports from structured scoring JSON. "
    "You do not invent observations."
)


def build_example(label: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": {
                    "score_json": label["score_json"],
                    "rubric_json": label["rubric_json"],
                    "resident_level": label.get("resident_level", "PGY3"),
                    "previous_attempts_summary": label.get("previous_attempts_summary", []),
                    "experimental_metrics": label.get("experimental_metrics", {}),
                },
            },
            {
                "role": "assistant",
                "content": {
                    "report_version": "v003",
                    "score_summary": label["report_json"].get("score_summary", {}),
                    "readiness_status": label["report_json"].get("readiness_status", {}),
                    "critical_findings": label["report_json"].get("critical_findings", []),
                    "strengths": label["report_json"].get("strengths", []),
                    "improvement_priorities": label["report_json"].get("improvement_priorities", []),
                    "next_practice_plan": label["report_json"].get("next_practice_plan", {}),
                    "markdown": label["report_json"].get("markdown", ""),
                },
            },
        ]
    }


def prepare_dataset(label_dir: Path, output_jsonl: Path) -> int:
    count = 0
    with output_jsonl.open("w") as output:
        for label_path in sorted(label_dir.glob("*.json")):
            label = json.loads(label_path.read_text())
            output.write(json.dumps(build_example(label)) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label_dir", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()
    count = prepare_dataset(args.label_dir, args.output_jsonl)
    print(f"Wrote {count} v003 LoRA example(s) to {args.output_jsonl}.")


if __name__ == "__main__":
    main()
