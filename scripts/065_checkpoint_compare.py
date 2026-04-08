#!/usr/bin/env python3
"""Quick side-by-side score comparison for 2+ checkpoints on five held-out videos."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.training.eval_v2 import evaluate_checkpoint, resolve_latest_test_split

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare checkpoints on a shared 5-video slice")
    parser.add_argument("checkpoints", nargs="+", help="Two or more checkpoint paths")
    parser.add_argument("--data", help="Held-out JSONL file; defaults to latest test split")
    parser.add_argument("--base-dir", default=".", help="Repository root")
    parser.add_argument("--max-videos", type=int, default=5, help="Number of held-out videos to compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("Provide at least two checkpoints to compare")

    data_path = Path(args.data) if args.data else resolve_latest_test_split(args.base_dir)
    results = []
    for checkpoint in args.checkpoints:
        results.append(
            evaluate_checkpoint(
                checkpoint_path=checkpoint,
                data_path=data_path,
                base_dir=args.base_dir,
                max_examples=args.max_videos,
                qualitative_samples=0,
            )
        )

    records_by_video = {}
    for result in results:
        checkpoint_name = Path(result["checkpoint"]).name
        for record in result["records"]:
            records_by_video.setdefault(record["video_id"], {})[checkpoint_name] = record

    table = Table(title=f"Checkpoint comparison on {data_path}")
    table.add_column("Video")
    table.add_column("Consensus", justify="right")
    table.add_column("Claude", justify="right")
    table.add_column("GPT-4o", justify="right")
    for checkpoint in args.checkpoints:
        table.add_column(Path(checkpoint).name, justify="right")

    for video_id in sorted(records_by_video):
        any_record = next(iter(records_by_video[video_id].values()))
        target = any_record.get("target_output") or {}
        teacher_scores = any_record.get("teacher_scores") or {}
        consensus_score = float(target.get("estimated_fls_score") or 0.0)
        claude_score = float((teacher_scores.get("teacher_claude") or {}).get("estimated_fls_score") or 0.0)
        gpt_score = float((teacher_scores.get("teacher_gpt4o") or {}).get("estimated_fls_score") or 0.0)
        baseline_error = min(
            abs(claude_score - consensus_score) if claude_score else float("inf"),
            abs(gpt_score - consensus_score) if gpt_score else float("inf"),
        )
        row = [video_id, f"{consensus_score:.1f}", f"{claude_score:.1f}", f"{gpt_score:.1f}"]

        for checkpoint in args.checkpoints:
            checkpoint_name = Path(checkpoint).name
            record = records_by_video[video_id].get(checkpoint_name)
            predicted = float(((record or {}).get("parsed_output") or {}).get("estimated_fls_score") or 0.0)
            error = abs(predicted - consensus_score)
            regression = " !" if baseline_error != float("inf") and error > baseline_error else ""
            row.append(f"{predicted:.1f}{regression}")
        table.add_row(*row)

    console.print(table)
    console.print("Rows marked with ! are worse than the best teacher baseline on that video.")


if __name__ == "__main__":
    main()