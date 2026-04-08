#!/usr/bin/env python3
"""Evaluate one or more checkpoints on held-out prompts with richer metrics.

This goes beyond validation loss by measuring score fidelity, phase detection,
penalty recognition, and qualitative coaching behavior on the same held-out set.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.training.eval_v2 import evaluate_checkpoint, resolve_latest_test_split

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one or more FLS checkpoints")
    parser.add_argument("checkpoints", nargs="+", help="One or more checkpoint paths")
    parser.add_argument("--data", help="Held-out JSONL file; defaults to latest data/training/*/test.jsonl")
    parser.add_argument("--base-dir", default=".", help="Repository root")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit evaluation set size")
    parser.add_argument(
        "--output",
        default="validation_results_v2.jsonl",
        help="JSONL file to append checkpoint summaries to",
    )
    parser.add_argument(
        "--qualitative-dir",
        default="memory/prompt_evals",
        help="Directory for saved qualitative outputs",
    )
    parser.add_argument(
        "--qualitative-samples",
        type=int,
        default=5,
        help="Number of coaching outputs to save per checkpoint",
    )
    return parser.parse_args()


def _format_metric(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data) if args.data else resolve_latest_test_split(args.base_dir)
    output_path = Path(args.base_dir) / args.output

    table = Table(title=f"Checkpoint Evaluation on {data_path}")
    table.add_column("Checkpoint", style="bold")
    table.add_column("Examples", justify="right")
    table.add_column("Parse")
    table.add_column("MAE(cons)", justify="right")
    table.add_column("MAE(claude)", justify="right")
    table.add_column("MAE(gpt)", justify="right")
    table.add_column("r(claude)", justify="right")
    table.add_column("r(gpt)", justify="right")
    table.add_column("Phase Acc", justify="right")
    table.add_column("Penalty F1", justify="right")

    for checkpoint in args.checkpoints:
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint,
            data_path=data_path,
            base_dir=args.base_dir,
            max_examples=args.max_examples,
            qualitative_dir=Path(args.base_dir) / args.qualitative_dir,
            qualitative_samples=args.qualitative_samples,
        )
        metrics = result["metrics"]
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "data_path": str(data_path),
            "qualitative_path": result["qualitative_path"],
            "metrics": metrics,
        }
        with open(output_path, "a") as handle:
            handle.write(json.dumps(row) + "\n")

        table.add_row(
            Path(checkpoint).name,
            str(metrics.get("examples")),
            _format_metric(metrics.get("parse_rate")),
            _format_metric(metrics.get("score_mae_consensus")),
            _format_metric(metrics.get("score_mae_claude")),
            _format_metric(metrics.get("score_mae_gpt4o")),
            _format_metric(metrics.get("pearson_r_claude")),
            _format_metric(metrics.get("pearson_r_gpt4o")),
            _format_metric(metrics.get("phase_accuracy")),
            _format_metric(metrics.get("penalty_f1")),
        )

    console.print(table)
    console.print(f"Results appended to {output_path}")


if __name__ == "__main__":
    main()