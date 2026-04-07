#!/usr/bin/env python3
"""Evaluate student model against teacher consensus scores."""
import argparse
import json
from rich.console import Console
from rich.table import Table
from src.training.evaluator import evaluate_student

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Evaluate student model")
    parser.add_argument("--student-scores", required=True, help="Dir with student score JSONs")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    result = evaluate_student(args.student_scores, args.base_dir)

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return

    table = Table(title="Student vs Teacher Evaluation")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Videos evaluated", str(result["videos_evaluated"]))
    table.add_row("Avg time error", f"{result['avg_time_error_seconds']}s")
    table.add_row("Avg FLS score error", f"{result['avg_fls_score_error']}")
    table.add_row("Avg phase error", f"{result['avg_phase_error_seconds']}s")
    table.add_row("Time agreement (≤10s)", f"{result['time_agreement_pct']}%")
    table.add_row("FLS agreement (≤20pts)", f"{result['fls_agreement_pct']}%")

    status = "[bold green]READY[/bold green]" if result["ready_for_promotion"] else "[bold red]NOT READY[/bold red]"
    table.add_row("Promotion status", status)

    console.print(table)

    # Per-video breakdown
    if result.get("per_video"):
        detail = Table(title="Per-Video Detail")
        detail.add_column("Video")
        detail.add_column("Time Err", justify="right")
        detail.add_column("FLS Err", justify="right")
        for vid, data in result["per_video"].items():
            detail.add_row(vid, f"{data['time_error']}s", f"{data['fls_error']}")
        console.print(detail)


if __name__ == "__main__":
    main()
