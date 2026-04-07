#!/usr/bin/env python3
"""Check for scoring drift between student and teacher models.

Run periodically. If drift exceeds threshold, triggers retraining.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from rich.console import Console
from src.memory.memory_store import MemoryStore

console = Console()

DRIFT_THRESHOLD_TIME = 15.0   # seconds
DRIFT_THRESHOLD_FLS = 25.0    # points
MIN_SAMPLES = 5               # minimum videos to evaluate drift


def main():
    parser = argparse.ArgumentParser(description="Check scoring drift")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--window", type=int, default=10, help="Number of recent videos to check")
    args = parser.parse_args()

    store = MemoryStore(args.base_dir)
    all_scores = store.get_all_scores()

    # Find videos scored by both student and teacher
    by_video = {}
    for s in all_scores:
        by_video.setdefault(s.video_id, {})[s.source] = s

    dual_scored = {
        vid: scores for vid, scores in by_video.items()
        if "student" in scores and ("consensus" in scores or "teacher_claude" in scores)
    }

    if len(dual_scored) < MIN_SAMPLES:
        console.print(f"[yellow]Only {len(dual_scored)} dual-scored videos (need {MIN_SAMPLES}). No drift check possible.[/yellow]")
        return

    # Check recent window
    recent = sorted(dual_scored.keys())[-args.window:]
    time_drifts = []
    fls_drifts = []

    for vid in recent:
        scores = dual_scored[vid]
        student = scores["student"]
        teacher = scores.get("consensus") or scores.get("teacher_claude")
        time_drifts.append(abs(student.completion_time_seconds - teacher.completion_time_seconds))
        fls_drifts.append(abs(student.estimated_fls_score - teacher.estimated_fls_score))

    avg_time_drift = sum(time_drifts) / len(time_drifts)
    avg_fls_drift = sum(fls_drifts) / len(fls_drifts)

    console.print(f"\n[bold]Drift Report[/bold] (last {len(recent)} videos)")
    console.print(f"  Avg time drift:  {avg_time_drift:.1f}s (threshold: {DRIFT_THRESHOLD_TIME}s)")
    console.print(f"  Avg FLS drift:   {avg_fls_drift:.1f} (threshold: {DRIFT_THRESHOLD_FLS})")

    needs_retrain = avg_time_drift > DRIFT_THRESHOLD_TIME or avg_fls_drift > DRIFT_THRESHOLD_FLS

    if needs_retrain:
        console.print("\n[bold red]⚠ DRIFT DETECTED — retraining recommended[/bold red]")
        console.print("  Run: python scripts/040_prepare_training_data.py --ver v2")
        console.print("  Then: train on RunPod with updated data")
    else:
        console.print("\n[bold green]✓ No significant drift[/bold green]")

    # Log result
    store._append_ledger("drift_check", {
        "window": len(recent),
        "avg_time_drift": round(avg_time_drift, 1),
        "avg_fls_drift": round(avg_fls_drift, 1),
        "needs_retrain": needs_retrain,
        "checked_at": datetime.utcnow().isoformat(),
    })


if __name__ == "__main__":
    main()
