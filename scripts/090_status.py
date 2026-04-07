#!/usr/bin/env python3
"""Show current system status."""
import argparse
from rich.console import Console
from rich.table import Table
from src.memory.memory_store import MemoryStore

console = Console()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    store = MemoryStore(args.base_dir)
    stats = store.get_stats()

    table = Table(title="FLS Training System Status")
    table.add_column("Component", style="bold")
    table.add_column("Count", justify="right")

    table.add_row("Scored videos", str(stats["total_scores"]))
    table.add_row("Feedback reports", str(stats["total_feedback"]))
    table.add_row("Corrections", str(stats["total_corrections"]))
    table.add_row("Ledger entries", str(stats["ledger_entries"]))
    table.add_row("Trainee profile", "✓" if stats["profile_exists"] else "✗")

    console.print(table)

    # Show trainee profile summary if exists
    profile = store.get_trainee_profile()
    if profile.total_attempts > 0:
        console.print(f"\n[bold]Trainee Summary:[/bold]")
        console.print(f"  Attempts: {profile.total_attempts}")
        console.print(f"  Best time: {profile.best_time_seconds:.0f}s")
        console.print(f"  Best FLS: {profile.best_fls_score:.0f}")
        console.print(f"  Baseline: {profile.baseline_time:.0f}s")
        console.print(f"  Current plateau: ~{profile.current_plateau_time:.0f}s")
        if profile.bottleneck_phase:
            console.print(f"  Bottleneck: {profile.bottleneck_phase}")


if __name__ == "__main__":
    main()
