#!/usr/bin/env python3
"""090_status.py — Show overall system status.

Usage:
    python scripts/090_status.py
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.learning_log import LearningLog
from src.memory.memory_store import MemoryStore

load_dotenv()
logging.basicConfig(level=logging.WARNING)
console = Console()


def main():
    store = MemoryStore()
    log = LearningLog()

    stats = store.get_stats()
    summary = log.summarize()

    table = Table(title="FLS-Training Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Videos ingested", str(stats["total_videos"]))
    table.add_row("Total scores", str(stats["total_scores"]))
    table.add_row("Critiques (consensus)", str(stats["total_critiques"]))
    table.add_row("Expert corrections", str(stats["total_corrections"]))
    table.add_row("Training runs", str(stats["total_training_runs"]))
    table.add_row("Total API cost", f"${stats['total_api_cost']:.4f}")
    table.add_row("Ledger events", str(summary["total_events"]))
    console.print(table)

    if summary.get("events_by_type"):
        ev_table = Table(title="Events by Type")
        ev_table.add_column("Event", style="cyan")
        ev_table.add_column("Count", style="green")
        for k, v in sorted(summary["events_by_type"].items()):
            ev_table.add_row(k, str(v))
        console.print(ev_table)

    # Show recent videos
    videos = store.list_videos()
    if videos:
        v_table = Table(title=f"Recent Videos (showing {min(10, len(videos))} of {len(videos)})")
        v_table.add_column("ID", style="cyan")
        v_table.add_column("Filename")
        v_table.add_column("Task")
        v_table.add_column("Duration")
        v_table.add_column("Ingested")
        for v in videos[:10]:
            v_table.add_row(
                v["id"], v["filename"], v["task"],
                f"{v['duration_seconds']:.1f}s", str(v["ingested_at"])
            )
        console.print(v_table)

    store.close()


if __name__ == "__main__":
    main()
