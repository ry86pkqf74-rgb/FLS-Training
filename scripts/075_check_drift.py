#!/usr/bin/env python3
"""075_check_drift.py — Check if model retraining is needed.

Usage:
    python scripts/075_check_drift.py
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.drift_detector import check_retrain_needed
from src.memory.learning_log import LearningLog
from src.memory.memory_store import MemoryStore

load_dotenv()
logging.basicConfig(level=logging.INFO)
console = Console()


def main():
    store = MemoryStore()
    log = LearningLog()

    result = check_retrain_needed(store, log)

    if result["should_retrain"]:
        console.print("[bold red]RETRAIN RECOMMENDED[/bold red]")
        for reason in result["reasons"]:
            console.print(f"  • {reason}")
    else:
        console.print("[bold green]No retraining needed[/bold green]")

    console.print("\n[bold]Stats:[/bold]")
    for k, v in result["stats"].items():
        console.print(f"  {k}: {v}")

    store.close()


if __name__ == "__main__":
    main()
