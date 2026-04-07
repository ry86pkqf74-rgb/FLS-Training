#!/usr/bin/env python3
"""040_prepare_training_data.py — Build training dataset from scored videos.

Usage:
    python scripts/040_prepare_training_data.py --version 1
    python scripts/040_prepare_training_data.py --version 2 --min-confidence 0.8
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.learning_log import LearningLog
from src.memory.memory_store import MemoryStore
from src.training.prepare_dataset import prepare_dataset

load_dotenv()
logging.basicConfig(level=logging.INFO)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument("--version", type=int, required=True, help="Dataset version number")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--include-coach", action="store_true",
                        help="Include coach feedback in training examples (Phase 2)")
    parser.add_argument("--video-dir", default="./videos")
    parser.add_argument("--output-dir", default="./data/training")
    parser.add_argument("--db", default="data/fls_training.duckdb")
    args = parser.parse_args()

    store = MemoryStore(args.db)
    log = LearningLog()

    console.print(f"[bold]Preparing training dataset v{args.version}...[/bold]")
    manifest = prepare_dataset(
        store, log,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        version=args.version,
        min_confidence=args.min_confidence,
        include_coach_feedback=args.include_coach,
    )

    if "error" in manifest:
        console.print(f"[red]Error: {manifest['error']}[/red]")
        sys.exit(1)

    console.print(f"[green]Dataset v{args.version} created:[/green]")
    console.print(f"  Train: {manifest['n_train']} examples")
    console.print(f"  Val:   {manifest['n_val']} examples")
    console.print(f"  Test:  {manifest['n_test']} examples")
    console.print(f"  Sources: {manifest['sources']}")

    store.close()


if __name__ == "__main__":
    main()
