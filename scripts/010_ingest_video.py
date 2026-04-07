#!/usr/bin/env python3
"""010_ingest_video.py — Register a new FLS training video.

Usage:
    python scripts/010_ingest_video.py --video /path/to/video.mov --task 5
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.video_metadata import ingest_video
from src.ingest.frame_extractor import get_video_info
from src.memory.memory_store import MemoryStore
from src.memory.learning_log import LearningLog
from src.scoring.schema import FLSTask

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
console = Console()

TASK_MAP = {
    "1": FLSTask.PEG_TRANSFER,
    "2": FLSTask.PATTERN_CUT,
    "3": FLSTask.CLIP_APPLY,
    "4": FLSTask.EXTRACORPOREAL,
    "5": FLSTask.INTRACORPOREAL,
}


def main():
    parser = argparse.ArgumentParser(description="Ingest an FLS training video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--task", required=True, choices=["1", "2", "3", "4", "5"],
                        help="FLS task number (1-5)")
    parser.add_argument("--db", default="data/fls_training.duckdb", help="DuckDB path")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        console.print(f"[red]Video not found: {video_path}[/red]")
        sys.exit(1)

    task = TASK_MAP[args.task]
    store = MemoryStore(args.db)
    log = LearningLog()

    # Ingest
    meta = ingest_video(video_path, task)
    store.insert_video(meta)
    log.log_video_ingested(meta.id, meta.filename, task.value)

    # Display result
    info = get_video_info(video_path)
    table = Table(title=f"Video Ingested: {meta.id}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("ID", meta.id)
    table.add_row("Filename", meta.filename)
    table.add_row("Task", task.value)
    table.add_row("Duration", f"{meta.duration_seconds}s")
    table.add_row("Resolution", meta.resolution)
    table.add_row("FPS", str(meta.fps))
    table.add_row("Hash", meta.file_hash)
    table.add_row("Ingested At", meta.ingested_at.isoformat())
    console.print(table)

    console.print(f"\n[bold green]Next:[/bold green] python scripts/020_score_frontier.py --video-id {meta.id} --video {args.video}")

    store.close()


if __name__ == "__main__":
    main()
