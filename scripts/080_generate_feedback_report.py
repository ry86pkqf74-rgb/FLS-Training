#!/usr/bin/env python3
"""080_generate_feedback_report.py — Generate a feedback report for a scored video.

Usage:
    python scripts/080_generate_feedback_report.py --video-id abc123
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback.feedback_generator import generate_feedback, feedback_to_markdown
from src.memory.memory_store import MemoryStore
from src.scoring.schema import ScoringResult

load_dotenv()
logging.basicConfig(level=logging.INFO)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Generate feedback report")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", help="Output markdown file path")
    parser.add_argument("--db", default="data/fls_training.duckdb")
    args = parser.parse_args()

    store = MemoryStore(args.db)

    # Get best score (prefer consensus, then corrected, then highest confidence)
    scores = store.get_scores_for_video(args.video_id)
    if not scores:
        console.print(f"[red]No scores found for video {args.video_id}[/red]")
        sys.exit(1)

    # Pick best source
    best = None
    for s in scores:
        if s["source"] == "critique_consensus":
            best = s
            break
    if not best:
        best = max(scores, key=lambda s: s.get("confidence_score", 0))

    # Reconstruct ScoringResult from raw_json
    raw = best.get("raw_json")
    if isinstance(raw, str):
        raw = json.loads(raw)
    score = ScoringResult.model_validate(raw)

    # Generate report
    report = generate_feedback(score)
    md = feedback_to_markdown(report)

    if args.output:
        Path(args.output).write_text(md)
        console.print(f"[green]Report saved to {args.output}[/green]")
    else:
        console.print(Markdown(md))

    store.close()


if __name__ == "__main__":
    main()
