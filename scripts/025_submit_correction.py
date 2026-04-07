#!/usr/bin/env python3
"""025_submit_correction.py — Record an expert correction to a model score.

Usage:
    python scripts/025_submit_correction.py \
        --video-id abc123 \
        --score-id def456 \
        --corrected-fields '{"knot_secure": true, "completion_time_seconds": 147}' \
        --corrector expert \
        --notes "Knot was secure on manual inspection"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.learning_log import LearningLog
from src.memory.memory_store import MemoryStore
from src.scoring.schema import Correction, CorrectorRole

load_dotenv()
logging.basicConfig(level=logging.INFO)
console = Console()


def main():
    parser = argparse.ArgumentParser(description="Submit an expert correction")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--score-id", required=True, help="ID of the score being corrected")
    parser.add_argument("--corrected-fields", required=True,
                        help="JSON string of corrected field:value pairs")
    parser.add_argument("--corrector", default="expert",
                        choices=["expert", "resident", "self"])
    parser.add_argument("--notes", default="")
    parser.add_argument("--db", default="data/fls_training.duckdb")
    args = parser.parse_args()

    try:
        fields = json.loads(args.corrected_fields)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON in --corrected-fields: {e}[/red]")
        sys.exit(1)

    store = MemoryStore(args.db)
    log = LearningLog()

    correction = Correction(
        video_id=args.video_id,
        original_score_id=args.score_id,
        corrector_role=CorrectorRole(args.corrector),
        corrected_fields=fields,
        notes=args.notes,
    )

    store.insert_correction(correction)
    path = log.save_correction(correction)

    console.print(f"[green]Correction {correction.id} saved[/green]")
    console.print(f"  Video: {args.video_id}")
    console.print(f"  Original score: {args.score_id}")
    console.print(f"  Fields corrected: {list(fields.keys())}")
    console.print(f"  File: {path}")

    store.close()


if __name__ == "__main__":
    main()
