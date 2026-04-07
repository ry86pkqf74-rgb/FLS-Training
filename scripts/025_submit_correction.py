#!/usr/bin/env python3
"""Submit expert corrections to a score."""
import argparse
import json
from datetime import datetime

from src.scoring.schema import CorrectionRecord
from src.memory.memory_store import MemoryStore


def main():
    parser = argparse.ArgumentParser(description="Submit correction to a score")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--score-id", required=True)
    parser.add_argument("--corrected-fields", required=True, help="JSON string of corrected fields")
    parser.add_argument("--corrector", default="expert")
    parser.add_argument("--notes", default="")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    store = MemoryStore(args.base_dir)
    fields = json.loads(args.corrected_fields)

    correction = CorrectionRecord(
        correction_id=f"corr_{args.video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        video_id=args.video_id,
        score_id=args.score_id,
        corrected_fields=fields,
        corrector=args.corrector,
        notes=args.notes,
    )

    path = store.save_correction(correction)
    print(f"Correction saved: {path}")
    print(f"  Fields corrected: {list(fields.keys())}")


if __name__ == "__main__":
    main()
