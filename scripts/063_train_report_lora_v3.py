#!/usr/bin/env python3
"""Training entrypoint placeholder for report LoRA v3.

Do not run this until generated v003 labels pass validation and a human reviews
the dataset. The script intentionally stops by default to prevent retraining on
old contradictory report style.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_jsonl", type=Path)
    parser.add_argument("--human-reviewed", action="store_true")
    args = parser.parse_args()

    if not args.human_reviewed:
        raise SystemExit(
            "Refusing to train report LoRA v3 until labels pass validation and human review. "
            "Re-run with --human-reviewed after approval."
        )

    raise SystemExit(
        "Training implementation is intentionally deferred. Wire this to the approved "
        "LoRA trainer only after human review of v003 labels."
    )


if __name__ == "__main__":
    main()
