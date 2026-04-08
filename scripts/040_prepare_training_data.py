#!/usr/bin/env python3
"""Prepare training data for fine-tuning on RunPod or unified JSONL export."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.memory.learning_log import LearningLog
from src.memory.memory_store import MemoryStore
from src.training.prepare_dataset import prepare_dataset
from src.training.data_prep import prepare_training_data


def _coerce_version_tag(ver: str) -> int:
    digits = "".join(ch for ch in str(ver) if ch.isdigit())
    return int(digits) if digits else 1


def main():
    parser = argparse.ArgumentParser(description="Prepare training datasets")
    parser.add_argument("--ver", default="v1", help="Dataset version tag")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--include-coach-feedback", action="store_true")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override")
    args = parser.parse_args()

    if args.include_coach_feedback:
        store = MemoryStore(args.base_dir)
        log = LearningLog(Path(args.base_dir) / "memory")
        output_dir = args.output_dir or "data/training"
        meta = prepare_dataset(
            store=store,
            log=log,
            video_dir=".",
            output_dir=output_dir,
            version=_coerce_version_tag(args.ver),
            min_confidence=args.min_confidence,
            train_split=1.0 - (args.val_split + 0.05),
            val_split=args.val_split,
            seed=args.seed,
            include_coach_feedback=True,
        )
        print(f"\nUnified dataset ready in {output_dir}")
        return

    meta = prepare_training_data(
        base_dir=args.base_dir,
        output_dir=args.output_dir or "training/data",
        version=args.ver,
        val_split=args.val_split,
        min_confidence=args.min_confidence,
        seed=args.seed,
    )

    print(f"\nReady for RunPod. Next steps:")
    print(f"  1. git add training/data/")
    print(f"  2. git commit -m 'feat: training data {args.ver}'")
    print(f"  3. git push")
    print(f"  4. On RunPod: python scripts/050_runpod_train.py --ver {args.ver}")


if __name__ == "__main__":
    main()
