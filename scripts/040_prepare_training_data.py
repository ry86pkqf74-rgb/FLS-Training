#!/usr/bin/env python3
"""Prepare training data for fine-tuning on RunPod."""
import argparse
from src.training.data_prep import prepare_training_data


def main():
    parser = argparse.ArgumentParser(description="Prepare training datasets")
    parser.add_argument("--ver", default="v1", help="Dataset version tag")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    meta = prepare_training_data(
        base_dir=args.base_dir,
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
