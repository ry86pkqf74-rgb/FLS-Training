#!/usr/bin/env python3
"""Run training on RunPod GPU pod.

Usage (on RunPod pod after git clone):
    pip install -e '.[training]'
    python scripts/050_runpod_train.py --ver v1

After training:
    git add models/ training/runs/
    git commit -m 'feat: student model v1 trained'
    git push
"""
import argparse
from src.training.runpod_trainer import TrainingConfig, run_full_training


def main():
    parser = argparse.ArgumentParser(description="Train FLS student model on RunPod")
    parser.add_argument("--ver", default="v1", help="Data version to train on")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--base-model", default="microsoft/Florence-2-large")
    args = parser.parse_args()

    config = TrainingConfig(
        base_model=args.base_model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        lora_r=args.lora_r,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        seed=args.seed,
        use_wandb=not args.no_wandb,
        scoring_train_path=f"training/data/scoring_train_{args.ver}.jsonl",
        scoring_val_path=f"training/data/scoring_val_{args.ver}.jsonl",
        coaching_train_path=f"training/data/coaching_train_{args.ver}.jsonl",
        coaching_val_path=f"training/data/coaching_val_{args.ver}.jsonl",
        output_dir=f"models/student_{args.ver}",
    )

    result = run_full_training(config)

    print("\n=== POST-TRAINING CHECKLIST ===")
    print("  git add models/ training/runs/")
    print("  git commit -m 'feat: student model trained'")
    print("  git push origin main")
    print("  # Then shut down RunPod pod to stop billing")


if __name__ == "__main__":
    main()
