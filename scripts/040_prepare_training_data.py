#!/usr/bin/env python3
"""Prepare training data for fine-tuning Qwen2.5-VL-7B-Instruct.

2026-04-08: now the only supported training-data builder. The legacy
``src/training/data_prep.py`` text-flattening path was quarantined in
`archive/deprecated/` during the hardening sprint.

Typical use (vision mode, v4):

    python scripts/040_prepare_training_data.py \\
        --ver v4 \\
        --frames-dir data/frames \\
        --max-frames 24 \\
        --group-by trainee \\
        --min-confidence 0.5 \\
        --include-coach-feedback

Text-only mode (legacy v3 iteration):

    python scripts/040_prepare_training_data.py --ver v3 --include-coach-feedback
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.memory.learning_log import LearningLog
from src.memory.memory_store import MemoryStore
from src.training.prepare_dataset import prepare_dataset


def _coerce_version_tag(ver: str) -> int:
    digits = "".join(ch for ch in str(ver) if ch.isdigit())
    return int(digits) if digits else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare FLS training datasets")
    parser.add_argument(
        "--ver",
        required=True,
        help="Dataset version tag, e.g. v4. REQUIRED — no silent default.",
    )
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--train-split", type=float, default=0.80)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Drop scores below this confidence. Default 0.5 (was 0.3 in v3).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--output-dir", default="data/training")
    parser.add_argument(
        "--include-coach-feedback",
        action="store_true",
        help="Attach coach feedback as a combined scoring+coaching target.",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help=(
            "Root of extracted frames. Expected layout: "
            "<frames-dir>/<video_id>/*.jpg. If omitted the builder emits "
            "text-only prompts (legacy v3 behaviour)."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=24,
        help="Max image blocks per training example. Ignored if --frames-dir unset.",
    )
    parser.add_argument(
        "--group-by",
        choices=["trainee", "video"],
        default="trainee",
        help="Hold-out strategy. 'trainee' is resident-aware, 'video' is legacy.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    base = Path(args.base_dir)
    store = MemoryStore(str(base))
    log = LearningLog(base / "memory")

    manifest = prepare_dataset(
        store=store,
        log=log,
        video_dir=str(base),
        output_dir=args.output_dir,
        version=_coerce_version_tag(args.ver),
        min_confidence=args.min_confidence,
        train_split=args.train_split,
        val_split=args.val_split,
        seed=args.seed,
        include_coach_feedback=args.include_coach_feedback,
        frames_dir=args.frames_dir,
        max_frames_per_sample=args.max_frames,
        group_by=args.group_by,
    )

    print()
    print(f"=== Dataset {args.ver} ready ===")
    print(f"  Total examples : {manifest.get('total_examples', 0)}")
    print(f"  Train/val/test : {manifest.get('n_train', 0)} / "
          f"{manifest.get('n_val', 0)} / {manifest.get('n_test', 0)}")
    print(f"  Vision mode    : {manifest.get('vision_mode', False)}")
    if manifest.get("vision_mode"):
        print(f"    with frames  : {manifest.get('vision_examples', 0)}")
        print(f"    text fallback: {manifest.get('vision_fallback_examples', 0)}")
    print(f"  Unique trainees: {manifest.get('unique_trainees', 0)}")
    print(f"  Split strategy : {manifest.get('split_strategy', 'unknown')}")
    print(f"  Source domains : {manifest.get('source_domains', {})}")
    print()
    print("Next steps:")
    print("  1. Validate: ls data/training/*_v*/")
    print("  2. Launch training via deploy/runpod_launch.sh on a GPU pod.")
    print("  3. Do NOT launch src/configs/finetune_lasana_v4.yaml until the")
    print("     gate in that file is cleared. See docs/LASANA_DEPLOYMENT_PLAN.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
