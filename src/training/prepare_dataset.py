"""Prepare training datasets from accumulated scores and corrections.

Builds JSONL files suitable for fine-tuning Qwen2.5-VL-7B-Instruct.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.memory.memory_store import MemoryStore
from src.memory.learning_log import LearningLog

logger = logging.getLogger(__name__)


def prepare_dataset(
    store: MemoryStore,
    log: LearningLog,
    video_dir: str | Path,
    output_dir: str | Path,
    version: int = 1,
    min_confidence: float = 0.7,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> dict:
    """Build train/val/test JSONL files from scored videos.

    Returns manifest dict with dataset stats.
    """
    # Load system prompt for training examples
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    system_prompt = (prompts_dir / "v001_task5_system.md").read_text()

    # Get training candidates
    candidates = store.get_training_candidates(min_confidence)
    logger.info(f"Found {len(candidates)} training candidates")

    if not candidates:
        logger.warning("No training candidates found. Score more videos first.")
        return {"error": "no_candidates", "n_candidates": 0}

    # Deduplicate by video_id (prefer corrections > consensus > raw)
    by_video: dict[str, dict] = {}
    SOURCE_PRIORITY = {"expert_correction": 0, "critique_consensus": 1,
                       "teacher_claude": 2, "teacher_gpt4o": 3, "student_model": 4}
    for c in candidates:
        vid = c["video_id"]
        source = c.get("source", "teacher_claude")
        priority = SOURCE_PRIORITY.get(source, 5)
        if vid not in by_video or priority < SOURCE_PRIORITY.get(by_video[vid].get("source", ""), 5):
            by_video[vid] = c

    samples = list(by_video.values())
    logger.info(f"Deduplicated to {len(samples)} unique videos")

    # Build training examples
    examples = []
    for sample in samples:
        raw_json = sample.get("raw_json")
        if not raw_json:
            continue

        # The raw_json contains the full ScoringResult
        if isinstance(raw_json, str):
            score_data = json.loads(raw_json)
        else:
            score_data = raw_json

        # Apply corrections if present
        corrected_fields = sample.get("corrected_fields")
        if corrected_fields:
            if isinstance(corrected_fields, str):
                corrected_fields = json.loads(corrected_fields)
            score_data.update(corrected_fields)

        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "[FRAMES_PLACEHOLDER] Score this FLS Task 5 video."},
                {"role": "assistant", "content": json.dumps(score_data, default=str)},
            ],
            "metadata": {
                "video_id": sample["video_id"],
                "source": sample.get("source", "unknown"),
                "confidence": sample.get("confidence_score", 0),
            },
        }
        examples.append(example)

    if not examples:
        return {"error": "no_valid_examples", "n_candidates": len(candidates)}

    # Split
    random.seed(seed)
    random.shuffle(examples)

    n_train = int(len(examples) * train_split)
    n_val = int(len(examples) * val_split)

    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]

    # Write output
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path(output_dir) / f"{date_str}_v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex, default=str) + "\n")
        logger.info(f"Wrote {len(split_data)} examples to {path}")

    # Manifest
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_examples": len(examples),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "min_confidence": min_confidence,
        "sources": {},
        "seed": seed,
    }
    for ex in examples:
        src = ex["metadata"]["source"]
        manifest["sources"][src] = manifest["sources"].get(src, 0) + 1

    manifest_path = out_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    log.append_event("dataset_prepared", {
        "version": version,
        "output_dir": str(out_dir),
        **manifest,
    })

    logger.info(f"Dataset v{version}: {len(train)} train, {len(val)} val, {len(test)} test")
    return manifest
