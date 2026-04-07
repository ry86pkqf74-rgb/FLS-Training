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

FEEDBACK_DIR = Path(__file__).parent.parent.parent / "memory" / "feedback"


def _load_coach_feedback(video_id: str) -> dict | None:
    """Load the most recent coach feedback for a video, if available."""
    if not FEEDBACK_DIR.exists():
        return None
    matches = sorted(
        [f for f in FEEDBACK_DIR.iterdir() if video_id in f.name and f.suffix == ".json"],
        reverse=True,
    )
    if not matches:
        return None
    try:
        data = json.loads(matches[0].read_text())
        # Strip internal metadata from training data
        data.pop("_meta", None)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load coach feedback for {video_id}: {e}")
        return None


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
    include_coach_feedback: bool = False,
) -> dict:
    """Build train/val/test JSONL files from scored videos.

    Args:
        include_coach_feedback: If True, append coach feedback to the assistant
            response so the student model learns to produce both scores and
            rich technique coaching in a single pass (Phase 2 of coach integration).

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

        # Optionally attach coach feedback for richer student training
        coach_feedback = None
        if include_coach_feedback:
            coach_feedback = _load_coach_feedback(sample["video_id"])

        # Build the assistant response: score JSON + optional coach JSON
        assistant_content = json.dumps(score_data, default=str)
        if coach_feedback:
            # Wrap both in a combined output so student learns to produce both
            combined = {
                "scoring_result": score_data,
                "coach_feedback": coach_feedback,
            }
            assistant_content = json.dumps(combined, default=str)

        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "[FRAMES_PLACEHOLDER] Score this FLS Task 5 video."},
                {"role": "assistant", "content": assistant_content},
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
        "include_coach_feedback": include_coach_feedback,
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
