#!/usr/bin/env python3
"""Prepare v003 multimodal SFT dataset (train/val/test) for v17 LoRA training.

This is the v003 successor to ``030_prep_sft_data.py`` and works directly
from the existing ``data/training/youtube_sft_v1`` JSONL — it does NOT
re-score videos. For each existing example we:

1. Recompute score math via the v003 ``recompute_score_from_components``.
2. Promote major penalties into ``critical_errors`` with the right gating
   flags (drain_avulsion etc. → forces_zero_score=True).
3. Add ``severity``, ``confidence_rationale``, ``cannot_determine`` and the
   ``task_specific_assessments`` block per the v003 schema.
4. Wrap the enriched target as the assistant message in chat format.

Outputs:
    data/training/youtube_sft_v003/{train,val,test}.jsonl
    data/training/youtube_sft_v003/manifest.json

Run this LOCALLY (CPU only, no GPU needed). The resulting JSONL is then
shipped to the RunPod GPU box for v17 multimodal LoRA training.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Make ``src`` importable regardless of cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rubrics.loader import canonical_task_id
from src.training.v003_target import enrich_to_v003_target, is_v003_target


SYSTEM_PROMPT = (
    "You are an expert FLS proctor AI. Score the trainee's performance on the "
    "indicated FLS task. Output rubric-faithful scoring JSON in v003 schema: "
    "include score_components with formula_applied, every penalty with "
    "points_deducted/severity/confidence/frame_evidence, critical_errors "
    "(forces_zero_score / blocks_proficiency_claim), cannot_determine, "
    "confidence_rationale, and task_specific_assessments. Do not invent "
    "evidence; never claim proficiency when critical errors are present."
)


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _augment_example(raw: dict) -> dict | None:
    """Convert a youtube_sft_v1-style record into a v003 chat example.

    The v1 record carries either a ``target`` dict (preferred) or the full
    raw teacher score. We always run the target through the v003 enrichment
    pipeline so the assistant content has the v003 contract.
    """
    target = raw.get("target") or raw
    task_id = canonical_task_id(
        raw.get("task_id") or target.get("task_id") or "task5"
    )
    enriched = enrich_to_v003_target(target, task_id)
    if not is_v003_target(enriched):
        return None

    user_text = (
        f"Task: {enriched.get('task_name') or task_id}\n"
        f"Task ID: {task_id}\n"
        "Score this performance and emit the v003 scoring JSON described in "
        "the system prompt."
    )

    user_content: list[dict] = []
    images = raw.get("images") or raw.get("frames") or []
    for img in images:
        if isinstance(img, str):
            user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": user_text})

    assistant_content = json.dumps(enriched, default=str)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content if user_content else user_text},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "video_id": raw.get("video_id") or target.get("video_id", ""),
            "task_id": task_id,
            "label_type": raw.get("label_type", "v003_enriched"),
            "training_score": enriched["score_components"]["total_fls_score"],
            "max_score": enriched["score_components"]["max_score"],
            "has_critical_error": bool(enriched["critical_errors"]),
            "schema_version": "v003",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data/training/youtube_sft_v1",
        help="Existing youtube_sft_v1 directory containing train/val/test.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/training/youtube_sft_v003",
        help="Where to write the v003 train/val/test JSONL.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    drop_stats: dict[str, int] = defaultdict(int)
    counts_by_task: dict[str, int] = defaultdict(int)
    counts_with_critical: int = 0

    for split_name in ("train", "val", "test"):
        src = args.source_dir / f"{split_name}.jsonl"
        for raw in _load_jsonl(src):
            example = _augment_example(raw)
            if example is None:
                drop_stats["enrichment_failed"] += 1
                continue
            splits[split_name].append(example)
            counts_by_task[example["metadata"]["task_id"]] += 1
            if example["metadata"]["has_critical_error"]:
                counts_with_critical += 1

    # If the source had no test split, carve one out of train.
    if not splits["test"] and splits["train"]:
        rng = random.Random(args.seed)
        rng.shuffle(splits["train"])
        carve = max(1, len(splits["train"]) // 20)  # ~5%
        splits["test"] = splits["train"][:carve]
        splits["train"] = splits["train"][carve:]

    total = sum(len(v) for v in splits.values())
    if total == 0:
        raise SystemExit(
            f"No examples produced from {args.source_dir}. "
            "Ensure scripts/030_prep_sft_data.py has been run on the score corpus."
        )

    for name, rows in splits.items():
        path = args.output_dir / f"{name}.jsonl"
        with path.open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=str) + "\n")
        print(f"  {name}.jsonl: {len(rows)} examples → {path}")

    manifest = {
        "dataset": "youtube_sft_v003",
        "source_dataset": str(args.source_dir),
        "schema_version": "v003",
        "system_prompt": SYSTEM_PROMPT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "totals": {name: len(rows) for name, rows in splits.items()},
        "task_counts": dict(counts_by_task),
        "examples_with_critical_error": counts_with_critical,
        "drop_stats": dict(drop_stats),
        "seed": args.seed,
    }
    with (args.output_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"\nManifest: {args.output_dir / 'manifest.json'}")
    print(f"Total v003 examples: {total}")
    print(f"  With critical errors: {counts_with_critical}")
    print(f"  Drop stats: {dict(drop_stats)}")


if __name__ == "__main__":
    main()
