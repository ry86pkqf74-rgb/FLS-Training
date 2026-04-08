#!/usr/bin/env python3
"""Identify videos with Claude scores but no GPT-4o score, then rescore them.

The 2026-04-08 baseline report showed 29 of 78 videos in the corpus have
only a Claude teacher score. That makes the "consensus" score for those
rows identical to Claude's score, which inflates consensus/Claude MAE
metrics and masks variance. Before any further training we want both
teachers covering the same videos.

Two modes:

1. ``--dry-run`` (default): list the missing video_ids only. Cheap,
   prints a cost estimate, does not touch the API.
2. ``--execute``: actually call GPT-4o via ``src.scoring.frontier_scorer``
   for each missing video_id. Writes new score files into
   ``memory/scores/`` matching the existing naming convention. Uses the
   ``OPENAI_API_KEY`` env var.

This script is deliberately conservative: it will abort if the number of
rescores exceeds ``--max-videos`` (default 40), so a runaway call cannot
drain an API key.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.training.schema_adapter import normalize_score

logger = logging.getLogger(__name__)

# Claude input tokens ~3k, GPT-4o output ~2k. Ballpark cost per scored
# video at April 2026 list prices: $0.015 input + $0.02 output ≈ $0.04.
# Keep this pessimistic so the printed estimate is a ceiling.
COST_PER_VIDEO_USD = 0.05


def _latest_per_source(base: Path) -> dict[str, dict[str, dict[str, Any]]]:
    latest: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for path in sorted((base / "memory" / "scores").rglob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        normalized = normalize_score(raw)
        if not normalized["video_id"]:
            continue
        src = normalized["source"].lower()
        if "gpt" in src:
            tag = "teacher_gpt"
        elif "claude" in src:
            tag = "teacher_claude"
        elif "consensus" in src:
            tag = "consensus"
        else:
            continue
        timestamp = str(raw.get("scored_at", ""))
        key = (normalized["video_id"], tag)
        previous = latest.get(key)
        if previous is None or timestamp >= previous[0]:
            latest[key] = (timestamp, {"normalized": normalized, "raw": raw, "path": path})
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (video_id, tag), (_, payload) in latest.items():
        grouped[video_id][tag] = payload
    return grouped


def _find_missing(grouped: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for video_id, records in grouped.items():
        if "teacher_gpt" in records:
            continue
        claude = records.get("teacher_claude")
        if not claude:
            continue
        raw = claude["raw"]
        missing.append({
            "video_id": video_id,
            "task_id": claude["normalized"]["task_id"] or "task5",
            "video_filename": raw.get("video_filename") or f"{video_id}.mp4",
            "source_path": str(claude["path"]),
        })
    return sorted(missing, key=lambda r: r["video_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Rescore Claude-only videos with GPT-4o")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--execute", action="store_true",
                        help="Actually call the OpenAI API. Default is dry-run.")
    parser.add_argument("--max-videos", type=int, default=40,
                        help="Abort if more than this many videos would be rescored.")
    parser.add_argument("--video-dir", default=None,
                        help="Directory containing mp4 files. Required for --execute.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    base = Path(args.base_dir)
    grouped = _latest_per_source(base)
    missing = _find_missing(grouped)

    print(f"=== Claude-only videos missing GPT-4o score ===")
    print(f"  total_videos_in_corpus : {len(grouped)}")
    print(f"  missing_gpt4o          : {len(missing)}")
    print(f"  estimated_cost         : ${len(missing) * COST_PER_VIDEO_USD:.2f} "
          f"(@ ${COST_PER_VIDEO_USD:.2f}/video)")
    print()

    if not missing:
        print("Nothing to do — every video already has a GPT-4o score.")
        return 0

    if len(missing) > args.max_videos:
        print(f"ABORT: {len(missing)} > --max-videos {args.max_videos}. Raise --max-videos "
              f"if you really want to rescore this many.")
        return 2

    if not args.execute:
        print("DRY RUN — pass --execute to actually rescore. Videos that would be rescored:")
        for entry in missing:
            print(f"  {entry['video_id']:30s}  ({entry['task_id']})  {entry['video_filename']}")
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        print("ABORT: OPENAI_API_KEY not set.")
        return 3
    if not args.video_dir:
        print("ABORT: --video-dir is required for --execute.")
        return 3

    # Deferred import so dry-run works without the OpenAI SDK installed
    from src.scoring.frontier_scorer import FrontierScorer  # type: ignore

    scorer = FrontierScorer(provider="openai")
    video_dir = Path(args.video_dir)
    scores_dir = base / "memory" / "scores"
    successes = 0
    failures: list[tuple[str, str]] = []
    for entry in missing:
        video_path = video_dir / entry["video_filename"]
        if not video_path.is_file():
            failures.append((entry["video_id"], f"video file missing: {video_path}"))
            continue
        try:
            result = scorer.score(video_path, task=entry["task_id"])
        except Exception as exc:  # pragma: no cover - runtime
            failures.append((entry["video_id"], str(exc)))
            continue

        stamp = result.scored_at.strftime("%Y%m%d%H%M%S")
        out_path = scores_dir / f"score_gpt_{entry['video_id']}_{stamp}.json"
        out_path.write_text(result.model_dump_json(indent=2))
        successes += 1
        logger.info("Rescored %s → %s", entry["video_id"], out_path.name)

    print()
    print(f"  rescored  : {successes}")
    print(f"  failures  : {len(failures)}")
    for video_id, reason in failures:
        print(f"    {video_id}: {reason}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
