#!/usr/bin/env python3
"""Build the locked faculty-rated gold evaluation set.

Strategy
--------
The v3 evaluation split was a random 10% slice of the full 78-video
corpus. That meant every hyperparameter tweak shuffled which videos
landed in the held-out fold, which is a hidden source of variance and
makes cross-run comparisons misleading.

This script produces a *locked* gold set: a small, deterministic list of
the highest-signal videos in the corpus, pinned to a file under
``data/training/gold/<timestamp>_gold.jsonl``. Training runs MUST
exclude these video_ids from the train split and evaluate against this
file as the MAE ground truth.

Selection criteria (each video must satisfy ALL):

1. A consensus record exists (not just a single teacher score).
2. Confidence_score >= ``--min-confidence`` (default 0.7).
3. Claude and GPT-4o records BOTH exist for the same video_id (so the
   consensus is genuinely reconciled, not a Claude passthrough).
4. The Claude↔GPT delta is within ``--max-teacher-delta`` FLS points
   (default 25) — very high teacher disagreement means the "ground
   truth" is noisy and should not anchor the eval.

The output is a JSONL of normalized score records plus a manifest
documenting the selection criteria and commit hash so the gold set is
reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.training.schema_adapter import normalize_score

logger = logging.getLogger(__name__)


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _latest_score_by_source(scores_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Return mapping video_id → {source_tag → normalized record}."""
    latest: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}

    for path in sorted(scores_dir.rglob("*.json")):
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        normalized = normalize_score(raw)
        video_id = normalized["video_id"]
        if not video_id:
            continue

        source = normalized["source"].lower()
        if "gpt" in source:
            tag = "teacher_gpt"
        elif "consensus" in source:
            tag = "consensus"
        elif "claude" in source:
            tag = "teacher_claude"
        else:
            continue

        timestamp = str(raw.get("scored_at") or raw.get("timestamp") or "")
        key = (video_id, tag)
        previous = latest.get(key)
        if previous is None or timestamp >= previous[0]:
            latest[key] = (timestamp, normalized)

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (video_id, tag), (_, record) in latest.items():
        grouped[video_id][tag] = record
    return grouped


def build_gold_set(
    base_dir: Path,
    *,
    min_confidence: float,
    max_teacher_delta: float,
    max_videos: int,
) -> dict[str, Any]:
    scores_dir = base_dir / "memory" / "scores"
    if not scores_dir.exists():
        raise FileNotFoundError(f"scores dir missing: {scores_dir}")

    grouped = _latest_score_by_source(scores_dir)

    selected: list[dict[str, Any]] = []
    rejections: dict[str, int] = defaultdict(int)

    for video_id, records in grouped.items():
        consensus = records.get("consensus")
        claude = records.get("teacher_claude")
        gpt = records.get("teacher_gpt")

        if not consensus:
            rejections["no_consensus"] += 1
            continue
        if consensus["confidence_score"] < min_confidence:
            rejections["low_confidence"] += 1
            continue
        if not claude or not gpt:
            rejections["single_teacher"] += 1
            continue
        delta = abs(claude["total_fls_score"] - gpt["total_fls_score"])
        if delta > max_teacher_delta:
            rejections["teacher_disagreement"] += 1
            continue

        selected.append({
            "video_id": video_id,
            "consensus_score": consensus["total_fls_score"],
            "claude_score": claude["total_fls_score"],
            "gpt_score": gpt["total_fls_score"],
            "teacher_delta": delta,
            "confidence": consensus["confidence_score"],
            "task_id": consensus["task_id"],
            "trainee_id": consensus.get("trainee_id"),
            "source_domain": consensus.get("source_domain"),
            "target": consensus["raw"],
        })

    # Sort by lowest teacher delta, then highest confidence — the gold
    # set should be our most agreed-upon, most confident videos.
    selected.sort(key=lambda r: (r["teacher_delta"], -r["confidence"]))

    if max_videos and len(selected) > max_videos:
        selected = selected[:max_videos]

    return {
        "selected": selected,
        "rejections": dict(rejections),
        "total_candidates": len(grouped),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build locked gold evaluation set")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--output-dir", default="data/training/gold")
    parser.add_argument("--min-confidence", type=float, default=0.7)
    parser.add_argument("--max-teacher-delta", type=float, default=25.0)
    parser.add_argument("--max-videos", type=int, default=30)
    parser.add_argument("--name", default=None, help="Optional tag for the output file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    base = Path(args.base_dir)
    result = build_gold_set(
        base,
        min_confidence=args.min_confidence,
        max_teacher_delta=args.max_teacher_delta,
        max_videos=args.max_videos,
    )
    selected = result["selected"]

    print("=== Gold set build ===")
    print(f"  candidates   : {result['total_candidates']}")
    print(f"  selected     : {len(selected)}")
    print("  rejections   :")
    for reason, count in sorted(result["rejections"].items()):
        print(f"    {reason:22s} {count}")

    if not selected:
        print("\nNO VIDEOS SELECTED — relax thresholds or harvest more teacher scores.")
        return 1

    out_dir = base / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.name}" if args.name else ""
    out_path = out_dir / f"{stamp}{suffix}_gold.jsonl"
    manifest_path = out_dir / f"{stamp}{suffix}_gold_manifest.json"

    with open(out_path, "w") as handle:
        for record in selected:
            handle.write(json.dumps(record, default=str) + "\n")

    video_ids = sorted(r["video_id"] for r in selected)
    video_id_hash = hashlib.sha256("\n".join(video_ids).encode()).hexdigest()[:12]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": _git_head(),
        "min_confidence": args.min_confidence,
        "max_teacher_delta": args.max_teacher_delta,
        "max_videos": args.max_videos,
        "total_candidates": result["total_candidates"],
        "selected": len(selected),
        "video_id_hash": video_id_hash,
        "video_ids": video_ids,
        "rejections": result["rejections"],
        "jsonl_path": str(out_path.relative_to(base)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {out_path.relative_to(base)}")
    print(f"Wrote {manifest_path.relative_to(base)}")
    print(f"video_id_hash: {video_id_hash}")
    print("\nIMPORTANT: add the video_ids in this gold set to the exclusion")
    print("list used by scripts/040_prepare_training_data.py before the next")
    print("training run (so the model never sees them during train).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
