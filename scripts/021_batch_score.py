#!/usr/bin/env python3
"""Batch-score harvested FLS videos with Claude Sonnet 4 and GPT-4o."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.frame_extractor import extract_frames, frames_to_base64
from src.scoring.frontier_scorer import score_with_claude, score_with_gpt


REPO_ROOT = Path(__file__).resolve().parent.parent
HARVEST_LOG = REPO_ROOT / "harvest_log.jsonl"
SCORES_DIR = REPO_ROOT / "memory" / "scores"
FRAMES_DIR = REPO_ROOT / "memory" / "frames"

load_dotenv()

TASK_MAP = {
    "task1": 1,
    "task1_peg_transfer": 1,
    "task2": 2,
    "task2_pattern_cut": 2,
    "task3": 3,
    "task3_endoloop": 3,
    "task3_ligating_loop": 3,
    "task4": 4,
    "task4_extracorporeal_knot": 4,
    "task4_extracorporeal_suture": 4,
    "task5": 5,
    "task5_intracorporeal_suturing": 5,
    "task5_intracorporeal_suture": 5,
}


def get_scored_videos() -> dict[str, set[str]]:
    """Return a mapping of video_id to completed teacher models."""
    scored: dict[str, set[str]] = {}
    if not SCORES_DIR.exists():
        return scored

    for score_path in SCORES_DIR.rglob("*.json"):
        name = score_path.stem
        for model in ("claude-sonnet-4", "gpt-4o"):
            marker = f"_{model}"
            if marker in name:
                video_id = name.split(marker)[0]
                scored.setdefault(video_id, set()).add(model)
                break
    return scored


def _save_frames_cache(video_id: str, frames_b64: list[str], metadata: dict) -> None:
    frame_dir = FRAMES_DIR / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    fps = float(metadata.get("fps") or 0)
    timestamps = []
    for index in metadata.get("extracted_indices", []):
        timestamps.append(round(index / fps, 2) if fps > 0 else 0.0)

    cache = {
        "video_id": video_id,
        "frames_b64": frames_b64,
        "frame_timestamps": timestamps,
        "metadata": metadata,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    (frame_dir / "frames.json").write_text(json.dumps(cache))


def _save_score(score) -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = SCORES_DIR / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = "claude-sonnet-4" if "claude" in score.model_name else "gpt-4o"
    out_path = out_dir / f"{score.video_id}_{model_slug}_{timestamp}.json"
    out_path.write_text(score.model_dump_json(indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-score harvested videos")
    parser.add_argument("--task", help="Filter to a specific task label from the harvest log")
    parser.add_argument("--max", type=int, default=30)
    parser.add_argument("--prompt-version", default="v002")
    parser.add_argument("--claude-only", action="store_true")
    parser.add_argument("--gpt-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=int, default=5, help="Seconds between API calls")
    args = parser.parse_args()

    if not HARVEST_LOG.exists():
        raise FileNotFoundError("harvest_log.jsonl not found. Run the harvester first.")

    scored = get_scored_videos()
    to_score: list[dict[str, object]] = []

    for line in HARVEST_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        video_id = str(entry.get("video_id") or "").strip()
        filepath = str(entry.get("filepath") or "").strip()
        task = str(entry.get("task") or "").strip()
        task_num = TASK_MAP.get(task)
        if not video_id or not filepath or not Path(filepath).exists() or task_num is None:
            continue
        if args.task and task != args.task:
            continue

        existing = scored.get(video_id, set())
        needs_claude = "claude-sonnet-4" not in existing and not args.gpt_only
        needs_gpt = "gpt-4o" not in existing and not args.claude_only
        if needs_claude or needs_gpt:
            to_score.append(
                {
                    "video_id": video_id,
                    "filepath": filepath,
                    "task": task,
                    "task_num": task_num,
                    "needs_claude": needs_claude,
                    "needs_gpt": needs_gpt,
                }
            )

    to_score = to_score[: args.max]
    print(f"Videos to score: {len(to_score)}")

    if args.dry_run:
        for item in to_score:
            flags = []
            if item["needs_claude"]:
                flags.append("Claude")
            if item["needs_gpt"]:
                flags.append("GPT-4o")
            print(
                f"  {str(item['video_id'])[:32]:32} task={item['task_num']} needs={'+'.join(flags)}"
            )
        return

    for index, item in enumerate(to_score, start=1):
        video_id = str(item["video_id"])
        filepath = str(item["filepath"])
        task_num = int(item["task_num"])

        print(f"\n{'=' * 72}")
        print(f"[{index}/{len(to_score)}] {video_id} (Task {task_num})")

        try:
            frames, metadata = extract_frames(filepath)
            frames_b64 = frames_to_base64(frames)
            _save_frames_cache(video_id, frames_b64, metadata)
            print(f"  Extracted {len(frames_b64)} frames")
        except Exception as exc:
            print(f"  SKIP (frame extraction failed): {exc}")
            continue

        if item["needs_claude"]:
            try:
                print(f"  Scoring with Claude (task={task_num}, {args.prompt_version})...")
                score = score_with_claude(
                    frames_b64=frames_b64,
                    video_id=video_id,
                    video_filename=Path(filepath).name,
                    prompt_version=args.prompt_version,
                    task=task_num,
                )
                out_path = _save_score(score)
                print(f"  Claude OK: {out_path.name}")
                time.sleep(args.delay)
            except Exception as exc:
                print(f"  Claude FAILED: {exc}")
                traceback.print_exc()

        if item["needs_gpt"]:
            try:
                print(f"  Scoring with GPT-4o (task={task_num}, {args.prompt_version})...")
                score = score_with_gpt(
                    frames_b64=frames_b64,
                    video_id=video_id,
                    video_filename=Path(filepath).name,
                    prompt_version=args.prompt_version,
                    task=task_num,
                )
                out_path = _save_score(score)
                print(f"  GPT-4o OK: {out_path.name}")
                time.sleep(args.delay)
            except Exception as exc:
                print(f"  GPT-4o FAILED: {exc}")
                traceback.print_exc()

    print("\nDone. Run: python scripts/030_run_consensus.py --prompt-version v002 --with-coach-feedback")


if __name__ == "__main__":
    main()