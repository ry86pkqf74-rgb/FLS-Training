#!/usr/bin/env python3
"""035_backfill_duckdb.py — Populate DuckDB from existing JSON scores and ledger.

Scans memory/scores/ for all score JSONs and memory/learning_ledger*.jsonl
for video metadata. Inserts everything into DuckDB so that
040_prepare_training_data.py has candidates to work with.

Safe to run multiple times — uses INSERT OR REPLACE.

Usage:
    python scripts/035_backfill_duckdb.py
    python scripts/035_backfill_duckdb.py --db data/fls_training.duckdb --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.memory_store import MemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()

SCORES_DIR = Path("memory/scores")
LEDGER_DIR = Path("memory")


def _parse_source_from_filename(filename: str) -> str:
    if "gpt-4o" in filename or "gpt4o" in filename:
        return "teacher_gpt4o"
    if "consensus" in filename:
        return "critique_consensus"
    if "student" in filename:
        return "student_model"
    return "teacher_claude"


def _parse_model_from_filename(filename: str) -> str:
    if "gpt-4o" in filename or "gpt4o" in filename:
        return "gpt-4o"
    if "claude" in filename:
        return "claude-sonnet-4-20250514"
    return "unknown"


def _parse_video_id_from_filename(filename: str) -> str:
    name = filename.replace(".json", "")
    for marker in ["_claude-sonnet-4", "_gpt-4o", "_consensus", "_student"]:
        if marker in name:
            return name.split(marker)[0]
    return name


def load_ledger_videos() -> dict[str, dict]:
    videos = {}
    for lf in sorted(LEDGER_DIR.glob("learning_ledger*.jsonl")):
        for line in lf.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("event_type") == "video_ingested":
                    d = entry["data"]
                    videos[d["video_id"]] = {
                        "id": d["video_id"],
                        "filename": d.get("filename", f"{d['video_id']}.mov"),
                        "task": d.get("task", "task5_intracorporeal_suture"),
                        "duration_seconds": d.get("duration_seconds", 0),
                        "resolution": d.get("resolution", "unknown"),
                        "fps": d.get("fps", 30.0),
                        "file_hash": d.get("file_hash", ""),
                        "ingested_at": entry.get("timestamp",
                                                  datetime.now(timezone.utc).isoformat()),
                    }
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping ledger line in {lf.name}: {e}")
    return videos


def load_score_files() -> list[dict]:
    scores = []
    if not SCORES_DIR.exists():
        return scores

    for date_dir in sorted(SCORES_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        for jf in sorted(date_dir.glob("*.json")):
            try:
                data = json.loads(jf.read_text())
                fn = jf.name
                video_id = data.get("video_id") or _parse_video_id_from_filename(fn)
                source = _parse_source_from_filename(fn)
                model = _parse_model_from_filename(fn)

                scores.append({
                    "id": jf.stem,
                    "video_id": video_id,
                    "source": source,
                    "model_name": model,
                    "model_version": "",
                    "prompt_version": "v001",
                    "scored_at": data.get("scored_at",
                                          datetime.now(timezone.utc).isoformat()),
                    "completion_time_seconds": data.get("completion_time_seconds", 0),
                    "estimated_penalties": data.get("estimated_penalties", 0),
                    "estimated_fls_score": data.get("estimated_fls_score", 0),
                    "confidence_score": data.get("confidence_score", 0),
                    "api_cost_usd": data.get("api_cost_usd", 0),
                    "latency_seconds": data.get("latency_seconds", 0),
                    "raw_json": json.dumps(data),
                })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping {jf}: {e}")
    return scores


def backfill(db_path: str, dry_run: bool = False):
    console.print("[bold]Loading ledger video metadata...[/bold]")
    videos = load_ledger_videos()
    console.print(f"  Found {len(videos)} videos in ledger files")

    console.print("[bold]Loading score JSON files...[/bold]")
    scores = load_score_files()
    console.print(f"  Found {len(scores)} score files")

    # Create video entries for scores missing from ledger
    for s in scores:
        vid = s["video_id"]
        if vid not in videos:
            videos[vid] = {
                "id": vid,
                "filename": f"{vid}.mov",
                "task": "task5_intracorporeal_suture",
                "duration_seconds": s["completion_time_seconds"],
                "resolution": "unknown",
                "fps": 30.0,
                "file_hash": "",
                "ingested_at": s["scored_at"],
            }

    if dry_run:
        console.print("\n[yellow]DRY RUN — no database changes[/yellow]")
        _print_summary(videos, scores)
        return

    console.print(f"\n[bold]Writing to {db_path}...[/bold]")
    store = MemoryStore(db_path)

    v_count = 0
    for meta in videos.values():
        try:
            store.conn.execute(
                "INSERT OR REPLACE INTO videos VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [meta["id"], meta["filename"], meta["task"],
                 meta["duration_seconds"], meta["resolution"],
                 meta["fps"], meta["file_hash"], meta["ingested_at"]],
            )
            v_count += 1
        except Exception as e:
            logger.warning(f"Failed to insert video {meta['id']}: {e}")

    s_count = 0
    for s in scores:
        try:
            store.conn.execute(
                "INSERT OR REPLACE INTO scores VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [s["id"], s["video_id"], s["source"], s["model_name"],
                 s["model_version"], s["prompt_version"], s["scored_at"],
                 s["completion_time_seconds"], s["estimated_penalties"],
                 s["estimated_fls_score"], s["confidence_score"],
                 s["api_cost_usd"], s["latency_seconds"], s["raw_json"]],
            )
            s_count += 1
        except Exception as e:
            logger.warning(f"Failed to insert score {s['id']}: {e}")

    store.close()

    console.print(f"\n[green]Backfill complete:[/green]")
    console.print(f"  Videos inserted: {v_count}")
    console.print(f"  Scores inserted: {s_count}")
    _print_summary(videos, scores)


def _print_summary(videos: dict, scores: list):
    by_video: dict[str, dict] = {}
    for s in scores:
        vid = s["video_id"]
        if vid not in by_video:
            by_video[vid] = {"claude": 0, "gpt4o": 0, "fls_scores": []}
        if "claude" in s["source"]:
            by_video[vid]["claude"] += 1
        elif "gpt4o" in s["source"]:
            by_video[vid]["gpt4o"] += 1
        if s["estimated_fls_score"]:
            by_video[vid]["fls_scores"].append(s["estimated_fls_score"])

    table = Table(title="Score Coverage")
    table.add_column("Video ID", style="cyan")
    table.add_column("Claude", justify="center")
    table.add_column("GPT-4o", justify="center")
    table.add_column("FLS Score", justify="right")
    table.add_column("Both Teachers", justify="center")

    for vid in sorted(by_video.keys()):
        info = by_video[vid]
        avg = sum(info["fls_scores"]) / len(info["fls_scores"]) if info["fls_scores"] else 0
        has_both = info["claude"] > 0 and info["gpt4o"] > 0
        table.add_row(
            vid,
            "✅" if info["claude"] > 0 else "❌",
            "✅" if info["gpt4o"] > 0 else "❌",
            f"{avg:.0f}" if avg else "-",
            "✅" if has_both else "❌",
        )

    console.print(table)
    total = len(by_video)
    both = sum(1 for v in by_video.values() if v["claude"] > 0 and v["gpt4o"] > 0)
    claude_only = sum(1 for v in by_video.values() if v["claude"] > 0 and v["gpt4o"] == 0)
    console.print(f"\n  Total: {total} | Both teachers: {both} | Claude-only: {claude_only}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill DuckDB from JSON scores")
    parser.add_argument("--db", default="data/fls_training.duckdb")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    backfill(args.db, args.dry_run)
