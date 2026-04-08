#!/usr/bin/env python3
"""Download FLS videos from the curated CSV target list."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
HARVEST_LOG = REPO_ROOT / "harvest_log.jsonl"
DOWNLOAD_DIR = Path.home() / "fls_harvested_videos"
CSV_PATH = REPO_ROOT / "data" / "harvest_targets.csv"
YT_DLP_CMD = [sys.executable, "-m", "yt_dlp"]


def get_already_harvested() -> set[str]:
    """Return URLs and video IDs already logged in harvest_log.jsonl."""
    harvested: set[str] = set()
    if not HARVEST_LOG.exists():
        return harvested

    for line in HARVEST_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        url = str(entry.get("url") or "").strip()
        video_id = str(entry.get("video_id") or entry.get("youtube_id") or "").strip()
        if url:
            harvested.add(url)
        if video_id:
            harvested.add(video_id)
    return harvested


def _build_download_cmd(url: str, task_dir: Path) -> list[str]:
    ffmpeg_available = shutil.which("ffmpeg") is not None
    if ffmpeg_available:
        format_selector = "bestvideo[height<=720]+bestaudio/best[height<=720]"
    else:
        format_selector = (
            "best[ext=mp4][height<=720]/best[ext=mp4]/best[height<=720]/best"
        )

    cmd = [
        *YT_DLP_CMD,
        "--format",
        format_selector,
        "--output",
        str(task_dir / "%(id)s.%(ext)s"),
        "--no-playlist",
        "--socket-timeout",
        "30",
        "--retries",
        "2",
        "--print",
        "after_move:filepath",
        url,
    ]
    if ffmpeg_available:
        cmd.extend(["--merge-output-format", "mp4"])
    return cmd


def download_video(url: str, task_dir: Path) -> tuple[Path | None, str | None]:
    """Download a single video with yt-dlp."""
    task_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        _build_download_cmd(url, task_dir),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "yt-dlp failed").strip()
        return None, stderr[:200]

    stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    for line in reversed(stdout_lines):
        candidate = Path(line)
        if candidate.exists() and candidate.is_file():
            return candidate, None

    recent_files = sorted(task_dir.glob("*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in recent_files:
        if candidate.is_file() and candidate.suffix.lower() in {".mp4", ".mkv", ".webm"}:
            return candidate, None

    return None, "File not found after download"


def _float_value(value: str | None) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest FLS videos from harvest_targets.csv")
    parser.add_argument("--task", help="Filter to a specific task label from the CSV")
    parser.add_argument("--max", type=int, default=50, help="Maximum number of videos to download")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-unclassified", dest="skip_unclassified", action="store_true")
    parser.add_argument("--include-unclassified", dest="skip_unclassified", action="store_false")
    parser.set_defaults(skip_unclassified=True)
    args = parser.parse_args()

    already = get_already_harvested()
    targets: list[dict[str, str]] = []

    with CSV_PATH.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = str(row.get("url") or "").strip()
            task = str(row.get("task") or "").strip()
            if not url:
                continue
            if url in already:
                continue
            if args.skip_unclassified and task == "unclassified":
                continue
            if args.task and task != args.task:
                continue
            targets.append(row)

    print(f"Found {len(targets)} new targets (after dedup vs {len(already)} logged ids/urls)")
    if args.task:
        print(f"Filtered to task: {args.task}")

    targets = targets[: args.max]
    print(f"Will download: {len(targets)}")

    if args.dry_run:
        for row in targets:
            print(f"  {row.get('task', ''):30} {row.get('url', '')}")
        return

    success = 0
    for index, row in enumerate(targets, start=1):
        url = str(row.get("url") or "").strip()
        task = str(row.get("task") or "unclassified").strip() or "unclassified"
        task_dir = DOWNLOAD_DIR / task

        title = str(row.get("title") or url)
        print(f"\n[{index}/{len(targets)}] {task}: {title[:80]}")
        filepath, err = download_video(url, task_dir)
        if err or filepath is None:
            print(f"  SKIP: {(err or 'unknown error')[:120]}")
            continue

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": url,
            "video_id": filepath.stem,
            "filename": filepath.name,
            "filepath": str(filepath),
            "task": task,
            "channel": row.get("channel", ""),
            "title": row.get("title", ""),
            "duration": _float_value(row.get("duration")),
            "source": "harvest_targets_csv",
            "estimated_skill_level": row.get("estimated_skill_level", "unknown"),
            "notes": row.get("notes", ""),
        }
        with HARVEST_LOG.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")

        success += 1
        print(f"  OK: {filepath.name}")

    print(f"\nDone: {success}/{len(targets)} downloaded")


if __name__ == "__main__":
    main()