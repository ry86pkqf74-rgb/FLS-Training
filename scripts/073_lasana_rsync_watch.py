#!/usr/bin/env python3
"""Watch extracted LASANA frame directories and rsync completed video trees to Contabo."""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG = logging.getLogger("lasana_rsync_watch")
RECEIPT_NAME = ".lasana_extract_complete.json"
STATE_DIRNAME = ".frame_rsync_state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch LASANA frames and rsync completed video_ids")
    parser.add_argument("--frames-root", required=True, help="Directory containing frames/<video_id>")
    parser.add_argument(
        "--dest-host",
        default=None,
        help="Remote Contabo host; defaults to CONTABO_HOST or s8-other-project",
    )
    parser.add_argument("--dest-dir", required=True, help="Remote directory that should receive <video_id>/")
    parser.add_argument(
        "--video-prefixes",
        default=None,
        help="Optional comma-separated video_id prefixes to sync",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Directory for per-video sync markers; defaults to <frames-root>/../.frame_rsync_state",
    )
    parser.add_argument("--rsync-bin", default="rsync", help="rsync executable")
    parser.add_argument("--watch", action="store_true", help="Poll continuously")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval for --watch")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def state_path(state_dir: Path, video_id: str) -> Path:
    return state_dir / f"{video_id}.json"


def count_frames(video_dir: Path) -> int:
    return sum(1 for _ in video_dir.glob("frame_*.jpg"))


def iter_completed_frame_dirs(frames_root: Path, video_prefixes: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for child in sorted(frames_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if video_prefixes and not any(child.name.startswith(prefix) for prefix in video_prefixes):
            continue
        if not (child / RECEIPT_NAME).is_file():
            continue
        if count_frames(child) == 0:
            continue
        candidates.append(child)
    return candidates


def build_rsync_command(rsync_bin: str, source_dir: Path, dest_host: str, dest_dir: str) -> list[str]:
    quoted_target = shlex.quote(f"{dest_dir.rstrip('/')}/{source_dir.name}")
    return [
        rsync_bin,
        "-az",
        "--partial",
        "--inplace",
        "--rsync-path",
        f"mkdir -p {quoted_target} && rsync",
        f"{source_dir}/",
        f"{dest_host}:{dest_dir.rstrip('/')}/{source_dir.name}/",
    ]


def sync_video_dir(
    *,
    frames_dir: Path,
    state_dir: Path,
    dest_host: str,
    dest_dir: str,
    rsync_bin: str,
) -> str:
    cmd = build_rsync_command(rsync_bin, frames_dir, dest_host, dest_dir)
    LOG.info("Syncing %s to %s:%s", frames_dir.name, dest_host, dest_dir)
    result = subprocess.run(cmd, check=False)
    payload = {
        "completed_at": utc_now(),
        "dest_dir": dest_dir,
        "dest_host": dest_host,
        "frame_count": count_frames(frames_dir),
        "frames_dir": str(frames_dir),
        "status": "synced" if result.returncode == 0 else "failed",
        "video_id": frames_dir.name,
    }
    write_json(state_path(state_dir, frames_dir.name), payload)
    return payload["status"]


def scan_once(
    *,
    frames_root: Path,
    state_dir: Path,
    dest_host: str,
    dest_dir: str,
    video_prefixes: list[str],
    rsync_bin: str,
) -> dict[str, int]:
    summary = {"synced": 0, "failed": 0}
    for frames_dir in iter_completed_frame_dirs(frames_root, video_prefixes):
        if state_path(state_dir, frames_dir.name).exists():
            continue
        status = sync_video_dir(
            frames_dir=frames_dir,
            state_dir=state_dir,
            dest_host=dest_host,
            dest_dir=dest_dir,
            rsync_bin=rsync_bin,
        )
        summary[status] += 1
    return summary


def run() -> int:
    args = parse_args()
    configure_logging()

    frames_root = Path(args.frames_root).expanduser().resolve()
    state_dir = (
        Path(args.state_dir).expanduser().resolve()
        if args.state_dir
        else (frames_root.parent / STATE_DIRNAME)
    )
    video_prefixes = parse_csv_arg(args.video_prefixes)
    dest_host = args.dest_host or __import__("os").environ.get("CONTABO_HOST", "s8-other-project")

    if not frames_root.is_dir():
        raise SystemExit(f"frames-root does not exist: {frames_root}")

    while True:
        summary = scan_once(
            frames_root=frames_root,
            state_dir=state_dir,
            dest_host=dest_host,
            dest_dir=args.dest_dir,
            video_prefixes=video_prefixes,
            rsync_bin=args.rsync_bin,
        )
        if not args.watch:
            return 0 if summary["failed"] == 0 else 1
        if summary == {"synced": 0, "failed": 0}:
            LOG.info("No newly completed LASANA frame directories to sync; sleeping %ss", args.poll_seconds)
        time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        LOG.info("Stopped by user")
        raise SystemExit(130)