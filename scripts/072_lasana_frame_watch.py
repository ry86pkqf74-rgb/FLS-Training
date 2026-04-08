#!/usr/bin/env python3
"""Watch laid-out LASANA video directories and trigger per-video frame extraction."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG = logging.getLogger("lasana_frame_watch")
RECEIPT_NAME = ".lasana_extract_complete.json"
STATE_DIRNAME = ".frame_watch_state"
VIDEO_FILENAMES = (
    "video.hevc",
    "video.h265",
    "video.mp4",
    "video.mkv",
    "video.mov",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch LASANA layout directories and trigger 068")
    parser.add_argument("--layout-dir", required=True, help="Directory containing <video_id>/video.hevc")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Processed LASANA output root passed through to scripts/068_lasana_extract_features.py",
    )
    parser.add_argument(
        "--video-prefixes",
        default=None,
        help="Optional comma-separated video_id prefixes to watch, e.g. lasana_suture",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Directory for per-video watcher state; defaults to <layout-dir>/.frame_watch_state",
    )
    parser.add_argument(
        "--extract-script",
        default=str(Path(__file__).with_name("068_lasana_extract_features.py")),
        help="Path to the frame extractor entrypoint",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to invoke the frame extractor",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate passed to 068")
    parser.add_argument("--watch", action="store_true", help="Poll continuously for new video_id folders")
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


def first_video_path(video_dir: Path) -> Path | None:
    for filename in VIDEO_FILENAMES:
        candidate = video_dir / filename
        if candidate.is_file():
            return candidate
    return None


def frames_dir_for(out_dir: Path, video_id: str) -> Path:
    return out_dir / "frames" / video_id


def completion_receipt_path(frames_dir: Path) -> Path:
    return frames_dir / RECEIPT_NAME


def state_path(state_dir: Path, video_id: str) -> Path:
    return state_dir / f"{video_id}.json"


def count_frames(frames_dir: Path) -> int:
    return sum(1 for _ in frames_dir.glob("frame_*.jpg"))


def has_frames(frames_dir: Path) -> bool:
    return count_frames(frames_dir) > 0


def iter_video_dirs(layout_dir: Path, video_prefixes: list[str]) -> list[tuple[str, Path, Path]]:
    candidates: list[tuple[str, Path, Path]] = []
    for child in sorted(layout_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if video_prefixes and not any(child.name.startswith(prefix) for prefix in video_prefixes):
            continue
        video_path = first_video_path(child)
        if video_path is None:
            continue
        candidates.append((child.name, child, video_path))
    return candidates


def mark_existing_complete(state_dir: Path, out_dir: Path, video_id: str, video_dir: Path, video_path: Path) -> None:
    frames_dir = frames_dir_for(out_dir, video_id)
    frame_count = count_frames(frames_dir)
    receipt = {
        "completed_at": utc_now(),
        "frame_count": frame_count,
        "frames_dir": str(frames_dir),
        "source_video": str(video_path),
        "status": "already_present",
        "video_dir": str(video_dir),
        "video_id": video_id,
    }
    write_json(completion_receipt_path(frames_dir), receipt)
    write_json(state_path(state_dir, video_id), receipt)


def build_extract_command(
    *,
    python_bin: str,
    extract_script: Path,
    staging_root: Path,
    out_dir: Path,
    fps: float,
) -> list[str]:
    return [
        python_bin,
        str(extract_script),
        "--frames-only",
        "--lasana-dir",
        str(staging_root),
        "--out-dir",
        str(out_dir),
        "--fps",
        str(fps),
    ]


def process_video_dir(
    *,
    video_id: str,
    video_dir: Path,
    video_path: Path,
    out_dir: Path,
    state_dir: Path,
    extract_script: Path,
    python_bin: str,
    fps: float,
) -> str:
    frames_dir = frames_dir_for(out_dir, video_id)
    if has_frames(frames_dir):
        mark_existing_complete(state_dir, out_dir, video_id, video_dir, video_path)
        return "skipped_existing"

    staging_parent = state_dir / "staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{video_id}_", dir=staging_parent) as tmpdir:
        staging_root = Path(tmpdir)
        (staging_root / video_id).symlink_to(video_dir, target_is_directory=True)
        cmd = build_extract_command(
            python_bin=python_bin,
            extract_script=extract_script,
            staging_root=staging_root,
            out_dir=out_dir,
            fps=fps,
        )
        LOG.info("Extracting frames for %s", video_id)
        result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        payload = {
            "completed_at": utc_now(),
            "frames_dir": str(frames_dir),
            "source_video": str(video_path),
            "status": "failed",
            "video_dir": str(video_dir),
            "video_id": video_id,
        }
        write_json(state_path(state_dir, video_id), payload)
        return "failed"

    if not has_frames(frames_dir):
        payload = {
            "completed_at": utc_now(),
            "frames_dir": str(frames_dir),
            "source_video": str(video_path),
            "status": "missing_frames_after_extract",
            "video_dir": str(video_dir),
            "video_id": video_id,
        }
        write_json(state_path(state_dir, video_id), payload)
        return "failed"

    payload = {
        "completed_at": utc_now(),
        "frame_count": count_frames(frames_dir),
        "frames_dir": str(frames_dir),
        "source_video": str(video_path),
        "status": "extracted",
        "video_dir": str(video_dir),
        "video_id": video_id,
    }
    write_json(completion_receipt_path(frames_dir), payload)
    write_json(state_path(state_dir, video_id), payload)
    return "extracted"


def scan_once(
    *,
    layout_dir: Path,
    out_dir: Path,
    state_dir: Path,
    video_prefixes: list[str],
    extract_script: Path,
    python_bin: str,
    fps: float,
) -> dict[str, int]:
    summary = {"extracted": 0, "skipped_existing": 0, "failed": 0}
    for video_id, video_dir, video_path in iter_video_dirs(layout_dir, video_prefixes):
        result = process_video_dir(
            video_id=video_id,
            video_dir=video_dir,
            video_path=video_path,
            out_dir=out_dir,
            state_dir=state_dir,
            extract_script=extract_script,
            python_bin=python_bin,
            fps=fps,
        )
        if result in summary:
            summary[result] += 1
    return summary


def run() -> int:
    args = parse_args()
    configure_logging()

    layout_dir = Path(args.layout_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    state_dir = (
        Path(args.state_dir).expanduser().resolve()
        if args.state_dir
        else (layout_dir / STATE_DIRNAME)
    )
    extract_script = Path(args.extract_script).expanduser().resolve()
    video_prefixes = parse_csv_arg(args.video_prefixes)

    if not layout_dir.is_dir():
        raise SystemExit(f"layout-dir does not exist: {layout_dir}")
    if not extract_script.is_file():
        raise SystemExit(f"extract-script does not exist: {extract_script}")

    while True:
        summary = scan_once(
            layout_dir=layout_dir,
            out_dir=out_dir,
            state_dir=state_dir,
            video_prefixes=video_prefixes,
            extract_script=extract_script,
            python_bin=args.python_bin,
            fps=args.fps,
        )
        if not args.watch:
            return 0 if summary["failed"] == 0 else 1
        if summary == {"extracted": 0, "skipped_existing": 0, "failed": 0}:
            LOG.info("No new laid-out LASANA videos found; sleeping %ss", args.poll_seconds)
        time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        LOG.info("Stopped by user")
        raise SystemExit(130)