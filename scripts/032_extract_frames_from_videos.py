#!/usr/bin/env python3
"""Extract evenly-spaced JPG frames from raw videos with ffmpeg.

Walks a directory of YouTube-style downloads (``<video_id>.<format>.mp4`` /
``.webm``), groups them by video_id, picks the highest-resolution variant for
each video_id, and extracts ``--n-frames`` evenly-spaced JPGs into
``memory/frames_v20/<video_id>/frame_NNN.jpg``.

Skips video_ids that already have frames in ``--existing-frames-dir``.

Run locally on the Mac (or anywhere with ffmpeg + the raw videos)::

    python scripts/032_extract_frames_from_videos.py \\
        --videos-dir ~/fls_harvested_videos \\
        --output-dir memory/frames_v20 \\
        --existing-frames-dir memory/frames \\
        --n-frames 8
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".mov"}
# Approximate format priority: prefer higher YouTube format codes (which map
# to higher quality) when multiple variants of the same video exist.
FORMAT_PRIORITY = ("f137", "f136", "f135", "f398", "f134", "f18", "f251", "f250")


def _video_id(path: Path) -> str:
    """``1b8jvVzo_Zk.f136.mp4`` -> ``1b8jvVzo_Zk``."""
    return path.stem.split(".", 1)[0]


def _format_rank(path: Path) -> int:
    name = path.name.lower()
    for i, fmt in enumerate(FORMAT_PRIORITY):
        if f".{fmt}." in name:
            return i
    return len(FORMAT_PRIORITY) + 1


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    try:
        return float(out.stdout.strip())
    except (TypeError, ValueError):
        return 0.0


def _extract_frames(video: Path, out_dir: Path, n_frames: int) -> int:
    duration = _ffprobe_duration(video)
    if duration <= 0:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    # Sample the central 90% of the clip to dodge title/credits frames.
    start = duration * 0.05
    end = duration * 0.95
    span = max(0.1, end - start)
    for i in range(n_frames):
        t = start + (span * i / max(1, n_frames - 1))
        out_path = out_dir / f"frame_{i:03d}.jpg"
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-ss", f"{t:.2f}",
            "-i", str(video),
            "-frames:v", "1",
            "-q:v", "3",
            "-y",
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and out_path.exists():
            extracted += 1
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-frames-dir", type=Path, default=None)
    parser.add_argument("--n-frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SystemExit("ffmpeg / ffprobe not found in PATH")

    by_video: dict[str, list[Path]] = defaultdict(list)
    for path in args.videos_dir.iterdir():
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            by_video[_video_id(path)].append(path)

    existing: set[str] = set()
    if args.existing_frames_dir and args.existing_frames_dir.exists():
        for d in args.existing_frames_dir.iterdir():
            if d.is_dir():
                existing.add(d.name)
                # Also strip leading "yt_" so we don't extract twice for the
                # same physical video under both naming conventions.
                if d.name.startswith("yt_"):
                    existing.add(d.name[len("yt_"):])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "videos_dir": str(args.videos_dir),
        "output_dir": str(args.output_dir),
        "existing_frames_dir": str(args.existing_frames_dir) if args.existing_frames_dir else None,
        "candidates": len(by_video),
        "skipped_existing": 0,
        "extracted": 0,
        "failed": 0,
        "details": [],
    }

    processed = 0
    for vid, variants in sorted(by_video.items()):
        if vid in existing:
            summary["skipped_existing"] += 1
            continue
        # Pick the best-format variant we have on disk.
        variants.sort(key=_format_rank)
        chosen = variants[0]
        out_dir = args.output_dir / vid
        try:
            n = _extract_frames(chosen, out_dir, args.n_frames)
        except Exception as exc:  # noqa: BLE001
            summary["failed"] += 1
            summary["details"].append({"video_id": vid, "error": str(exc)})
            continue
        if n > 0:
            summary["extracted"] += 1
            summary["details"].append({"video_id": vid, "source": chosen.name, "frames": n})
        else:
            summary["failed"] += 1
        processed += 1
        if args.limit and processed >= args.limit:
            break

    summary["details"] = summary["details"][:30]  # keep the report compact
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
