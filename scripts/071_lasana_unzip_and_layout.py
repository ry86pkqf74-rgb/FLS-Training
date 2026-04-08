#!/usr/bin/env python3
"""Watch LASANA task-level zip archives and lay them out per trial.

The downloader writes complete archives such as ``SutureAndKnot_left.zip`` to a
raw directory once the final rename from ``.part`` happens. This script turns
those task-level archives into a stable per-trial layout that the frame
extractor can consume immediately:

  <out-dir>/<video_id>/video.hevc

Where ``video_id`` matches the task-qualified identities produced by
``069_ingest_lasana_to_store.py`` (for example ``lasana_suture_kiourf``).
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zipfile import BadZipFile, ZipFile, ZipInfo


LOG = logging.getLogger("lasana_unzip")
RAW_ARCHIVE_GLOB = "*.zip"
STATE_DIRNAME = ".archive_state"
MANIFEST_NAME = "manifest.csv"
VIDEO_EXTS = {".hevc", ".h265", ".mp4", ".mkv", ".mov"}
TASK_MAP = {
    "BalloonResection": "lasana_balloon",
    "CircleCutting": "lasana_circle",
    "PegTransfer": "lasana_peg",
    "SutureAndKnot": "lasana_suture",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unzip LASANA task archives into per-trial layout")
    parser.add_argument("--raw-dir", required=True, help="Directory containing completed LASANA *.zip archives")
    parser.add_argument("--out-dir", required=True, help="Directory where per-trial video folders are written")
    parser.add_argument("--task", default=None, help="Optional task filter, e.g. SutureAndKnot")
    parser.add_argument("--watch", action="store_true", help="Poll for new *.zip archives indefinitely")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval for --watch mode")
    parser.add_argument("--max-archives", type=int, default=0, help="Process at most N matching archives per scan")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract trials even if video files already exist")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_task_name(raw: str | None) -> str:
    return "".join(ch.lower() for ch in str(raw or "") if ch.isalnum())


def infer_task_from_archive(archive_path: Path) -> str | None:
    prefix = archive_path.stem.split("_", 1)[0]
    if prefix in TASK_MAP:
        return prefix
    return None


def task_slug(task_name: str) -> str:
    return TASK_MAP[task_name]


def deterministic_video_id(task_name: str, trial_id: str) -> str:
    return f"{task_slug(task_name)}_{trial_id}"


def normalize_video_suffix(suffix: str) -> str:
    lowered = suffix.lower()
    if lowered in {".hevc", ".h265"}:
        return ".hevc"
    return lowered


def trial_id_from_member(member_name: str) -> str:
    return Path(member_name).stem


def iter_video_members(handle: ZipFile) -> list[ZipInfo]:
    members: list[ZipInfo] = []
    for info in handle.infolist():
        if info.is_dir():
            continue
        name = Path(info.filename).name
        if not name or name.startswith(".") or name.startswith("__MACOSX"):
            continue
        if Path(name).suffix.lower() not in VIDEO_EXTS:
            continue
        members.append(info)
    members.sort(key=lambda item: item.filename)
    return members


def destination_path(out_dir: Path, task_name: str, member_name: str) -> tuple[str, Path]:
    trial_id = trial_id_from_member(member_name)
    video_id = deterministic_video_id(task_name, trial_id)
    suffix = normalize_video_suffix(Path(member_name).suffix)
    return video_id, out_dir / video_id / f"video{suffix}"


def state_marker_path(out_dir: Path, archive_path: Path) -> Path:
    return out_dir / STATE_DIRNAME / f"{archive_path.name}.done"


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["record_key"]: row for row in rows if row.get("record_key")}


def write_manifest(path: Path, rows: dict[str, dict[str, str]]) -> None:
    fieldnames = [
        "record_key",
        "archive",
        "task_name",
        "trial_id",
        "video_id",
        "member_name",
        "local_path",
        "bytes",
        "status",
        "error",
        "updated_at",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])


def manifest_row(
    *,
    archive_name: str,
    task_name: str,
    trial_id: str,
    video_id: str,
    member_name: str,
    local_path: Path,
    status: str,
    bytes_written: int | None,
    error_text: str = "",
) -> dict[str, str]:
    return {
        "record_key": f"{archive_name}:{video_id}",
        "archive": archive_name,
        "task_name": task_name,
        "trial_id": trial_id,
        "video_id": video_id,
        "member_name": member_name,
        "local_path": str(local_path),
        "bytes": "" if bytes_written is None else str(bytes_written),
        "status": status,
        "error": error_text,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def extract_archive(
    archive_path: Path,
    out_dir: Path,
    *,
    overwrite: bool = False,
) -> list[dict[str, str]]:
    task_name = infer_task_from_archive(archive_path)
    if task_name is None:
        raise ValueError(f"Unsupported LASANA archive name: {archive_path.name}")

    rows: list[dict[str, str]] = []
    with ZipFile(archive_path) as handle:
        members = iter_video_members(handle)
        if not members:
            raise ValueError(f"Archive contains no supported video members: {archive_path.name}")

        for info in members:
            member_basename = Path(info.filename).name
            trial_id = trial_id_from_member(member_basename)
            video_id, destination = destination_path(out_dir, task_name, member_basename)
            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists() and not overwrite:
                rows.append(
                    manifest_row(
                        archive_name=archive_path.name,
                        task_name=task_name,
                        trial_id=trial_id,
                        video_id=video_id,
                        member_name=member_basename,
                        local_path=destination,
                        status="skipped_existing",
                        bytes_written=destination.stat().st_size,
                    )
                )
                continue

            tmp_path = destination.with_suffix(destination.suffix + ".part")
            if tmp_path.exists():
                tmp_path.unlink()

            with handle.open(info) as src, tmp_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            tmp_path.replace(destination)
            rows.append(
                manifest_row(
                    archive_name=archive_path.name,
                    task_name=task_name,
                    trial_id=trial_id,
                    video_id=video_id,
                    member_name=member_basename,
                    local_path=destination,
                    status="extracted",
                    bytes_written=destination.stat().st_size,
                )
            )

    return rows


def discover_archives(raw_dir: Path, task_filter: str | None = None) -> list[Path]:
    requested = normalize_task_name(task_filter) if task_filter else None
    archives: list[Path] = []
    for archive_path in sorted(raw_dir.glob(RAW_ARCHIVE_GLOB)):
        task_name = infer_task_from_archive(archive_path)
        if task_name is None:
            continue
        if requested and normalize_task_name(task_name) != requested:
            continue
        archives.append(archive_path)
    return archives


def process_available_archives(
    *,
    raw_dir: Path,
    out_dir: Path,
    task_filter: str | None,
    max_archives: int,
    overwrite: bool,
) -> int:
    archives = discover_archives(raw_dir, task_filter)
    if max_archives > 0:
        archives = archives[:max_archives]

    manifest_path = out_dir / MANIFEST_NAME
    status_rows = read_manifest(manifest_path)
    processed = 0

    for archive_path in archives:
        marker = state_marker_path(out_dir, archive_path)
        if marker.exists() and not overwrite:
            continue

        LOG.info("Processing %s", archive_path.name)
        try:
            rows = extract_archive(archive_path, out_dir, overwrite=overwrite)
            for row in rows:
                status_rows[row["record_key"]] = row
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(datetime.now(timezone.utc).isoformat() + "\n")
            processed += 1
        except (BadZipFile, ValueError, OSError) as exc:
            LOG.warning("Failed to process %s: %s", archive_path.name, exc)
            status_rows[f"{archive_path.name}:archive"] = {
                "record_key": f"{archive_path.name}:archive",
                "archive": archive_path.name,
                "task_name": infer_task_from_archive(archive_path) or "",
                "trial_id": "",
                "video_id": "",
                "member_name": "",
                "local_path": str(out_dir),
                "bytes": "",
                "status": "failed",
                "error": str(exc),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        finally:
            write_manifest(manifest_path, status_rows)

    return processed


def run() -> int:
    args = parse_args()
    configure_logging()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"raw-dir does not exist: {raw_dir}")

    while True:
        processed = process_available_archives(
            raw_dir=raw_dir,
            out_dir=out_dir,
            task_filter=args.task,
            max_archives=args.max_archives,
            overwrite=args.overwrite,
        )
        if not args.watch:
            return 0
        if processed == 0:
            LOG.info("No new complete LASANA archives found; sleeping %ss", args.poll_seconds)
        time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    try:
        sys.exit(run())
    except KeyboardInterrupt:
        LOG.info("Stopped by user")
        sys.exit(130)