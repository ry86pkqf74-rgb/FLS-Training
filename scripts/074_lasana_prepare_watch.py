#!/usr/bin/env python3
"""Gate LASANA dataset preparation on complete local frame availability on Contabo."""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LOG = logging.getLogger("lasana_prepare_watch")
TASK_SLUGS = [
    "lasana_balloon",
    "lasana_circle",
    "lasana_peg",
    "lasana_suture",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 040 only after all LASANA frames are local")
    parser.add_argument("--base-dir", default=".", help="Repository base directory on Contabo")
    parser.add_argument(
        "--scores-dir",
        default="memory/scores/lasana",
        help="LASANA score directory used to infer expected local frame trees",
    )
    parser.add_argument(
        "--frames-root",
        default="data/external/lasana_processed/frames",
        help="Local LASANA frames root on Contabo",
    )
    parser.add_argument(
        "--prepare-script",
        default="scripts/040_prepare_training_data.py",
        help="Training data builder to trigger once LASANA is fully local",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to invoke the dataset builder",
    )
    parser.add_argument(
        "--dataset-version",
        required=True,
        help="Version tag passed to 040 via --ver once all LASANA tasks are present",
    )
    parser.add_argument(
        "--state-path",
        default="data/training/.lasana_prepare_state.json",
        help="Marker written after a successful LASANA-only 040 build",
    )
    parser.add_argument(
        "--prepare-arg",
        action="append",
        default=[],
        help="Extra argument appended verbatim to the 040 command; may be repeated",
    )
    parser.add_argument("--watch", action="store_true", help="Poll continuously")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Polling interval for --watch")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def task_slug_for(payload: dict[str, Any]) -> str | None:
    metadata = payload.get("metadata") or {}
    for candidate in (
        payload.get("task"),
        payload.get("task_id"),
        metadata.get("task_id"),
        metadata.get("task"),
    ):
        if isinstance(candidate, str) and candidate.startswith("lasana_"):
            return candidate
    return None


def video_id_for(payload: dict[str, Any]) -> str | None:
    candidate = payload.get("video_id")
    if isinstance(candidate, str) and candidate:
        return candidate
    return None


def count_frames(video_dir: Path) -> int:
    return sum(1 for _ in video_dir.glob("frame_*.jpg"))


def load_expected_video_ids(scores_dir: Path) -> dict[str, set[str]]:
    expected = {task_slug: set() for task_slug in TASK_SLUGS}
    if not scores_dir.is_dir():
        return expected
    for path in sorted(scores_dir.rglob("*.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        task_slug = task_slug_for(payload)
        video_id = video_id_for(payload)
        if task_slug in expected and video_id:
            expected[task_slug].add(video_id)
    return expected


def build_task_summary(scores_dir: Path, frames_root: Path) -> dict[str, dict[str, Any]]:
    expected = load_expected_video_ids(scores_dir)
    summary: dict[str, dict[str, Any]] = {}
    for task_slug, video_ids in expected.items():
        ready = sorted(
            video_id
            for video_id in video_ids
            if (frames_root / video_id).is_dir() and count_frames(frames_root / video_id) > 0
        )
        missing = sorted(video_id for video_id in video_ids if video_id not in ready)
        summary[task_slug] = {
            "expected": len(video_ids),
            "missing": missing,
            "ready": len(ready),
        }
    return summary


def all_tasks_ready(summary: dict[str, dict[str, Any]]) -> bool:
    for task_slug in TASK_SLUGS:
        task_summary = summary.get(task_slug) or {"expected": 0, "ready": 0}
        if task_summary["expected"] == 0 or task_summary["ready"] != task_summary["expected"]:
            return False
    return True


def build_prepare_command(
    *,
    python_bin: str,
    prepare_script: Path,
    base_dir: Path,
    frames_root: Path,
    dataset_version: str,
    extra_args: list[str],
) -> list[str]:
    return [
        python_bin,
        str(prepare_script),
        "--base-dir",
        str(base_dir),
        "--ver",
        dataset_version,
        "--include-sources",
        "lasana",
        "--respect-existing-splits",
        "--frames-dir",
        str(frames_root),
        *extra_args,
    ]


def scan_once(
    *,
    base_dir: Path,
    scores_dir: Path,
    frames_root: Path,
    prepare_script: Path,
    python_bin: str,
    dataset_version: str,
    state_path: Path,
    extra_args: list[str],
) -> tuple[bool, dict[str, dict[str, Any]]]:
    summary = build_task_summary(scores_dir, frames_root)
    if state_path.exists():
        return False, summary
    if not all_tasks_ready(summary):
        return False, summary

    cmd = build_prepare_command(
        python_bin=python_bin,
        prepare_script=prepare_script,
        base_dir=base_dir,
        frames_root=frames_root,
        dataset_version=dataset_version,
        extra_args=extra_args,
    )
    LOG.info("All four LASANA task trees are local; triggering 040 for %s", dataset_version)
    result = subprocess.run(cmd, check=False)
    payload = {
        "completed_at": utc_now(),
        "command": cmd,
        "dataset_version": dataset_version,
        "frames_root": str(frames_root),
        "scores_dir": str(scores_dir),
        "status": "prepared" if result.returncode == 0 else "failed",
        "task_summary": summary,
    }
    write_json(state_path, payload)
    return result.returncode == 0, summary


def run() -> int:
    args = parse_args()
    configure_logging()

    base_dir = Path(args.base_dir).expanduser().resolve()
    scores_dir = (base_dir / args.scores_dir).resolve()
    frames_root = (base_dir / args.frames_root).resolve()
    prepare_script = (base_dir / args.prepare_script).resolve()
    state_path = (base_dir / args.state_path).resolve()

    if not prepare_script.is_file():
        raise SystemExit(f"prepare-script does not exist: {prepare_script}")

    while True:
        triggered, summary = scan_once(
            base_dir=base_dir,
            scores_dir=scores_dir,
            frames_root=frames_root,
            prepare_script=prepare_script,
            python_bin=args.python_bin,
            dataset_version=args.dataset_version,
            state_path=state_path,
            extra_args=args.prepare_arg,
        )
        if not args.watch:
            if triggered and load_json(state_path).get("status") == "failed":
                return 1
            return 0
        if not triggered:
            task_bits = ", ".join(
                f"{task}:{summary.get(task, {}).get('ready', 0)}/{summary.get(task, {}).get('expected', 0)}"
                for task in TASK_SLUGS
            )
            LOG.info("Waiting for complete LASANA frame tree on Contabo (%s); sleeping %ss", task_bits, args.poll_seconds)
        time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        LOG.info("Stopped by user")
        raise SystemExit(130)