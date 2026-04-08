#!/usr/bin/env python3
"""Thin streaming orchestrator for the LASANA archive -> frames -> dataset flow.

This wrapper keeps the existing stage scripts intact and wires them together
for the two live hosts:

- Hetzner: run 070 for its assigned tasks, feed completed archives through
  071, extract frames via 068, then rsync newly extracted frame directories
  to Contabo.
- Contabo: run 070 for SutureAndKnot, feed completed archives through 071,
  extract local frames via 068, and trigger 040 once all score-backed LASANA
  frame directories are present locally.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


LOG = logging.getLogger("lasana_stream_watch")
DEFAULT_REQUIRED_TASKS = (
    "lasana_balloon",
    "lasana_circle",
    "lasana_peg",
    "lasana_suture",
)
STATE_FILENAME = "watch_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch and stream the LASANA ingest pipeline")
    parser.add_argument("--host-role", choices=["contabo", "hetzner"], required=True)
    parser.add_argument("--base-dir", default=".", help="Repository root")
    parser.add_argument("--raw-dir", required=True, help="Directory where 070 writes *.zip archives")
    parser.add_argument("--layout-dir", required=True, help="Directory where 071 emits <video_id>/video.hevc")
    parser.add_argument("--processed-dir", required=True, help="Directory where 068 writes frames/ and manifest.csv")
    parser.add_argument("--watch", action="store_true", help="Poll indefinitely instead of running one scan")
    parser.add_argument("--poll-seconds", type=int, default=30, help="Polling interval for --watch")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame extraction rate passed to 068")
    parser.add_argument("--max-archives", type=int, default=0, help="Limit archives processed by 071 per scan")
    parser.add_argument("--overwrite-layout", action="store_true", help="Pass overwrite=True to 071 archive extraction")

    parser.add_argument("--run-downloader", action="store_true", help="Invoke 070 before each scan")
    parser.add_argument(
        "--manifest-path",
        default="data/external/lasana/_meta/bitstreams.json",
        help="Path to the checked-in HAL manifest used by 070",
    )
    parser.add_argument(
        "--download-task",
        action="append",
        default=[],
        help="Task name to pass to 070; repeat for multiple tasks",
    )
    parser.add_argument("--resume-downloads", action="store_true", help="Pass --resume to 070")

    parser.add_argument("--rsync-dest", default=None, help="Remote rsync destination for extracted frames")
    parser.add_argument("--rsync-bin", default="rsync", help="rsync binary to invoke")
    parser.add_argument(
        "--rsync-extra-arg",
        action="append",
        default=[],
        help="Extra argument to append to each rsync invocation",
    )

    parser.add_argument(
        "--prepare-when-ready",
        action="store_true",
        help="On Contabo, invoke 040 once all required LASANA frame dirs are present locally",
    )
    parser.add_argument("--prepare-ver", default="lasana_v1", help="Dataset version tag for 040")
    parser.add_argument("--prepare-output-dir", default="data/training", help="Output directory for 040")
    parser.add_argument("--prepare-min-confidence", type=float, default=0.5)
    parser.add_argument("--prepare-max-frames", type=int, default=24)
    parser.add_argument(
        "--prepare-group-by",
        choices=["trainee", "video"],
        default="video",
        help="group-by passed to 040 if 040 is triggered",
    )
    parser.add_argument(
        "--required-task",
        action="append",
        default=list(DEFAULT_REQUIRED_TASKS),
        help="Required LASANA task ids used by the Contabo 040 gate",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def resolve_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def load_script_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_complete_frame_video_ids(frames_root: Path) -> set[str]:
    if not frames_root.is_dir():
        return set()
    complete: set[str] = set()
    for entry in frames_root.iterdir():
        if entry.is_dir() and any(entry.glob("frame_*.jpg")):
            complete.add(entry.name)
    return complete


def load_state(processed_dir: Path) -> dict[str, Any]:
    state_path = processed_dir / ".stream_state" / STATE_FILENAME
    if not state_path.is_file():
        return {"synced_video_ids": [], "prepare_completed": False}
    try:
        payload = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"synced_video_ids": [], "prepare_completed": False}
    payload.setdefault("synced_video_ids", [])
    payload.setdefault("prepare_completed", False)
    return payload


def save_state(processed_dir: Path, state: dict[str, Any]) -> None:
    state_path = processed_dir / ".stream_state" / STATE_FILENAME
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def run_downloader_once(args: argparse.Namespace, base_dir: Path, runner=subprocess.run) -> None:
    if not args.run_downloader:
        return

    downloader = base_dir / "scripts" / "070_lasana_download.py"
    tasks = args.download_task or [None]
    for task in tasks:
        cmd = [
            sys.executable,
            str(downloader),
            "--manifest-path",
            str(resolve_path(base_dir, args.manifest_path)),
            "--out-dir",
            str(resolve_path(base_dir, args.raw_dir)),
        ]
        if args.resume_downloads:
            cmd.append("--resume")
        if task:
            cmd.extend(["--task", task])

        LOG.info("Running downloader: %s", " ".join(cmd))
        completed = runner(cmd, cwd=str(base_dir), check=False)
        if completed.returncode not in {0, 1}:
            raise RuntimeError(f"070_lasana_download.py failed with rc={completed.returncode}")


def process_archives_once(layout_module, raw_dir: Path, layout_dir: Path, args: argparse.Namespace) -> int:
    return int(
        layout_module.process_available_archives(
            raw_dir=raw_dir,
            out_dir=layout_dir,
            task_filter=None,
            max_archives=args.max_archives,
            overwrite=args.overwrite_layout,
        )
    )


def extract_frames_once(extractor_module, layout_dir: Path, processed_dir: Path, fps: float) -> list[str]:
    before = collect_complete_frame_video_ids(processed_dir / "frames")
    extractor_module.phase1_frames(
        SimpleNamespace(
            lasana_dir=str(layout_dir),
            out_dir=str(processed_dir),
            fps=fps,
            max_trials=0,
        )
    )
    after = collect_complete_frame_video_ids(processed_dir / "frames")
    return sorted(after - before)


def sync_frame_dirs(
    *,
    frames_root: Path,
    rsync_dest: str,
    synced_video_ids: set[str],
    rsync_bin: str,
    rsync_extra_args: list[str],
    runner=subprocess.run,
) -> list[str]:
    completed = collect_complete_frame_video_ids(frames_root)
    pending = sorted(completed - synced_video_ids)
    synced_now: list[str] = []
    for video_id in pending:
        src_dir = frames_root / video_id
        cmd = [rsync_bin, "-az", "--partial", *rsync_extra_args, f"{src_dir}/", f"{rsync_dest.rstrip('/')}/{video_id}/"]
        LOG.info("Syncing frames for %s", video_id)
        completed = runner(cmd, check=False)
        if completed.returncode == 0:
            synced_now.append(video_id)
        else:
            LOG.warning("rsync failed for %s with rc=%d", video_id, completed.returncode)
    return synced_now


def load_expected_video_ids(base_dir: Path, required_tasks: list[str]) -> dict[str, set[str]]:
    scores_dir = base_dir / "memory" / "scores" / "lasana"
    expected = {task: set() for task in required_tasks}
    if not scores_dir.is_dir():
        return expected

    for path in scores_dir.rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("source") or "") != "lasana":
            continue
        task_id = str(payload.get("task_id") or ((payload.get("metadata") or {}).get("task_id") or "")).strip()
        video_id = str(payload.get("video_id") or "").strip()
        if task_id in expected and video_id:
            expected[task_id].add(video_id)
    return expected


def assess_prepare_readiness(base_dir: Path, frames_root: Path, required_tasks: list[str]) -> dict[str, Any]:
    expected = load_expected_video_ids(base_dir, required_tasks)
    local_frames = collect_complete_frame_video_ids(frames_root)
    tasks: dict[str, dict[str, int]] = {}
    ready = True

    for task in required_tasks:
        task_expected = expected.get(task, set())
        present = len(task_expected & local_frames)
        total = len(task_expected)
        missing = max(total - present, 0)
        tasks[task] = {"expected": total, "present": present, "missing": missing}
        if total == 0 or missing > 0:
            ready = False

    return {
        "ready": ready,
        "tasks": tasks,
        "local_frame_dirs": len(local_frames),
    }


def build_prepare_command(args: argparse.Namespace, base_dir: Path, processed_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(base_dir / "scripts" / "040_prepare_training_data.py"),
        "--base-dir",
        str(base_dir),
        "--ver",
        args.prepare_ver,
        "--output-dir",
        str(resolve_path(base_dir, args.prepare_output_dir)),
        "--include-sources",
        "lasana",
        "--respect-existing-splits",
        "--frames-dir",
        str(processed_dir / "frames"),
        "--max-frames",
        str(args.prepare_max_frames),
        "--min-confidence",
        str(args.prepare_min_confidence),
        "--group-by",
        args.prepare_group_by,
    ]


def maybe_run_prepare(
    *,
    args: argparse.Namespace,
    base_dir: Path,
    processed_dir: Path,
    state: dict[str, Any],
    readiness: dict[str, Any],
    runner=subprocess.run,
) -> bool:
    if not args.prepare_when_ready or state.get("prepare_completed"):
        return False
    if not readiness.get("ready"):
        return False

    cmd = build_prepare_command(args, base_dir, processed_dir)
    LOG.info("All LASANA frame trees are local; running 040: %s", " ".join(cmd))
    completed = runner(cmd, cwd=str(base_dir), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"040_prepare_training_data.py failed with rc={completed.returncode}")

    state["prepare_completed"] = True
    state["prepare_command"] = cmd
    state["prepare_completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return True


def validate_args(args: argparse.Namespace) -> None:
    if args.host_role == "hetzner" and args.prepare_when_ready:
        raise SystemExit("Hetzner must not run 040; use --prepare-when-ready only on Contabo")
    if args.host_role == "hetzner" and not args.rsync_dest:
        raise SystemExit("Hetzner mode requires --rsync-dest so extracted frames reach Contabo")


def run() -> int:
    args = parse_args()
    configure_logging()
    validate_args(args)

    base_dir = resolve_path(Path.cwd(), args.base_dir)
    raw_dir = resolve_path(base_dir, args.raw_dir)
    layout_dir = resolve_path(base_dir, args.layout_dir)
    processed_dir = resolve_path(base_dir, args.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    layout_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    extractor_module = load_script_module(base_dir / "scripts" / "068_lasana_extract_features.py", "lasana_extract")
    layout_module = load_script_module(base_dir / "scripts" / "071_lasana_unzip_and_layout.py", "lasana_layout")
    state = load_state(processed_dir)

    while True:
        run_downloader_once(args, base_dir)
        processed_archives = process_archives_once(layout_module, raw_dir, layout_dir, args)
        new_frames = extract_frames_once(extractor_module, layout_dir, processed_dir, args.fps)
        if new_frames:
            LOG.info("Extracted %d new LASANA frame directories", len(new_frames))

        if args.host_role == "hetzner" and args.rsync_dest:
            synced_video_ids = set(state.get("synced_video_ids", []))
            synced_now = sync_frame_dirs(
                frames_root=processed_dir / "frames",
                rsync_dest=args.rsync_dest,
                synced_video_ids=synced_video_ids,
                rsync_bin=args.rsync_bin,
                rsync_extra_args=args.rsync_extra_arg,
            )
            if synced_now:
                state["synced_video_ids"] = sorted(synced_video_ids | set(synced_now))
                save_state(processed_dir, state)
        else:
            readiness = assess_prepare_readiness(base_dir, processed_dir / "frames", args.required_task)
            LOG.info("Contabo readiness: %s", readiness["tasks"])
            if maybe_run_prepare(
                args=args,
                base_dir=base_dir,
                processed_dir=processed_dir,
                state=state,
                readiness=readiness,
            ):
                save_state(processed_dir, state)

        if not args.watch:
            return 0
        if processed_archives == 0 and not new_frames:
            LOG.info("No new LASANA work detected; sleeping %ss", max(args.poll_seconds, 1))
        time.sleep(max(args.poll_seconds, 1))


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        LOG.info("Stopped by user")
        raise SystemExit(130)