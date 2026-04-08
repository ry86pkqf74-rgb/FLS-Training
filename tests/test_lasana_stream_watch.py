from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "072_lasana_stream_watch.py"
    spec = importlib.util.spec_from_file_location("lasana_stream_watch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Completed:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode


def test_extract_frames_once_detects_new_video_ids(tmp_path):
    module = _load_module()

    layout_dir = tmp_path / "layout"
    processed_dir = tmp_path / "processed"
    (layout_dir / "lasana_suture_kiourf").mkdir(parents=True)

    class FakeExtractor:
        @staticmethod
        def phase1_frames(args):
            assert args.lasana_dir == str(layout_dir)
            assert args.out_dir == str(processed_dir)
            frame_dir = processed_dir / "frames" / "lasana_suture_kiourf"
            frame_dir.mkdir(parents=True, exist_ok=True)
            (frame_dir / "frame_0001.jpg").write_bytes(b"jpg")

    new_frames = module.extract_frames_once(FakeExtractor, layout_dir, processed_dir, fps=1.0)

    assert new_frames == ["lasana_suture_kiourf"]


def test_sync_frame_dirs_only_pushes_unsynced_video_ids(tmp_path):
    module = _load_module()

    frames_root = tmp_path / "frames"
    for video_id in ("lasana_balloon_a", "lasana_circle_b"):
        frame_dir = frames_root / video_id
        frame_dir.mkdir(parents=True)
        (frame_dir / "frame_0001.jpg").write_bytes(b"jpg")

    commands: list[list[str]] = []

    def fake_runner(cmd, check=False):
        del check
        commands.append(cmd)
        return _Completed(0)

    synced_now = module.sync_frame_dirs(
        frames_root=frames_root,
        rsync_dest="root@contabo:/data/fls/lasana_processed/frames",
        synced_video_ids={"lasana_balloon_a"},
        rsync_bin="rsync",
        rsync_extra_args=["--mkpath"],
        runner=fake_runner,
    )

    assert synced_now == ["lasana_circle_b"]
    assert commands == [[
        "rsync",
        "-az",
        "--partial",
        "--mkpath",
        f"{frames_root / 'lasana_circle_b'}/",
        "root@contabo:/data/fls/lasana_processed/frames/lasana_circle_b/",
    ]]


def test_assess_prepare_readiness_requires_all_score_backed_tasks(tmp_path):
    module = _load_module()

    base_dir = tmp_path / "repo"
    scores_dir = base_dir / "memory" / "scores" / "lasana"
    frames_root = base_dir / "data" / "external" / "lasana_processed" / "frames"
    scores_dir.mkdir(parents=True)
    frames_root.mkdir(parents=True)

    payloads = {
        "lasana_balloon": "lasana_balloon_alpha",
        "lasana_circle": "lasana_circle_beta",
        "lasana_peg": "lasana_peg_gamma",
        "lasana_suture": "lasana_suture_delta",
    }
    for task_id, video_id in payloads.items():
        (scores_dir / f"{video_id}.json").write_text(json.dumps({"source": "lasana", "task_id": task_id, "video_id": video_id}))

    for video_id in list(payloads.values())[:3]:
        frame_dir = frames_root / video_id
        frame_dir.mkdir(parents=True)
        (frame_dir / "frame_0001.jpg").write_bytes(b"jpg")

    readiness = module.assess_prepare_readiness(base_dir, frames_root, list(payloads))
    assert readiness["ready"] is False
    assert readiness["tasks"]["lasana_suture"]["missing"] == 1

    final_dir = frames_root / payloads["lasana_suture"]
    final_dir.mkdir(parents=True)
    (final_dir / "frame_0001.jpg").write_bytes(b"jpg")

    readiness = module.assess_prepare_readiness(base_dir, frames_root, list(payloads))
    assert readiness["ready"] is True
    assert readiness["tasks"]["lasana_suture"] == {"expected": 1, "present": 1, "missing": 0}


def test_maybe_run_prepare_runs_040_once_when_ready(tmp_path):
    module = _load_module()

    base_dir = tmp_path / "repo"
    processed_dir = base_dir / "processed"
    processed_dir.mkdir(parents=True)
    args = SimpleNamespace(
        prepare_when_ready=True,
        prepare_ver="lasana_stream_v1",
        prepare_output_dir="data/training",
        prepare_max_frames=24,
        prepare_min_confidence=0.5,
        prepare_group_by="video",
    )
    state = {"prepare_completed": False, "synced_video_ids": []}
    commands: list[list[str]] = []

    def fake_runner(cmd, cwd=None, check=False):
        del cwd, check
        commands.append(cmd)
        return _Completed(0)

    ran = module.maybe_run_prepare(
        args=args,
        base_dir=base_dir,
        processed_dir=processed_dir,
        state=state,
        readiness={"ready": True, "tasks": {}},
        runner=fake_runner,
    )

    assert ran is True
    assert state["prepare_completed"] is True
    assert commands == [[
        module.sys.executable,
        str(base_dir / "scripts" / "040_prepare_training_data.py"),
        "--base-dir",
        str(base_dir),
        "--ver",
        "lasana_stream_v1",
        "--output-dir",
        str(base_dir / "data" / "training"),
        "--include-sources",
        "lasana",
        "--respect-existing-splits",
        "--frames-dir",
        str(processed_dir / "frames"),
        "--max-frames",
        "24",
        "--min-confidence",
        "0.5",
        "--group-by",
        "video",
    ]]