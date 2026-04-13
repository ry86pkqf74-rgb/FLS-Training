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


class _Process:
    def __init__(self, poll_results: list[int | None]):
        self._poll_results = list(poll_results)

    def poll(self):
        if self._poll_results:
            return self._poll_results.pop(0)
        return None


def test_build_downloader_command_includes_task_and_resume(tmp_path):
    module = _load_module()

    args = SimpleNamespace(
        manifest_path="data/external/lasana/_meta/bitstreams.json",
        raw_dir="/tmp/raw",
        resume_downloads=True,
    )

    cmd = module.build_downloader_command(args, tmp_path, "SutureAndKnot")

    assert cmd == [
        module.sys.executable,
        str(tmp_path / "scripts" / "070_lasana_download.py"),
        "--manifest-path",
        str(tmp_path / "data" / "external" / "lasana" / "_meta" / "bitstreams.json"),
        "--out-dir",
        str(Path("/tmp/raw").resolve()),
        "--resume",
        "--task",
        "SutureAndKnot",
    ]


def test_run_downloader_once_starts_background_job_in_watch_mode(tmp_path):
    module = _load_module()

    args = SimpleNamespace(
        run_downloader=True,
        watch=True,
        download_task=["SutureAndKnot"],
        manifest_path="data/external/lasana/_meta/bitstreams.json",
        raw_dir="raw",
        resume_downloads=True,
    )
    active_downloads = {}
    popen_calls: list[tuple[list[str], str | None]] = []

    def fake_popen(cmd, cwd=None):
        popen_calls.append((cmd, cwd))
        return _Process([None])

    def fail_runner(*args, **kwargs):
        raise AssertionError("runner should not be used in watch mode")

    module.run_downloader_once(
        args,
        tmp_path,
        runner=fail_runner,
        popen=fake_popen,
        active_downloads=active_downloads,
    )

    assert list(active_downloads) == ["SutureAndKnot"]
    assert popen_calls == [(
        [
            module.sys.executable,
            str(tmp_path / "scripts" / "070_lasana_download.py"),
            "--manifest-path",
            str(tmp_path / "data" / "external" / "lasana" / "_meta" / "bitstreams.json"),
            "--out-dir",
            str(tmp_path / "raw"),
            "--resume",
            "--task",
            "SutureAndKnot",
        ],
        str(tmp_path),
    )]


def test_run_downloader_once_does_not_duplicate_running_background_job(tmp_path):
    module = _load_module()

    args = SimpleNamespace(
        run_downloader=True,
        watch=True,
        download_task=["SutureAndKnot"],
        manifest_path="data/external/lasana/_meta/bitstreams.json",
        raw_dir="raw",
        resume_downloads=False,
    )
    active_downloads = {"SutureAndKnot": _Process([None])}

    def fail_popen(*args, **kwargs):
        raise AssertionError("popen should not be called while downloader is still running")

    module.run_downloader_once(
        args,
        tmp_path,
        popen=fail_popen,
        active_downloads=active_downloads,
    )

    assert list(active_downloads) == ["SutureAndKnot"]


def test_run_downloader_once_restarts_background_job_after_success(tmp_path):
    module = _load_module()

    args = SimpleNamespace(
        run_downloader=True,
        watch=True,
        download_task=["SutureAndKnot"],
        manifest_path="data/external/lasana/_meta/bitstreams.json",
        raw_dir="raw",
        resume_downloads=False,
    )
    active_downloads = {"SutureAndKnot": _Process([0])}
    spawned: list[_Process] = []

    def fake_popen(cmd, cwd=None):
        del cmd, cwd
        proc = _Process([None])
        spawned.append(proc)
        return proc

    module.run_downloader_once(
        args,
        tmp_path,
        popen=fake_popen,
        active_downloads=active_downloads,
    )

    assert active_downloads["SutureAndKnot"] is spawned[0]


def test_run_downloader_once_raises_on_background_failure(tmp_path):
    module = _load_module()

    args = SimpleNamespace(
        run_downloader=True,
        watch=True,
        download_task=["SutureAndKnot"],
        manifest_path="data/external/lasana/_meta/bitstreams.json",
        raw_dir="raw",
        resume_downloads=False,
    )
    active_downloads = {"SutureAndKnot": _Process([2])}

    try:
        module.run_downloader_once(args, tmp_path, active_downloads=active_downloads)
    except RuntimeError as exc:
        assert "rc=2" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


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