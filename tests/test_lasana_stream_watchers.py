from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


def _load_module(script_name: str, module_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frame_watch_extracts_one_video_and_writes_receipts(monkeypatch, tmp_path):
    module = _load_module("072_lasana_frame_watch.py", "lasana_frame_watch")

    layout_dir = tmp_path / "layout"
    video_dir = layout_dir / "lasana_suture_kiourf"
    video_dir.mkdir(parents=True)
    (video_dir / "video.hevc").write_bytes(b"hevc")

    out_dir = tmp_path / "processed"
    state_dir = tmp_path / "state"

    def fake_run(cmd, check=False):
        del check
        assert cmd[0]
        frames_dir = out_dir / "frames" / "lasana_suture_kiourf"
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "frame_0001.jpg").write_bytes(b"jpg")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    summary = module.scan_once(
        layout_dir=layout_dir,
        out_dir=out_dir,
        state_dir=state_dir,
        video_prefixes=["lasana_suture"],
        extract_script=Path("/tmp/fake_068.py"),
        python_bin="python3",
        fps=1.0,
    )

    assert summary == {"extracted": 1, "skipped_existing": 0, "failed": 0}
    receipt = json.loads(
        (out_dir / "frames" / "lasana_suture_kiourf" / ".lasana_extract_complete.json").read_text()
    )
    assert receipt["status"] == "extracted"
    assert receipt["frame_count"] == 1
    state = json.loads((state_dir / "lasana_suture_kiourf.json").read_text())
    assert state["video_id"] == "lasana_suture_kiourf"


def test_frame_watch_marks_existing_frames_without_rerunning(monkeypatch, tmp_path):
    module = _load_module("072_lasana_frame_watch.py", "lasana_frame_watch_existing")

    layout_dir = tmp_path / "layout"
    video_dir = layout_dir / "lasana_suture_existing"
    video_dir.mkdir(parents=True)
    (video_dir / "video.hevc").write_bytes(b"hevc")

    out_dir = tmp_path / "processed"
    frames_dir = out_dir / "frames" / "lasana_suture_existing"
    frames_dir.mkdir(parents=True)
    (frames_dir / "frame_0001.jpg").write_bytes(b"jpg")
    state_dir = tmp_path / "state"

    def fail_run(*args, **kwargs):
        raise AssertionError("extractor should not run when frames already exist")

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    summary = module.scan_once(
        layout_dir=layout_dir,
        out_dir=out_dir,
        state_dir=state_dir,
        video_prefixes=[],
        extract_script=Path("/tmp/fake_068.py"),
        python_bin="python3",
        fps=1.0,
    )

    assert summary == {"extracted": 0, "skipped_existing": 1, "failed": 0}
    receipt = json.loads((frames_dir / ".lasana_extract_complete.json").read_text())
    assert receipt["status"] == "already_present"


def test_rsync_and_prepare_watchers_gate_on_completion(monkeypatch, tmp_path):
    rsync_module = _load_module("073_lasana_rsync_watch.py", "lasana_rsync_watch")
    prepare_module = _load_module("074_lasana_prepare_watch.py", "lasana_prepare_watch")

    frames_root = tmp_path / "frames"
    sync_video_dir = frames_root / "lasana_balloon_alpha"
    sync_video_dir.mkdir(parents=True)
    (sync_video_dir / "frame_0001.jpg").write_bytes(b"jpg")
    (sync_video_dir / ".lasana_extract_complete.json").write_text("{}\n")
    rsync_state_dir = tmp_path / "rsync_state"

    rsync_calls: list[list[str]] = []

    def fake_rsync(cmd, check=False):
        del check
        rsync_calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(rsync_module.subprocess, "run", fake_rsync)

    rsync_summary = rsync_module.scan_once(
        frames_root=frames_root,
        state_dir=rsync_state_dir,
        dest_host="contabo-host",
        dest_dir="/srv/fls/lasana_frames",
        video_prefixes=["lasana_balloon"],
        rsync_bin="rsync",
    )

    assert rsync_summary == {"synced": 1, "failed": 0}
    assert rsync_calls[0][-1] == "contabo-host:/srv/fls/lasana_frames/lasana_balloon_alpha/"

    base_dir = tmp_path / "repo"
    scores_dir = base_dir / "memory" / "scores" / "lasana"
    frames_root_local = base_dir / "data" / "external" / "lasana_processed" / "frames"
    prepare_script = base_dir / "scripts" / "040_prepare_training_data.py"
    prepare_script.parent.mkdir(parents=True, exist_ok=True)
    prepare_script.write_text("#!/usr/bin/env python3\n")

    for task_slug in prepare_module.TASK_SLUGS:
        video_id = f"{task_slug}_trial"
        scores_dir.mkdir(parents=True, exist_ok=True)
        (scores_dir / f"score_{video_id}.json").write_text(
            json.dumps({"video_id": video_id, "metadata": {"task_id": task_slug}})
        )
        if task_slug != "lasana_suture":
            video_frames_dir = frames_root_local / video_id
            video_frames_dir.mkdir(parents=True, exist_ok=True)
            (video_frames_dir / "frame_0001.jpg").write_bytes(b"jpg")

    prepare_calls: list[list[str]] = []

    def fake_prepare(cmd, check=False):
        del check
        prepare_calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(prepare_module.subprocess, "run", fake_prepare)

    state_path = base_dir / "data" / "training" / ".lasana_prepare_state.json"
    triggered, summary = prepare_module.scan_once(
        base_dir=base_dir,
        scores_dir=scores_dir,
        frames_root=frames_root_local,
        prepare_script=prepare_script,
        python_bin="python3",
        dataset_version="lasana_stream_v1",
        state_path=state_path,
        extra_args=[],
    )

    assert triggered is False
    assert summary["lasana_suture"]["ready"] == 0
    assert not prepare_calls

    suture_frames_dir = frames_root_local / "lasana_suture_trial"
    suture_frames_dir.mkdir(parents=True, exist_ok=True)
    (suture_frames_dir / "frame_0001.jpg").write_bytes(b"jpg")

    triggered, summary = prepare_module.scan_once(
        base_dir=base_dir,
        scores_dir=scores_dir,
        frames_root=frames_root_local,
        prepare_script=prepare_script,
        python_bin="python3",
        dataset_version="lasana_stream_v1",
        state_path=state_path,
        extra_args=["--max-frames", "24"],
    )

    assert triggered is True
    assert summary["lasana_suture"]["ready"] == 1
    assert prepare_calls
    assert prepare_calls[0][0:4] == ["python3", str(prepare_script), "--base-dir", str(base_dir)]
    state = json.loads(state_path.read_text())
    assert state["status"] == "prepared"