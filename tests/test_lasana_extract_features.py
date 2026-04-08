from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "068_lasana_extract_features.py"
    spec = importlib.util.spec_from_file_location("lasana_extract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_phase1_frames_accepts_task_qualified_trial_layout(monkeypatch, tmp_path):
    module = _load_module()

    lasana_dir = tmp_path / "layout"
    clip_dir = lasana_dir / "lasana_suture_kiourf"
    clip_dir.mkdir(parents=True)
    (clip_dir / "video.hevc").write_bytes(b"fake-hevc")

    def fake_decode(bitstream, out_dir, fps, left_only=True):
        del bitstream, fps, left_only
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "frame_0001.jpg").write_bytes(b"jpg")
        (out_dir / "frame_0002.jpg").write_bytes(b"jpg")
        return (2, 0)

    monkeypatch.setattr(module, "decode_one_trial", fake_decode)

    manifest_path = module.phase1_frames(
        SimpleNamespace(
            lasana_dir=str(lasana_dir),
            out_dir=str(tmp_path / "processed"),
            fps=1.0,
            max_trials=0,
        )
    )

    with manifest_path.open() as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["video_id"] == "lasana_suture_kiourf"
    assert rows[0]["bitstream"] == "lasana_suture_kiourf/video.hevc"
    assert rows[0]["frames_dir"] == "frames/lasana_suture_kiourf"
