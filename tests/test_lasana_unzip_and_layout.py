from __future__ import annotations

import importlib.util
from pathlib import Path
from zipfile import ZipFile


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "071_lasana_unzip_and_layout.py"
    spec = importlib.util.spec_from_file_location("lasana_layout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_archive_task_qualifies_member_layout(tmp_path):
    module = _load_module()

    archive_path = tmp_path / "SutureAndKnot_left.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/kiourf.h265", b"synthetic-hevc-bytes")

    out_dir = tmp_path / "layout"
    rows = module.extract_archive(archive_path, out_dir)

    destination = out_dir / "lasana_suture_kiourf" / "video.hevc"
    assert destination.read_bytes() == b"synthetic-hevc-bytes"
    assert len(rows) == 1
    assert rows[0]["archive"] == "SutureAndKnot_left.zip"
    assert rows[0]["task_name"] == "SutureAndKnot"
    assert rows[0]["trial_id"] == "kiourf"
    assert rows[0]["video_id"] == "lasana_suture_kiourf"
    assert rows[0]["status"] == "extracted"