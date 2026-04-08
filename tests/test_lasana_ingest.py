from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "069_ingest_lasana_to_store.py"
    spec = importlib.util.spec_from_file_location("lasana_ingest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_qualified_identity_prevents_cross_task_collision():
    module = _load_module()

    balloon_id = module.deterministic_score_id("BalloonResection", "kiourf")
    suture_id = module.deterministic_score_id("SutureAndKnot", "kiourf")
    balloon_video = module.deterministic_video_id("BalloonResection", "kiourf")
    suture_video = module.deterministic_video_id("SutureAndKnot", "kiourf")
    balloon_frame_dir = module.frame_dir_name("BalloonResection", "kiourf")
    suture_frame_dir = module.frame_dir_name("SutureAndKnot", "kiourf")

    assert balloon_id != suture_id
    assert balloon_video != suture_video
    assert balloon_frame_dir != suture_frame_dir
    assert balloon_id == "score_lasana_balloon_kiourf"
    assert suture_id == "score_lasana_suture_kiourf"