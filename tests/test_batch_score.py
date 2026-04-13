from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "021_batch_score.py"
    spec = importlib.util.spec_from_file_location("batch_score", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_task_number_prefers_harvest_targets_csv():
    module = _load_module()

    task_num, source = module.resolve_task_number(
        "abc123",
        "task5_intracorporeal_suture",
        {"abc123": "task2_pattern_cut"},
        4,
        "This looks like extracorporeal knot tying.",
    )

    assert task_num == 2
    assert source == "harvest_targets.csv:task2_pattern_cut"


def test_parse_teacher_score_filename_supports_current_and_legacy_layouts(tmp_path):
    module = _load_module()

    current_claude = tmp_path / "score_claude_yt_Example123_20260408010101.json"
    current_gpt = tmp_path / "score_gpt_yt_Example123_20260408010101.json"
    legacy = tmp_path / "yt_Example123_claude-sonnet-4_20260408010101.json"

    assert module.parse_teacher_score_filename(current_claude) == ("yt_Example123", "claude-sonnet-4")
    assert module.parse_teacher_score_filename(current_gpt) == ("yt_Example123", "gpt-4o")
    assert module.parse_teacher_score_filename(legacy) == ("yt_Example123", "claude-sonnet-4")


def test_resolve_local_video_path_repairs_stale_harvest_log_paths(tmp_path, monkeypatch):
    module = _load_module()

    harvest_root = tmp_path / "fls_harvested_videos"
    harvest_root.mkdir()
    local_video = harvest_root / "IwrNTRVXuJQ.f136.mp4"
    local_video.write_bytes(b"video")

    monkeypatch.setattr(module, "HARVEST_VIDEO_ROOTS", (harvest_root,))

    index = module.build_local_video_index()
    resolved = module.resolve_local_video_path(
        {
            "video_id": "IwrNTRVXuJQ",
            "filepath": "/Users/loganglosser/fls_harvested_videos/task3_endoloop/IwrNTRVXuJQ.mp4",
        },
        index,
    )

    assert resolved == local_video


def test_infer_task_from_technique_summary_uses_keywords():
    module = _load_module()

    assert module.infer_task_from_technique_summary("Clean peg transfer with outbound and return pass") == 1
    assert module.infer_task_from_technique_summary("Strong pattern cut but drifted outside the circle") == 2
    assert module.infer_task_from_technique_summary("The endoloop cinch was delayed but successful") == 3
    assert module.infer_task_from_technique_summary("Good extracorporeal tying with knot pusher control") == 4
    assert module.infer_task_from_technique_summary("Efficient intracorporeal suturing and square knots") == 5


def test_all_recorded_teacher_scores_zero_requires_all_models_zero(tmp_path):
    module = _load_module()

    zero_path = tmp_path / "zero.json"
    positive_path = tmp_path / "positive.json"
    zero_path.write_text(json.dumps({"estimated_fls_score": 0.0}), encoding="utf-8")
    positive_path.write_text(json.dumps({"estimated_fls_score": 2.5}), encoding="utf-8")

    assert module.all_recorded_teacher_scores_zero(
        "vid_zero",
        {"vid_zero": {"claude-sonnet-4": zero_path, "gpt-4o": zero_path}},
    )
    assert not module.all_recorded_teacher_scores_zero(
        "vid_mixed",
        {"vid_mixed": {"claude-sonnet-4": zero_path, "gpt-4o": positive_path}},
    )


def test_teacher_models_needing_task_rescore_detects_mismatches(tmp_path):
    module = _load_module()

    task1 = tmp_path / "task1.json"
    task3 = tmp_path / "task3.json"
    missing = tmp_path / "missing.json"
    task1.write_text(json.dumps({"task_id": "task1", "estimated_fls_score": 10.0}), encoding="utf-8")
    task3.write_text(json.dumps({"task_id": "task3", "estimated_fls_score": 12.0}), encoding="utf-8")
    missing.write_text(json.dumps({"estimated_fls_score": 0.0}), encoding="utf-8")

    assert module.teacher_models_needing_task_rescore(
        "vid",
        1,
        {"vid": {"claude-sonnet-4": task1, "gpt-4o": task3}},
    ) == {"gpt-4o"}
    assert module.teacher_models_needing_task_rescore(
        "vid_missing",
        2,
        {"vid_missing": {"claude-sonnet-4": missing}},
    ) == {"claude-sonnet-4"}


def test_load_frames_cache_reads_cached_frames(tmp_path, monkeypatch):
    module = _load_module()

    frames_root = tmp_path / "frames"
    cache_dir = frames_root / "abc123"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / "frames.json"
    cache_path.write_text(
        json.dumps(
            {
                "video_id": "abc123",
                "frames_b64": ["one", "two"],
                "frame_timestamps": [0.0, 1.0],
                "metadata": {"fps": 30},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "FRAMES_DIR", frames_root)

    assert module._load_frames_cache("abc123") == (["one", "two"], {"fps": 30})


def test_detect_content_task_mismatch_flags_wrong_task_language(tmp_path):
    module = _load_module()

    score_path = tmp_path / "score.json"
    score_path.write_text(
        json.dumps(
            {
                "technique_summary": "Video demonstrates competent extracorporeal suturing technique but represents wrong task entirely.",
                "penalties": [
                    {
                        "description": "Video shows extracorporeal suturing (Task 4) rather than peg transfer (Task 1)",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert module.detect_content_task_mismatch(1, score_path) == "explicit task mismatch language"


def test_detect_content_task_mismatch_flags_instructional_content(tmp_path):
    module = _load_module()

    score_path = tmp_path / "score.json"
    score_path.write_text(
        json.dumps(
            {
                "technique_summary": "This appears to be an educational demonstration rather than a timed performance attempt.",
            }
        ),
        encoding="utf-8",
    )

    assert module.detect_content_task_mismatch(3, score_path) == "instructional/demo content"