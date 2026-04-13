from __future__ import annotations

from src.scoring.frontier_scorer import _extract_openai_text, _normalize_phase_value, _prepare_scoring_payload


def test_normalize_phase_value_maps_task_specific_labels():
    assert _normalize_phase_value("setup") == "idle"
    assert _normalize_phase_value("loop_deploy") == "needle_load"
    assert _normalize_phase_value("transfer_midair") == "suture_placement"
    assert _normalize_phase_value("knot_formation") == "first_throw"
    assert _normalize_phase_value("final_knot_tightening") == "third_throw"


def test_prepare_scoring_payload_coerces_none_score_component_and_phases():
    payload = _prepare_scoring_payload(
        {
            "score_components": {
                "time_score": None,
                "penalty_deductions": None,
                "total_fls_score": None,
            },
            "frame_analyses": [
                {"frame_number": 1, "phase": "setup", "description": "intro"},
                {"frame_number": 2, "phase": "reverse_transfer", "description": "move"},
            ],
            "phase_timings": [
                {
                    "phase": "final_knot_tightening",
                    "start_seconds": 0,
                    "end_seconds": 1,
                    "duration_seconds": 1,
                }
            ],
            "completion_time_seconds": None,
            "estimated_penalties": None,
            "estimated_fls_score": None,
            "confidence_score": None,
        },
        task=3,
    )

    assert payload["score_components"]["total_fls_score"] == 0.0
    assert payload["frame_analyses"][0]["phase"] == "idle"
    assert payload["frame_analyses"][1]["phase"] == "suture_placement"
    assert payload["phase_timings"][0]["phase"] == "third_throw"


def test_extract_openai_text_supports_string_and_part_lists():
    assert _extract_openai_text('{"ok": true}') == '{"ok": true}'
    assert _extract_openai_text([
        {"type": "output_text", "text": '{"a": 1}'},
        {"type": "output_text", "text": '{"b": 2}'},
    ]) == '{"a": 1}\n{"b": 2}'