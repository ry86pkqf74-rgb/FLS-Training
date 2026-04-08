from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import src.training.prepare_dataset as dataset_module


class DummyLog:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def append_event(self, event_type: str, payload: dict) -> None:
        self.events.append((event_type, payload))


class FixedDatetime:
    @staticmethod
    def now(tz=None):
        return datetime(2026, 4, 8, 12, 0, 0, tzinfo=tz or timezone.utc)


def _candidate(
    *,
    video_id: str,
    source: str,
    task_id: str,
    confidence: float = 0.9,
    split: str | None = None,
) -> dict:
    raw_json = {
        "id": f"score_{video_id}",
        "video_id": video_id,
        "source": source,
        "task_id": task_id,
        "estimated_fls_score": 4.0,
        "confidence_score": confidence,
        "metadata": {"source_domain": source},
    }
    if split:
        raw_json["split"] = split
        raw_json["metadata"]["split"] = split
    return {
        "video_id": video_id,
        "source": source,
        "confidence_score": confidence,
        "raw_json": raw_json,
        "score_id": f"score_{video_id}",
    }


def _read_bytes(base_dir: Path, version_tag: str) -> dict[str, bytes]:
    dataset_dir = base_dir / "data" / "training" / f"2026-04-08_{version_tag}"
    return {
        name: (dataset_dir / name).read_bytes()
        for name in ("train.jsonl", "val.jsonl", "test.jsonl", "manifest.yaml")
    }


def _patched_loader_factory(candidates: list[dict]):
    def _loader(*args, include_sources=None, exclude_sources=None, **kwargs):
        del args, kwargs
        rows = copy.deepcopy(candidates)
        if include_sources:
            allowed = {item.lower() for item in include_sources}
            rows = [row for row in rows if row["source"].lower() in allowed]
        if exclude_sources:
            denied = {item.lower() for item in exclude_sources}
            rows = [row for row in rows if row["source"].lower() not in denied]
        return rows

    return _loader


def test_prepare_dataset_defaults_are_byte_identical_with_explicit_new_defaults(monkeypatch, tmp_path):
    candidates = [
        _candidate(video_id="yt_alpha", source="teacher_claude", task_id="task5"),
        _candidate(video_id="yt_beta", source="teacher_claude", task_id="task5"),
        _candidate(video_id="yt_gamma", source="teacher_claude", task_id="task5"),
    ]

    monkeypatch.setattr(dataset_module, "datetime", FixedDatetime)
    monkeypatch.setattr(
        dataset_module,
        "_load_training_candidates",
        _patched_loader_factory(candidates),
    )

    base_one = tmp_path / "one"
    base_two = tmp_path / "two"
    base_one.mkdir()
    base_two.mkdir()

    store_one = type("Store", (), {"base": base_one})()
    store_two = type("Store", (), {"base": base_two})()

    dataset_module.prepare_dataset(
        store=cast(Any, store_one),
        log=cast(Any, DummyLog()),
        video_dir=base_one,
        output_dir=base_one / "data" / "training",
        version=4,
        version_tag="v4",
        min_confidence=0.5,
        group_by="video",
        seed=7,
    )
    dataset_module.prepare_dataset(
        store=cast(Any, store_two),
        log=cast(Any, DummyLog()),
        video_dir=base_two,
        output_dir=base_two / "data" / "training",
        version=4,
        version_tag="v4",
        min_confidence=0.5,
        group_by="video",
        seed=7,
        include_sources=[],
        exclude_sources=[],
        respect_existing_splits=False,
    )

    assert _read_bytes(base_one, "v4") == _read_bytes(base_two, "v4")


def test_prepare_dataset_filters_sources_and_honors_declared_splits(monkeypatch, tmp_path):
    candidates = [
        _candidate(video_id="lasana_a", source="lasana", task_id="lasana_peg", split="train"),
        _candidate(video_id="lasana_b", source="lasana", task_id="lasana_suture", split="val"),
        _candidate(video_id="yt_alpha", source="teacher_claude", task_id="task5"),
    ]

    monkeypatch.setattr(dataset_module, "datetime", FixedDatetime)
    monkeypatch.setattr(
        dataset_module,
        "_load_training_candidates",
        _patched_loader_factory(candidates),
    )

    base_dir = tmp_path / "build"
    base_dir.mkdir()
    store = type("Store", (), {"base": base_dir})()

    manifest = dataset_module.prepare_dataset(
        store=cast(Any, store),
        log=cast(Any, DummyLog()),
        video_dir=base_dir,
        output_dir=base_dir / "data" / "training",
        version=1,
        version_tag="lasana_v1",
        min_confidence=0.5,
        group_by="video",
        seed=11,
        include_sources=["lasana"],
        respect_existing_splits=True,
    )

    dataset_dir = Path(manifest["output_dir"])
    train_rows = [json.loads(line) for line in (dataset_dir / "train.jsonl").read_text().splitlines() if line]
    val_rows = [json.loads(line) for line in (dataset_dir / "val.jsonl").read_text().splitlines() if line]
    test_rows = [json.loads(line) for line in (dataset_dir / "test.jsonl").read_text().splitlines() if line]

    assert {row["metadata"]["source"] for row in train_rows + val_rows + test_rows} == {"lasana"}
    assert {row["metadata"]["task"] for row in train_rows + val_rows} == {"lasana_peg", "lasana_suture"}
    assert all(row["metadata"]["task_id"].startswith("lasana_") for row in train_rows + val_rows)
    assert not test_rows
    assert len(train_rows) == 1
    assert len(val_rows) == 1
    assert manifest["sources"] == {"lasana": 2}
    assert manifest["split_strategy"] == "existing_split+video"