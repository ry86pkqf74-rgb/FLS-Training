#!/usr/bin/env python3
"""Ingest LASANA human labels into the MemoryStore score corpus.

This script writes one deterministic score JSON per LASANA trial under
``memory/scores/`` so re-runs update records in place instead of creating
duplicates. The authoritative source labels remain the LASANA human ratings.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from src.memory.memory_store import MemoryStore


TASK_MAP = {
    "BalloonResection": "lasana_balloon",
    "CircleCutting": "lasana_circle",
    "PegTransfer": "lasana_peg",
    "SutureAndKnot": "lasana_suture",
}

ASPECT_FIELDS = [
    "GRS",
    "bimanual_dexterity",
    "depth_perception",
    "efficiency",
    "tissue_handling",
]

RATER_STD_NORMALIZER = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest LASANA annotations into MemoryStore")
    parser.add_argument("--base-dir", default=".", help="Repository base directory")
    parser.add_argument(
        "--annotations-dir",
        default="data/external/lasana/annotations/Annotation",
        help="Directory containing LASANA annotation CSVs",
    )
    parser.add_argument(
        "--frames-root",
        default="data/external/lasana_processed/frames",
        help="Root directory where LASANA frame folders will live",
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASK_MAP),
        default=None,
        help="Optional single-task ingest filter",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="Limit ingested trials for smoke testing; 0 means all matched rows",
    )
    return parser.parse_args()


def read_semicolon_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        return list(reader)


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() == "true"


def parse_duration_to_seconds(value: str | None) -> float:
    if not value:
        return 0.0
    parts = [int(part) for part in value.strip().split(":")]
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    else:
        return float(parts[0]) if parts else 0.0
    return float(hours * 3600 + minutes * 60 + seconds)


def numeric_mean(values: list[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return mean(cleaned)


def numeric_std(values: list[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return 0.0
    return pstdev(cleaned)


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def task_error_fields(main_row: dict[str, str]) -> dict[str, bool]:
    excluded = {
        "id",
        "duration",
        "frame_count",
        "GRS",
        "bimanual_dexterity",
        "depth_perception",
        "efficiency",
        "tissue_handling",
    }
    return {
        key: parse_bool(value)
        for key, value in main_row.items()
        if key not in excluded
    }


def load_rater_rows(annotations_dir: Path, task_name: str) -> dict[str, list[dict[str, str]]]:
    by_trial: dict[str, list[dict[str, str]]] = {}
    for rater_index in range(4):
        path = annotations_dir / f"{task_name}_rater{rater_index}.csv"
        for row in read_semicolon_csv(path):
            trial_id = row.get("id", "").strip()
            if not trial_id:
                continue
            by_trial.setdefault(trial_id, []).append(row)
    return by_trial


def load_split_rows(annotations_dir: Path, task_name: str) -> dict[str, str]:
    rows = read_semicolon_csv(annotations_dir / f"{task_name}_split.csv")
    return {row["id"].strip(): row["split"].strip() for row in rows if row.get("id")}


def deterministic_score_id(trial_id: str) -> str:
    return f"score_lasana_{trial_id}"


def score_output_path(store: MemoryStore, trial_id: str) -> Path:
    return store.scores_dir / "lasana" / f"{deterministic_score_id(trial_id)}.json"


def remove_legacy_duplicates(store: MemoryStore, trial_id: str, canonical_path: Path) -> None:
    for path in store.scores_dir.rglob("*.json"):
        if path == canonical_path:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if str(payload.get("source") or "") != "lasana":
            continue
        metadata = payload.get("metadata") or {}
        payload_trial = metadata.get("trial_id") or payload.get("trial_id")
        if payload_trial == trial_id:
            path.unlink(missing_ok=True)


def build_score_record(
    *,
    task_name: str,
    trial_id: str,
    main_row: dict[str, str],
    split: str,
    rater_rows: list[dict[str, str]],
    frames_root: str,
) -> dict[str, Any]:
    per_metric_means: dict[str, float | None] = {}
    per_metric_stds: dict[str, float | None] = {}
    for field in ASPECT_FIELDS:
        values = [parse_float(row.get(field)) for row in rater_rows]
        per_metric_means[field] = numeric_mean(values)
        per_metric_stds[field] = numeric_std(values)

    grs_z = parse_float(main_row.get("GRS"))
    if grs_z is None:
        grs_z = 0.0

    score_std = per_metric_stds["GRS"]
    if score_std is None:
        score_std = 2.0

    normalized_rater_std = clip(score_std / RATER_STD_NORMALIZER, 0.0, 1.0)
    confidence = clip(1.0 - normalized_rater_std, 0.5, 1.0)
    grs_rescaled = clip(3.0 + float(grs_z or 0.0), 1.0, 5.0)
    frame_dir = str(Path(frames_root) / trial_id)

    metadata = {
        "trial_id": trial_id,
        "task_id": TASK_MAP[task_name],
        "task_name": task_name,
        "source_domain": "lasana",
        "dataset": "lasana",
        "split": split,
        "duration": main_row.get("duration", ""),
        "frame_count": int(main_row.get("frame_count") or 0),
        "frame_dir": frame_dir,
        "notes": (
            "LASANA labels ingested from human raters. "
            "fls_score is populated from grs_rescaled = clip(3 + grs_z, 1, 5). "
            f"confidence uses 1 - clip(score_std / {RATER_STD_NORMALIZER:.1f}, 0, 1), then clipped to [0.5, 1.0]."
        ),
        "label_stats": {
            "grs_z": grs_z,
            "grs_rescaled": grs_rescaled,
            "grs_raw_mean": per_metric_means["GRS"],
            "score_std": score_std,
            "normalized_rater_std": normalized_rater_std,
            "means": {
                "bimanual_dexterity": per_metric_means["bimanual_dexterity"],
                "depth_perception": per_metric_means["depth_perception"],
                "efficiency": per_metric_means["efficiency"],
                "tissue_handling": per_metric_means["tissue_handling"],
            },
            "stds": {
                "bimanual_dexterity": per_metric_stds["bimanual_dexterity"],
                "depth_perception": per_metric_stds["depth_perception"],
                "efficiency": per_metric_stds["efficiency"],
                "tissue_handling": per_metric_stds["tissue_handling"],
            },
        },
        "task_errors": task_error_fields(main_row),
    }

    return {
        "id": deterministic_score_id(trial_id),
        "video_id": f"lasana_{trial_id}",
        "video_filename": f"{trial_id}.mkv",
        "video_hash": "",
        "source": "lasana",
        "model_name": "lasana_human_rater_mean",
        "model_version": "lasana_human_v1",
        "prompt_version": "human_labels",
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "task_id": TASK_MAP[task_name],
        "split": split,
        "trial_id": trial_id,
        "penalties": [],
        "score_components": None,
        "phases_detected": [],
        "reasoning": "Human LASANA label aggregate.",
        "frame_analyses": [],
        "completion_time_seconds": parse_duration_to_seconds(main_row.get("duration")),
        "phase_timings": [],
        "knot_assessments": [],
        "suture_placement": None,
        "drain_assessment": None,
        "estimated_penalties": 0.0,
        "estimated_fls_score": grs_rescaled,
        "confidence_score": confidence,
        "technique_summary": "",
        "improvement_suggestions": [],
        "strengths": [],
        "comparison_to_previous": {},
        "superseded": False,
        "superseded_by": None,
        "superseded_at": None,
        "superseded_reason": "",
        "metadata": metadata,
        "grs_z": grs_z,
        "grs_rescaled": grs_rescaled,
        "score_std": score_std,
        "bimanual_dexterity": per_metric_means["bimanual_dexterity"],
        "depth_perception": per_metric_means["depth_perception"],
        "efficiency": per_metric_means["efficiency"],
        "tissue_handling": per_metric_means["tissue_handling"],
        "fls_score": grs_rescaled,
    }


def ingest_task(
    *,
    store: MemoryStore,
    annotations_dir: Path,
    frames_root: str,
    task_name: str,
    max_trials: int,
) -> tuple[int, int]:
    main_rows = read_semicolon_csv(annotations_dir / f"{task_name}.csv")
    split_rows = load_split_rows(annotations_dir, task_name)
    rater_rows = load_rater_rows(annotations_dir, task_name)

    written = 0
    skipped = 0
    for index, main_row in enumerate(main_rows, start=1):
        if max_trials and index > max_trials:
            break
        trial_id = main_row.get("id", "").strip()
        if not trial_id:
            skipped += 1
            continue
        split = split_rows.get(trial_id)
        if split not in {"train", "val", "test"}:
            skipped += 1
            continue

        record = build_score_record(
            task_name=task_name,
            trial_id=trial_id,
            main_row=main_row,
            split=split,
            rater_rows=rater_rows.get(trial_id, []),
            frames_root=frames_root,
        )

        output_path = score_output_path(store, trial_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        remove_legacy_duplicates(store, trial_id, output_path)
        output_path.write_text(json.dumps(record, indent=2, default=str))
        written += 1

    return written, skipped


def main() -> int:
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    annotations_dir = (base_dir / args.annotations_dir).resolve()
    frames_root = args.frames_root

    if not annotations_dir.is_dir():
        raise SystemExit(f"Annotations dir not found: {annotations_dir}")

    store = MemoryStore(base_dir)
    tasks = [args.task] if args.task else list(TASK_MAP)

    total_written = 0
    total_skipped = 0
    for task_name in tasks:
        written, skipped = ingest_task(
            store=store,
            annotations_dir=annotations_dir,
            frames_root=frames_root,
            task_name=task_name,
            max_trials=args.max_trials,
        )
        total_written += written
        total_skipped += skipped

    store._append_ledger(
        "lasana_ingested",
        {
            "tasks": tasks,
            "written": total_written,
            "skipped": total_skipped,
            "frames_root": frames_root,
        },
    )

    print(f"LASANA ingest complete: wrote {total_written} records, skipped {total_skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())