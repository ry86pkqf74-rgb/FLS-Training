"""Dataset lineage sidecars for FLS-Training.

For each `*.jsonl` dataset file produced by `scripts/040_prepare_training_data.py`
this module writes a companion `<name>.meta.json` file describing provenance,
licensing posture, validation status, and skill-level distribution. The sidecar
is the audit trail that lets us answer questions like "which of these frames
came from a proprietary clinic recording?" and "were any of the rejected videos
still in the training split?".

The sidecar schema (kept stable — downstream tooling reads this verbatim):

    {
      "manifest_id":        str,
      "created_at":         ISO-8601 UTC,
      "created_by":         "scripts/040_prepare_training_data.py",
      "total_samples":      int,
      "sources":            [ {source_name, video_count, license_class,
                               commercial_ok, trainee_count, notes}, ... ],
      "validation_summary": {accepted, quarantined, rejected, acceptance_rate},
      "split_strategy":     str,
      "held_out_trainees":  [str, ...],
      "skill_level_distribution": {novice_lt400, intermediate_400_480,
                                   advanced_480_plus}
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Source classification + licensing metadata
# --------------------------------------------------------------------------- #

# video_ids matching any of these patterns are treated as "self_recorded".
_SELF_RECORDED_PATTERNS = (
    re.compile(r"^V_?\d+(_|$)", re.IGNORECASE),        # V31, V_8, V31_video
    re.compile(r"^post\d*V?_?\d+", re.IGNORECASE),     # post2, postV7, post_practice
    re.compile(r"^lap_pre", re.IGNORECASE),            # lap_pre_video
    re.compile(r"^post-?practice", re.IGNORECASE),
)

_SOURCE_NOTES = {
    "self_recorded": "Single trainee, 3 equipment setups",
    "youtube_harvest": "Downloaded via yt-dlp, no explicit license",
}

_SOURCE_LICENSE = {
    "self_recorded": {
        "license_class": "proprietary",
        "commercial_ok": True,
    },
    "youtube_harvest": {
        "license_class": "fair_use_research",
        "commercial_ok": False,
    },
}


def classify_source(video_id: str, known_self_recorded: set[str] | None = None) -> str:
    """Return 'self_recorded' or 'youtube_harvest' for a video_id.

    Classification order:
        1. Explicit membership in `known_self_recorded` (DuckDB videos table).
        2. Heuristic regex match against self-recorded filename patterns.
        3. Default: youtube_harvest (any 11-char YouTube id or other string).
    """
    if known_self_recorded and video_id in known_self_recorded:
        return "self_recorded"
    for pat in _SELF_RECORDED_PATTERNS:
        if pat.match(video_id):
            return "self_recorded"
    return "youtube_harvest"


# --------------------------------------------------------------------------- #
# DuckDB helpers (best-effort — all lookups degrade to empty on failure)
# --------------------------------------------------------------------------- #

def _db_path() -> Path:
    env = os.environ.get("FLS_DB_PATH")
    if env:
        return Path(env)
    return Path("data/fls_training.duckdb")


def _open_db():
    try:
        import duckdb  # type: ignore
    except ImportError:
        logger.warning("duckdb not installed; lineage sidecar will omit DB-derived fields")
        return None
    path = _db_path()
    if not path.exists():
        logger.warning("DuckDB not found at %s; lineage sidecar will omit DB-derived fields", path)
        return None
    try:
        return duckdb.connect(str(path), read_only=True)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("failed to open DuckDB at %s: %s", path, exc)
        return None


def _known_self_recorded_ids(conn) -> set[str]:
    if conn is None:
        return set()
    try:
        rows = conn.execute("SELECT id FROM videos").fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def _scores_by_video(conn, video_ids: Iterable[str]) -> dict[str, float]:
    """Return the newest teacher FLS score per video_id."""
    if conn is None:
        return {}
    ids = [v for v in video_ids if v]
    if not ids:
        return {}
    # Detect which column names this scores table uses
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info('scores')").fetchall()}
    except Exception:
        return {}
    fls_col = "estimated_fls_score" if "estimated_fls_score" in cols else "fls_score"
    if fls_col not in cols:
        return {}
    placeholders = ",".join("?" for _ in ids)
    query = f"""
        SELECT video_id, {fls_col}, scored_at
        FROM scores
        WHERE video_id IN ({placeholders})
          AND source IN ('teacher_claude', 'teacher_gpt4o', 'teacher_gpt',
                         'critique_consensus', 'consensus')
        ORDER BY scored_at DESC
    """
    try:
        rows = conn.execute(query, ids).fetchall()
    except Exception:
        return {}
    latest: dict[str, float] = {}
    for vid, score, _ts in rows:
        if vid in latest or score is None:
            continue
        latest[vid] = float(score)
    return latest


def _validation_summary(conn, video_ids: Iterable[str]) -> dict[str, Any]:
    if conn is None:
        return _empty_validation_summary()
    # Does the validations table exist?
    try:
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    except Exception:
        return _empty_validation_summary()
    if "validations" not in tables:
        return _empty_validation_summary()

    ids = list({v for v in video_ids if v})
    if not ids:
        return _empty_validation_summary()

    placeholders = ",".join("?" for _ in ids)
    try:
        rows = conn.execute(
            f"SELECT validation_status FROM validations WHERE video_id IN ({placeholders})",
            ids,
        ).fetchall()
    except Exception:
        return _empty_validation_summary()

    counts = Counter(r[0] for r in rows if r[0])
    accepted = counts.get("ACCEPTED", 0)
    quarantined = counts.get("QUARANTINED", 0)
    rejected = counts.get("REJECTED", 0)
    total = accepted + quarantined + rejected
    rate = round(accepted / total, 4) if total else 0.0
    return {
        "accepted": accepted,
        "quarantined": quarantined,
        "rejected": rejected,
        "acceptance_rate": rate,
    }


def _empty_validation_summary() -> dict[str, Any]:
    return {
        "accepted": 0,
        "quarantined": 0,
        "rejected": 0,
        "acceptance_rate": 0.0,
    }


# --------------------------------------------------------------------------- #
# JSONL iteration
# --------------------------------------------------------------------------- #

def _iter_video_ids(jsonl_path: Path) -> list[str]:
    ids: list[str] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = record.get("metadata") or {}
            vid = meta.get("video_id") if isinstance(meta, dict) else None
            if not vid:
                vid = record.get("video_id")
            if vid:
                ids.append(str(vid))
    return ids


# --------------------------------------------------------------------------- #
# Skill-level distribution
# --------------------------------------------------------------------------- #

def _skill_distribution(video_ids: Iterable[str], scores: dict[str, float]) -> dict[str, int]:
    buckets = {
        "novice_lt400": 0,
        "intermediate_400_480": 0,
        "advanced_480_plus": 0,
    }
    counted: set[str] = set()
    for vid in video_ids:
        if vid in counted:
            continue
        counted.add(vid)
        score = scores.get(vid)
        if score is None:
            continue
        if score < 400:
            buckets["novice_lt400"] += 1
        elif score < 480:
            buckets["intermediate_400_480"] += 1
        else:
            buckets["advanced_480_plus"] += 1
    return buckets


# --------------------------------------------------------------------------- #
# Sidecar assembly + write
# --------------------------------------------------------------------------- #

def _coerce_version_int(ver: Any) -> int:
    digits = "".join(ch for ch in str(ver) if ch.isdigit())
    return int(digits) if digits else 1


def build_sidecar(
    *,
    jsonl_path: Path,
    version: Any,
    split_strategy: str,
    held_out_trainees: list[str] | None,
    created_by: str = "scripts/040_prepare_training_data.py",
    conn=None,
) -> dict[str, Any]:
    """Build the lineage sidecar dict for a single jsonl file."""
    video_ids = _iter_video_ids(jsonl_path)
    total_samples = len(video_ids)
    unique_ids = sorted(set(video_ids))

    known_self_recorded = _known_self_recorded_ids(conn)

    # Count UNIQUE videos per source (samples can repeat a video_id)
    source_to_videos: dict[str, set[str]] = {"self_recorded": set(), "youtube_harvest": set()}
    for vid in unique_ids:
        source = classify_source(vid, known_self_recorded)
        source_to_videos.setdefault(source, set()).add(vid)

    sources_block: list[dict[str, Any]] = []
    for name in ("self_recorded", "youtube_harvest"):
        videos = source_to_videos.get(name, set())
        if not videos and name != "self_recorded":
            continue
        licensing = _SOURCE_LICENSE[name]
        if name == "self_recorded":
            trainee_count = 1 if videos else 0
        else:
            # Without a surgeon registry, the only safe assumption is
            # "≤ video_count distinct trainees". Report video_count as the
            # upper bound.
            trainee_count = len(videos)
        sources_block.append(
            {
                "source_name": name,
                "video_count": len(videos),
                "license_class": licensing["license_class"],
                "commercial_ok": licensing["commercial_ok"],
                "trainee_count": trainee_count,
                "notes": _SOURCE_NOTES[name],
            }
        )

    validation_summary = _validation_summary(conn, unique_ids)
    scores = _scores_by_video(conn, unique_ids)
    skill_dist = _skill_distribution(unique_ids, scores)

    version_int = _coerce_version_int(version)
    manifest_id = f"dataset_task5_v{version_int:03d}"

    return {
        "manifest_id": manifest_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": created_by,
        "total_samples": total_samples,
        "sources": sources_block,
        "validation_summary": validation_summary,
        "split_strategy": split_strategy,
        "held_out_trainees": list(held_out_trainees or []),
        "skill_level_distribution": skill_dist,
    }


def _sidecar_path(jsonl_path: Path) -> Path:
    """train.jsonl → train.meta.json (matches spec *.meta.json pattern)."""
    return jsonl_path.with_suffix("").with_suffix(".meta.json")


def write_sidecars(
    *,
    output_dir: Path,
    version: Any,
    split_strategy: str,
    held_out_trainees: list[str] | None,
    created_by: str = "scripts/040_prepare_training_data.py",
) -> list[Path]:
    """Write a `.meta.json` sidecar for every `.jsonl` in `output_dir`.

    Returns the list of sidecar paths written.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        logger.warning("lineage: output_dir %s does not exist; skipping", output_dir)
        return []

    jsonl_files = sorted(output_dir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("lineage: no .jsonl files in %s", output_dir)
        return []

    conn = _open_db()
    written: list[Path] = []
    try:
        for jsonl_path in jsonl_files:
            sidecar = build_sidecar(
                jsonl_path=jsonl_path,
                version=version,
                split_strategy=split_strategy,
                held_out_trainees=held_out_trainees,
                created_by=created_by,
                conn=conn,
            )
            path = _sidecar_path(jsonl_path)
            path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
            written.append(path)
            logger.info("lineage: wrote %s (%d samples)", path, sidecar["total_samples"])
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    return written


__all__ = [
    "classify_source",
    "build_sidecar",
    "write_sidecars",
]
