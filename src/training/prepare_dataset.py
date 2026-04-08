"""Prepare training datasets from accumulated scores and corrections.

Builds JSONL files suitable for fine-tuning Qwen2.5-VL-7B-Instruct.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.memory.memory_store import MemoryStore

if TYPE_CHECKING:
    from src.memory.learning_log import LearningLog

logger = logging.getLogger(__name__)

FEEDBACK_DIR = Path(__file__).parent.parent.parent / "memory" / "feedback"
SCORES_DIR = Path(__file__).parent.parent.parent / "memory" / "scores"
COMPARISONS_DIR = Path(__file__).parent.parent.parent / "memory" / "comparisons"
CORRECTIONS_DIR = Path(__file__).parent.parent.parent / "memory" / "corrections"

TASK_LABELS = {
    "task1": "FLS Task 1 peg transfer",
    "task2": "FLS Task 2 pattern cut",
    "task3": "FLS Task 3 ligating loop",
    "task4": "FLS Task 4 extracorporeal suture",
    "task5": "FLS Task 5 intracorporeal suture",
}


def _canonical_task_id(task_id: str | None) -> str:
    raw = str(task_id or "task5").strip().lower()
    aliases = {
        "task1_peg_transfer": "task1",
        "task2_pattern_cut": "task2",
        "task3_endoloop": "task3",
        "task3_ligating_loop": "task3",
        "task4_extracorporeal_knot": "task4",
        "task4_extracorporeal_suture": "task4",
        "task5_intracorporeal_suturing": "task5",
        "task5_intracorporeal_suture": "task5",
    }
    if raw in aliases:
        return aliases[raw]
    if raw.isdigit():
        return f"task{raw}"
    if raw.startswith("task"):
        return raw.split("_", 1)[0]
    return "task5"


def _task_user_prompt(task_id: str | None) -> str:
    canonical = _canonical_task_id(task_id)
    label = TASK_LABELS.get(canonical, TASK_LABELS["task5"])
    return f"[FRAMES_PLACEHOLDER] Score this {label} video.\nTask ID: {canonical}"


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _normalize_source(source: str | None, filename: str) -> str:
    raw = (source or "").lower()
    name = filename.lower()
    if "consensus" in raw or "critique_consensus" in raw or "_consensus_" in name:
        return "critique_consensus"
    if "gpt" in raw or "chatgpt" in raw or "gpt-4o" in name or "gpt4o" in name:
        return "teacher_gpt4o"
    if "student" in raw or "_student" in name:
        return "student_model"
    if "correction" in raw:
        return "expert_correction"
    return "teacher_claude"


def _iter_json_files(base_dir: Path) -> list[Path]:
    if not base_dir.exists():
        return []

    files = sorted(base_dir.glob("*.json"))
    for child in sorted(base_dir.iterdir()):
        if child.is_dir():
            files.extend(sorted(child.glob("*.json")))
    return files


def _default_video_filename(video_id: str) -> str:
    stem = video_id[:-6] if video_id.endswith("_video") else video_id
    return f"{stem}.mov"


def _build_consensus_score(video_id: str, payload: dict[str, Any], file_path: Path) -> dict[str, Any] | None:
    consensus_score = payload.get("consensus_score") or {}
    if not consensus_score:
        return None

    del file_path

    meta = payload.get("_meta") or {}
    trace = payload.get("trace") or {}
    scored_at = meta.get("timestamp") or trace.get("timestamp") or datetime.now(timezone.utc).isoformat()
    timestamp_slug = scored_at.replace("-", "").replace(":", "").replace("+", "_").replace("T", "_")

    return {
        "id": f"score_consensus_{video_id}_{timestamp_slug}",
        "video_id": video_id,
        "video_filename": _default_video_filename(video_id),
        "video_hash": "",
        "source": "critique_consensus",
        "model_name": meta.get("model") or "manual-consensus",
        "model_version": meta.get("model") or "manual-consensus",
        "prompt_version": payload.get("prompt_version") or "v002",
        "scored_at": scored_at,
        "task_id": _canonical_task_id(consensus_score.get("task_id") or payload.get("task_id")),
        "frame_analyses": [],
        "completion_time_seconds": consensus_score.get("completion_time_seconds", 0),
        "phase_timings": consensus_score.get("phase_timings", []),
        "knot_assessments": consensus_score.get("knot_assessments", []),
        "suture_placement": consensus_score.get("suture_placement"),
        "drain_assessment": consensus_score.get("drain_assessment"),
        "estimated_penalties": consensus_score.get("estimated_penalties", 0),
        "estimated_fls_score": consensus_score.get("estimated_fls_score", 0),
        "confidence_score": consensus_score.get("confidence_score", payload.get("agreement_score", 0.0)),
        "technique_summary": consensus_score.get("technique_summary", ""),
        "improvement_suggestions": consensus_score.get("improvement_suggestions", []),
        "strengths": consensus_score.get("strengths", []),
        "comparison_to_previous": {},
    }


def _load_corrections_by_video() -> dict[str, dict[str, Any]]:
    latest: dict[str, tuple[datetime, dict[str, Any]]] = {}
    for path in _iter_json_files(CORRECTIONS_DIR):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        video_id = data.get("video_id")
        if not video_id:
            continue

        corrected_at = _parse_timestamp(data.get("corrected_at"))
        current = latest.get(video_id)
        if current is None or corrected_at >= current[0]:
            latest[video_id] = (corrected_at, data)

    return {video_id: item[1] for video_id, item in latest.items()}


def _load_training_candidates(base_dir: Path, min_confidence: float) -> list[dict[str, Any]]:
    del base_dir  # file-backed memory lives under the repository root paths above

    latest_by_video_source: dict[tuple[str, str], tuple[datetime, dict[str, Any]]] = {}

    for path in _iter_json_files(SCORES_DIR):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        video_id = data.get("video_id")
        if not video_id:
            continue

        source = _normalize_source(data.get("source"), path.name)
        confidence = float(data.get("confidence_score", 0) or 0)
        if confidence < min_confidence:
            continue

        candidate = {
            "video_id": video_id,
            "source": source,
            "confidence_score": confidence,
            "raw_json": data,
            "score_id": data.get("id"),
        }
        scored_at = _parse_timestamp(data.get("scored_at"))
        key = (video_id, source)
        current = latest_by_video_source.get(key)
        if current is None or scored_at >= current[0]:
            latest_by_video_source[key] = (scored_at, candidate)

    for path in _iter_json_files(COMPARISONS_DIR):
        if "_consensus_" not in path.name:
            continue
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        video_id = (payload.get("_meta") or {}).get("video_id") or path.name.split("_consensus_")[0]
        consensus = _build_consensus_score(video_id, payload, path)
        if not consensus:
            continue

        confidence = float(consensus.get("confidence_score", 0) or 0)
        if confidence < min_confidence:
            continue

        candidate = {
            "video_id": video_id,
            "source": "critique_consensus",
            "confidence_score": confidence,
            "raw_json": consensus,
            "score_id": consensus.get("id"),
        }
        scored_at = _parse_timestamp(consensus.get("scored_at"))
        key = (video_id, "critique_consensus")
        current = latest_by_video_source.get(key)
        if current is None or scored_at >= current[0]:
            latest_by_video_source[key] = (scored_at, candidate)

    corrections_by_video = _load_corrections_by_video()
    candidates = [item[1] for item in latest_by_video_source.values()]
    expanded_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        expanded_candidates.append(candidate)
        correction = corrections_by_video.get(candidate["video_id"])
        if correction:
            expanded_candidates.append({
                **candidate,
                "source": "expert_correction",
                "corrected_fields": correction.get("corrected_fields", {}),
            })

    return expanded_candidates


def _load_coach_feedback(video_id: str) -> dict | None:
    """Load the most recent coach feedback for a video, if available."""
    if not FEEDBACK_DIR.exists():
        return None
    matches = sorted(
        [f for f in FEEDBACK_DIR.iterdir() if video_id in f.name and f.suffix == ".json"],
        reverse=True,
    )
    if not matches:
        return None
    try:
        data = json.loads(matches[0].read_text())
        # Strip internal metadata from training data
        data.pop("_meta", None)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load coach feedback for {video_id}: {e}")
        return None


def prepare_dataset(
    store: MemoryStore,
    log: LearningLog,
    video_dir: str | Path,
    output_dir: str | Path,
    version: int = 1,
    min_confidence: float = 0.7,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
    include_coach_feedback: bool = False,
) -> dict:
    """Build train/val/test JSONL files from scored videos.

    Args:
        include_coach_feedback: If True, append coach feedback to the assistant
            response so the student model learns to produce both scores and
            rich technique coaching in a single pass (Phase 2 of coach integration).

    Returns manifest dict with dataset stats.
    """
    # Load system prompt for training examples
    prompts_dir = Path(__file__).parent.parent.parent / "prompts"
    universal_prompt = prompts_dir / "v002_universal_scoring_system.md"
    system_prompt = universal_prompt.read_text() if universal_prompt.exists() else (prompts_dir / "v001_task5_system.md").read_text()

    # Get training candidates
    candidates = _load_training_candidates(store.base, min_confidence)
    logger.info(f"Found {len(candidates)} training candidates")

    if not candidates:
        logger.warning("No training candidates found. Score more videos first.")
        return {"error": "no_candidates", "n_candidates": 0}

    # Deduplicate by video_id (prefer corrections > consensus > raw)
    by_video: dict[str, dict] = {}
    SOURCE_PRIORITY = {"expert_correction": 0, "critique_consensus": 1,
                       "teacher_claude": 2, "teacher_gpt4o": 3, "student_model": 4}
    for c in candidates:
        vid = c["video_id"]
        source = c.get("source", "teacher_claude")
        priority = SOURCE_PRIORITY.get(source, 5)
        if vid not in by_video or priority < SOURCE_PRIORITY.get(by_video[vid].get("source", ""), 5):
            by_video[vid] = c

    samples = list(by_video.values())
    logger.info(f"Deduplicated to {len(samples)} unique videos")

    # Build training examples
    examples = []
    for sample in samples:
        raw_json = sample.get("raw_json")
        if not raw_json:
            continue

        # The raw_json contains the full ScoringResult
        if isinstance(raw_json, str):
            score_data = json.loads(raw_json)
        else:
            score_data = raw_json

        # Apply corrections if present
        corrected_fields = sample.get("corrected_fields")
        if corrected_fields:
            if isinstance(corrected_fields, str):
                corrected_fields = json.loads(corrected_fields)
            score_data.update(corrected_fields)

        # Optionally attach coach feedback for richer student training
        coach_feedback = None
        if include_coach_feedback:
            coach_feedback = _load_coach_feedback(sample["video_id"])

        # Build the assistant response: score JSON + optional coach JSON
        assistant_content = json.dumps(score_data, default=str)
        if coach_feedback:
            # Wrap both in a combined output so student learns to produce both
            combined = {
                "scoring_result": score_data,
                "coach_feedback": coach_feedback,
            }
            assistant_content = json.dumps(combined, default=str)

        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _task_user_prompt(score_data.get("task_id"))},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "video_id": sample["video_id"],
                "source": sample.get("source", "unknown"),
                "confidence": sample.get("confidence_score", 0),
                "task": _canonical_task_id(score_data.get("task_id")),
            },
        }
        examples.append(example)

    if not examples:
        return {"error": "no_valid_examples", "n_candidates": len(candidates)}

    # Split
    random.seed(seed)
    random.shuffle(examples)

    n_train = int(len(examples) * train_split)
    n_val = int(len(examples) * val_split)

    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]

    # Write output
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path(output_dir) / f"{date_str}_v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex, default=str) + "\n")
        logger.info(f"Wrote {len(split_data)} examples to {path}")

    # Manifest
    manifest = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_examples": len(examples),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "min_confidence": min_confidence,
        "include_coach_feedback": include_coach_feedback,
        "sources": {},
        "seed": seed,
    }
    for ex in examples:
        src = ex["metadata"]["source"]
        manifest["sources"][src] = manifest["sources"].get(src, 0) + 1

    manifest_path = out_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    log.append_event("dataset_prepared", {
        "version": version,
        "output_dir": str(out_dir),
        **manifest,
    })

    logger.info(f"Dataset v{version}: {len(train)} train, {len(val)} val, {len(test)} test")
    return manifest
