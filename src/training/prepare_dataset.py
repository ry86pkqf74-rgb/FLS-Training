"""Prepare training datasets from accumulated scores and corrections.

Builds JSONL files suitable for fine-tuning Qwen2.5-VL-7B-Instruct.

Two output modes:

* **Text-only (legacy)** — default when no ``frames_dir`` is supplied. The
  user message contains a placeholder like ``[FRAMES_PLACEHOLDER]`` and no
  image content blocks. Matches the proven April 2026 v3 deploy path.
  Useful for cheap local iteration and for ablations, but a model trained
  this way has never seen a pixel.

* **Vision (v4+)** — enabled by passing ``frames_dir=data/frames`` (or any
  directory containing one subfolder per ``video_id`` with extracted JPEGs).
  The user message becomes a list of content blocks: up to
  ``max_frames_per_sample`` ``{"type": "image", "image": <path>}`` entries
  followed by a single ``{"type": "text", "text": ...}`` block. This is
  the format ``src.training.finetune_vlm`` consumes via
  ``UnslothVisionDataCollator``.

Resident-aware splits: every example carries ``metadata.trainee_id`` and
``metadata.source_domain`` derived through
``src.training.schema_adapter``. When ``group_by='trainee'`` is passed
(the new default), the val/test splits hold out entire trainees rather
than individual videos so the model is not rewarded for memorising a
single resident's style.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import yaml

from src.memory.memory_store import MemoryStore
from src.training.schema_adapter import (
    canonical_task_id as adapter_canonical_task_id,
    get_source_domain,
    get_total_score,
    get_trainee_id,
)

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

# Task id is resolved via the schema adapter now. The local helper is a
# thin wrapper that keeps the old default-to-task5 behaviour required by
# historical records that had no task_id field.
def _canonical_task_id(task_id: str | None) -> str:
    return adapter_canonical_task_id(task_id, default="task5")


def _task_user_text(task_id: str | None) -> str:
    canonical = _canonical_task_id(task_id)
    label = TASK_LABELS.get(canonical, TASK_LABELS["task5"])
    return (
        f"Score this {label} attempt. Analyse the sampled frames "
        f"and return a strict-JSON ScoringResult matching the schema in "
        f"the system prompt.\nTask ID: {canonical}"
    )


# Image file extensions the frame sampler will accept. JPEG preferred to
# keep payloads small; PNG and WEBP tolerated.
_FRAME_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _sample_frame_paths(
    frames_dir: Path | None,
    video_id: str,
    max_frames: int,
) -> list[Path]:
    """Return up to ``max_frames`` frame paths for ``video_id``.

    The layout expected is ``<frames_dir>/<video_id>/*.jpg`` (or any
    ``_FRAME_EXTS`` extension). If no frames exist the function returns an
    empty list — the caller then falls back to text-only mode for that
    specific video so a missing video does not nuke the entire run.
    """
    if frames_dir is None or max_frames <= 0:
        return []
    video_dir = frames_dir / video_id
    if not video_dir.is_dir():
        return []
    candidates = sorted(
        [p for p in video_dir.iterdir() if p.suffix.lower() in _FRAME_EXTS]
    )
    if not candidates:
        return []
    if len(candidates) <= max_frames:
        return candidates
    # Even stride so the sample spans the full clip.
    step = len(candidates) / max_frames
    return [candidates[int(i * step)] for i in range(max_frames)]


def _build_user_content(
    task_id: str | None,
    frame_paths: list[Path],
) -> Any:
    """Render the chat `user` message — text string or content-block list."""
    text = _task_user_text(task_id)
    if not frame_paths:
        return text
    blocks: list[dict[str, Any]] = [
        {"type": "image", "image": str(path)} for path in frame_paths
    ]
    blocks.append({"type": "text", "text": text})
    return blocks


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


def _split_examples(
    examples: list[dict[str, Any]],
    *,
    train_split: float,
    val_split: float,
    seed: int,
    group_by: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resident-aware train/val/test split.

    ``group_by='trainee'`` holds out entire trainees. Falls back to
    per-video splitting if fewer than four distinct trainees are present
    — with only a couple of trainees a resident-aware split would
    allocate zero examples to val. ``group_by='video'`` forces the
    legacy behaviour.
    """
    rng = random.Random(seed)

    def _group_key(ex: dict[str, Any]) -> str:
        meta = ex.get("metadata") or {}
        if group_by == "trainee" and meta.get("trainee_id"):
            return f"trainee::{meta['trainee_id']}"
        return f"video::{meta.get('video_id', 'unknown')}"

    groups: dict[str, list[dict[str, Any]]] = {}
    for ex in examples:
        groups.setdefault(_group_key(ex), []).append(ex)

    trainee_groups = [k for k in groups if k.startswith("trainee::")]
    if group_by == "trainee" and len(trainee_groups) < 4:
        logger.info(
            "Not enough distinct trainees (%d) for resident-aware split; "
            "falling back to per-video split.",
            len(trainee_groups),
        )
        return _split_examples(
            examples,
            train_split=train_split,
            val_split=val_split,
            seed=seed,
            group_by="video",
        )

    keys = sorted(groups.keys())
    rng.shuffle(keys)

    n_total = len(keys)
    n_train = int(n_total * train_split)
    n_val = max(1, int(n_total * val_split)) if n_total > 1 else 0
    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])
    test_keys = set(keys[n_train + n_val :])

    train = [ex for k in train_keys for ex in groups[k]]
    val = [ex for k in val_keys for ex in groups[k]]
    test = [ex for k in test_keys for ex in groups[k]]
    return train, val, test


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
    frames_dir: str | Path | None = None,
    max_frames_per_sample: int = 24,
    group_by: str = "trainee",
    exclude_video_ids: Iterable[str] | None = None,
) -> dict:
    """Build train/val/test JSONL files from scored videos.

    New optional args (2026-04-08 hardening sprint):

    * ``frames_dir`` — root directory of extracted frames. If set, each
      example's user message becomes a content-block list with up to
      ``max_frames_per_sample`` image blocks followed by a text block.
      If left ``None`` (default), the old text-only prompt is emitted.
    * ``max_frames_per_sample`` — per-example image cap.
    * ``group_by`` — how to build the val/test split. ``"trainee"``
      (default) holds out entire trainees. ``"video"`` reverts to the
      legacy per-video split. Falls back to ``"video"`` if the corpus has
      no ``trainee_id`` metadata.

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

    # Drop gold-set / held-out videos so the evaluation corpus never leaks
    # into the training split. Applied post-dedup so the excluded count is
    # logged against unique videos, not raw score records.
    if exclude_video_ids:
        exclude_set = {str(v) for v in exclude_video_ids if v}
        if exclude_set:
            before = len(samples)
            samples = [s for s in samples if s.get("video_id") not in exclude_set]
            dropped = before - len(samples)
            logger.info(
                "Excluded %d held-out videos (gold set); %d samples remain",
                dropped,
                len(samples),
            )

    frames_root = Path(frames_dir) if frames_dir else None
    if frames_root is not None and not frames_root.exists():
        logger.warning(
            "frames_dir %s does not exist; falling back to text-only prompts",
            frames_root,
        )
        frames_root = None

    vision_mode = frames_root is not None
    vision_samples = 0
    text_fallback_samples = 0

    # Build training examples
    examples: list[dict[str, Any]] = []
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

        frame_paths = _sample_frame_paths(
            frames_root, sample["video_id"], max_frames_per_sample
        )
        user_content = _build_user_content(score_data.get("task_id"), frame_paths)
        if frame_paths:
            vision_samples += 1
        elif vision_mode:
            text_fallback_samples += 1

        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "metadata": {
                "video_id": sample["video_id"],
                "source": sample.get("source", "unknown"),
                "confidence": sample.get("confidence_score", 0),
                "task_id": _canonical_task_id(score_data.get("task_id")),
                "trainee_id": get_trainee_id(score_data),
                "source_domain": get_source_domain(score_data),
                "target_score": get_total_score(score_data),
                "num_frames": len(frame_paths),
                "vision": bool(frame_paths),
            },
        }
        examples.append(example)

    if not examples:
        return {"error": "no_valid_examples", "n_candidates": len(candidates)}

    if vision_mode:
        logger.info(
            "Vision mode: %d examples with frames, %d fell back to text-only",
            vision_samples,
            text_fallback_samples,
        )

    # Split — resident-aware when possible
    train, val, test = _split_examples(
        examples,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
        group_by=group_by,
    )

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
        "vision_mode": vision_mode,
        "vision_examples": vision_samples,
        "vision_fallback_examples": text_fallback_samples,
        "frames_dir": str(frames_root) if frames_root else None,
        "max_frames_per_sample": max_frames_per_sample if vision_mode else 0,
        "split_strategy": group_by,
        "sources": {},
        "source_domains": {},
        "unique_trainees": 0,
        "seed": seed,
    }
    trainee_set: set[str] = set()
    for ex in examples:
        meta = ex["metadata"]
        src = meta.get("source", "unknown")
        manifest["sources"][src] = manifest["sources"].get(src, 0) + 1
        domain = meta.get("source_domain") or "unknown"
        manifest["source_domains"][domain] = manifest["source_domains"].get(domain, 0) + 1
        if meta.get("trainee_id"):
            trainee_set.add(str(meta["trainee_id"]))
    manifest["unique_trainees"] = len(trainee_set)

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
