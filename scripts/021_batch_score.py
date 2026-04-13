#!/usr/bin/env python3
"""Batch-score harvested FLS videos with Claude Sonnet 4 and GPT-4o."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


REPO_ROOT = Path(__file__).resolve().parent.parent
HARVEST_LOG = REPO_ROOT / "harvest_log.jsonl"
HARVEST_TARGETS_CSV = REPO_ROOT / "data" / "harvest_targets.csv"
SCORES_DIR = REPO_ROOT / "memory" / "scores"
FRAMES_DIR = REPO_ROOT / "memory" / "frames"
HARVEST_VIDEO_ROOTS = (Path.home() / "fls_harvested_videos",)
VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi"}

load_dotenv()

TASK_MAP = {
    "task1": 1,
    "task1_peg_transfer": 1,
    "task2": 2,
    "task2_pattern_cut": 2,
    "task3": 3,
    "task3_endoloop": 3,
    "task3_ligating_loop": 3,
    "task4": 4,
    "task4_extracorporeal_knot": 4,
    "task4_extracorporeal_suture": 4,
    "task5": 5,
    "task5_intracorporeal_suturing": 5,
    "task5_intracorporeal_suture": 5,
}


def extract_video_id_from_youtube_url(url: str) -> str | None:
    if not url or "v=" not in url:
        return None
    return url.split("v=", 1)[1].split("&")[0].strip() or None


def load_harvest_targets_csv() -> dict[str, str]:
    """Map YouTube video_id -> task label (e.g. task1_peg_transfer)."""
    if not HARVEST_TARGETS_CSV.exists():
        return {}

    out: dict[str, str] = {}
    with HARVEST_TARGETS_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = (row.get("url") or "").strip()
            task = (row.get("task") or "").strip()
            video_id = extract_video_id_from_youtube_url(url)
            if video_id and task:
                out[video_id] = task
                out[video_id.lower()] = task
    return out


def _video_id_from_local_filename(path: Path) -> str | None:
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        return None
    token = path.name.split(".", 1)[0].strip()
    return token or None


def build_local_video_index() -> dict[str, list[Path]]:
    """Index local harvested videos by video_id across known roots."""
    index: dict[str, list[Path]] = defaultdict(list)
    for root in HARVEST_VIDEO_ROOTS:
        if not root.exists():
            continue
        for candidate in root.rglob("*"):
            if not candidate.is_file():
                continue
            video_id = _video_id_from_local_filename(candidate)
            if not video_id:
                continue
            index[video_id].append(candidate)
            index[video_id.lower()].append(candidate)
    return dict(index)


def _video_sort_key(path: Path) -> tuple[int, int, str]:
    suffix_rank = {".mp4": 0, ".webm": 1, ".mov": 2, ".mkv": 3, ".avi": 4}
    name = path.name.lower()
    if ".f" in name:
        quality_hint = 0
    else:
        quality_hint = 1
    return (suffix_rank.get(path.suffix.lower(), 9), quality_hint, name)


def resolve_local_video_path(
    entry: dict[str, object], video_index: dict[str, list[Path]]
) -> Path | None:
    """Resolve a playable local file even when harvest log paths are stale."""
    raw_path = str(entry.get("filepath") or "").strip()
    if raw_path:
        path = Path(raw_path)
        if path.exists():
            return path
        basename_candidate = Path.home() / "fls_harvested_videos" / path.name
        if basename_candidate.exists():
            return basename_candidate

    video_id = str(entry.get("video_id") or entry.get("youtube_id") or "").strip()
    if not video_id:
        return None

    candidates = video_index.get(video_id) or video_index.get(video_id.lower()) or []
    if not candidates:
        return None
    return sorted(candidates, key=_video_sort_key)[0]


def harvest_entry_task_label(entry: dict[str, object]) -> str:
    """Task label from harvest row, falling back to legacy Task 5 markers."""
    raw = str(entry.get("task") or "").strip()
    if raw:
        return raw
    if entry.get("task5_confidence"):
        return "task5_intracorporeal_suture"
    return ""


def parse_teacher_score_filename(path: Path) -> tuple[str | None, str | None]:
    """Return (video_id, model_slug) for files like ``{id}_claude-sonnet-4_*.json``."""
    name = path.name
    if name.startswith("score_claude_") and name.endswith(".json"):
        stem = path.stem
        video_id = stem[len("score_claude_") :].rsplit("_", 1)[0]
        return video_id or None, "claude-sonnet-4"
    if name.startswith("score_gpt_") and name.endswith(".json"):
        stem = path.stem
        video_id = stem[len("score_gpt_") :].rsplit("_", 1)[0]
        return video_id or None, "gpt-4o"
    for slug in ("claude-sonnet-4", "gpt-4o"):
        marker = f"_{slug}_"
        if marker in name:
            return name.split(marker, 1)[0], slug
    return None, None


def index_teacher_scores_by_video() -> dict[str, dict[str, Path]]:
    """Map video_id -> model_slug -> latest score path."""
    grouped: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    if not SCORES_DIR.exists():
        return {}

    for path in SCORES_DIR.rglob("*.json"):
        video_id, slug = parse_teacher_score_filename(path)
        if not video_id or not slug:
            continue
        grouped[video_id][slug].append(path)

    return {
        video_id: {
            slug: max(paths, key=lambda candidate: candidate.stat().st_mtime)
            for slug, paths in model_paths.items()
        }
        for video_id, model_paths in grouped.items()
    }


def read_score_fls(path: Path) -> float:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    return float(data.get("estimated_fls_score") or 0.0)


def read_technique_summary(path: Path) -> str | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    text = (data.get("technique_summary") or data.get("reasoning") or "").strip()
    return text or None


def task_num_from_score_task_id_field(task_id: str) -> int | None:
    """Map ``task1`` / ``task5_intracorporeal_suture`` style ids to 1-5."""
    if not task_id or not str(task_id).strip().lower().startswith("task"):
        return None
    rest = str(task_id).strip().lower()[4:]
    if not rest:
        return None
    digit = rest[0]
    if digit.isdigit():
        task_num = int(digit)
        if 1 <= task_num <= 5:
            return task_num
    return None


def read_task_num_from_score_path(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return task_num_from_score_task_id_field(str(data.get("task_id") or ""))


def latest_technique_summary_for_video(
    video_id: str, index: dict[str, dict[str, Path]]
) -> str | None:
    paths = list((index.get(video_id) or {}).values())
    if not paths:
        return None
    latest = max(paths, key=lambda candidate: candidate.stat().st_mtime)
    return read_technique_summary(latest)


def read_task_num_from_score_index(video_id: str, index: dict[str, dict[str, Path]]) -> int | None:
    paths = list((index.get(video_id) or {}).values())
    if not paths:
        return None
    latest = max(paths, key=lambda candidate: candidate.stat().st_mtime)
    return read_task_num_from_score_path(latest)


def all_recorded_teacher_scores_zero(video_id: str, index: dict[str, dict[str, Path]]) -> bool:
    """True if every existing teacher score file has estimated_fls_score == 0."""
    models = index.get(video_id)
    if not models:
        return False
    for path in models.values():
        if read_score_fls(path) > 0.0:
            return False
    return True


def teacher_models_needing_task_rescore(
    video_id: str, expected_task_num: int, index: dict[str, dict[str, Path]]
) -> set[str]:
    """Return teacher model slugs whose latest saved task_id disagrees with the expected task."""
    needs: set[str] = set()
    for model_slug, path in (index.get(video_id) or {}).items():
        actual_task_num = read_task_num_from_score_path(path)
        if actual_task_num is None or actual_task_num != expected_task_num:
            needs.add(model_slug)
    return needs


def _score_text_parts(data: dict[str, object]) -> list[str]:
    parts: list[str] = []
    for key in ("technique_summary", "reasoning"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    for penalty in data.get("penalties") or []:
        if isinstance(penalty, dict):
            description = penalty.get("description")
            if isinstance(description, str) and description.strip():
                parts.append(description.strip())

    for frame in data.get("frame_analyses") or []:
        if not isinstance(frame, dict):
            continue
        for key in ("description", "technique_notes"):
            value = frame.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value.strip())
    return parts


def detect_content_task_mismatch(expected_task_num: int, score_path: Path) -> str | None:
    """Return a human-readable reason if score text suggests wrong-task or demo content."""
    try:
        data = json.loads(score_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    combined = " ".join(_score_text_parts(data)).lower()
    if not combined:
        return None

    if "task mismatch" in combined or "wrong task" in combined or "not task" in combined:
        return "explicit task mismatch language"

    for task_num in (1, 2, 3, 4, 5):
        if task_num == expected_task_num:
            continue
        if f"task {task_num}" in combined or f"task{task_num}" in combined:
            return f"explicit mention of other task {task_num}"

    inferred = infer_task_from_technique_summary(combined)
    if inferred is not None and inferred != expected_task_num:
        return f"content heuristic inferred task {inferred}"

    if any(token in combined for token in ("instructional", "educational", "demonstration")):
        return "instructional/demo content"

    return None


def infer_task_from_technique_summary(text: str) -> int | None:
    """Infer FLS task 1-5 from a prior model write-up."""
    if not text:
        return None
    lowered = text.lower()
    explicit = [
        int(match.group(1))
        for match in re.finditer(r"\b(?:fls\s+)?task\s*([1-5])\b", lowered)
    ]
    if len(explicit) == 1:
        return explicit[0]
    if len(explicit) > 1 and len(set(explicit)) == 1:
        return explicit[0]
    if len(explicit) > 1:
        return None

    if any(
        keyword in lowered
        for keyword in (
            "peg transfer",
            "six objects",
            "six rubber",
            "return transfer",
            "outbound and return",
        )
    ):
        return 1
    if any(
        keyword in lowered
        for keyword in ("pattern cut", "circular pattern", "precision cutting", "cut a circle")
    ):
        return 2
    if any(keyword in lowered for keyword in ("ligating loop", "endoloop", "loop was", "cinch")):
        return 3
    if "intracorporeal" in lowered:
        return 5
    if any(keyword in lowered for keyword in ("extracorporeal", "knot pusher")):
        return 4
    return None


def resolve_task_number(
    video_id: str,
    log_task: str,
    csv_map: dict[str, str],
    score_task_num: int | None,
    technique_summary: str | None,
) -> tuple[int, str]:
    """Pick task index 1-5 for scoring rubric selection."""
    label = csv_map.get(video_id) or csv_map.get(video_id.lower())
    if label and label in TASK_MAP:
        return TASK_MAP[label], f"harvest_targets.csv:{label}"
    if log_task and log_task in TASK_MAP:
        return TASK_MAP[log_task], f"harvest_log:{log_task}"
    if score_task_num is not None:
        return score_task_num, f"score_json.task_id->task{score_task_num}"
    if technique_summary:
        inferred = infer_task_from_technique_summary(technique_summary)
        if inferred is not None:
            return inferred, "technique_summary"
    raise ValueError(
        f"Cannot resolve task for video_id={video_id!r} log_task={log_task!r} "
        f"score_task_num={score_task_num!r}"
    )


def load_harvest_video_entries() -> dict[str, dict[str, object]]:
    """Deduplicate harvest log by video_id, keeping the last row."""
    if not HARVEST_LOG.exists():
        return {}

    latest: dict[str, dict[str, object]] = {}
    for line in HARVEST_LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        video_id = str(entry.get("video_id") or entry.get("youtube_id") or "").strip()
        filepath = str(entry.get("filepath") or "").strip()
        if not video_id or not filepath:
            continue
        latest[video_id] = entry
    return latest


def get_scored_videos() -> dict[str, set[str]]:
    """Return a mapping of video_id to completed teacher models."""
    scored: dict[str, set[str]] = {}
    if not SCORES_DIR.exists():
        return scored

    for score_path in SCORES_DIR.rglob("*.json"):
        video_id, model = parse_teacher_score_filename(score_path)
        if video_id and model:
            scored.setdefault(video_id, set()).add(model)
    return scored


def _save_frames_cache(video_id: str, frames_b64: list[str], metadata: dict) -> None:
    frame_dir = FRAMES_DIR / video_id
    frame_dir.mkdir(parents=True, exist_ok=True)

    fps = float(metadata.get("fps") or 0)
    timestamps = []
    for index in metadata.get("extracted_indices", []):
        timestamps.append(round(index / fps, 2) if fps > 0 else 0.0)

    cache = {
        "video_id": video_id,
        "frames_b64": frames_b64,
        "frame_timestamps": timestamps,
        "metadata": metadata,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    (frame_dir / "frames.json").write_text(json.dumps(cache))


def _frames_cache_path(video_id: str) -> Path:
    return FRAMES_DIR / video_id / "frames.json"


def _load_frames_cache(video_id: str) -> tuple[list[str], dict] | None:
    cache_path = _frames_cache_path(video_id)
    if not cache_path.exists():
        return None
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    frames_b64 = cache.get("frames_b64") or []
    if not isinstance(frames_b64, list) or not frames_b64:
        return None
    metadata = cache.get("metadata") or {}
    return frames_b64, metadata


def _build_task_audit(
    entries: dict[str, dict[str, object]],
    csv_map: dict[str, str],
    score_index: dict[str, dict[str, Path]],
    video_index: dict[str, list[Path]],
    task_filter: set[int] | None,
) -> dict[str, object]:
    unresolved: list[str] = []
    mismatch_rows: list[dict[str, object]] = []
    content_mismatch_rows: list[dict[str, object]] = []
    zero_total = 0
    zero_runnable = 0

    for video_id, entry in entries.items():
        log_task = harvest_entry_task_label(entry)
        summary = latest_technique_summary_for_video(video_id, score_index)
        score_task_num = read_task_num_from_score_index(video_id, score_index)
        try:
            task_num, source = resolve_task_number(
                video_id, log_task, csv_map, score_task_num, summary
            )
        except ValueError:
            unresolved.append(video_id)
            continue

        if task_filter is not None and task_num not in task_filter:
            continue

        mismatch_models = teacher_models_needing_task_rescore(video_id, task_num, score_index)
        for model_slug in sorted(mismatch_models):
            score_path = (score_index.get(video_id) or {}).get(model_slug)
            mismatch_rows.append(
                {
                    "video_id": video_id,
                    "model": model_slug,
                    "expected_task": task_num,
                    "actual_task": read_task_num_from_score_path(score_path) if score_path else None,
                    "source": source,
                    "score_path": str(score_path) if score_path else "",
                }
            )

        for model_slug, score_path in sorted((score_index.get(video_id) or {}).items()):
            reason = detect_content_task_mismatch(task_num, score_path)
            if not reason:
                continue
            try:
                data = json.loads(score_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            content_mismatch_rows.append(
                {
                    "video_id": video_id,
                    "model": model_slug,
                    "expected_task": task_num,
                    "reason": reason,
                    "score_path": str(score_path),
                    "summary": str(data.get("technique_summary") or "")[:180],
                    "source": source,
                }
            )

        if not all_recorded_teacher_scores_zero(video_id, score_index):
            continue

        zero_total += 1
        resolved_path = resolve_local_video_path(entry, video_index)
        if resolved_path is not None or _frames_cache_path(video_id).exists():
            zero_runnable += 1

    return {
        "unresolved_videos": unresolved,
        "task_mismatches": mismatch_rows,
        "content_task_mismatches": content_mismatch_rows,
        "zero_score_total": zero_total,
        "zero_score_runnable": zero_runnable,
    }


def _save_score(score) -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = SCORES_DIR / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    model_slug = "claude-sonnet-4" if "claude" in score.model_name else "gpt-4o"
    out_path = out_dir / f"{score.video_id}_{model_slug}_{timestamp}.json"
    out_path.write_text(score.model_dump_json(indent=2))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-score harvested videos. Task routing: data/harvest_targets.csv, then "
            "harvest_log task, then task_id from the latest teacher score JSON, then "
            "technique_summary / reasoning heuristics."
        )
    )
    parser.add_argument("--task", help="Filter to a harvest task label (e.g. task1_peg_transfer)")
    parser.add_argument("--max", type=int, default=30)
    parser.add_argument("--prompt-version", default="v002")
    parser.add_argument("--claude-only", action="store_true")
    parser.add_argument("--gpt-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=int, default=5, help="Seconds between API calls")
    parser.add_argument(
        "--zero-score-only",
        action="store_true",
        help=(
            "Only re-score videos whose existing teacher JSON scores are all estimated_fls_score==0. "
            "Forces both models unless --claude-only / --gpt-only is set."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        metavar="N[,N,...]",
        help="Only include these resolved task numbers, for example 1,2,3,4.",
    )
    parser.add_argument(
        "--task-mismatch-only",
        action="store_true",
        help=(
            "Only re-score teacher outputs whose latest saved task_id does not match the resolved "
            "task for that video."
        ),
    )
    parser.add_argument(
        "--audit-task-routing",
        action="store_true",
        help=(
            "Report unresolved tasks, teacher score task mismatches, and runnable zero-score videos; "
            "do not score anything."
        ),
    )
    args = parser.parse_args()

    task_filter: set[int] | None = None
    if args.tasks:
        task_filter = set()
        for part in args.tasks.replace(" ", "").split(","):
            if not part:
                continue
            task_num = int(part)
            if task_num not in (1, 2, 3, 4, 5):
                raise SystemExit(f"--tasks: invalid task number {part!r}")
            task_filter.add(task_num)

    if not HARVEST_LOG.exists():
        raise FileNotFoundError("harvest_log.jsonl not found. Run the harvester first.")

    csv_map = load_harvest_targets_csv()
    video_index = build_local_video_index()
    score_index = index_teacher_scores_by_video()
    scored = get_scored_videos()
    entries = load_harvest_video_entries()

    if args.audit_task_routing:
        audit = _build_task_audit(entries, csv_map, score_index, video_index, task_filter)
        print(
            json.dumps(
                {
                    "unresolved_count": len(audit["unresolved_videos"]),
                    "unresolved_samples": audit["unresolved_videos"][:10],
                    "task_mismatch_count": len(audit["task_mismatches"]),
                    "task_mismatch_samples": audit["task_mismatches"][:10],
                    "content_mismatch_count": len(audit["content_task_mismatches"]),
                    "content_mismatch_samples": audit["content_task_mismatches"][:10],
                    "zero_score_total": audit["zero_score_total"],
                    "zero_score_runnable": audit["zero_score_runnable"],
                },
                indent=2,
            )
        )
        return

    to_score: list[dict[str, object]] = []

    for video_id, entry in entries.items():
        log_task = harvest_entry_task_label(entry)

        summary = latest_technique_summary_for_video(video_id, score_index)
        score_task_num = read_task_num_from_score_index(video_id, score_index)
        try:
            task_num, task_source = resolve_task_number(
                video_id, log_task, csv_map, score_task_num, summary
            )
        except ValueError as exc:
            print(f"SKIP (task): {video_id} - {exc}", file=sys.stderr)
            continue

        if args.task:
            want = TASK_MAP.get(args.task)
            if want is None or task_num != want:
                continue

        if task_filter is not None and task_num not in task_filter:
            continue

        resolved_path = resolve_local_video_path(entry, video_index)
        has_cached_frames = _frames_cache_path(video_id).exists()
        if resolved_path is None and not has_cached_frames:
            continue

        mismatch_models = teacher_models_needing_task_rescore(video_id, task_num, score_index)

        needs_claude = False
        needs_gpt = False
        if args.zero_score_only or args.task_mismatch_only:
            if args.zero_score_only and all_recorded_teacher_scores_zero(video_id, score_index):
                needs_claude = needs_claude or not args.gpt_only
                needs_gpt = needs_gpt or not args.claude_only
            if args.task_mismatch_only:
                needs_claude = needs_claude or (
                    "claude-sonnet-4" in mismatch_models and not args.gpt_only
                )
                needs_gpt = needs_gpt or ("gpt-4o" in mismatch_models and not args.claude_only)
        else:
            existing = scored.get(video_id, set())
            needs_claude = "claude-sonnet-4" not in existing and not args.gpt_only
            needs_gpt = "gpt-4o" not in existing and not args.claude_only

        if needs_claude or needs_gpt:
            to_score.append(
                {
                    "video_id": video_id,
                    "filepath": str(resolved_path) if resolved_path is not None else "",
                    "log_task": log_task,
                    "task_num": task_num,
                    "task_source": task_source,
                    "has_cached_frames": has_cached_frames,
                    "needs_claude": needs_claude,
                    "needs_gpt": needs_gpt,
                }
            )

    to_score = to_score[: args.max]
    print(f"Videos to score: {len(to_score)}")

    if args.dry_run:
        for item in to_score:
            flags = []
            if item["needs_claude"]:
                flags.append("Claude")
            if item["needs_gpt"]:
                flags.append("GPT-4o")
            print(
                f"  {str(item['video_id'])[:32]:32} task={item['task_num']} "
                f"src={item['task_source'][:40]:40} needs={'+'.join(flags)}"
            )
        return

    from src.ingest.frame_extractor import extract_frames, frames_to_base64
    from src.scoring.frontier_scorer import score_with_claude, score_with_gpt

    for index, item in enumerate(to_score, start=1):
        video_id = str(item["video_id"])
        filepath = str(item["filepath"])
        task_num = int(item["task_num"])
        task_source = str(item["task_source"])

        print(f"\n{'=' * 72}")
        print(f"[{index}/{len(to_score)}] {video_id} (Task {task_num}, {task_source})")

        cached = _load_frames_cache(video_id) if bool(item["has_cached_frames"]) else None
        if cached is not None:
            frames_b64, metadata = cached
            print(f"  Loaded {len(frames_b64)} cached frames")
        else:
            if not filepath:
                print("  SKIP (no local video or cached frames available)")
                continue
            try:
                frames, metadata = extract_frames(filepath)
                frames_b64 = frames_to_base64(frames)
                _save_frames_cache(video_id, frames_b64, metadata)
                print(f"  Extracted {len(frames_b64)} frames")
            except Exception as exc:
                print(f"  SKIP (frame extraction failed): {exc}")
                continue

        if item["needs_claude"]:
            try:
                print(f"  Scoring with Claude (task={task_num}, {args.prompt_version})...")
                score = score_with_claude(
                    frames_b64=frames_b64,
                    video_id=video_id,
                    video_filename=Path(filepath).name,
                    prompt_version=args.prompt_version,
                    task=task_num,
                )
                out_path = _save_score(score)
                print(f"  Claude OK: {out_path.name}")
                time.sleep(args.delay)
            except Exception as exc:
                print(f"  Claude FAILED: {exc}")
                traceback.print_exc()

        if item["needs_gpt"]:
            try:
                print(f"  Scoring with GPT-4o (task={task_num}, {args.prompt_version})...")
                score = score_with_gpt(
                    frames_b64=frames_b64,
                    video_id=video_id,
                    video_filename=Path(filepath).name,
                    prompt_version=args.prompt_version,
                    task=task_num,
                )
                out_path = _save_score(score)
                print(f"  GPT-4o OK: {out_path.name}")
                time.sleep(args.delay)
            except Exception as exc:
                print(f"  GPT-4o FAILED: {exc}")
                traceback.print_exc()

    print("\nDone. Run: python scripts/030_run_consensus.py --prompt-version v002 --with-coach-feedback")


if __name__ == "__main__":
    main()
