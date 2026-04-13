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
    with HARVEST_TARGETS_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("url") or "").strip()
            task = (row.get("task") or "").strip()
            vid = extract_video_id_from_youtube_url(url)
            if vid and task:
                out[vid] = task
                out[vid.lower()] = task
    return out


def harvest_entry_task_label(entry: dict[str, object]) -> str:
    """Task label from log row: ``task`` field, else legacy task5 harvest markers."""
    raw = str(entry.get("task") or "").strip()
    if raw:
        return raw
    if entry.get("task5_confidence"):
        return "task5_intracorporeal_suture"
    return ""


def parse_teacher_score_filename(path: Path) -> tuple[str | None, str | None]:
    """Return (video_id, model_slug) for files like ``{id}_claude-sonnet-4_*.json``."""
    name = path.name
    for slug in ("claude-sonnet-4", "gpt-4o"):
        marker = f"_{slug}_"
        if marker in name:
            return name.split(marker, 1)[0], slug
    return None, None


def index_teacher_scores_by_video() -> dict[str, dict[str, Path]]:
    """video_id -> model_slug -> path (latest mtime per model)."""
    acc: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    if not SCORES_DIR.exists():
        return {}
    for path in SCORES_DIR.rglob("*.json"):
        video_id, slug = parse_teacher_score_filename(path)
        if not video_id or not slug:
            continue
        acc[video_id][slug].append(path)
    out: dict[str, dict[str, Path]] = {}
    for vid, models in acc.items():
        out[vid] = {
            slug: max(paths, key=lambda p: p.stat().st_mtime) for slug, paths in models.items()
        }
    return out


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
    """Map ``task1`` / ``task5_intracorporeal_suture`` style ids to 1–5."""
    if not task_id or not str(task_id).strip().lower().startswith("task"):
        return None
    rest = str(task_id).strip().lower()[4:]
    if not rest:
        return None
    digit = rest[0]
    if digit.isdigit():
        n = int(digit)
        if 1 <= n <= 5:
            return n
    return None


def read_task_num_from_score_path(path: Path) -> int | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return task_num_from_score_task_id_field(str(data.get("task_id") or ""))


def latest_technique_summary_for_video(video_id: str, index: dict[str, dict[str, Path]]) -> str | None:
    paths = list((index.get(video_id) or {}).values())
    if not paths:
        return None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    return read_technique_summary(latest)


def read_task_num_from_score_index(video_id: str, index: dict[str, dict[str, Path]]) -> int | None:
    paths = list((index.get(video_id) or {}).values())
    if not paths:
        return None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
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


def infer_task_from_technique_summary(text: str) -> int | None:
    """Infer FLS task 1–5 from a prior model write-up (fallback when labels are wrong)."""
    if not text:
        return None
    t = text.lower()
    explicit = [int(m.group(1)) for m in re.finditer(r"\b(?:fls\s+)?task\s*([1-5])\b", t)]
    if len(explicit) == 1:
        return explicit[0]
    if len(explicit) > 1 and len(set(explicit)) == 1:
        return explicit[0]
    if len(explicit) > 1:
        return None

    # Keyword hints (single task cues)
    if any(
        k in t
        for k in (
            "peg transfer",
            "six objects",
            "six rubber",
            "return transfer",
            "outbound and return",
        )
    ):
        return 1
    if any(k in t for k in ("pattern cut", "circular pattern", "precision cutting", "cut a circle")):
        return 2
    if any(k in t for k in ("ligating loop", "endoloop", "loop was", "cinch")):
        return 3
    if "intracorporeal" in t:
        return 5
    if any(k in t for k in ("extracorporeal", "knot pusher")):
        return 4
    return None


def resolve_task_number(
    video_id: str,
    log_task: str,
    csv_map: dict[str, str],
    score_task_num: int | None,
    technique_summary: str | None,
) -> tuple[int, str]:
    """Pick task index 1–5 for scoring rubric / task_context.

    Precedence: ``harvest_targets.csv`` (authoritative for Tasks 1–4 harvest), harvest log /
    legacy markers, ``task_id`` from the latest teacher score JSON, then ``technique_summary``.
    """
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
    """Deduplicate harvest log by video_id (last line wins)."""
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
        vid = str(entry.get("video_id") or entry.get("youtube_id") or "").strip()
        filepath = str(entry.get("filepath") or "").strip()
        if not vid or not filepath:
            continue
        latest[vid] = entry
    return latest


def get_scored_videos() -> dict[str, set[str]]:
    """Return a mapping of video_id to completed teacher models."""
    scored: dict[str, set[str]] = {}
    if not SCORES_DIR.exists():
        return scored

    for score_path in SCORES_DIR.rglob("*.json"):
        name = score_path.stem
        for model in ("claude-sonnet-4", "gpt-4o"):
            marker = f"_{model}"
            if marker in name:
                video_id = name.split(marker)[0]
                scored.setdefault(video_id, set()).add(model)
                break
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
            "Batch-score harvested videos. Task routing: data/harvest_targets.csv (authoritative "
            "for Tasks 1–4 targets), harvest_log task (and legacy task5_confidence rows), "
            "then task_id from the latest teacher score JSON, then technique_summary / reasoning heuristics."
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
            "Forces both models (unless --claude-only / --gpt-only). Uses task routing from CSV/summary."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        metavar="N[,N,...]",
        help="Only include these task numbers (e.g. 1,2,3,4). Matches resolved task after CSV/log routing.",
    )
    args = parser.parse_args()

    task_filter: set[int] | None = None
    if args.tasks:
        task_filter = set()
        for part in args.tasks.replace(" ", "").split(","):
            if not part:
                continue
            n = int(part)
            if n not in (1, 2, 3, 4, 5):
                raise SystemExit(f"--tasks: invalid task number {part!r}")
            task_filter.add(n)

    if not HARVEST_LOG.exists():
        raise FileNotFoundError("harvest_log.jsonl not found. Run the harvester first.")

    csv_map = load_harvest_targets_csv()
    score_index = index_teacher_scores_by_video()
    scored = get_scored_videos()
    entries = load_harvest_video_entries()
    to_score: list[dict[str, object]] = []

    for video_id, entry in entries.items():
        filepath = str(entry.get("filepath") or "").strip()
        log_task = harvest_entry_task_label(entry)
        if not Path(filepath).exists():
            continue

        summary = latest_technique_summary_for_video(video_id, score_index)
        score_task_num = read_task_num_from_score_index(video_id, score_index)
        try:
            task_num, task_source = resolve_task_number(
                video_id, log_task, csv_map, score_task_num, summary
            )
        except ValueError as exc:
            print(f"SKIP (task): {video_id} — {exc}", file=sys.stderr)
            continue

        if args.task:
            want = TASK_MAP.get(args.task)
            if want is None or task_num != want:
                continue

        if task_filter is not None and task_num not in task_filter:
            continue

        if args.zero_score_only:
            if not all_recorded_teacher_scores_zero(video_id, score_index):
                continue
            needs_claude = not args.gpt_only
            needs_gpt = not args.claude_only
        else:
            existing = scored.get(video_id, set())
            needs_claude = "claude-sonnet-4" not in existing and not args.gpt_only
            needs_gpt = "gpt-4o" not in existing and not args.claude_only

        if needs_claude or needs_gpt:
            to_score.append(
                {
                    "video_id": video_id,
                    "filepath": filepath,
                    "log_task": log_task,
                    "task_num": task_num,
                    "task_source": task_source,
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
        task_src = str(item["task_source"])

        print(f"\n{'=' * 72}")
        print(f"[{index}/{len(to_score)}] {video_id} (Task {task_num}, {task_src})")

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