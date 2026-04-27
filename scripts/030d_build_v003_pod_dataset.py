#!/usr/bin/env python3
"""Build a task-stratified v003 multimodal training dataset on the pod.

Reads every score record in ``memory/scores/`` and produces train/val/test JSONL
in ``/workspace/v003_multimodal/`` with:

* task_id resolved (backfilled from data/harvest_targets.csv when needed)
* assistant content in v003 shape via ``enrich_to_v003_target``
* user content carrying a per-task rubric summary so the LoRA learns
  task-specific scoring (not a generic "FLS score")
* up to 8 image refs per example when frames exist under
  ``/workspace/v003_frames/<video_id>/frame_*.jpg`` (text-only fallback otherwise)
* stratified split per task_id (80/10/10)
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace/FLS-Training")
sys.path.insert(0, str(ROOT))

from src.rubrics.loader import canonical_task_id  # noqa: E402
from src.training.v003_target import enrich_to_v003_target  # noqa: E402

SCORES_DIR = ROOT / "memory" / "scores"
HARVEST_CSV = ROOT / "data" / "harvest_targets.csv"
SYSTEM_PROMPT_FILE = ROOT / "prompts" / "v002_universal_scoring_system.md"

FRAMES_ROOT = Path("/workspace/v003_frames")
OUT_DIR = Path("/workspace/v003_multimodal")
MAX_FRAMES_PER_SAMPLE = 8
SEED = 42

VALID_TASKS = {"task1", "task2", "task3", "task4", "task5", "task6"}
SOURCE_PRIORITY = {
    "consensus": 0,
    "teacher_claude": 1,
    "teacher_gpt": 2,
    "claude_only_high_conf": 3,
}


def _csv_lookup() -> dict[str, str]:
    out: dict[str, str] = {}
    if not HARVEST_CSV.exists():
        return out
    with HARVEST_CSV.open() as f:
        for row in csv.DictReader(f):
            url = row.get("url", "")
            task = (row.get("task") or "").strip()
            m = re.search(r"[?&]v=([^&]+)", url)
            if m and task:
                out[m.group(1)] = task
                out[f"yt_{m.group(1)}"] = task
    return out


def _resolve_task(score: dict, csv_lookup: dict[str, str]) -> str | None:
    raw = (score.get("task_id") or "").strip()
    canonical = canonical_task_id(raw or "task5")
    if (not raw) or raw.lower() in {"unclassified", "unknown", ""}:
        vid = score.get("video_id", "")
        cand = csv_lookup.get(vid) or csv_lookup.get(vid.replace("yt_", ""))
        if cand:
            canonical = canonical_task_id(cand)
    return canonical if canonical in VALID_TASKS else None


def _best_per_video() -> dict[str, dict]:
    by_video: dict[str, dict] = {}
    for path in SCORES_DIR.rglob("*.json"):
        if "quarantine" in str(path):
            continue
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        vid = d.get("video_id")
        if not vid:
            continue
        score_fls = float(d.get("estimated_fls_score") or 0)
        avulsed = (d.get("drain_assessment") or {}).get("drain_avulsed")
        if score_fls <= 0 and not avulsed:
            continue
        prio = SOURCE_PRIORITY.get(d.get("source", "teacher_claude"), 9)
        if vid not in by_video or prio < SOURCE_PRIORITY.get(by_video[vid].get("source", ""), 9):
            by_video[vid] = d
    return by_video


def _user_content(target: dict, video_id: str) -> tuple[list | str, int]:
    """Return user content + number of attached frames."""
    images: list[dict] = []
    frames_dir = FRAMES_ROOT / video_id
    if frames_dir.is_dir():
        jpgs = sorted(frames_dir.glob("frame_*.jpg"))
        if len(jpgs) > MAX_FRAMES_PER_SAMPLE:
            step = len(jpgs) / MAX_FRAMES_PER_SAMPLE
            jpgs = [jpgs[int(i * step)] for i in range(MAX_FRAMES_PER_SAMPLE)]
        images = [{"type": "image", "image": f"file://{p}"} for p in jpgs]

    task_name = target.get("task_name", "")
    max_score = int(target["score_components"]["max_score"])
    max_time = int(target.get("max_time_seconds") or 0)
    official = target.get("official_fls_task")

    rubric_summary = (
        f"Task {target['task_id']}: {task_name}. "
        f"Max score {max_score} (denominator). "
        f"Max time {max_time} s. "
        f"Score formula: max_score - completion_time - penalties (auto-zero on auto_fail). "
        f"Official FLS task: {official}."
    )
    user_text = (
        f"{rubric_summary}\n"
        "Score this performance and emit a complete v003 scoring JSON: "
        "score_components (with formula_applied), each penalty with "
        "points_deducted/severity/confidence/frame_evidence, critical_errors "
        "(forces_zero_score / blocks_proficiency_claim where appropriate), "
        "cannot_determine, confidence_rationale, task_specific_assessments. "
        "Do not invent evidence; never claim proficiency when critical errors exist."
    )
    if images:
        return images + [{"type": "text", "text": user_text}], len(images)
    return user_text, 0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_lookup = _csv_lookup()
    by_video = _best_per_video()
    system_prompt = (
        SYSTEM_PROMPT_FILE.read_text()
        if SYSTEM_PROMPT_FILE.exists()
        else "You are an FLS proctor. Output v003 scoring JSON."
    )

    per_task: dict[str, list[dict]] = defaultdict(list)
    drops: dict[str, int] = defaultdict(int)
    vision_count = 0
    text_only_count = 0

    for vid, score in by_video.items():
        task_id = _resolve_task(score, csv_lookup)
        if task_id is None:
            drops["no_task"] += 1
            continue
        score["task_id"] = task_id
        try:
            target = enrich_to_v003_target(score, task_id)
        except Exception as exc:  # noqa: BLE001
            drops[f"enrich_error_{type(exc).__name__}"] += 1
            continue

        user_content, n_frames = _user_content(target, vid)
        if n_frames:
            vision_count += 1
        else:
            text_only_count += 1

        example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(target, default=str)},
            ],
            "metadata": {
                "video_id": vid,
                "task_id": task_id,
                "training_score": target["score_components"]["total_fls_score"],
                "max_score": target["score_components"]["max_score"],
                "has_critical_error": bool(target["critical_errors"]),
                "vision": bool(n_frames),
                "n_frames": n_frames,
                "schema_version": "v003",
            },
        }
        per_task[task_id].append(example)

    rng = random.Random(SEED)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for task_id, rows in per_task.items():
        rng.shuffle(rows)
        n = len(rows)
        n_test = max(1, n // 10)
        n_val = max(1, n // 10)
        splits["test"].extend(rows[:n_test])
        splits["val"].extend(rows[n_test:n_test + n_val])
        splits["train"].extend(rows[n_test + n_val:])

    for name, rows in splits.items():
        rng.shuffle(rows)
        with (OUT_DIR / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  {name}.jsonl: {len(rows)}")

    manifest = {
        "dataset": "v003_multimodal",
        "schema_version": "v003",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "totals": {k: len(v) for k, v in splits.items()},
        "task_distribution": {t: len(rows) for t, rows in per_task.items()},
        "vision_examples": vision_count,
        "text_only_examples": text_only_count,
        "drops": dict(drops),
        "max_frames_per_sample": MAX_FRAMES_PER_SAMPLE,
        "system_prompt_file": str(SYSTEM_PROMPT_FILE),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
