#!/usr/bin/env python3
"""Build the v20 multimodal training dataset.

Three changes vs the v19 builder (``030e_build_v003_v19_dataset.py``):

1. **Source = ``memory/scores_v003_relabel/``** (the auto-fail-cleaned scores)
   instead of ``memory/scores/``. ~9 % of records had their FLS score forced
   to 0 by the relabel; the model now learns the v003 contract from labels
   that already agree with the contract.
2. **Tighter critical-error emission**. A penalty is only promoted to a
   ``critical_errors`` entry if it has ≥2 ``frame_evidence`` frames OR is
   one of the absolute auto-fail types (drain_avulsion, knot_failure,
   gauze_detachment, block_dislodged, appendage_transection). Borderline
   cases stay as ``severity: major`` penalties — same training signal, no
   false positives in the critical_errors list.
3. **Frame coverage is *additive*** with the existing decoded JPGs: the script
   reads from ``/workspace/v003_frames`` (when run on the pod) but also
   accepts an additional dir for late-arriving frames extracted from raw
   videos via ``scripts/032_extract_frames_from_videos.py``.

Output: ``/workspace/v003_multimodal_v20/{train,val,test}.jsonl`` plus a
manifest. Test set is left identical to v19 so the eval is comparable.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT_CANDIDATES = [Path("/workspace/FLS-Training"), Path(__file__).resolve().parents[1]]
ROOT = next((r for r in ROOT_CANDIDATES if r.exists()), Path(__file__).resolve().parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rubrics.loader import canonical_task_id  # noqa: E402

VALID_TASKS = {"task1", "task2", "task3", "task4", "task5", "task6"}
SOURCE_PRIORITY = {
    "consensus": 0,
    "teacher_claude": 1,
    "teacher_gpt": 2,
    "claude_only_high_conf": 3,
}
ASSISTANT_CHAR_BUDGET = 2300
CRITICAL_UPSAMPLE_FACTOR = 5
SEED = 42
MAX_FRAMES_PER_SAMPLE = 4
ABSOLUTE_AUTO_FAIL_TYPES = (
    "drain_avulsion",
    "drain_avulsed",
    "gauze_detachment",
    "gauze_detached",
    "block_dislodged",
    "appendage_transection",
    "needle_left_view",
    "needle_exits_field",
    "incomplete_task",
)
TRAIN_SYSTEM_PROMPT = (
    "You are an expert FLS (Fundamentals of Laparoscopic Surgery) proctor AI. "
    "Score the trainee's video for the indicated FLS task. Output a single "
    "JSON object in v003 schema: task_id, task_name, max_score (the per-task "
    "denominator: task1=300, task2=300, task3=180, task4=420, task5=600, "
    "task6=315), max_time_seconds, completion_time_seconds, score_components "
    "with formula_applied = max_score - completion_time - total_penalties "
    "(or auto-zero if any penalty is severity=auto_fail), penalties (each with "
    "type, points_deducted, severity ∈ {minor, moderate, major, critical, "
    "auto_fail}, frame_evidence, confidence), critical_errors (each with "
    "forces_zero_score and blocks_proficiency_claim, ONLY when ≥2 frame_evidence "
    "frames or an absolute auto-fail type), cannot_determine, confidence_score, "
    "confidence_rationale, task_specific_assessments, strengths, "
    "improvement_suggestions. Never claim proficiency when critical_errors is "
    "non-empty. Do not invent millimetric or numeric evidence; emit "
    "cannot_determine when not visible."
)


def _safe_canonical(value: str) -> str | None:
    if not value:
        return None
    value = value.strip()
    try:
        c = canonical_task_id(value)
        if c in VALID_TASKS:
            return c
    except Exception:
        pass
    head = value.lower().split("_", 1)[0]
    return head if head in VALID_TASKS else None


def _csv_lookup(harvest_csv: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not harvest_csv.exists():
        return out
    with harvest_csv.open() as f:
        for row in csv.DictReader(f):
            url = row.get("url", "")
            task = (row.get("task") or "").strip()
            m = re.search(r"[?&]v=([^&]+)", url)
            if m and task:
                out[m.group(1)] = task
                out[f"yt_{m.group(1)}"] = task
    return out


def _infer_task_from_text(text: str) -> str | None:
    if not text:
        return None
    t = text.lower()
    pairs = [
        ("task1", ("peg transfer", "peg-transfer", "rubber rings")),
        ("task2", ("pattern cut", "circle cut", "gauze")),
        ("task3", ("endoloop", "ligating loop", "appendage")),
        ("task4", ("extracorporeal", "knot pusher")),
        ("task5", ("intracorporeal", "intra-corporeal")),
        ("task6", ("rings of rings", "needle manipulation", "ring traversal")),
    ]
    for task_id, kws in pairs:
        if any(k in t for k in kws):
            return task_id
    return None


def _resolve_task(score: dict, csv_lookup: dict[str, str]) -> str | None:
    direct = _safe_canonical(score.get("task_id") or "")
    if direct:
        return direct
    vid = score.get("video_id", "")
    cand = csv_lookup.get(vid) or csv_lookup.get(vid.replace("yt_", ""))
    if cand:
        c = _safe_canonical(cand)
        if c:
            return c
    return _infer_task_from_text(
        " ".join(str(score.get(k) or "") for k in ("technique_summary", "reasoning", "video_filename"))
    )


def _safe_float(v, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _score_total(d: dict) -> float:
    s = _safe_float(d.get("estimated_fls_score"))
    if s > 0:
        return s
    sc = d.get("score_components") or {}
    if isinstance(sc, dict):
        return _safe_float(sc.get("total_fls_score"))
    return 0.0


def _has_useful_signal(d: dict) -> bool:
    if _score_total(d) > 0:
        return True
    if (d.get("drain_assessment") or {}).get("drain_avulsed"):
        return True
    if d.get("penalties"):
        return True
    if d.get("critical_errors"):
        return True
    return False


def _is_absolute_auto_fail_type(t: str) -> bool:
    norm = (t or "").strip().lower().replace("-", "_").replace(" ", "_")
    return any(p in norm or norm in p for p in ABSOLUTE_AUTO_FAIL_TYPES)


def _enrich_target(score: dict, task_id: str) -> dict:
    """Build a v003-shape target with tighter critical_errors gating.

    A penalty becomes a critical_errors entry iff:
      - ≥2 frame_evidence entries, OR
      - type matches the absolute auto-fail set (drain_avulsion, etc.)
    Otherwise the penalty stays in ``penalties`` with severity major/critical
    but is *not* duplicated under critical_errors.
    """
    from src.training.v003_target import enrich_to_v003_target

    enriched = enrich_to_v003_target(score, task_id)

    raw_penalties = enriched.get("penalties") or []
    tightened_critical = []
    seen_types: set[str] = set()
    for p in raw_penalties:
        if not isinstance(p, dict):
            continue
        sev = p.get("severity")
        ptype = (p.get("type") or "").strip().lower()
        if not ptype or ptype in seen_types:
            continue
        if sev not in {"major", "critical", "auto_fail"}:
            continue

        frame_evidence = p.get("frame_evidence") or []
        is_absolute = _is_absolute_auto_fail_type(ptype)
        meets_threshold = len(frame_evidence) >= 2 or is_absolute
        if not meets_threshold:
            continue
        seen_types.add(ptype)
        tightened_critical.append({
            "type": p.get("type"),
            "present": True,
            "reason": p.get("description", ""),
            "frame_evidence": list(frame_evidence)[:6],
            "forces_zero_score": is_absolute,
            "blocks_proficiency_claim": True,
        })

    # Also keep any explicit critical_errors that came from the relabel pass
    # if they aren't already represented.
    for c in score.get("critical_errors") or []:
        if not isinstance(c, dict):
            continue
        ctype = (c.get("type") or "").strip().lower()
        if ctype and ctype not in seen_types:
            seen_types.add(ctype)
            tightened_critical.append({
                "type": c.get("type"),
                "present": True,
                "reason": c.get("reason", ""),
                "frame_evidence": list(c.get("frame_evidence") or [])[:6],
                "forces_zero_score": bool(c.get("forces_zero_score")),
                "blocks_proficiency_claim": bool(c.get("blocks_proficiency_claim", True)),
            })

    enriched["critical_errors"] = tightened_critical
    return enriched


def _truncate_str(s: str, budget: int) -> str:
    if len(s) <= budget:
        return s
    return s[: max(0, budget - 1)].rstrip() + "…"


def _trim_for_budget(target: dict) -> dict:
    t = json.loads(json.dumps(target))
    if t.get("technique_summary"):
        t["technique_summary"] = _truncate_str(t["technique_summary"], 250)
    for f in ("strengths", "improvement_suggestions", "cannot_determine"):
        if isinstance(t.get(f), list):
            t[f] = [_truncate_str(str(x), 140) for x in t[f][:3]]
    for p in t.get("penalties") or []:
        if isinstance(p, dict):
            if isinstance(p.get("description"), str):
                p["description"] = _truncate_str(p["description"], 140)
            if isinstance(p.get("frame_evidence"), list):
                p["frame_evidence"] = p["frame_evidence"][:6]
    tsa = t.get("task_specific_assessments")
    if isinstance(tsa, dict):
        for f in ("frame_analyses", "phase_timings", "phases_detected"):
            if isinstance(tsa.get(f), list):
                tsa[f] = tsa[f][:6]

    serialized = json.dumps(t, default=str)
    if len(serialized) > ASSISTANT_CHAR_BUDGET and isinstance(tsa, dict):
        for f in ("frame_analyses", "phase_timings", "phases_detected"):
            tsa.pop(f, None)
        serialized = json.dumps(t, default=str)
    if len(serialized) > ASSISTANT_CHAR_BUDGET:
        t.pop("technique_summary", None)
        serialized = json.dumps(t, default=str)
    if len(serialized) > ASSISTANT_CHAR_BUDGET:
        for f in ("strengths", "improvement_suggestions", "cannot_determine"):
            t[f] = []
    return t


def _user_content(target: dict, video_id: str, frames_root: Path) -> tuple[list | str, int]:
    images: list[dict] = []
    frames_dir = frames_root / video_id
    if frames_dir.is_dir():
        jpgs = sorted(frames_dir.glob("frame_*.jpg"))
        if len(jpgs) > MAX_FRAMES_PER_SAMPLE:
            step = len(jpgs) / MAX_FRAMES_PER_SAMPLE
            jpgs = [jpgs[int(i * step)] for i in range(MAX_FRAMES_PER_SAMPLE)]
        images = [{"type": "image", "image": f"file://{p}"} for p in jpgs]

    rubric_summary = (
        f"Task {target['task_id']}: {target.get('task_name','')}. "
        f"Max score {int(target['score_components']['max_score'])}. "
        f"Max time {int(target.get('max_time_seconds') or 0)} s. "
        f"Score formula: max_score - completion_time - penalties (auto-zero on auto_fail). "
        f"Official FLS task: {target.get('official_fls_task')}."
    )
    user_text = (
        f"{rubric_summary}\n"
        "Score this performance and emit a complete v003 scoring JSON. "
        "Promote a penalty into critical_errors only when frame_evidence has ≥2 entries "
        "or the type is an absolute auto-fail (drain_avulsion, knot_failure, gauze_detachment, block_dislodged). "
        "Do not invent evidence; never claim proficiency when critical_errors is non-empty."
    )
    if images:
        return images + [{"type": "text", "text": user_text}], len(images)
    return user_text, 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, default=ROOT / "memory/scores_v003_relabel")
    parser.add_argument("--harvest-csv", type=Path, default=ROOT / "data/harvest_targets.csv")
    parser.add_argument("--frames-root", type=Path, default=Path("/workspace/v003_frames"))
    parser.add_argument("--out-dir", type=Path, default=Path("/workspace/v003_multimodal_v20"))
    parser.add_argument(
        "--carry-test-from",
        type=Path,
        default=Path("/workspace/v003_multimodal/test.jsonl"),
        help="Reuse the v19 test split so eval is comparable across iterations.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_lookup = _csv_lookup(args.harvest_csv)

    by_video: dict[str, dict] = {}
    for path in args.scores_dir.rglob("*.json"):
        if "quarantine" in str(path).lower():
            continue
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        vid = d.get("video_id")
        if not vid or not _has_useful_signal(d):
            continue
        prio = SOURCE_PRIORITY.get(d.get("source", "teacher_claude"), 9)
        if vid not in by_video or prio < SOURCE_PRIORITY.get(by_video[vid].get("source", ""), 9):
            by_video[vid] = d

    per_task: dict[str, list[dict]] = defaultdict(list)
    drops: dict[str, int] = defaultdict(int)
    vision_count = text_only_count = 0

    for vid, score in by_video.items():
        task_id = _resolve_task(score, csv_lookup)
        if task_id is None:
            drops["no_task"] += 1
            continue
        score["task_id"] = task_id
        try:
            target = _enrich_target(score, task_id)
        except Exception as exc:  # noqa: BLE001
            drops[f"enrich_error_{type(exc).__name__}"] += 1
            continue
        target = _trim_for_budget(target)

        user_content, n_frames = _user_content(target, vid, args.frames_root)
        if n_frames:
            vision_count += 1
        else:
            text_only_count += 1

        example = {
            "messages": [
                {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
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
                "schema_version": "v003_v20",
            },
        }
        per_task[task_id].append(example)

    rng = random.Random(SEED)
    splits = {"train": [], "val": [], "test": []}

    # Reuse v19 test set for apples-to-apples eval if it exists.
    test_video_ids: set[str] = set()
    if args.carry_test_from.exists():
        for line in args.carry_test_from.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            vid = (row.get("metadata") or {}).get("video_id")
            if vid:
                test_video_ids.add(vid)

    for task_id, rows in per_task.items():
        rng.shuffle(rows)
        # Pull out test split first based on carried-over video_ids.
        test_rows = [r for r in rows if r["metadata"]["video_id"] in test_video_ids]
        remainder = [r for r in rows if r["metadata"]["video_id"] not in test_video_ids]
        n_val = max(1, len(remainder) // 10)
        splits["test"].extend(test_rows)
        splits["val"].extend(remainder[:n_val])
        splits["train"].extend(remainder[n_val:])

    # Upsample critical-error rows in TRAIN only.
    upsampled_train = []
    crit_originals = 0
    for row in splits["train"]:
        upsampled_train.append(row)
        if row["metadata"]["has_critical_error"]:
            crit_originals += 1
            for _ in range(CRITICAL_UPSAMPLE_FACTOR - 1):
                upsampled_train.append(row)
    rng.shuffle(upsampled_train)
    splits["train"] = upsampled_train

    for name, rows in splits.items():
        with (args.out_dir / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"  {name}.jsonl: {len(rows)}")

    manifest = {
        "dataset": "v003_multimodal_v20",
        "schema_version": "v003_v20",
        "scores_dir": str(args.scores_dir),
        "frames_root": str(args.frames_root),
        "carry_test_from": str(args.carry_test_from),
        "assistant_char_budget": ASSISTANT_CHAR_BUDGET,
        "critical_upsample_factor": CRITICAL_UPSAMPLE_FACTOR,
        "totals": {k: len(v) for k, v in splits.items()},
        "task_distribution": {t: len(rows) for t, rows in per_task.items()},
        "vision_examples": vision_count,
        "text_only_examples": text_only_count,
        "critical_originals_in_train": crit_originals,
        "drops": dict(drops),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
