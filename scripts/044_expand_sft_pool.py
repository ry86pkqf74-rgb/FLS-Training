#!/usr/bin/env python3
"""Expand the YouTube-side v002 SFT pool by:
  1. Folding in 2026-04-08_v4 (144 train / 27 val / 9 test rows) after
     converting their old score_components shape (time_score, penalty_deductions,
     total_fls_score) to the v002 shape (max_score, time_used, total_penalties,
     total_fls_score, formula_applied).
  2. Folding in gold/20260408_123618_final_gold.jsonl (33 rows) with the same
     schema conversion.
  3. Optionally merging LASANA rescore outputs under /data/fls/scored/lasana_v002
     (produced by scripts/080_lasana_rescore_v002.py) once available. Pass the
     path with --lasana-dir.

Outputs (train/val/test .jsonl) are v002-schema-clean and VL-ready. Each
example is a dict with:
  video_id, task_id, target (v002 ScoringResult dict), frames (list of paths),
  consensus_conf (if available)

Downstream: wire these into 043_train_qwen_vl_v3.py as the new --train-path, or
replace /workspace/yt_train.jsonl on the pod.

Task max_score reference (v002):
  task1_peg_transfer: 300, task2_pattern_cut: 300, task3_endoloop: 300,
  task4_extracorporeal_knot: 600, task5_intracorporeal_suturing: 600
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TASK_MAX_SCORE = {
    "task1_peg_transfer": 300,
    "task2_pattern_cut": 300,
    "task3_endoloop": 300,
    "task3_ligating_loop": 300,
    "task4_extracorporeal_knot": 600,
    "task4_extracorporeal_suture": 600,
    "task5_intracorporeal_suturing": 600,
    "task5_intracorporeal_suture": 600,
}
# Canonical aliases
TASK_CANON = {
    "task1": "task1_peg_transfer",
    "task2": "task2_pattern_cut",
    "task3": "task3_endoloop",
    "task3_ligating_loop": "task3_endoloop",
    "task4": "task4_extracorporeal_knot",
    "task4_extracorporeal_suture": "task4_extracorporeal_knot",
    "task5": "task5_intracorporeal_suturing",
    "task5_intracorporeal_suture": "task5_intracorporeal_suturing",
}


def canon_task(t: str | None) -> str | None:
    if not t: return None
    return TASK_CANON.get(t, t)


def old_to_v002_target(target: dict, task_id: str) -> dict | None:
    """Convert old score_components {time_score, penalty_deductions, total_fls_score}
    into the v002 shape. Returns None on conversion failure."""
    if not isinstance(target, dict): return None
    sc_old = target.get("score_components", {})
    if not isinstance(sc_old, dict): return None

    total = sc_old.get("total_fls_score")
    penalties = sc_old.get("penalty_deductions", sc_old.get("total_penalties"))
    time_score = sc_old.get("time_score")

    task_canon = canon_task(task_id)
    max_score = TASK_MAX_SCORE.get(task_canon, 300)

    # v002 uses: total_fls_score = max_score - time_used - total_penalties
    # Old schema: total_fls_score = time_score - penalty_deductions
    #             where time_score = max_score - time_used
    # So: time_used = max_score - time_score
    time_used = None
    if isinstance(time_score, (int, float)):
        time_used = max(0.0, float(max_score) - float(time_score))

    if total is None or penalties is None:
        return None

    # Clamp FLS to [0, max_score] per FLS rubric (minimum score is 0).
    # Without this, rows with time_used + penalties > max_score produced
    # negative labels (-200, -265, etc.) teaching the model nonsense.
    total_clamped = max(0.0, min(float(max_score), float(total)))
    new_sc = {
        "max_score": max_score,
        "time_used": time_used if time_used is not None else 0.0,
        "total_penalties": float(penalties),
        "total_fls_score": total_clamped,
        "formula_applied": (
            f"{max_score} - {time_used:.1f} - {float(penalties):.1f} = {total_clamped:.1f}"
            if time_used is not None else f"total={total_clamped}"
        ),
    }
    total = total_clamped

    # Build canonical v002 target — SAME shape as LASANA rows — so the model
    # sees a single output schema across all sources. Drop everything else.
    video_id = target.get("video_id") or "unknown"
    canonical = {
        "id": target.get("id") or f"score_v4_{video_id}",
        "video_id": video_id,
        "video_filename": target.get("video_filename") or f"{video_id}.mp4",
        "source": target.get("source") or "v4",
        "model_name": target.get("model_name") or "claude-sonnet-4-20250514",
        "model_version": target.get("model_version") or "claude-sonnet-4-20250514",
        "prompt_version": target.get("prompt_version") or "v002",
        "task_id": task_canon,
        "completion_time_seconds": float(target.get("completion_time_seconds") or (time_used or 0.0)),
        "penalties": target.get("penalties") or [],
        "score_components": new_sc,
        "estimated_fls_score": float(total),
        "confidence": float(target.get("confidence") or target.get("confidence_score") or 0.5),
    }
    return canonical


def row_from_v4(row: dict, frames_root: Path) -> dict | None:
    """2026-04-08_v4 rows are chat-format (messages). Extract video_id + target."""
    video_id = None
    task_id = None
    target_candidate = None
    user_task = None
    for m in row.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            if m.get("role") == "user":
                # user prompt carries "Task ID: taskN" (exact form used by 030)
                um = re.search(r'Task ID:\s*(task\d\w*)', c)
                if um: user_task = um.group(1)
            if m.get("role") == "assistant":
                try:
                    obj = json.loads(c)
                except Exception:
                    jm = re.search(r"\{.*\}", c, re.S)
                    obj = None
                    if jm:
                        try: obj = json.loads(jm.group(0))
                        except Exception: pass
                if obj is not None:
                    # v4 wraps the target in {"scoring_result": {...}}
                    target_candidate = obj.get("scoring_result") if isinstance(obj, dict) and "scoring_result" in obj else obj
    if not target_candidate: return None
    # Prefer the target's own task_id; fall back to the user-prompt task label
    task_id = canon_task(target_candidate.get("task_id") or user_task)
    if not task_id: return None
    video_id = target_candidate.get("video_id") or "unknown"
    new_target = old_to_v002_target(target_candidate, task_id)
    if new_target is None: return None
    frames = resolve_frames(video_id, frames_root)
    return {
        "video_id": video_id, "task_id": task_id, "target": new_target,
        "frames": frames, "source": "v4",
        "consensus_conf": target_candidate.get("confidence", 0.5),
    }


def row_from_gold(row: dict, frames_root: Path) -> dict | None:
    video_id = row.get("video_id") or "unknown"
    task_id = canon_task(row.get("task_id") or row.get("target", {}).get("task_id"))
    if not task_id: return None
    new_target = old_to_v002_target(row.get("target", {}), task_id)
    if new_target is None: return None
    return {
        "video_id": video_id, "task_id": task_id, "target": new_target,
        "frames": resolve_frames(video_id, frames_root), "source": "gold",
        "consensus_conf": row.get("confidence", 1.0),
    }


def row_from_lasana(path: Path, frames_root: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    # Filename: lasana_<prefix>_<slug>_claude-sonnet-4.json
    stem = path.stem
    video_id = stem.rsplit("_claude-sonnet-4", 1)[0]
    task_id = canon_task(d.get("task_id"))
    if not task_id: return None
    sc = d.get("score_components")
    if not isinstance(sc, dict) or sc.get("total_fls_score") is None:
        return None
    # Clamp FLS to [0, max_score]: Claude teacher occasionally returns
    # negatives when time_used + penalties > max_score.
    _ms = sc.get("max_score") or 300
    sc["total_fls_score"] = max(0.0, min(float(_ms), float(sc["total_fls_score"])))
    # Build a v002-canonical target (match v4/YT row shape) so the model sees
    # ONE output schema across all sources. Drop LASANA-native fields that
    # don't appear in v4 rows: frame_analyses, task_name, max_time_seconds,
    # phase_timings, task_specific_assessments, strengths, improvement_suggestions,
    # technique_summary, cannot_determine, confidence_rationale, and all _debug keys.
    canonical = {
        "id": f"score_lasana_{video_id}",
        "video_id": video_id,
        "video_filename": f"{video_id}.mp4",
        "source": "lasana_rescore",
        "model_name": "claude-sonnet-4-20250514",
        "model_version": "claude-sonnet-4-20250514",
        "prompt_version": "v002",
        "task_id": task_id,
        "completion_time_seconds": float(d.get("completion_time_seconds") or sc.get("time_used") or 0.0),
        "penalties": d.get("penalties") or [],
        "score_components": sc,
        "estimated_fls_score": float(sc.get("total_fls_score")),
        "confidence": float(d.get("confidence", 0.7)),
    }
    return {
        "video_id": video_id, "task_id": task_id, "target": canonical,
        "frames": resolve_lasana_frames(video_id, frames_root),
        "source": "lasana_rescore",
        "consensus_conf": float(d.get("confidence", 0.7)),
    }


def resolve_frames(video_id: str, root: Path) -> list[str]:
    """Return sorted frame paths for a YouTube video (if found)."""
    for cand in (root / video_id, root / f"yt_{video_id}"):
        if cand.exists():
            return sorted(str(p) for p in cand.glob("*.jpg"))
    return []


def resolve_lasana_frames(video_id: str, root: Path) -> list[str]:
    d = root / video_id
    if d.exists():
        return sorted(str(p) for p in d.glob("frame_*.jpg"))
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v4-dir", default=str(REPO / "data/training/2026-04-08_v4"))
    ap.add_argument("--gold-file", default=str(
        REPO / "data/training/gold/20260408_123618_final_gold.jsonl"))
    ap.add_argument("--lasana-dir", default="",
                    help="LASANA scored dir (e.g. /data/fls/scored/lasana_v002 on S8)")
    ap.add_argument("--yt-frames-root", default=str(REPO / "memory/frames"))
    ap.add_argument("--lasana-frames-root", default="/data/fls/lasana_processed/frames")
    ap.add_argument("--out-dir", default=str(REPO / "data/training/youtube_sft_v2"))
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--require-frames", action="store_true",
                    help="drop rows with no resolved frames (recommended for VL)")
    args = ap.parse_args()

    all_rows: list[dict] = []
    dropped = {"v4_schema": 0, "v4_noframe": 0, "gold_schema": 0, "gold_noframe": 0,
               "lasana_schema": 0, "lasana_noframe": 0, "low_conf": 0}

    yt_frames_root = Path(args.yt_frames_root)
    lasana_frames_root = Path(args.lasana_frames_root)

    # v4
    v4_dir = Path(args.v4_dir)
    for split in ("train", "val", "test"):
        p = v4_dir / f"{split}.jsonl"
        if not p.exists(): continue
        for line in p.read_text().splitlines():
            row = json.loads(line)
            r = row_from_v4(row, yt_frames_root)
            if r is None: dropped["v4_schema"] += 1; continue
            if args.require_frames and not r["frames"]:
                dropped["v4_noframe"] += 1; continue
            all_rows.append(r)
    print(f"v4: kept {sum(1 for r in all_rows if r['source']=='v4')} | "
          f"schema-dropped {dropped['v4_schema']} noframe {dropped['v4_noframe']}")

    # gold
    gold = Path(args.gold_file)
    if gold.exists():
        for line in gold.read_text().splitlines():
            row = json.loads(line)
            r = row_from_gold(row, yt_frames_root)
            if r is None: dropped["gold_schema"] += 1; continue
            if args.require_frames and not r["frames"]:
                dropped["gold_noframe"] += 1; continue
            all_rows.append(r)
    print(f"gold: kept {sum(1 for r in all_rows if r['source']=='gold')} | "
          f"schema-dropped {dropped['gold_schema']} noframe {dropped['gold_noframe']}")

    # lasana
    if args.lasana_dir:
        ld = Path(args.lasana_dir)
        if ld.exists():
            for p in sorted(ld.glob("*_claude-sonnet-4.json")):
                r = row_from_lasana(p, lasana_frames_root)
                if r is None: dropped["lasana_schema"] += 1; continue
                if args.require_frames and not r["frames"]:
                    dropped["lasana_noframe"] += 1; continue
                all_rows.append(r)
        print(f"lasana: kept {sum(1 for r in all_rows if r['source']=='lasana_rescore')}"
              f" | schema-dropped {dropped['lasana_schema']} "
              f"noframe {dropped['lasana_noframe']}")

    # conf filter
    before = len(all_rows)
    all_rows = [r for r in all_rows if (r.get("consensus_conf") or 0) >= args.min_conf]
    dropped["low_conf"] = before - len(all_rows)

    # dedupe by video_id (keep highest-confidence row)
    by_vid: dict[str, dict] = {}
    for r in all_rows:
        v = r["video_id"]
        if v not in by_vid or (r.get("consensus_conf") or 0) > (by_vid[v].get("consensus_conf") or 0):
            by_vid[v] = r
    all_rows = list(by_vid.values())

    print(f"\nTotal after dedup + conf>={args.min_conf}: {len(all_rows)}")
    print(f"Dropped: {dropped}")

    # By task
    from collections import Counter
    task_counts = Counter(r["task_id"] for r in all_rows)
    for t, c in sorted(task_counts.items()):
        print(f"  {t}: {c}")

    # Shuffle + stratified-ish split by task_id
    random.seed(args.seed)
    by_task: dict[str, list] = {}
    for r in all_rows:
        by_task.setdefault(r["task_id"], []).append(r)

    train, val, test = [], [], []
    for task, rows in by_task.items():
        random.shuffle(rows)
        n = len(rows)
        n_test = max(1, int(n * args.test_frac)) if n >= 10 else 0
        n_val  = max(1, int(n * args.val_frac)) if n >= 5 else 0
        test.extend(rows[:n_test])
        val.extend(rows[n_test:n_test+n_val])
        train.extend(rows[n_test+n_val:])

    random.shuffle(train); random.shuffle(val); random.shuffle(test)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("val", val), ("test", test)]:
        p = out_dir / f"yt_{name}.jsonl"
        with p.open("w") as f:
            for r in rows: f.write(json.dumps(r, default=str) + "\n")
        print(f"  wrote {p} ({len(rows)} rows)")

    manifest = {
        "generated_by": "044_expand_sft_pool.py",
        "sources": {
            "v4": sum(1 for r in all_rows if r["source"] == "v4"),
            "gold": sum(1 for r in all_rows if r["source"] == "gold"),
            "lasana_rescore": sum(1 for r in all_rows if r["source"] == "lasana_rescore"),
        },
        "dropped": dropped,
        "task_counts": dict(task_counts),
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "min_conf": args.min_conf,
        "require_frames": args.require_frames,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
