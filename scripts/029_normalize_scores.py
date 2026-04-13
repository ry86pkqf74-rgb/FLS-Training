#!/usr/bin/env python3
"""
029_normalize_scores.py — Post-hoc normalizer for scorer JSON output.

Fixes:
- Missing `video_classification` field (observed in ~70% of Sonnet output during
  batch 2, ~1% of Haiku). Infers classification from available signals.
- Missing/null `scoreable` boolean (derive from classification).
- Validates `score_components.total_fls_score <= max_score` (never rewrites
  scores, just flags outliers in the normalization log).

Inference rules for `video_classification` when absent:
  1. If `cannot_determine` is True OR frame_analyses is empty -> "unusable"
  2. Else if `scoreable` is False -> "unusable"
  3. Else if `task_id` starts with "task" AND total_fls_score is a number -> "performance"
  4. Else -> "unclassified"

Idempotent: files already containing `video_classification` are left untouched
unless --force is passed.

Usage:
  python3 029_normalize_scores.py [--scores-dir DIR] [--dry-run] [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SCORES_DIR = Path("/opt/fls-training/memory/scores")


def infer_classification(d: dict) -> str:
    """Best-effort inference when the model omitted video_classification."""
    if d.get("cannot_determine") is True:
        return "unusable"
    frame_analyses = d.get("frame_analyses") or []
    if len(frame_analyses) == 0:
        return "unusable"
    if d.get("scoreable") is False:
        return "unusable"
    task_id = d.get("task_id") or ""
    sc = d.get("score_components") or {}
    total = sc.get("total_fls_score")
    if task_id.startswith("task") and isinstance(total, (int, float)):
        return "performance"
    return "unclassified"


def derive_scoreable(classification: str, existing) -> bool:
    """Set scoreable based on classification if not already provided."""
    if isinstance(existing, bool):
        return existing
    return classification in ("performance", "expert_demo")


def validate_score_bounds(d: dict):
    """Return a warning string if total_fls_score > max_score, else None."""
    sc = d.get("score_components") or {}
    total = sc.get("total_fls_score")
    mx = sc.get("max_score")
    if isinstance(total, (int, float)) and isinstance(mx, (int, float)):
        if total > mx:
            return f"total_fls_score={total} > max_score={mx}"
    return None


def normalize_file(path, force, dry_run):
    summary = {
        "path": str(path),
        "changed": False,
        "added_classification": None,
        "added_scoreable": None,
        "warning": None,
        "error": None,
    }
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        summary["error"] = f"read/parse: {e}"
        return summary

    changed = False
    if "video_classification" not in d or force:
        if "video_classification" not in d:
            cls = infer_classification(d)
            d["video_classification"] = cls
            d.setdefault("_normalizer", {})["classification_inferred"] = cls
            d["_normalizer"]["normalized_at"] = datetime.now(timezone.utc).isoformat()
            summary["added_classification"] = cls
            changed = True

    cls = d.get("video_classification", "unclassified")
    if "scoreable" not in d or d.get("scoreable") is None:
        d["scoreable"] = derive_scoreable(cls, d.get("scoreable"))
        summary["added_scoreable"] = d["scoreable"]
        changed = True

    w = validate_score_bounds(d)
    if w:
        summary["warning"] = w

    if changed and not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(d, indent=2, default=str))
        tmp.replace(path)
    summary["changed"] = changed
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--subdir", type=str, default=None)
    args = p.parse_args()

    root = args.scores_dir
    if args.subdir:
        root = root / args.subdir
    if not root.exists():
        print(f"ERROR: scores dir not found: {root}", file=sys.stderr)
        return 2

    files = sorted(root.rglob("*.json"))
    print(f"Scanning {len(files)} score files under {root}")

    inferred_counts = Counter()
    changed = 0
    warnings = 0
    errors = 0
    warning_samples = []

    for f in files:
        s = normalize_file(f, force=args.force, dry_run=args.dry_run)
        if s["error"]:
            errors += 1
            continue
        if s["changed"]:
            changed += 1
        if s["added_classification"]:
            inferred_counts[s["added_classification"]] += 1
        if s["warning"]:
            warnings += 1
            if len(warning_samples) < 10:
                warning_samples.append(f"  {f.name}: {s['warning']}")

    mode = "DRY RUN" if args.dry_run else "APPLIED"
    print(f"\n=== {mode} ===")
    print(f"Files scanned:           {len(files)}")
    print(f"Files changed:           {changed}")
    print(f"Inferred classifications: {dict(inferred_counts)}")
    print(f"Score-bound warnings:    {warnings}")
    print(f"Parse errors:            {errors}")
    if warning_samples:
        print("Warning samples:")
        for s in warning_samples:
            print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
