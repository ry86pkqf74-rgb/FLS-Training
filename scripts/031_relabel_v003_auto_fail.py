#!/usr/bin/env python3
"""Apply the v003 auto-fail rule to every legacy teacher score in memory/scores.

Why this exists: the v17/v18/v19 teachers were not consistent about zeroing
the score when drain_avulsion / knot_failure / gauze_detachment / block_dislodged
appeared. The v003 spec is unambiguous: those penalties force the score to 0.
Our v19 model now correctly applies that rule, so the eval looked worse than
reality (the labels themselves were the inconsistency).

This script does *not* re-call any LLM. It is a pure dictionary transformation
over already-scored JSON files.

For each score record under ``memory/scores/`` (skipping the quarantine dir),
when *any* penalty type matches the auto-fail set, we:

* set ``estimated_fls_score`` and ``score_components.total_fls_score`` to 0.0
* set ``score_components.formula_applied`` to "automatic zero due to auto-fail penalty"
* mark the matched penalty with ``severity = "auto_fail"``
* append (idempotently) a ``critical_errors`` entry with
  ``forces_zero_score = True`` and ``blocks_proficiency_claim = True``

Outputs are written under ``memory/scores_v003_relabel/`` mirroring the same
directory layout, so the original files are preserved and v17/v18/v19 datasets
remain reproducible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


AUTO_FAIL_PATTERNS = (
    "drain_avulsion",
    "drain_avulsed",
    "gauze_detachment",
    "gauze_detached",
    "block_dislodged",
    "needle_left_view",
    "needle_exits_field",
    "appendage_transection",
    "incomplete_task",
)


def _matches_auto_fail(value: str) -> str | None:
    norm = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not norm:
        return None
    for pat in AUTO_FAIL_PATTERNS:
        if pat in norm or norm in pat:
            return pat
    return None


def _apply(score: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return (relabeled_score, did_change)."""
    penalties = score.get("penalties") or []
    matched: list[tuple[int, str]] = []
    for i, p in enumerate(penalties):
        if not isinstance(p, dict):
            continue
        if p.get("severity") == "auto_fail":
            matched.append((i, p.get("type", "auto_fail")))
            continue
        m = _matches_auto_fail(p.get("type", ""))
        if m:
            matched.append((i, m))

    if not matched:
        return score, False

    # Mutate.
    out = json.loads(json.dumps(score))  # deep copy

    # Mark each matched penalty as auto_fail.
    for i, _ in matched:
        if isinstance(out["penalties"][i], dict):
            out["penalties"][i]["severity"] = "auto_fail"

    out["estimated_fls_score"] = 0.0
    sc = out.get("score_components") or {}
    if isinstance(sc, dict):
        sc["total_fls_score"] = 0.0
        sc["formula_applied"] = "automatic zero due to auto-fail penalty"
        out["score_components"] = sc
    else:
        out["score_components"] = {
            "total_fls_score": 0.0,
            "formula_applied": "automatic zero due to auto-fail penalty",
        }

    # Append (idempotent) critical_errors entries for each matched penalty.
    crits = list(out.get("critical_errors") or [])
    existing_types = {(c.get("type") or "").strip().lower() for c in crits if isinstance(c, dict)}
    for _, t in matched:
        if t in existing_types:
            continue
        crits.append({
            "type": t,
            "present": True,
            "reason": "auto-fail penalty present (v003 relabel)",
            "frame_evidence": [],
            "forces_zero_score": True,
            "blocks_proficiency_claim": True,
        })
        existing_types.add(t)
    out["critical_errors"] = crits

    return out, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, default=Path("memory/scores"))
    parser.add_argument("--output-dir", type=Path, default=Path("memory/scores_v003_relabel"))
    args = parser.parse_args()

    in_root = args.scores_dir.resolve()
    out_root = args.output_dir.resolve()

    total = 0
    changed = 0
    skipped = 0

    for src in in_root.rglob("*.json"):
        if "quarantine" in str(src).lower():
            continue
        rel = src.relative_to(in_root)
        try:
            score = json.loads(src.read_text())
        except Exception:
            skipped += 1
            continue
        total += 1
        new_score, did_change = _apply(score)
        target = out_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(new_score, default=str))
        if did_change:
            changed += 1

    print(json.dumps({
        "scores_dir": str(in_root),
        "output_dir": str(out_root),
        "total_scanned": total,
        "auto_fail_relabeled": changed,
        "json_decode_failures": skipped,
    }, indent=2))


if __name__ == "__main__":
    main()
