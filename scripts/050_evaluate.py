"""Evaluate student model against frontier consensus scores.

Compares a fine-tuned student's predictions to the frontier teacher consensus
on a held-out set, computing field-level agreement and FLS score MAE.

Usage:
    python scripts/050_evaluate.py \
        --student-scores data/eval/student_predictions.jsonl \
        --frontier-scores data/eval/frontier_consensus.jsonl \
        --config src/configs/finetune_task5_v2.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# Fields to compare for field-level agreement
NUMERIC_FIELDS = [
    "completion_time_seconds",
    "estimated_fls_score",
    "estimated_penalties",
]
NUMERIC_TOLERANCE = {
    "completion_time_seconds": 10.0,  # within 10s = agreement
    "estimated_fls_score": 15.0,      # within 15 points = agreement
    "estimated_penalties": 5.0,       # within 5 points = agreement
}

CATEGORICAL_FIELDS = [
    "drain_assessment.gap_visible",
    "drain_assessment.drain_avulsed",
    "drain_assessment.slit_closure_quality",
]

KNOT_FIELDS = [
    "is_surgeon_knot",
    "hand_switched",
    "appears_secure",
]


def _get_nested(data: dict, dotpath: str):
    """Get a value from a nested dict using dot notation."""
    keys = dotpath.split(".")
    v = data
    for k in keys:
        if isinstance(v, dict):
            v = v.get(k)
        else:
            return None
    return v


def _compare_numeric(student_val, frontier_val, tolerance: float) -> bool:
    if student_val is None or frontier_val is None:
        return student_val == frontier_val
    try:
        return abs(float(student_val) - float(frontier_val)) <= tolerance
    except (TypeError, ValueError):
        return False


def _compare_knots(student_knots: list[dict], frontier_knots: list[dict]) -> tuple[int, int]:
    """Compare knot assessments throw-by-throw. Returns (agreed, total)."""
    agreed = 0
    total = 0
    for i in range(min(len(student_knots), len(frontier_knots))):
        sk, fk = student_knots[i], frontier_knots[i]
        for field in KNOT_FIELDS:
            sv = sk.get(field)
            fv = fk.get(field)
            if sv is None and fv is None:
                continue  # both null = skip (e.g. is_surgeon_knot on throw 2)
            total += 1
            if sv == fv:
                agreed += 1
    return agreed, total


def _compare_phases(student_phases: list[dict], frontier_phases: list[dict]) -> tuple[int, int]:
    """Compare phase ordering. Returns (matched, total)."""
    s_names = [p.get("phase", "") for p in student_phases]
    f_names = [p.get("phase", "") for p in frontier_phases]
    total = max(len(s_names), len(f_names))
    if total == 0:
        return 0, 0
    matched = sum(1 for a, b in zip(s_names, f_names) if a == b)
    return matched, total


def evaluate_pair(student: dict, frontier: dict) -> dict:
    """Evaluate a single student prediction against frontier consensus."""
    agreed = 0
    total = 0
    disagreements = []

    # Numeric fields
    for field in NUMERIC_FIELDS:
        sv = student.get(field)
        fv = frontier.get(field)
        tol = NUMERIC_TOLERANCE.get(field, 10.0)
        total += 1
        if _compare_numeric(sv, fv, tol):
            agreed += 1
        else:
            disagreements.append({
                "field": field, "student": sv, "frontier": fv,
                "tolerance": tol, "type": "numeric",
            })

    # Categorical fields
    for field in CATEGORICAL_FIELDS:
        sv = _get_nested(student, field)
        fv = _get_nested(frontier, field)
        total += 1
        if sv == fv:
            agreed += 1
        else:
            disagreements.append({
                "field": field, "student": sv, "frontier": fv, "type": "categorical",
            })

    # Knot assessments
    s_knots = student.get("knot_assessments", [])
    f_knots = frontier.get("knot_assessments", [])
    k_agreed, k_total = _compare_knots(s_knots, f_knots)
    agreed += k_agreed
    total += k_total

    # Phase ordering
    s_phases = student.get("phase_timings", [])
    f_phases = frontier.get("phase_timings", [])
    p_agreed, p_total = _compare_phases(s_phases, f_phases)
    agreed += p_agreed
    total += p_total

    # FLS score MAE
    s_fls = student.get("estimated_fls_score")
    f_fls = frontier.get("estimated_fls_score")
    fls_mae = abs(float(s_fls or 0) - float(f_fls or 0))

    field_agreement = agreed / total if total > 0 else 0.0

    return {
        "video_id": student.get("video_id", frontier.get("video_id", "unknown")),
        "field_agreement": round(field_agreement, 4),
        "fields_agreed": agreed,
        "fields_total": total,
        "fls_score_mae": round(fls_mae, 2),
        "student_fls": s_fls,
        "frontier_fls": f_fls,
        "disagreements": disagreements,
    }


def evaluate_batch(
    student_path: str | Path,
    frontier_path: str | Path,
) -> dict:
    """Evaluate all student predictions against frontier consensus."""
    with open(student_path) as f:
        students = [json.loads(line) for line in f if line.strip()]
    with open(frontier_path) as f:
        frontiers = [json.loads(line) for line in f if line.strip()]

    # Index frontier by video_id
    frontier_by_vid = {}
    for fr in frontiers:
        vid = fr.get("video_id")
        if vid:
            frontier_by_vid[vid] = fr

    results = []
    for st in students:
        vid = st.get("video_id")
        if vid not in frontier_by_vid:
            logger.warning(f"No frontier consensus for {vid}, skipping")
            continue
        results.append(evaluate_pair(st, frontier_by_vid[vid]))

    if not results:
        return {"error": "no_matching_pairs", "n_evaluated": 0}

    avg_agreement = sum(r["field_agreement"] for r in results) / len(results)
    avg_mae = sum(r["fls_score_mae"] for r in results) / len(results)
    worst_agreement = min(r["field_agreement"] for r in results)

    return {
        "n_evaluated": len(results),
        "avg_field_agreement": round(avg_agreement, 4),
        "avg_fls_score_mae": round(avg_mae, 2),
        "worst_field_agreement": round(worst_agreement, 4),
        "per_video": results,
    }


def check_promotion(eval_summary: dict, config: dict) -> dict:
    """Check if student meets promotion criteria from config."""
    threshold_agreement = config.get("promotion_threshold_field_agreement", 0.90)
    threshold_mae = config.get("promotion_threshold_score_mae", 15.0)

    meets_agreement = eval_summary["avg_field_agreement"] >= threshold_agreement
    meets_mae = eval_summary["avg_fls_score_mae"] <= threshold_mae

    promoted = meets_agreement and meets_mae

    return {
        "promoted": promoted,
        "avg_field_agreement": eval_summary["avg_field_agreement"],
        "threshold_agreement": threshold_agreement,
        "meets_agreement": meets_agreement,
        "avg_fls_score_mae": eval_summary["avg_fls_score_mae"],
        "threshold_mae": threshold_mae,
        "meets_mae": meets_mae,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate student vs frontier consensus")
    parser.add_argument("--student-scores", required=True, help="JSONL of student predictions")
    parser.add_argument("--frontier-scores", required=True, help="JSONL of frontier consensus")
    parser.add_argument("--config", default="src/configs/finetune_task5_v2.yaml")
    parser.add_argument("--output", default=None, help="Write results JSON to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config = {}
    if Path(args.config).exists():
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    summary = evaluate_batch(args.student_scores, args.frontier_scores)
    promotion = check_promotion(summary, config)

    print("\n=== Evaluation Summary ===")
    print(f"  Videos evaluated:      {summary['n_evaluated']}")
    print(f"  Avg field agreement:   {summary['avg_field_agreement']:.1%}")
    print(f"  Avg FLS score MAE:     {summary['avg_fls_score_mae']:.1f}")
    print(f"  Worst video agreement: {summary['worst_field_agreement']:.1%}")

    print("\n=== Promotion Check ===")
    print(f"  Agreement: {promotion['avg_field_agreement']:.1%} (threshold {promotion['threshold_agreement']:.1%}) {'✅' if promotion['meets_agreement'] else '❌'}")
    print(f"  MAE:       {promotion['avg_fls_score_mae']:.1f} (threshold {promotion['threshold_mae']:.1f}) {'✅' if promotion['meets_mae'] else '❌'}")
    print(f"  PROMOTED:  {'✅ YES' if promotion['promoted'] else '❌ NO'}")

    if args.output:
        out = {"evaluation": summary, "promotion": promotion}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output}")
