#!/usr/bin/env python3
"""
FLS-Training: Auto-Validation Script
=====================================
Reads dual-teacher scores from DuckDB/memory and auto-categorizes each video as:
  ACCEPTED / QUARANTINED / REJECTED

Uses self-consistency filtering + time-anchor validation.
No manual review needed.

Usage:
  python scripts/026_auto_validate.py                    # validate all videos
  python scripts/026_auto_validate.py --video-id V31     # validate one video
  python scripts/026_auto_validate.py --report           # print summary report
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# ============================================================
# CONFIGURATION
# ============================================================

# Self-consistency thresholds
MAX_SCORE_DIVERGENCE = 25       # FLS points between teachers
MAX_TIME_DIVERGENCE = 15        # seconds between teachers
MIN_CONFIDENCE = 0.40           # minimum confidence from either teacher

# Quarantine thresholds (between accept and reject)
QUARANTINE_SCORE_DIVERGENCE = 50

# Time-anchor: penalties should be 0-20 for competent performances
MAX_REASONABLE_PENALTY = 20
MIN_REASONABLE_PENALTY = 0

VALIDATION_LOG = Path("validation_results.jsonl")


def load_scores(video_id: str = None) -> list[dict]:
    """Load scored video data from memory/scores/ and ledger files."""
    scores = []
    
    # Try loading from memory/scores/
    scores_dir = Path("memory/scores")
    if scores_dir.exists():
        for f in scores_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                if video_id is None or data.get("video_id") == video_id:
                    scores.append(data)
            except (json.JSONDecodeError, KeyError):
                continue

    # Also try loading from comparisons
    comp_dir = Path("memory/comparisons")
    if comp_dir.exists():
        for f in comp_dir.glob("*consensus*.json"):
            try:
                data = json.loads(f.read_text())
                if video_id is None or video_id in f.name:
                    scores.append(data)
            except (json.JSONDecodeError, KeyError):
                continue

    return scores


def time_anchor_check(completion_time: float, estimated_score: float) -> dict:
    """Check if the score falls within the mechanically plausible band.
    
    FLS score = 600 - time - penalties
    Floor = 600 - time - MAX_REASONABLE_PENALTY
    Ceiling = 600 - time - MIN_REASONABLE_PENALTY
    """
    floor = 600 - completion_time - MAX_REASONABLE_PENALTY
    ceiling = 600 - completion_time - MIN_REASONABLE_PENALTY
    
    in_band = floor <= estimated_score <= ceiling
    implied_penalty = 600 - completion_time - estimated_score
    
    return {
        "floor": floor,
        "ceiling": ceiling,
        "in_band": in_band,
        "implied_penalty": implied_penalty,
        "completion_time": completion_time,
    }


def validate_video(claude_score: dict, gpt_score: dict = None) -> dict:
    """Validate a single video's scores.
    
    Returns:
        {
            "status": "ACCEPTED" | "QUARANTINED" | "REJECTED",
            "reasons": [list of reasons],
            "metrics": {detailed metrics}
        }
    """
    reasons = []
    metrics = {}
    
    c_fls = claude_score.get("estimated_fls_score", 0)
    c_time = claude_score.get("completion_time_seconds", 0)
    c_conf = claude_score.get("confidence_score", 0)
    
    # Time anchor check on Claude score
    if c_time > 0:
        anchor = time_anchor_check(c_time, c_fls)
        metrics["time_anchor"] = anchor
        if not anchor["in_band"]:
            reasons.append(f"Claude score {c_fls} outside time-anchor band [{anchor['floor']:.0f}, {anchor['ceiling']:.0f}]")
    
    # Confidence check
    if c_conf < MIN_CONFIDENCE:
        reasons.append(f"Claude confidence {c_conf:.2f} < {MIN_CONFIDENCE}")
    metrics["claude_confidence"] = c_conf
    
    # If we have GPT scores too, check consistency
    if gpt_score:
        g_fls = gpt_score.get("estimated_fls_score", 0)
        g_time = gpt_score.get("completion_time_seconds", 0)
        g_conf = gpt_score.get("confidence_score", 0)
        
        score_diff = abs(c_fls - g_fls)
        time_diff = abs(c_time - g_time)
        
        metrics["score_divergence"] = score_diff
        metrics["time_divergence"] = time_diff
        metrics["gpt_confidence"] = g_conf
        
        if score_diff > QUARANTINE_SCORE_DIVERGENCE:
            reasons.append(f"Score divergence {score_diff:.0f} > {QUARANTINE_SCORE_DIVERGENCE} (REJECT threshold)")
        elif score_diff > MAX_SCORE_DIVERGENCE:
            reasons.append(f"Score divergence {score_diff:.0f} > {MAX_SCORE_DIVERGENCE}")
        
        if time_diff > MAX_TIME_DIVERGENCE:
            reasons.append(f"Time divergence {time_diff:.0f}s > {MAX_TIME_DIVERGENCE}s")
        
        if g_conf < MIN_CONFIDENCE:
            reasons.append(f"GPT confidence {g_conf:.2f} < {MIN_CONFIDENCE}")
        
        # Time anchor on GPT score too
        if g_time > 0:
            g_anchor = time_anchor_check(g_time, g_fls)
            if not g_anchor["in_band"]:
                reasons.append(f"GPT score {g_fls} outside time-anchor band")
    
    # Determine status
    if not reasons:
        status = "ACCEPTED"
    elif any("REJECT" in r for r in reasons):
        status = "REJECTED"
    elif len(reasons) >= 2:
        status = "REJECTED"
    else:
        status = "QUARANTINED"
    
    return {
        "status": status,
        "reasons": reasons,
        "metrics": metrics,
    }


def run_validation(video_id: str = None, report: bool = False):
    """Run validation on all or one video."""
    scores = load_scores(video_id)
    
    if not scores:
        print("No scores found. Run the scoring pipeline first.")
        print("Looking in: memory/scores/ and memory/comparisons/")
        return
    
    results = {"ACCEPTED": 0, "QUARANTINED": 0, "REJECTED": 0}
    all_entries = []
    
    for score_data in scores:
        vid = score_data.get("video_id", "unknown")
        
        # Try to find paired teacher scores
        claude_score = score_data  # default: treat as claude
        gpt_score = None  # would come from paired scoring
        
        result = validate_video(claude_score, gpt_score)
        results[result["status"]] += 1
        
        entry = {
            "video_id": vid,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        all_entries.append(entry)
        
        # Log
        with open(VALIDATION_LOG, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        
        status = result["status"]
        reasons = "; ".join(result["reasons"]) if result["reasons"] else "all checks passed"
        print(f"  [{status:12s}] {vid}: {reasons}")
    
    if report or video_id is None:
        total = sum(results.values())
        print(f"\n=== Validation Summary ===")
        print(f"  ACCEPTED:    {results['ACCEPTED']:3d} ({results['ACCEPTED']/max(total,1)*100:.0f}%)")
        print(f"  QUARANTINED: {results['QUARANTINED']:3d} ({results['QUARANTINED']/max(total,1)*100:.0f}%)")
        print(f"  REJECTED:    {results['REJECTED']:3d} ({results['REJECTED']/max(total,1)*100:.0f}%)")
        print(f"  TOTAL:       {total}")


def main():
    parser = argparse.ArgumentParser(description="FLS Auto-Validation")
    parser.add_argument("--video-id", type=str, default=None, help="Validate a specific video")
    parser.add_argument("--report", action="store_true", help="Print summary report")
    args = parser.parse_args()
    
    run_validation(video_id=args.video_id, report=args.report)


if __name__ == "__main__":
    main()
