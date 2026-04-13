#!/usr/bin/env python3
"""028_classify_and_extract_teaching.py — Classify scored videos and extract teaching content.

Reads all score JSONs, classifies each as:
  - PERFORMANCE: actual FLS task execution (use for scoring training)
  - EXPERT_DEMO: expert demonstration with teaching value (use for coaching training)
  - INSTRUCTIONAL: pure educational/lecture content (use for coaching training)
  - UNUSABLE: unrelated content, equipment demos, etc.

For EXPERT_DEMO and INSTRUCTIONAL videos, extracts structured teaching content
that can be used to train the coaching/feedback model.

Output:
  - memory/video_classifications.jsonl (one record per video)
  - data/training/teaching_examples.jsonl (extracted teaching content for coaching model)
"""

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
SCORES_DIR = REPO / "memory" / "scores"
OUTPUT_CLASSIFICATIONS = REPO / "memory" / "video_classifications.jsonl"
OUTPUT_TEACHING = REPO / "data" / "training" / "teaching_examples.jsonl"


def get_fls_score(d):
    s = d.get("estimated_fls_score") or d.get("fls_score")
    if s and float(s) > 0:
        return float(s)
    sc = d.get("score_components", {})
    if isinstance(sc, dict):
        s = sc.get("total_fls_score")
        if s and float(s) > 0:
            return float(s)
    return 0.0


def get_confidence(d):
    return float(d.get("confidence_score") or d.get("confidence") or 0)


def classify_video(claude_score, gpt_score=None):
    """Classify a video based on its score data."""
    ts = str(claude_score.get("technique_summary", "")).lower()
    fls = get_fls_score(claude_score)
    conf = get_confidence(claude_score)
    fa = claude_score.get("frame_analyses", []) or []
    strengths = claude_score.get("strengths", []) or []
    suggestions = claude_score.get("improvement_suggestions", []) or []
    
    # Check frame descriptions for teaching signals
    frame_text = " ".join(
        str(f.get("description", "")) + " " + str(f.get("technique_notes", ""))
        for f in fa
    ).lower()
    
    all_text = ts + " " + frame_text
    
    # Classification rules
    edu_keywords = [
        "educational video", "not a performance", "instructional",
        "not an fls", "educational content", "product demo",
        "presentation slides", "lecture", "course implementation",
        "promotional content", "virtual reality training session"
    ]
    demo_keywords = [
        "expert demo", "expert-level performance demonstrat",
        "appears to be instructional demonstration",
        "tabletop educational demonstration",
        "educational demonstration", "demonstration of proper",
        "knot tying demonstration"
    ]
    
    is_pure_educational = any(kw in all_text for kw in edu_keywords)
    is_demo = any(kw in all_text for kw in demo_keywords)
    
    if is_pure_educational and fls == 0:
        classification = "INSTRUCTIONAL"
    elif is_pure_educational and fls > 0:
        # Scored but identified as educational — could be expert demo with score
        classification = "EXPERT_DEMO"
    elif is_demo and fls > 400:
        classification = "EXPERT_DEMO"
    elif is_demo and fls == 0:
        classification = "INSTRUCTIONAL"
    elif fls == 0 and conf >= 0.9:
        # High confidence zero = model is sure this isn't scoreable
        classification = "UNUSABLE"
    elif fls == 0:
        classification = "UNUSABLE"
    else:
        classification = "PERFORMANCE"
    
    return classification


def extract_teaching_content(video_id, task_id, claude_score, classification):
    """Extract structured teaching content from demo/instructional videos."""
    fa = claude_score.get("frame_analyses", []) or []
    strengths = claude_score.get("strengths", []) or []
    suggestions = claude_score.get("improvement_suggestions", []) or []
    ts = claude_score.get("technique_summary", "")
    phase_timings = claude_score.get("phase_timings", []) or []
    tsa = claude_score.get("task_specific_assessments", {}) or {}
    penalties = claude_score.get("penalties", []) or []
    
    # Extract technique demonstrations from frame analyses
    technique_demos = []
    for f in fa:
        desc = str(f.get("description", ""))
        notes = str(f.get("technique_notes", ""))
        phase = str(f.get("phase", ""))
        if desc or notes:
            technique_demos.append({
                "frame": f.get("frame_number"),
                "phase": phase,
                "what_is_shown": desc,
                "technique_insight": notes,
            })
    
    teaching_record = {
        "video_id": video_id,
        "task_id": task_id,
        "classification": classification,
        "technique_summary": ts,
        "demonstrated_strengths": strengths,
        "teaching_points": suggestions,
        "technique_demonstrations": technique_demos,
        "phase_timings": phase_timings,
        "task_specific_details": tsa if isinstance(tsa, dict) else {},
        "penalties_shown": penalties if isinstance(penalties, list) else [],
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # For expert demos with scores, also include benchmarking data
    fls = get_fls_score(claude_score)
    if fls > 0:
        teaching_record["expert_benchmark"] = {
            "fls_score": fls,
            "completion_time": claude_score.get("completion_time_seconds"),
            "confidence": get_confidence(claude_score),
        }
    
    return teaching_record


def main():
    # Collect latest Claude scores per video
    by_video = {}
    for path in sorted(SCORES_DIR.rglob("*.json")):
        if "consensus" in path.name.lower() or "_quarantine" in str(path):
            continue
        try:
            d = json.load(open(path))
            src = d.get("source", "")
            if "claude" not in src:
                continue
            vid = d.get("video_id", "")
            if not vid:
                continue
            scored_at = d.get("scored_at", "")
            if vid not in by_video or scored_at > by_video[vid].get("scored_at", ""):
                by_video[vid] = d
        except:
            continue
    
    print(f"Total videos with Claude scores: {len(by_video)}")
    
    counts = defaultdict(int)
    teaching_examples = []
    
    OUTPUT_CLASSIFICATIONS.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEACHING.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_CLASSIFICATIONS, "w") as clf:
        for vid in sorted(by_video):
            d = by_video[vid]
            classification = classify_video(d)
            task_id = d.get("task_id", "unknown")
            fls = get_fls_score(d)
            conf = get_confidence(d)
            
            record = {
                "video_id": vid,
                "task_id": task_id,
                "classification": classification,
                "fls_score": fls,
                "confidence": conf,
                "classified_at": datetime.now(timezone.utc).isoformat(),
            }
            clf.write(json.dumps(record) + "\n")
            counts[classification] += 1
            
            # Extract teaching content from demos and instructional videos
            if classification in ("EXPERT_DEMO", "INSTRUCTIONAL"):
                teaching = extract_teaching_content(vid, task_id, d, classification)
                teaching_examples.append(teaching)
    
    with open(OUTPUT_TEACHING, "w") as tf:
        for t in teaching_examples:
            tf.write(json.dumps(t, default=str) + "\n")
    
    print(f"\nClassification results:")
    for cls, count in sorted(counts.items()):
        print(f"  {cls}: {count}")
    print(f"\nTeaching examples extracted: {len(teaching_examples)}")
    print(f"\nOutput:")
    print(f"  Classifications: {OUTPUT_CLASSIFICATIONS}")
    print(f"  Teaching data: {OUTPUT_TEACHING}")


if __name__ == "__main__":
    main()
