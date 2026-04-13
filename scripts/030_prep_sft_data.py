#!/usr/bin/env python3
"""Prepare SFT training data from validated YouTube scores.

Takes all ACCEPTED dual-scored videos and generates conversation-format
JSONL for Qwen2.5-VL fine-tuning.

Run on Hetzner after scoring + validation completes.
"""
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

SCORES_DIR = Path("/opt/fls-training/memory/scores")
OUTPUT_DIR = Path("/opt/fls-training/data/training/youtube_sft_v1")

def safe_float(val, default=0.0):
    try: return float(val)
    except (TypeError, ValueError): return default

def get_fls(d):
    s = safe_float(d.get("estimated_fls_score"))
    if s > 0: return s
    sc = d.get("score_components", {})
    if isinstance(sc, dict):
        s = safe_float(sc.get("total_fls_score"))
        if s > 0: return s
    return 0.0

def get_conf(d):
    return safe_float(d.get("confidence_score") or d.get("confidence"))

# Load all scores grouped by video
by_video = defaultdict(list)
for f in sorted(SCORES_DIR.rglob("*.json")):
    if "consensus" in f.name or "quarantine" in str(f):
        continue
    try:
        d = json.loads(f.read_text())
        vid = d.get("video_id", "")
        if vid:
            by_video[vid].append(d)
    except:
        pass

print(f"Total videos with scores: {len(by_video)}")

# Find ACCEPTED pairs and build consensus
training_examples = []
for vid, scores in by_video.items():
    # Get best Claude and best Teacher B
    claude_scores = [s for s in scores if "claude" in s.get("source", "") and get_fls(s) > 0]
    tb_scores = [s for s in scores if ("haiku" in s.get("source", "") or "gpt" in s.get("source", "")) and get_fls(s) > 0]
    
    if not claude_scores:
        continue
    
    best_claude = max(claude_scores, key=lambda s: get_conf(s))
    c_fls = get_fls(best_claude)
    c_conf = get_conf(best_claude)
    
    if tb_scores:
        best_tb = max(tb_scores, key=lambda s: get_conf(s))
        tb_fls = get_fls(best_tb)
        tb_conf = get_conf(best_tb)
        delta = abs(c_fls - tb_fls)
        avg_conf = (c_conf + tb_conf) / 2
        
        # ACCEPTED: dual-scored with delta<=30, avg_conf>=0.5
        if delta <= 30 and avg_conf >= 0.5:
            consensus_fls = (c_fls + tb_fls) / 2
            consensus_conf = avg_conf
            label_type = "dual_teacher_consensus"
        else:
            continue  # Skip quarantined/rejected
    elif c_conf >= 0.75:
        # Claude-only with high confidence
        consensus_fls = c_fls
        consensus_conf = c_conf
        label_type = "claude_only_high_conf"
    else:
        continue
    
    # Build training example from the Claude score (most detailed)
    d = best_claude
    task_id = d.get("task_id", "")
    
    # Build target JSON (what the student should learn to output)
    target = {
        "task_id": task_id,
        "task_name": d.get("task_name", ""),
        "completion_time_seconds": d.get("completion_time_seconds", 0),
        "score_components": d.get("score_components", {}),
        "confidence": consensus_conf,
        "technique_summary": d.get("technique_summary", ""),
        "strengths": d.get("strengths", []),
        "improvement_suggestions": d.get("improvement_suggestions", []),
        "penalties": d.get("penalties", []),
    }
    
    # Override score with consensus
    if isinstance(target["score_components"], dict):
        target["score_components"]["total_fls_score"] = round(consensus_fls, 1)
    target["estimated_fls_score"] = round(consensus_fls, 1)
    
    training_examples.append({
        "video_id": vid,
        "task_id": task_id,
        "label_type": label_type,
        "consensus_fls": round(consensus_fls, 1),
        "consensus_conf": round(consensus_conf, 2),
        "delta": round(delta, 1) if tb_scores else None,
        "target": target,
    })

print(f"Training examples: {len(training_examples)}")
print(f"  dual_teacher_consensus: {sum(1 for e in training_examples if e['label_type'] == 'dual_teacher_consensus')}")
print(f"  claude_only_high_conf: {sum(1 for e in training_examples if e['label_type'] == 'claude_only_high_conf')}")

# Split 90/10 train/val
import random
random.seed(42)
random.shuffle(training_examples)
split = int(len(training_examples) * 0.9)
train = training_examples[:split]
val = training_examples[split:]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_DIR / "train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(ex, default=str) + "\n")
with open(OUTPUT_DIR / "val.jsonl", "w") as f:
    for ex in val:
        f.write(json.dumps(ex, default=str) + "\n")

# Task distribution
task_counts = defaultdict(int)
for ex in training_examples:
    task_counts[ex["task_id"]] += 1

manifest = {
    "dataset": "youtube_sft_v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "total": len(training_examples),
    "train": len(train),
    "val": len(val),
    "task_counts": dict(task_counts),
    "label_types": {
        "dual_teacher_consensus": sum(1 for e in training_examples if e["label_type"] == "dual_teacher_consensus"),
        "claude_only_high_conf": sum(1 for e in training_examples if e["label_type"] == "claude_only_high_conf"),
    },
}
with open(OUTPUT_DIR / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2, default=str)

print(f"\nOutput: {OUTPUT_DIR}")
print(f"  train.jsonl: {len(train)}")
print(f"  val.jsonl: {len(val)}")
print(f"\nTask distribution:")
for t, c in sorted(task_counts.items()):
    print(f"  {t}: {c}")
print(json.dumps(manifest, indent=2))
