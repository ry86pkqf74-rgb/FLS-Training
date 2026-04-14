#!/usr/bin/env python3
"""Prepare SFT training data from validated YouTube scores.

Takes all ACCEPTED dual-scored videos and generates conversation-format
JSONL for Qwen2.5-VL fine-tuning.

Changes in v2 (2026-04-14):
- Backfill task_id from data/harvest_targets.csv when scorer output is missing it.
- Drop examples that are still unclassified/empty after backfill (prevents the
  regression where the student learns to emit empty task_id).

Run on Hetzner after scoring + validation completes.
"""
import csv
import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

REPO_ROOT = Path("/opt/fls-training")
SCORES_DIR = REPO_ROOT / "memory/scores"
OUTPUT_DIR = REPO_ROOT / "data/training/youtube_sft_v1"
HARVEST_CSV = REPO_ROOT / "data/harvest_targets.csv"

VALID_TASKS = {
    "task1_peg_transfer",
    "task2_pattern_cut",
    "task3_endoloop",
    "task4_extracorporeal_knot",
    "task5_intracorporeal_suturing",
    # tolerate older 'suture' suffix variant:
    "task5_intracorporeal_suture",
}

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

def load_task_lookup():
    """Map yt_<id> and <id> -> canonical task_id from harvest_targets.csv."""
    lookup = {}
    if not HARVEST_CSV.exists():
        return lookup
    with open(HARVEST_CSV) as f:
        for row in csv.DictReader(f):
            url = row.get("url", "")
            task = (row.get("task") or "").strip()
            m = re.search(r"[?&]v=([^&]+)", url)
            if not m or not task or task not in VALID_TASKS:
                continue
            yt_id = m.group(1)
            lookup[yt_id] = task
            lookup[f"yt_{yt_id}"] = task
    return lookup

def canonical_task(tid, video_id, lookup):
    """Return canonical task_id or None if unusable."""
    tid = (tid or "").strip()
    if tid == "task5_intracorporeal_suture":
        tid = "task5_intracorporeal_suturing"
    if tid in VALID_TASKS:
        return tid
    # Fallback to harvest CSV lookup
    fallback = lookup.get(video_id) or lookup.get(video_id.replace("yt_", ""))
    if fallback == "task5_intracorporeal_suture":
        fallback = "task5_intracorporeal_suturing"
    if fallback in VALID_TASKS:
        return fallback
    return None

def main():
    task_lookup = load_task_lookup()
    print(f"Harvest task lookup loaded: {len(task_lookup)} entries")

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

    training_examples = []
    drop_stats = defaultdict(int)
    for vid, scores in by_video.items():
        claude_scores = [s for s in scores if "claude" in s.get("source", "") and get_fls(s) > 0]
        tb_scores = [s for s in scores if ("haiku" in s.get("source", "") or "gpt" in s.get("source", "")) and get_fls(s) > 0]
        if not claude_scores:
            drop_stats["no_claude"] += 1
            continue
        best_claude = max(claude_scores, key=lambda s: get_conf(s))
        c_fls = get_fls(best_claude)
        c_conf = get_conf(best_claude)

        label_type = None
        delta = None
        if tb_scores:
            best_tb = max(tb_scores, key=lambda s: get_conf(s))
            tb_fls = get_fls(best_tb)
            tb_conf = get_conf(best_tb)
            delta = abs(c_fls - tb_fls)
            avg_conf = (c_conf + tb_conf) / 2
            if delta <= 30 and avg_conf >= 0.5:
                consensus_fls = (c_fls + tb_fls) / 2
                consensus_conf = avg_conf
                label_type = "dual_teacher_consensus"
            else:
                drop_stats["dual_rejected"] += 1
                continue
        elif c_conf >= 0.75:
            consensus_fls = c_fls
            consensus_conf = c_conf
            label_type = "claude_only_high_conf"
        else:
            drop_stats["claude_low_conf"] += 1
            continue

        # NEW: canonical task_id with fallback to harvest CSV
        raw_task = best_claude.get("task_id", "")
        task_id = canonical_task(raw_task, vid, task_lookup)
        if task_id is None:
            drop_stats["empty_or_unclassified_task"] += 1
            continue

        # Build target JSON
        d = best_claude
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
        if isinstance(target["score_components"], dict):
            target["score_components"]["total_fls_score"] = round(consensus_fls, 1)
        target["estimated_fls_score"] = round(consensus_fls, 1)

        training_examples.append({
            "video_id": vid,
            "task_id": task_id,
            "label_type": label_type,
            "consensus_fls": round(consensus_fls, 1),
            "consensus_conf": round(consensus_conf, 2),
            "delta": round(delta, 1) if delta is not None else None,
            "target": target,
        })

    print(f"Training examples: {len(training_examples)}")
    print(f"  dual_teacher_consensus: {sum(1 for e in training_examples if e['label_type'] == 'dual_teacher_consensus')}")
    print(f"  claude_only_high_conf: {sum(1 for e in training_examples if e['label_type'] == 'claude_only_high_conf')}")
    print(f"Drop stats: {dict(drop_stats)}")

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
        "drop_stats": dict(drop_stats),
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"  train.jsonl: {len(train)}")
    print(f"  val.jsonl: {len(val)}")
    print(f"\nTask distribution:")
    for t, c in sorted(task_counts.items()):
        print(f"  {t}: {c}")

if __name__ == "__main__":
    main()
