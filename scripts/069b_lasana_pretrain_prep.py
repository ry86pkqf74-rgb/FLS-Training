#!/usr/bin/env python3
"""LASANA pre-training data prep — v2 with correct frame directory mapping."""
import csv, json, math, os, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ANNOTATIONS_DIR = Path("/opt/fls-training/data/external/lasana/annotations/Annotation")
FRAMES_DIR = Path("/data/fls/lasana_processed/frames")
OUTPUT_DIR = Path("/data/fls/training/lasana_pretrain_v2")

# Map annotation task names to frame directory prefixes
TASK_MAP = {
    "PegTransfer":      {"prefix": "lasana_peg",     "fls_task": "task1_peg_transfer", "max_score": 300},
    "CircleCutting":    {"prefix": "lasana_circle",   "fls_task": "task2_pattern_cut",  "max_score": 300},
    "BalloonResection": {"prefix": "lasana_balloon",  "fls_task": "task2_pattern_cut",  "max_score": 300},
    "SutureAndKnot":    {"prefix": "lasana_suture",   "fls_task": "task5_intracorporeal_suture", "max_score": 600},
}

def load_annotations():
    all_trials = {}
    for task_name, info in TASK_MAP.items():
        csv_path = ANNOTATIONS_DIR / f"{task_name}.csv"
        split_path = ANNOTATIONS_DIR / f"{task_name}_split.csv"
        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found"); continue

        splits = {}
        if split_path.exists():
            with open(split_path, newline="") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    splits[row["id"]] = row.get("split", "train")

        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f, delimiter=";"):
                tid = row["id"]
                vid = f"{info['prefix']}_{tid}"  # e.g., lasana_peg_blaox
                dur_parts = row.get("duration", "0:00:00").split(":")
                if len(dur_parts) == 3:
                    dur = int(dur_parts[0])*3600 + int(dur_parts[1])*60 + int(dur_parts[2])
                else:
                    dur = 0

                sub = {}
                for c in ["bimanual_dexterity","depth_perception","efficiency","tissue_handling"]:
                    if c in row and row[c]: sub[c] = float(row[c])
                
                errs = {}
                skip_cols = {"id","duration","frame_count","GRS","bimanual_dexterity","depth_perception","efficiency","tissue_handling"}
                for c in row:
                    if c not in skip_cols and row[c] in ("True","False"):
                        errs[c] = row[c] == "True"

                all_trials[vid] = {
                    "video_id": vid, "trial_id": tid, "lasana_task": task_name,
                    "fls_task": info["fls_task"], "max_score": info["max_score"],
                    "grs_zscore": float(row.get("GRS", 0)),
                    "sub_scores": sub, "errors": errs,
                    "duration_seconds": dur,
                    "frame_count": int(row.get("frame_count", 0)),
                    "split": splits.get(tid, "train"),
                }
        print(f"  {task_name}: {sum(1 for t in all_trials.values() if t['lasana_task']==task_name)} trials")
    return all_trials

def find_frames(video_id, max_frames=23):
    fd = FRAMES_DIR / video_id
    if not fd.exists(): return []
    frames = sorted(fd.glob("frame_*.jpg"))
    if not frames: frames = sorted(fd.glob("*.jpg"))
    if len(frames) <= max_frames: return [str(f) for f in frames]
    step = len(frames) / (max_frames - 3)
    uniform = [str(frames[int(i*step)]) for i in range(max_frames-3)]
    final = [str(f) for f in frames[-3:]]
    return uniform + final

def grs_to_fls(grs, max_score):
    pct = 1 / (1 + math.exp(-grs))
    return round(max_score * 0.10 + (max_score * 0.82) * pct, 1)

def main():
    print("=== LASANA Pre-training Data Prep v2 ===\n")
    trials = load_annotations()
    print(f"Total: {len(trials)} trials\n")

    examples = {"train": [], "val": [], "test": []}
    matched = unmatched = 0
    for vid, t in sorted(trials.items()):
        frames = find_frames(vid)
        if not frames: unmatched += 1; continue
        matched += 1
        fls_est = grs_to_fls(t["grs_zscore"], t["max_score"])
        errs_active = [k for k,v in t["errors"].items() if v]
        
        example = {
            "video_id": vid,
            "task_id": t["fls_task"],
            "frame_paths": frames,
            "n_frames": len(frames),
            "target": {
                "video_classification": "performance",
                "task_id": t["fls_task"],
                "source_dataset": "lasana",
                "ground_truth": {
                    "grs_zscore": t["grs_zscore"],
                    "sub_scores": t["sub_scores"],
                    "errors": t["errors"],
                    "label_source": "human_raters_tu_dresden",
                },
                "estimated_fls_score": fls_est,
                "completion_time_seconds": t["duration_seconds"],
                "confidence": 0.95,
                "score_components": {
                    "max_score": t["max_score"],
                    "grs_zscore": t["grs_zscore"],
                    "total_fls_score": fls_est,
                    "label_type": "human_grs_converted",
                },
                "technique_summary": f"LASANA {t['lasana_task']} trial. GRS={t['grs_zscore']:.3f}. "
                    f"Duration={t['duration_seconds']}s. Errors: {errs_active if errs_active else 'none'}.",
                "strengths": [f"{k}: z={v:.2f}" for k,v in t["sub_scores"].items() if v > 0.5],
                "improvement_suggestions": [f"Improve {k} (z={v:.2f})" for k,v in t["sub_scores"].items() if v < -0.5],
            },
            "metadata": {
                "source": "lasana", "lasana_task": t["lasana_task"],
                "trial_id": t["trial_id"], "grs_zscore": t["grs_zscore"],
                "duration_seconds": t["duration_seconds"],
            },
        }
        examples[t["split"]].append(example)

    print(f"Matched: {matched} | Missing frames: {unmatched}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for split, data in examples.items():
        p = OUTPUT_DIR / f"{split}.jsonl"
        with open(p, "w") as f:
            for ex in data: f.write(json.dumps(ex, default=str) + "\n")
        print(f"  {split}: {len(data)} -> {p}")
        total += len(data)

    # Task/GRS stats
    tc = defaultdict(int)
    gs = defaultdict(list)
    for s, data in examples.items():
        for ex in data:
            tc[ex["task_id"]] += 1
            gs[ex["task_id"]].append(ex["metadata"]["grs_zscore"])

    manifest = {
        "dataset": "lasana_pretrain_v2", "total": total,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "splits": {s: len(d) for s,d in examples.items()},
        "task_counts": dict(tc),
        "grs_stats": {t: {"n": len(v), "mean": sum(v)/len(v), "min": min(v), "max": max(v)}
                      for t,v in gs.items()},
    }
    with open(OUTPUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\n=== DONE: {total} examples ===")
    print(json.dumps(manifest, indent=2, default=str))

if __name__ == "__main__":
    main()
