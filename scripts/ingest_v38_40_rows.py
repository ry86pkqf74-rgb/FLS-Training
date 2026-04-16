#!/usr/bin/env python3
"""Build v12 training rows from V38-V40 scored JSONs (task5)."""
import json, re, glob
from pathlib import Path

rows = []

def parse_sc(sc_str, time_used, max_score):
    total_match = re.search(r"total_fls_score=([0-9.]+)", str(sc_str))
    total = float(total_match.group(1)) if total_match else 0.0
    penalties = round(max(0.0, max_score - time_used - total), 2)
    return {"max_score": max_score, "time_used": time_used, "total_penalties": penalties,
            "total_fls_score": total, "formula_applied": f"{max_score} - {time_used} - {penalties} = {total}"}

today = "2026-04-15"
for vid in ["V38_video", "V39_video", "V40_video"]:
    files = glob.glob(f"/data/fls/scored/{today}/{vid}_claude-sonnet-4_*.json")
    if not files:
        print(f"[SKIP] {vid}")
        continue
    d = json.load(open(files[0]))
    time_s = float(d.get("completion_time_seconds") or 0.0)
    sc_dict = parse_sc(d.get("score_components",""), time_s, 600)
    conf = 0.55 if sc_dict["total_penalties"] > 100 else 0.70
    target = dict(d)
    target["score_components"] = sc_dict
    target["estimated_fls_score"] = sc_dict["total_fls_score"]
    for k in ["frame_analyses","phase_timings","knot_assessments","comparison_to_previous",
               "superseded","superseded_by","superseded_at","superseded_reason",
               "phases_detected","reasoning"]:
        target.pop(k, None)
    row = {"video_id": vid, "task_id": "task5_intracorporeal_suturing",
           "target": target, "frames": [], "source": "teacher_claude", "consensus_conf": conf}
    rows.append(row)
    print(f"  {vid}: time={time_s}s  score={sc_dict['total_fls_score']}  pen={sc_dict['total_penalties']}  conf={conf}")

Path("/tmp/v38_40_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
print(f"Wrote {len(rows)} rows")
