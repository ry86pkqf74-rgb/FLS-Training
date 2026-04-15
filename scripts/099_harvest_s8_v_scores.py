#!/usr/bin/env python3
"""Harvest V*/post*/lap* video scores from S8 into a v002-shaped JSONL."""
import json, glob, os, sys

by_vid = {}
rank = {"consensus": 3, "claude-sonnet-4": 2, "gpt-4o": 1}
for p in sorted(glob.glob("/data/fls/scored/2026-04-07/*.json")
              + glob.glob("/data/fls/scored/2026-04-08/*.json")):
    base = os.path.basename(p)
    src = None
    vid = None
    for m in rank:
        if "_"+m in base:
            src = m
            vid = base.split("_"+m)[0]
            break
    if not vid or not vid.startswith(("V","post","lap")):
        continue
    try: d = json.load(open(p))
    except: continue
    task = (d.get("task_id") or d.get("task") or
            (d.get("consensus_score",{}).get("task_id") if isinstance(d.get("consensus_score"),dict) else None))
    # infer task from notes / default V*/post*/lap* -> task5
    if not task:
        notes = (str(d.get("recording_notes","")) + " " + str(d.get("technique_summary",""))).lower()
        if "peg" in notes: task = "task1"
        elif "pattern cut" in notes or "gauze" in notes: task = "task2"
        elif "endoloop" in notes: task = "task3"
        elif "extracorporeal" in notes: task = "task4"
        elif "intracorporeal" in notes or "suturing" in notes or "suture" in notes: task = "task5"
        else: task = "task5"
    score = (d.get("estimated_fls_score")
             or (d.get("consensus_score",{}).get("total_fls_score") if isinstance(d.get("consensus_score"),dict) else None)
             or (d.get("consensus_score") if isinstance(d.get("consensus_score"),(int,float)) else None)
             or d.get("final_score"))
    conf = d.get("confidence_score") or d.get("confidence") or (
        d.get("final_arbitration",{}).get("confidence") if isinstance(d.get("final_arbitration"),dict) else None)
    raw_pen = d.get("penalties") if d.get("penalties") is not None else d.get("estimated_penalties")
    if isinstance(raw_pen, (int, float)):
        pen_list = []; pen_total = float(raw_pen)
    elif isinstance(raw_pen, list):
        pen_list = raw_pen; pen_total = None
    else:
        pen_list = []; pen_total = None
    cur = {"path": p, "src": src, "rank": rank[src], "task": task, "score": score, "conf": conf,
           "penalties": pen_list, "pen_total": pen_total,
           "ctime": d.get("completion_time_seconds") or 0}
    if vid not in by_vid or cur["rank"] > by_vid[vid]["rank"]:
        by_vid[vid] = cur

have_score = {v:r for v,r in by_vid.items() if r["score"] is not None}
have_task = {v:r for v,r in by_vid.items() if r["task"]}
usable = {v:r for v,r in by_vid.items() if r["score"] is not None and r["task"]}
print(f"total vids: {len(by_vid)}")
print(f"with score: {len(have_score)}")
print(f"with task: {len(have_task)}")
print(f"usable: {len(usable)}")
from collections import Counter
print("tasks:", Counter(r["task"] for r in usable.values()))
print("sources:", Counter(r["src"] for r in usable.values()))

TASK_MAX = {"task1": 300, "task2": 300, "task3": 180, "task4": 420, "task5": 600}
TASK_CANON = {"task1":"task1","task2":"task2","task3":"task3","task4":"task4","task5":"task5",
              "task1_peg_transfer":"task1","task2_pattern_cut":"task2","task3_endoloop":"task3",
              "task4_extracorporeal_knot":"task4","task5_intracorporeal_suturing":"task5"}

out = []
for vid, r in usable.items():
    task_short = TASK_CANON.get(r["task"], r["task"])
    if task_short not in TASK_MAX: continue
    max_score = TASK_MAX[task_short]
    penalties = r["penalties"] or []
    if r.get("pen_total") is not None:
        total_pen = float(r["pen_total"])
    else:
        total_pen = sum((p.get("deduction",0) * p.get("count",1)) if isinstance(p,dict) else 0 for p in penalties)
    total_fls = max(0.0, min(float(max_score), float(r["score"])))
    time_used = max(0.0, float(max_score) - float(total_fls) - float(total_pen))
    conf = float(r["conf"] or 0.7)
    out.append({
        "video_id": vid, "consensus_score": total_fls, "claude_score": total_fls,
        "gpt_score": None, "teacher_delta": 0, "confidence": conf,
        "task_id": task_short, "trainee_id": None, "source_domain": "user_video",
        "target": {
            "id": f"score_s8harvest_{vid}", "video_id": vid,
            "video_filename": f"{vid}.mp4", "source": r["src"],
            "model_name": "claude-sonnet-4-20250514" if r["src"] == "claude-sonnet-4" else r["src"],
            "prompt_version": "v002", "task_id": task_short,
            "completion_time_seconds": float(r["ctime"] or time_used),
            "penalties": penalties,
            "score_components": {
                "time_score": float(max_score) - time_used,
                "penalty_deductions": float(total_pen),
                "total_fls_score": total_fls, "max_score": max_score,
            },
            "estimated_fls_score": total_fls,
            "confidence": conf, "confidence_score": conf,
        },
    })

print(f"built {len(out)} rows")
with open("/tmp/s8_v_harvest.jsonl","w") as f:
    for r in out: f.write(json.dumps(r)+"\n")
print("wrote /tmp/s8_v_harvest.jsonl")
