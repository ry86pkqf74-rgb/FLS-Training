"""Build a task-stratified v003 multimodal training dataset.

For every score record in memory/scores/:
  1. Backfill task_id from data/harvest_targets.csv if missing.
  2. Drop unclassified examples.
  3. Run the score through src.training.v003_target.enrich_to_v003_target so the
     assistant content carries the v003 contract (formula_applied, critical_errors,
     severity, cannot_determine, confidence_rationale, task_specific_assessments).
  4. If /workspace/v003_frames/<video_id>/ has JPGs, embed up to N image refs.
     Otherwise fall back to text-only (still useful for schema learning).
  5. Stratified split per task (80/10/10 train/val/test).

Outputs /workspace/v003_multimodal/{train,val,test}.jsonl + manifest.json.
"""
import csv, json, os, random, re, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/workspace/FLS-Training")
sys.path.insert(0, str(ROOT))
from src.training.v003_target import enrich_to_v003_target  # type: ignore
from src.rubrics.loader import canonical_task_id, load_rubric, get_task_max_score, get_task_max_time, is_official_fls_task  # type: ignore

SCORES_DIR = ROOT / "memory" / "scores"
HARVEST_CSV = ROOT / "data" / "harvest_targets.csv"
FRAMES_ROOT = Path("/workspace/v003_frames")
OUT_DIR = Path("/workspace/v003_multimodal")
SYSTEM_PROMPT_FILE = ROOT / "prompts" / "v002_universal_scoring_system.md"
MAX_FRAMES_PER_SAMPLE = 8
SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Backfill lookup yt_id -> task
csv_lookup = {}
if HARVEST_CSV.exists():
    with HARVEST_CSV.open() as f:
        for row in csv.DictReader(f):
            url = row.get("url","")
            task = (row.get("task") or "").strip()
            m = re.search(r"[?&]v=([^&]+)", url)
            if m and task:
                csv_lookup[m.group(1)] = task
                csv_lookup[f"yt_{m.group(1)}"] = task

VALID_TASKS = {"task1","task2","task3","task4","task5","task6"}

def resolve_task(score):
    raw = (score.get("task_id") or "").strip()
    canonical = canonical_task_id(raw or "task5")
    # If raw was empty or unclassified, try CSV.
    if not raw or raw.lower() in {"unclassified","unknown",""}:
        vid = score.get("video_id","")
        cand = csv_lookup.get(vid) or csv_lookup.get(vid.replace("yt_",""))
        if cand:
            canonical = canonical_task_id(cand)
    if canonical not in VALID_TASKS:
        return None
    return canonical

# Choose best score per video (prefer consensus > claude > gpt)
SOURCE_PRIORITY = {"consensus": 0, "teacher_claude": 1, "teacher_gpt": 2, "claude_only_high_conf": 3}
by_video = {}
for path in SCORES_DIR.rglob("*.json"):
    if "quarantine" in str(path):
        continue
    try:
        d = json.loads(path.read_text())
    except Exception:
        continue
    vid = d.get("video_id")
    if not vid:
        continue
    score_fls = float(d.get("estimated_fls_score") or 0)
    if score_fls <= 0:
        # Allow zero-score records only if drain avulsion / auto-fail signal is present.
        if not d.get("drain_assessment", {}).get("drain_avulsed"):
            continue
    src = d.get("source","teacher_claude")
    prio = SOURCE_PRIORITY.get(src, 9)
    if vid not in by_video or prio < SOURCE_PRIORITY.get(by_video[vid].get("source",""),9):
        by_video[vid] = d

system_prompt = SYSTEM_PROMPT_FILE.read_text() if SYSTEM_PROMPT_FILE.exists() else "You are an FLS proctor. Output v003 scoring JSON."

per_task = defaultdict(list)
drops = defaultdict(int)
vision_count = 0
text_only_count = 0

for vid, score in by_video.items():
    task_id = resolve_task(score)
    if task_id is None:
        drops["no_task"] += 1
        continue
    score["task_id"] = task_id
    try:
        target = enrich_to_v003_target(score, task_id)
    except Exception as e:
        drops[f"enrich_error_{type(e).__name__}"] += 1
        continue

    # User content with frames if available.
    frames_dir = FRAMES_ROOT / vid
    images = []
    if frames_dir.is_dir():
        jpgs = sorted(frames_dir.glob("frame_*.jpg"))
        if jpgs:
            # Subsample to MAX_FRAMES_PER_SAMPLE evenly.
            if len(jpgs) > MAX_FRAMES_PER_SAMPLE:
                step = len(jpgs) / MAX_FRAMES_PER_SAMPLE
                jpgs = [jpgs[int(i*step)] for i in range(MAX_FRAMES_PER_SAMPLE)]
            images = [{"type":"image","image": f"file://{p}"} for p in jpgs]
    rubric_summary = (
        f"Task {task_id}: {target[\"task_name\"]}. "
        f"Max score {int(target[\"max_score\"])} (denominator). "
        f"Max time {int(target[\"max_time_seconds\"])} s. "
        f"Score formula: max_score - completion_time - penalties (auto-zero on auto_fail). "
        f"Official FLS task: {target[\"official_fls_task\"]}."
    )
    user_text = (
        f"{rubric_summary}\n"
        "Score this performance and emit a complete v003 scoring JSON: "
        "score_components (with formula_applied), each penalty with "
        "points_deducted/severity/confidence/frame_evidence, critical_errors "
        "(forces_zero_score / blocks_proficiency_claim where appropriate), "
        "cannot_determine, confidence_rationale, task_specific_assessments. "
        "Do not invent evidence; never claim proficiency when critical errors exist."
    )
    user_content = images + [{"type":"text","text": user_text}] if images else user_text
    if images:
        vision_count += 1
    else:
        text_only_count += 1

    example = {
        "messages": [
            {"role":"system","content": system_prompt},
            {"role":"user","content": user_content},
            {"role":"assistant","content": json.dumps(target, default=str)},
        ],
        "metadata": {
            "video_id": vid,
            "task_id": task_id,
            "training_score": target["score_components"]["total_fls_score"],
            "max_score": target["score_components"]["max_score"],
            "has_critical_error": bool(target["critical_errors"]),
            "vision": bool(images),
            "n_frames": len(images),
            "schema_version": "v003",
        },
    }
    per_task[task_id].append(example)

# Stratified split per task.
rng = random.Random(SEED)
splits = {"train": [], "val": [], "test": []}
for task_id, rows in per_task.items():
    rng.shuffle(rows)
    n = len(rows)
    n_val = max(1, n // 10)
    n_test = max(1, n // 10)
    splits["test"].extend(rows[:n_test])
    splits["val"].extend(rows[n_test:n_test+n_val])
    splits["train"].extend(rows[n_test+n_val:])

for name, rows in splits.items():
    rng.shuffle(rows)
    with (OUT_DIR / f"{name}.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"  {name}.jsonl: {len(rows)}")

manifest = {
    "dataset": "v003_multimodal",
    "schema_version": "v003",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "totals": {k: len(v) for k,v in splits.items()},
    "task_distribution": {t: len(rows) for t,rows in per_task.items()},
    "vision_examples": vision_count,
    "text_only_examples": text_only_count,
    "drops": dict(drops),
    "max_frames_per_sample": MAX_FRAMES_PER_SAMPLE,
    "system_prompt_file": str(SYSTEM_PROMPT_FILE),
}
(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
print(json.dumps(manifest, indent=2, default=str))
