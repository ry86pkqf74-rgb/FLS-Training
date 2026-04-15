#!/usr/bin/env python3
"""Rescore LASANA videos through Claude Sonnet using the v002 universal scoring prompt.

Why this isn't using src/scoring/frontier_scorer.py:
  - ScoringResult's Phase enum is hardcoded to task5 (suturing) phases, so
    pydantic validation blows up on task1/task2 scores. That's a repo bug;
    this script bypasses it by saving the raw JSON response directly. A
    downstream prep step can load the JSON without the Phase enum constraint.
  - The OpenAI key on S8 currently returns 401, so GPT teacher is offline.
    This script is Claude-only until a working OPENAI_API_KEY is installed.

Task mapping (LASANA prefix -> v002 task_id):
  lasana_peg_*     -> task1_peg_transfer     (direct equivalent)
  lasana_circle_*  -> task2_pattern_cut      (pattern/circle cut proxy)
  lasana_suture_*  -> task5_intracorporeal_suturing (near equivalent)
  lasana_balloon_* -> SKIP (no FLS analog; forcing would corrupt schema)

Output: /data/fls/scored/lasana_v002/<video_id>_claude-sonnet-4.json
        Raw Claude response JSON; downstream prep should parse without enum
        validation. Idempotent (skips videos with an existing output file).

Launch on S8:
  tmux new -s lasana_rescore
  cd /opt/FLS-Training && source .venv/bin/activate
  python scripts/080_lasana_rescore_v002.py --workers 8 --max-frames 8
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

FRAMES_ROOT = Path("/data/fls/lasana_processed/frames")
OUT_ROOT = Path("/data/fls/scored/lasana_v002")
LOG_FILE = Path("/data/fls/logs/lasana_rescore_v002.log")
PROMPT_FILE = REPO_ROOT / "prompts" / "v002_universal_scoring_system.md"

PREFIX_TO_TASK = {
    "peg":    "task1_peg_transfer",
    "circle": "task2_pattern_cut",
    "suture": "task5_intracorporeal_suturing",
}

MODEL = "claude-sonnet-4-20250514"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def list_lasana_videos() -> list[tuple[str, str]]:
    out = []
    for d in sorted(FRAMES_ROOT.iterdir()):
        if not d.is_dir(): continue
        vid = d.name
        parts = vid.split("_", 2)
        if len(parts) < 3 or parts[0] != "lasana": continue
        if parts[1] not in PREFIX_TO_TASK: continue
        out.append((vid, PREFIX_TO_TASK[parts[1]]))
    return out


def sample_frames_b64(video_id: str, max_frames: int) -> list[str]:
    vdir = FRAMES_ROOT / video_id
    frames = sorted(vdir.glob("frame_*.jpg"))
    if not frames: return []
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        frames = [frames[int(i * step)] for i in range(max_frames)]
    return [base64.b64encode(p.read_bytes()).decode() for p in frames]


def _parse_json(text: str) -> dict:
    # strip markdown fences if present
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m: text = m.group(1)
    # first balanced { ... }
    start = text.find("{")
    if start < 0: raise ValueError("no json object found")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    raise ValueError("unbalanced json braces")


def score_one_claude(video_id: str, task_id: str, frames_b64: list[str],
                     system_prompt: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()

    content = []
    for i, b64 in enumerate(frames_b64):
        content.append({"type": "image",
                        "source": {"type": "base64",
                                   "media_type": "image/jpeg", "data": b64}})
        content.append({"type": "text", "text": f"Frame {i+1} of {len(frames_b64)}"})
    content.append({"type": "text",
                    "text": f"Task: {task_id}. Analyze these frames and output a single "
                            "strict-JSON ScoringResult per the v002 universal scoring "
                            "schema. Output JSON only, no markdown fences."})

    resp = client.messages.create(
        model=MODEL, max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    raw_text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    data = _parse_json(raw_text)
    # Ensure task_id is correct — the prompt is universal so the model may
    # misclassify. LASANA filename-derived task is ground truth here.
    data["task_id"] = task_id
    data["_rescored_at"] = datetime.now(timezone.utc).isoformat()
    data["_source"] = "lasana_rescored"
    data["_model"] = MODEL
    data["_prompt_version"] = "v002"
    data["_raw_claude_response"] = raw_text[:2000]  # keep a slice for debugging
    return data


def already_scored(video_id: str) -> bool:
    return (OUT_ROOT / f"{video_id}_claude-sonnet-4.json").exists()


def save_score(video_id: str, data: dict) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / f"{video_id}_claude-sonnet-4.json").write_text(
        json.dumps(data, indent=2, default=str))


def worker(vid: str, task_id: str, max_frames: int, system_prompt: str,
           force: bool = False) -> dict:
    out = {"video_id": vid, "status": None}
    try:
        if not force and already_scored(vid):
            out["status"] = "skip-existing"; return out
        b64s = sample_frames_b64(vid, max_frames)
        if not b64s:
            out["status"] = "no-frames"; return out
        data = score_one_claude(vid, task_id, b64s, system_prompt)
        save_score(vid, data)
        out["status"] = "ok"
    except Exception as e:
        out["status"] = f"err: {type(e).__name__}: {e}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-frames", type=int, default=8,
                    help="default for all tasks; overridden by --suture-frames for suturing")
    ap.add_argument("--suture-frames", type=int, default=24,
                    help="frame budget for task5 suturing (needs more coverage — 600s task)")
    ap.add_argument("--max-videos", type=int, default=0, help="0 = all")
    ap.add_argument("--task", choices=list(PREFIX_TO_TASK.keys()) + ["all"], default="all")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing per-video output files")
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("FATAL: ANTHROPIC_API_KEY not set", file=sys.stderr); sys.exit(1)
    if not PROMPT_FILE.exists():
        print(f"FATAL: prompt file missing: {PROMPT_FILE}", file=sys.stderr); sys.exit(1)
    system_prompt = PROMPT_FILE.read_text()

    videos = list_lasana_videos()
    if args.task != "all":
        t_filter = PREFIX_TO_TASK[args.task]
        videos = [v for v in videos if v[1] == t_filter]
    if args.shuffle: random.shuffle(videos)
    if args.max_videos: videos = videos[: args.max_videos]

    log(f"Queued {len(videos)} videos | workers={args.workers} max_frames={args.max_frames} "
        f"suture_frames={args.suture_frames} task={args.task} force={args.force}")
    log(f"Output dir: {OUT_ROOT}  Model: {MODEL}")

    ok = err = skip = noframe = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        def _mf(t):
            return args.suture_frames if t == "task5_intracorporeal_suturing" else args.max_frames
        futs = {ex.submit(worker, v, t, _mf(t), system_prompt, args.force): v
                for v, t in videos}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            s = r["status"]
            if s == "ok": ok += 1
            elif s == "skip-existing": skip += 1
            elif s == "no-frames": noframe += 1
            else:
                err += 1
                log(f"  [{i}/{len(videos)}] {r['video_id']} {s[:180]}")
            if i % 10 == 0 or i == len(videos):
                rate = i / max(time.time() - t0, 1)
                eta_m = (len(videos) - i) / max(rate, 0.001) / 60
                log(f"  [{i}/{len(videos)}] rate={rate:.2f}/s eta={eta_m:.1f}m "
                    f"ok={ok} skip={skip} noframe={noframe} err={err}")
    log(f"DONE in {(time.time()-t0)/60:.1f}m | ok={ok} skip={skip} noframe={noframe} err={err}")


if __name__ == "__main__":
    main()
