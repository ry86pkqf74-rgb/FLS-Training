#!/usr/bin/env python3
"""030_run_consensus.py — Generate critique consensus for videos with both teacher scores.

Loads Teacher A (Claude) + Teacher B (GPT-4o) JSONs, calls the Critique Agent
with the multi-turn consensus prompt, and saves structured consensus output.

Usage:
    python scripts/030_run_consensus.py                          # all eligible videos
    python scripts/030_run_consensus.py --video-id V22_video     # single video
    python scripts/030_run_consensus.py --dry-run                # list eligible, don't call API
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback.coach_agent import generate_coach_feedback

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCORES_DIR = Path("memory/scores")
COMPARISONS_DIR = Path("memory/comparisons")
FEEDBACK_DIR = Path("memory/feedback")
FRAMES_DIR = Path("memory/frames")
HARVEST_LOG = Path("harvest_log.jsonl")
PROMPTS_DIR = Path("prompts")
LEDGER_DIR = Path("memory")
RUBRICS_DIR = Path("rubrics")

TASK_RUBRIC_FILES = {
    "task1": "task1_peg_transfer.yaml",
    "task2": "task2_pattern_cut.yaml",
    "task3": "task3_ligating_loop.yaml",
    "task4": "task4_extracorporeal_suture.yaml",
    "task5": "task5_intracorporeal_suture.yaml",
}


def _canonical_task_id(task: str) -> str:
    task_name = str(task).strip().lower()
    alias_map = {
        "task1_peg_transfer": "task1",
        "task2_pattern_cut": "task2",
        "task3_endoloop": "task3",
        "task3_ligating_loop": "task3",
        "task4_extracorporeal_knot": "task4",
        "task4_extracorporeal_suture": "task4",
        "task5_intracorporeal_suturing": "task5",
        "task5_intracorporeal_suture": "task5",
    }
    if task_name in alias_map:
        return alias_map[task_name]
    if task_name.isdigit():
        return f"task{task_name}"
    if task_name.startswith("task"):
        return task_name.split("_", 1)[0]
    return f"task{task_name}"


def _load_rubric(task: str) -> dict:
    task_id = _canonical_task_id(task)
    rubric_path = RUBRICS_DIR / TASK_RUBRIC_FILES[task_id]
    return yaml.safe_load(rubric_path.read_text()) or {}


def _build_task_context(task: str) -> str:
    rubric = _load_rubric(task)
    penalty_names = ", ".join(p.get("name", "") for p in rubric.get("penalties", []))
    return (
        f"Task ID: {rubric.get('task_id', _canonical_task_id(task))}\n"
        f"Task Name: {rubric.get('name', '')}\n"
        f"Maximum Time: {rubric.get('max_time_seconds', 'unknown')} seconds\n"
        f"Proficiency Target: {rubric.get('proficiency_time_seconds', 'unknown')} seconds\n"
        f"Score Formula: {rubric.get('score_formula', '')}\n"
        f"Penalty Categories: {penalty_names}"
    )


def _infer_task(claude_json: dict, gpt_json: dict, cli_task: str | None) -> str:
    if cli_task:
        return _canonical_task_id(cli_task)
    for payload in (claude_json, gpt_json):
        task_id = payload.get("task_id") or payload.get("task")
        if task_id:
            return _canonical_task_id(str(task_id))
    return "task5"


def _score_file_meta(payload: dict) -> dict:
    return {
        "completion_time_seconds": payload.get("completion_time_seconds"),
        "estimated_penalties": payload.get("estimated_penalties"),
        "estimated_fls_score": payload.get("estimated_fls_score"),
        "confidence": payload.get("confidence_score", payload.get("confidence")),
        "task_id": payload.get("task_id"),
    }


def _extract_response_text(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts).strip()


def _agreement_level(delta_fls: float) -> str:
    if delta_fls <= 1.0:
        return "very_strong"
    if delta_fls <= 5.0:
        return "strong"
    if delta_fls <= 12.0:
        return "moderate"
    return "weak"


def _build_teacher_comparison(
    video_id: str,
    claude_json: dict,
    gpt_json: dict,
    claude_path: Path,
    gpt_path: Path,
    consensus: dict,
    task_id: str,
) -> dict:
    claude_score = float(claude_json.get("estimated_fls_score") or 0.0)
    gpt_score = float(gpt_json.get("estimated_fls_score") or 0.0)
    claude_time = float(claude_json.get("completion_time_seconds") or 0.0)
    gpt_time = float(gpt_json.get("completion_time_seconds") or 0.0)
    claude_penalties = float(claude_json.get("estimated_penalties") or 0.0)
    gpt_penalties = float(gpt_json.get("estimated_penalties") or 0.0)

    consensus_score = consensus.get("consensus_score") or {}
    return {
        "id": f"comparison_{video_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "video_id": video_id,
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "teachers": {
            "claude": {
                "score_file": str(claude_path),
                **_score_file_meta(claude_json),
            },
            "gpt-4o": {
                "score_file": str(gpt_path),
                **_score_file_meta(gpt_json),
            },
        },
        "deltas": {
            "delta_completion_time_seconds": round(abs(claude_time - gpt_time), 2),
            "delta_penalties": round(abs(claude_penalties - gpt_penalties), 2),
            "delta_fls_score": round(abs(claude_score - gpt_score), 2),
            "agreement_level": _agreement_level(abs(claude_score - gpt_score)),
        },
        "consensus_summary": {
            "estimated_fls_score": consensus_score.get("estimated_fls_score"),
            "completion_time_seconds": consensus_score.get("completion_time_seconds"),
            "confidence_score": consensus_score.get("confidence_score", consensus.get("overall_confidence")),
        },
    }


def _save_consensus_score(video_id: str, consensus: dict, task_id: str, prompt_version: str) -> Path | None:
    consensus_score = consensus.get("consensus_score") or {}
    if not consensus_score:
        return None

    meta = consensus.get("_meta") or {}
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = SCORES_DIR / date_str
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "id": f"score_consensus_{video_id}_{timestamp}",
        "video_id": video_id,
        "video_filename": f"{video_id}.mp4",
        "video_hash": "",
        "source": "consensus",
        "model_name": meta.get("model") or "claude-sonnet-4-20250514",
        "model_version": meta.get("model") or "claude-sonnet-4-20250514",
        "prompt_version": prompt_version,
        "scored_at": meta.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        **consensus_score,
    }
    if "confidence_score" not in payload:
        payload["confidence_score"] = consensus.get("overall_confidence", consensus.get("agreement_score", 0.0))

    out_path = out_dir / f"{video_id}_consensus_{timestamp}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    return out_path


def _load_cached_frames(video_id: str) -> tuple[list[str], list[float]]:
    cache_path = FRAMES_DIR / video_id / "frames.json"
    if not cache_path.exists():
        return [], []
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return [], []
    return cache.get("frames_b64", []), cache.get("frame_timestamps", [])


def _load_harvest_entry(video_id: str) -> dict:
    if not HARVEST_LOG.exists():
        return {}
    for line in HARVEST_LOG.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("video_id") == video_id:
            return entry
    return {}


def _coach_skill_level(raw_skill: str) -> str:
    raw = (raw_skill or "").strip().lower()
    if raw in {"novice", "intermediate", "advanced"}:
        return raw
    if raw == "expert":
        return "advanced"
    return "intermediate"


def _load_trainee_history() -> list[dict]:
    history = []
    seen_videos = set()
    for path in sorted(SCORES_DIR.rglob("*.json")):
        if "claude" not in path.name:
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        video_id = data.get("video_id")
        if not video_id or video_id in seen_videos:
            continue
        seen_videos.add(video_id)
        history.append(
            {
                "video_id": video_id,
                "fls_score": data.get("estimated_fls_score"),
                "completion_time_seconds": data.get("completion_time_seconds"),
                "confidence": data.get("confidence_score"),
            }
        )
    return history


def _save_coach_feedback(video_id: str, feedback: dict) -> Path:
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = FEEDBACK_DIR / f"{video_id}_coach_{timestamp}.json"
    out_path.write_text(json.dumps(feedback, indent=2, default=str))
    return out_path


def _load_consensus_prompt(version: str, task: str) -> str:
    if version.startswith("v002"):
        return (PROMPTS_DIR / "v002_consensus_system.md").read_text()
    return (PROMPTS_DIR / f"{version}_{_canonical_task_id(task)}_consensus.md").read_text()


def find_score_file(video_id: str, source_hint: str) -> Path | None:
    """Find the score JSON for a video + source."""
    for date_dir in sorted(SCORES_DIR.iterdir(), reverse=True):
        if not date_dir.is_dir():
            continue
        for f in sorted(date_dir.iterdir(), reverse=True):
            if video_id in f.name and source_hint in f.name and f.suffix == ".json":
                return f
    return None


def find_eligible_videos() -> list[dict]:
    """Find all videos with both Claude and GPT-4o scores but no consensus yet."""
    # Collect all claude and gpt4o video IDs
    claude_files: dict[str, Path] = {}
    gpt_files: dict[str, Path] = {}

    for date_dir in sorted(SCORES_DIR.iterdir()):
        if not date_dir.is_dir():
            continue
        for f in sorted(date_dir.iterdir()):
            if not f.name.endswith(".json"):
                continue
            # Extract video_id from filename
            name = f.stem
            if "claude" in name:
                vid = name.split("_claude")[0]
                claude_files[vid] = f
            elif "gpt-4o" in name:
                vid = name.split("_gpt-4o")[0]
                gpt_files[vid] = f

    # Find existing consensus files
    existing_consensus = set()
    if COMPARISONS_DIR.exists():
        for f in COMPARISONS_DIR.iterdir():
            if "consensus" in f.name and f.suffix == ".json":
                # Extract video_id
                vid = f.name.split("_consensus")[0]
                existing_consensus.add(vid)

    # Build eligible list
    eligible = []
    for vid in sorted(set(claude_files.keys()) & set(gpt_files.keys())):
        if vid in existing_consensus:
            continue  # already have consensus
        eligible.append({
            "video_id": vid,
            "claude_path": claude_files[vid],
            "gpt_path": gpt_files[vid],
        })

    return eligible


def run_consensus(
    video_id: str,
    claude_json: dict,
    gpt_json: dict,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.3,
    prompt_version: str = "v001",
    task: str = "task5",
) -> dict:
    """Call the Critique Agent to produce consensus from two teacher scores."""
    import anthropic

    system_prompt = _load_consensus_prompt(prompt_version, task)
    task_context = _build_task_context(task)

    if prompt_version.startswith("v002"):
        user_message = f"""{task_context}

## Claude Score JSON:
```json
{json.dumps(claude_json, indent=2)}
```

## GPT-4o Score JSON:
```json
{json.dumps(gpt_json, indent=2)}
```

Merge these into the consensus JSON."""

        client = anthropic.Anthropic()
        t0 = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        elapsed = time.time() - t0

        raw_text = _extract_response_text(response.content)
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            raw_text = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            ).strip()

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as e:
            logger.error(f"Consensus returned invalid JSON for {video_id}: {e}")
            return {
                "error": "json_parse_failure",
                "video_id": video_id,
                "raw_response": raw_text[:3000],
            }

        result["_meta"] = {
            "video_id": video_id,
            "model": model,
            "final_round": 1,
            "elapsed_seconds": round(elapsed, 2),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd_approx": round(
                response.usage.input_tokens * 3 / 1_000_000
                + response.usage.output_tokens * 15 / 1_000_000, 4
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_version": prompt_version,
        }
        return result

    # Build round 1 prompt
    user_message = f"""{task_context}

## Teacher A (Claude Sonnet 4):
```json
{json.dumps(claude_json, indent=2)}
```

## Teacher B (GPT-4o):
```json
{json.dumps(gpt_json, indent=2)}
```

## Video Metadata:
- Video ID: {video_id}
- Duration: {claude_json.get('completion_time_seconds', 'unknown')}s (Teacher A estimate)
- Frame timestamps from Teacher A's frame_analyses

This is Round 1. Produce the consensus JSON."""

    # Replace {round_number} in system prompt
    round1_system = system_prompt.replace("{round_number}", "1")

    client = anthropic.Anthropic()

    t0 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=round1_system,
        messages=[{"role": "user", "content": user_message}],
    )
    elapsed = time.time() - t0

    raw_text = _extract_response_text(response.content)

    # Strip markdown fences
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        raw_text = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        ).strip()

    try:
        result = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.error(f"Consensus returned invalid JSON for {video_id}: {e}")
        return {
            "error": "json_parse_failure",
            "video_id": video_id,
            "raw_response": raw_text[:3000],
        }

    # Attach metadata
    result["_meta"] = {
        "video_id": video_id,
        "model": model,
        "round": 1,
        "elapsed_seconds": round(elapsed, 2),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd_approx": round(
            response.usage.input_tokens * 3 / 1_000_000
            + response.usage.output_tokens * 15 / 1_000_000, 4
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Check if round 2 needed
    agreement = result.get("agreement_score", 0)
    early_stop = float(os.getenv("CONSENSUS_EARLY_STOP", "0.92"))
    needs_rebuttal = result.get("needs_rebuttal", [])

    if agreement >= early_stop or not needs_rebuttal:
        result["_meta"]["final_round"] = 1
        result["_meta"]["early_stopped"] = agreement >= early_stop
        logger.info(
            f"  {video_id}: Round 1 agreement={agreement:.2f} "
            f"{'(early-stop)' if agreement >= early_stop else '(no rebuttals needed)'}"
        )
    else:
        # Round 2
        logger.info(
            f"  {video_id}: Round 1 agreement={agreement:.2f}, "
            f"{len(needs_rebuttal)} fields need rebuttal — running Round 2..."
        )
        round2_system = system_prompt.replace("{round_number}", "2")
        round2_message = f"""This is Round 2 (Final Arbitration). Here is the Round 1 output:
```json
{json.dumps(result, indent=2, default=str)}
```

The following fields were flagged as needs_rebuttal:
{json.dumps(needs_rebuttal, indent=2)}

Make your final ruling on each disputed field. Output the definitive consensus JSON."""

        t0 = time.time()
        r2_response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=round2_system,
            messages=[{"role": "user", "content": round2_message}],
        )
        r2_elapsed = time.time() - t0

        r2_text = _extract_response_text(r2_response.content)
        if r2_text.startswith("```"):
            lines = r2_text.split("\n")
            r2_text = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
            ).strip()

        try:
            r2_result = json.loads(r2_text)
            # Merge round 2 into result
            r2_result["_meta"] = {
                **result["_meta"],
                "final_round": 2,
                "early_stopped": False,
                "round2_elapsed_seconds": round(r2_elapsed, 2),
                "round2_input_tokens": r2_response.usage.input_tokens,
                "round2_output_tokens": r2_response.usage.output_tokens,
                "total_cost_usd_approx": round(
                    result["_meta"]["cost_usd_approx"]
                    + r2_response.usage.input_tokens * 3 / 1_000_000
                    + r2_response.usage.output_tokens * 15 / 1_000_000, 4
                ),
            }
            result = r2_result
        except json.JSONDecodeError:
            logger.warning(f"  {video_id}: Round 2 JSON parse failed, using Round 1 result")
            result["_meta"]["final_round"] = 1
            result["_meta"]["round2_error"] = "json_parse_failure"

    return result


def append_ledger(video_id: str, consensus: dict):
    """Append consensus event to learning ledger."""
    score_payload = consensus.get("consensus_score", {}) or {}
    fls_score = score_payload.get("estimated_fls_score")
    if fls_score is None:
        fls_score = (score_payload.get("score_components") or {}).get("total_fls_score")

    agreement_score = consensus.get("agreement_score")
    if agreement_score is None:
        agreement_score = consensus.get("overall_confidence")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "consensus_generated",
        "data": {
            "video_id": video_id,
            "agreement_score": agreement_score,
            "rounds": consensus.get("_meta", {}).get("final_round", 1),
            "fls_score": fls_score,
            "cost_usd": consensus.get("_meta", {}).get("cost_usd_approx"),
        },
    }
    ledger_path = LEDGER_DIR / "learning_ledger.jsonl"
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate consensus from dual teacher scores")
    parser.add_argument("--video-id", help="Run for a single video")
    parser.add_argument("--task", help="Task id or task number override")
    parser.add_argument("--prompt-version", default="v002")
    parser.add_argument("--dry-run", action="store_true", help="List eligible videos, don't call API")
    parser.add_argument("--force", action="store_true", help="Re-run even if consensus exists")
    parser.add_argument("--with-coach-feedback", action="store_true", help="Generate coach feedback JSON after each consensus")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

    model = args.model or os.getenv("CRITIQUE_MODEL", "claude-sonnet-4-20250514")

    if args.video_id:
        # Single video mode
        claude_path = find_score_file(args.video_id, "claude")
        gpt_path = find_score_file(args.video_id, "gpt-4o")
        if not claude_path or not gpt_path:
            print(f"ERROR: Need both Claude and GPT-4o scores for {args.video_id}")
            print(f"  Claude: {'✅ ' + str(claude_path) if claude_path else '❌ missing'}")
            print(f"  GPT-4o: {'✅ ' + str(gpt_path) if gpt_path else '❌ missing'}")
            sys.exit(1)
        targets = [{"video_id": args.video_id, "claude_path": claude_path, "gpt_path": gpt_path}]
    else:
        targets = find_eligible_videos()
        if args.force:
            # Include all videos with both teachers, even if consensus exists
            all_both = []
            claude_files = {}
            gpt_files = {}
            for date_dir in sorted(SCORES_DIR.iterdir()):
                if not date_dir.is_dir():
                    continue
                for f in sorted(date_dir.iterdir()):
                    if not f.name.endswith(".json"):
                        continue
                    name = f.stem
                    if "claude" in name:
                        vid = name.split("_claude")[0]
                        claude_files[vid] = f
                    elif "gpt-4o" in name:
                        vid = name.split("_gpt-4o")[0]
                        gpt_files[vid] = f
            for vid in sorted(set(claude_files.keys()) & set(gpt_files.keys())):
                all_both.append({
                    "video_id": vid,
                    "claude_path": claude_files[vid],
                    "gpt_path": gpt_files[vid],
                })
            targets = all_both

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Found {len(targets)} videos to process:\n")
    for t in targets:
        print(f"  {t['video_id']}")

    if args.dry_run or not targets:
        return

    print(f"\nUsing model: {model}")
    print(f"Starting consensus generation...\n")

    total_cost = 0.0
    successes = 0
    failures = 0

    for i, target in enumerate(targets, 1):
        vid = target["video_id"]
        print(f"[{i}/{len(targets)}] {vid}...")

        claude_json = json.loads(target["claude_path"].read_text())
        gpt_json = json.loads(target["gpt_path"].read_text())
        task_id = _infer_task(claude_json, gpt_json, args.task)

        try:
            consensus = run_consensus(
                vid,
                claude_json,
                gpt_json,
                model=model,
                prompt_version=args.prompt_version,
                task=task_id,
            )
        except Exception as e:
            logger.error(f"  {vid}: API call failed: {e}")
            failures += 1
            continue

        if "error" in consensus:
            logger.error(f"  {vid}: {consensus['error']}")
            failures += 1
            continue

        # Save
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = COMPARISONS_DIR / f"{vid}_consensus_{ts}.json"
        out_path.write_text(json.dumps(consensus, indent=2, default=str))
        comparison = _build_teacher_comparison(
            vid,
            claude_json,
            gpt_json,
            target["claude_path"],
            target["gpt_path"],
            consensus,
            task_id,
        )
        comparison_path = COMPARISONS_DIR / f"{vid}_teacher_comparison_{ts}.json"
        comparison_path.write_text(json.dumps(comparison, indent=2, default=str))
        consensus_score_path = _save_consensus_score(vid, consensus, task_id, args.prompt_version)

        # Ledger
        append_ledger(vid, consensus)

        meta = consensus.get("_meta", {})
        agreement = consensus.get("agreement_score")
        if agreement is None:
            agreement = consensus.get("overall_confidence", 0)
        score_payload = consensus.get("consensus_score", {}) or {}
        fls = score_payload.get("estimated_fls_score")
        if fls is None:
            fls = (score_payload.get("score_components") or {}).get("total_fls_score", "?")
        cost = meta.get("cost_usd_approx", meta.get("total_cost_usd_approx", 0))
        total_cost += cost
        successes += 1

        print(
            f"  ✅ agreement={agreement:.2f} | FLS={fls} | "
            f"rounds={meta.get('final_round', '?')} | "
            f"${cost:.4f} | saved → {out_path.name}"
        )
        print(f"     comparison → {comparison_path.name}")
        if consensus_score_path:
            print(f"     score → {consensus_score_path.name}")

        if args.with_coach_feedback:
            harvest_entry = _load_harvest_entry(vid)
            frames_b64, frame_timestamps = _load_cached_frames(vid)
            try:
                coach_feedback = generate_coach_feedback(
                    consensus_json=consensus.get("consensus_score") or {},
                    teacher_a_json=claude_json,
                    teacher_b_json=gpt_json,
                    frame_b64s=frames_b64,
                    frame_timestamps=frame_timestamps,
                    trainee_history=_load_trainee_history(),
                    prompt_version=args.prompt_version,
                    task_id=task_id,
                    skill_level=_coach_skill_level(str(harvest_entry.get("estimated_skill_level") or "intermediate")),
                )
                if "error" in coach_feedback:
                    logger.warning(f"  {vid}: coach feedback failed: {coach_feedback['error']}")
                else:
                    coach_path = _save_coach_feedback(vid, coach_feedback)
                    print(f"     coach → {coach_path.name}")
            except Exception as exc:
                logger.warning(f"  {vid}: coach feedback exception: {exc}")

        # Rate limit courtesy
        if i < len(targets):
            time.sleep(1)

    print(f"\n{'='*50}")
    print(f"Done: {successes} succeeded, {failures} failed")
    print(f"Total API cost: ~${total_cost:.4f}")
    print(f"Consensus files saved to {COMPARISONS_DIR}/")


if __name__ == "__main__":
    main()
