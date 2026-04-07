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

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCORES_DIR = Path("memory/scores")
COMPARISONS_DIR = Path("memory/comparisons")
PROMPTS_DIR = Path("prompts")
LEDGER_DIR = Path("memory")


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
) -> dict:
    """Call the Critique Agent to produce consensus from two teacher scores."""
    import anthropic

    system_prompt = (PROMPTS_DIR / "v001_task5_consensus.md").read_text()

    # Build round 1 prompt
    user_message = f"""## Teacher A (Claude Sonnet 4):
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

## FLS Task 5 Rubric Summary:
- Max time: 600s (score = 600 - time - penalties)
- 3 throws required: first is surgeon's knot (double), throws 2-3 are single
- Hand must switch between throws
- Suture through marked points, drain slit must close
- Penalties: deviation from marks, gap in drain, insecure knot, avulsed drain

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

    raw_text = response.content[0].text.strip()

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

        r2_text = r2_response.content[0].text.strip()
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
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "consensus_generated",
        "data": {
            "video_id": video_id,
            "agreement_score": consensus.get("agreement_score"),
            "rounds": consensus.get("_meta", {}).get("final_round", 1),
            "fls_score": (consensus.get("consensus_score", {}) or {}).get(
                "estimated_fls_score"
            ),
            "cost_usd": consensus.get("_meta", {}).get("cost_usd_approx"),
        },
    }
    ledger_path = LEDGER_DIR / "learning_ledger.jsonl"
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate consensus from dual teacher scores")
    parser.add_argument("--video-id", help="Run for a single video")
    parser.add_argument("--dry-run", action="store_true", help="List eligible videos, don't call API")
    parser.add_argument("--force", action="store_true", help="Re-run even if consensus exists")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)

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

        try:
            consensus = run_consensus(vid, claude_json, gpt_json, model=model)
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

        # Ledger
        append_ledger(vid, consensus)

        meta = consensus.get("_meta", {})
        agreement = consensus.get("agreement_score", 0)
        fls = (consensus.get("consensus_score", {}) or {}).get("estimated_fls_score", "?")
        cost = meta.get("cost_usd_approx", meta.get("total_cost_usd_approx", 0))
        total_cost += cost
        successes += 1

        print(
            f"  ✅ agreement={agreement:.2f} | FLS={fls} | "
            f"rounds={meta.get('final_round', '?')} | "
            f"${cost:.4f} | saved → {out_path.name}"
        )

        # Rate limit courtesy
        if i < len(targets):
            time.sleep(1)

    print(f"\n{'='*50}")
    print(f"Done: {successes} succeeded, {failures} failed")
    print(f"Total API cost: ~${total_cost:.4f}")
    print(f"Consensus files saved to {COMPARISONS_DIR}/")


if __name__ == "__main__":
    main()
