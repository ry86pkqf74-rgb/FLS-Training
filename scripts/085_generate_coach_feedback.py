#!/usr/bin/env python3
"""085_generate_coach_feedback.py — Generate rich technique coaching for a scored video.

This runs AFTER scoring (020) and consensus. It calls a dedicated Coach Agent
(separate from the scoring pipeline) to produce pedagogically rich, frame-grounded
technique feedback that goes beyond the official FLS rubric.

Usage:
    python scripts/085_generate_coach_feedback.py --video-id V22_video
    python scripts/085_generate_coach_feedback.py --video-id V22_video --with-frames ~/videos/V22.mov
    python scripts/085_generate_coach_feedback.py --video-id V22_video --output memory/feedback/V22_coach.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.json import JSON as RichJSON

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback.coach_agent import generate_coach_feedback
from src.memory.memory_store import MemoryStore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def _load_score_json(score_dir: Path, video_id: str, source_hint: str) -> dict | None:
    """Find and load a score JSON from memory/scores/."""
    for f in sorted(score_dir.rglob("*.json"), reverse=True):
        if video_id in f.name and source_hint in f.name:
            return json.loads(f.read_text())
    return None


def _load_trainee_history(score_dir: Path) -> list[dict]:
    """Load summary of all scored videos for progress context."""
    history = []
    seen_videos = set()
    for f in sorted(score_dir.rglob("*.json")):
        if "claude" not in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            vid = data.get("video_id") or f.stem.split("_claude")[0]
            if vid in seen_videos:
                continue
            seen_videos.add(vid)
            history.append({
                "video_id": vid,
                "fls_score": data.get("estimated_fls_score"),
                "completion_time_seconds": data.get("completion_time_seconds"),
                "confidence": data.get("confidence_score"),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return history


def main():
    parser = argparse.ArgumentParser(description="Generate coach feedback for a scored video")
    parser.add_argument("--video-id", required=True, help="Video ID to generate coaching for")
    parser.add_argument("--with-frames", help="Path to video file (extracts frames for richer coaching)")
    parser.add_argument("--output", help="Output JSON path (default: memory/feedback/{video_id}_coach.json)")
    parser.add_argument("--model", help="Override coach model")
    parser.add_argument("--no-history", action="store_true", help="Skip trainee history context")
    parser.add_argument("--db", default="data/fls_training.duckdb")
    args = parser.parse_args()

    score_dir = Path("memory/scores")
    feedback_dir = Path("memory/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)

    # Load consensus or best available score
    console.print(f"[bold]Loading scores for {args.video_id}...[/bold]")

    # Try consensus first, then claude, then gpt-4o
    consensus = _load_score_json(score_dir, args.video_id, "consensus")
    teacher_a = _load_score_json(score_dir, args.video_id, "claude")
    teacher_b = _load_score_json(score_dir, args.video_id, "gpt-4o")

    # Use best available as consensus if no explicit consensus exists
    if not consensus:
        consensus = teacher_a or teacher_b
        if consensus:
            console.print("[yellow]No consensus score found, using best teacher score[/yellow]")
        else:
            console.print(f"[red]No scores found for {args.video_id}[/red]")
            sys.exit(1)

    console.print(f"  FLS Score: {consensus.get('estimated_fls_score', '?')}")
    console.print(f"  Time: {consensus.get('completion_time_seconds', '?')}s")
    console.print(f"  Confidence: {consensus.get('confidence_score', '?')}")

    # Extract frames if video provided
    frame_b64s = []
    frame_timestamps = []
    if args.with_frames:
        console.print(f"[bold]Extracting frames from {args.with_frames}...[/bold]")
        try:
            from src.ingest.frame_extractor import extract_uniform_frames, extract_final_state_frames
            video_path = Path(args.with_frames)
            uniform = extract_uniform_frames(video_path, n_frames=15)  # fewer for coach
            final = extract_final_state_frames(video_path, n_frames=3)
            all_frames = uniform + final
            frame_b64s = [f.image_b64 for f in all_frames]
            frame_timestamps = [f.timestamp_seconds for f in all_frames]
            console.print(f"  Extracted {len(all_frames)} frames")
        except Exception as e:
            console.print(f"[yellow]Frame extraction failed: {e}. Proceeding without frames.[/yellow]")

    # Load trainee history
    trainee_history = None
    if not args.no_history:
        trainee_history = _load_trainee_history(score_dir)
        if trainee_history:
            console.print(f"[bold]Loaded history: {len(trainee_history)} previous sessions[/bold]")

    # Call coach agent
    console.print(f"\n[bold green]Generating coach feedback...[/bold green]")
    feedback = generate_coach_feedback(
        consensus_json=consensus,
        teacher_a_json=teacher_a,
        teacher_b_json=teacher_b,
        frame_b64s=frame_b64s,
        frame_timestamps=frame_timestamps,
        trainee_history=trainee_history,
        model=args.model,
    )

    if "error" in feedback:
        console.print(f"[red]Coach error: {feedback['error']}[/red]")
        if "raw_response" in feedback:
            console.print(feedback["raw_response"][:500])
        sys.exit(1)

    # Save
    output_path = Path(args.output) if args.output else (
        feedback_dir / f"{args.video_id}_coach_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(feedback, indent=2))
    console.print(f"\n[green]Saved to {output_path}[/green]")

    # Print summary
    console.print(f"\n[bold]═══ Coach Feedback Summary ═══[/bold]")
    oa = feedback.get("overall_assessment", {})
    console.print(f"  {oa.get('session_headline', '')}")

    console.print(f"\n[bold]Strengths:[/bold]")
    for s in feedback.get("strengths_to_reinforce", [])[:3]:
        console.print(f"  ✅ {s.get('observation', '')}")

    console.print(f"\n[bold]Priority Technique Coaching:[/bold]")
    for tc in feedback.get("technique_coaching", [])[:3]:
        pri = tc.get("priority", "")
        icon = "🔴" if pri == "high" else "🟡" if pri == "medium" else "🟢"
        console.print(f"  {icon} [{tc.get('category', '')}] {tc.get('observation', '')}")
        console.print(f"     → {tc.get('correction', '')}")
        if tc.get("drill"):
            console.print(f"     🏋️ Drill: {tc['drill']}")

    pp = feedback.get("practice_plan", {})
    if pp:
        console.print(f"\n[bold]Next Session Focus:[/bold] {pp.get('session_focus', '')}")
        pd = pp.get("primary_drill", {})
        if pd:
            console.print(f"  Drill: {pd.get('name', '')} ({pd.get('target_time_minutes', '?')} min)")

    meta = feedback.get("_meta", {})
    console.print(f"\n[dim]Model: {meta.get('model', '?')} | "
                  f"Time: {meta.get('elapsed_seconds', '?')}s | "
                  f"Cost: ~${meta.get('cost_usd_approx', '?')}[/dim]")


if __name__ == "__main__":
    main()
