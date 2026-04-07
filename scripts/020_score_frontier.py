#!/usr/bin/env python3
"""Score a video using frontier models (Claude + GPT-4o) with critique consensus."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.scoring.frontier_scorer import score_with_claude, score_with_gpt, run_critique
from src.memory.memory_store import MemoryStore

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Score video with frontier models")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--task", default="5")
    parser.add_argument("--prompt-version", default="v001")
    parser.add_argument("--skip-gpt", action="store_true", help="Skip GPT-4o (Claude only)")
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    store = MemoryStore(args.base_dir)
    video_path = Path(args.video)

    # Load cached frames
    frames_path = Path(args.base_dir) / "memory" / "frames" / args.video_id / "frames.json"
    if frames_path.exists():
        cache = json.loads(frames_path.read_text())
        frames_b64 = cache["frames_b64"]
        console.print(f"[green]Loaded {len(frames_b64)} cached frames[/green]")
    else:
        console.print("[yellow]No cached frames — extracting from video...[/yellow]")
        from src.ingest.frame_extractor import extract_frames, frames_to_base64
        frames, meta = extract_frames(video_path)
        frames_b64 = frames_to_base64(frames)

    # Build history summary
    all_scores = store.get_all_scores()
    history_lines = []
    for s in sorted(all_scores, key=lambda x: x.scored_at)[-10:]:
        history_lines.append(f"{s.video_id}: {s.completion_time_seconds:.0f}s / {s.estimated_fls_score:.0f} FLS")
    history_summary = "\n".join(history_lines) if history_lines else ""
    score_b = None

    # Score with Claude
    console.print("\n[bold blue]Scoring with Claude...[/bold blue]")
    score_a = score_with_claude(
        frames_b64=frames_b64,
        video_id=args.video_id,
        video_filename=video_path.name,
        video_hash="",
        prompt_version=args.prompt_version,
        task=args.task,
        previous_scores_summary=history_summary,
    )
    store.save_score(score_a)
    console.print(f"  Claude: {score_a.completion_time_seconds:.0f}s / {score_a.estimated_fls_score:.0f} FLS (conf: {score_a.confidence_score:.2f})")

    if not args.skip_gpt:
        # Score with GPT-4o
        console.print("\n[bold green]Scoring with GPT-4o...[/bold green]")
        try:
            score_b = score_with_gpt(
                frames_b64=frames_b64,
                video_id=args.video_id,
                video_filename=video_path.name,
                video_hash="",
                prompt_version=args.prompt_version,
                task=args.task,
                previous_scores_summary=history_summary,
            )
            store.save_score(score_b)
            console.print(f"  GPT-4o: {score_b.completion_time_seconds:.0f}s / {score_b.estimated_fls_score:.0f} FLS (conf: {score_b.confidence_score:.2f})")

            # Run critique
            console.print("\n[bold magenta]Running critique agent...[/bold magenta]")
            try:
                consensus = run_critique(
                    score_a=score_a,
                    score_b=score_b,
                    video_id=args.video_id,
                    prompt_version=args.prompt_version,
                    task=args.task,
                )
            except Exception as exc:
                console.print(f"[yellow]Critique failed, using higher-confidence teacher score: {exc}[/yellow]")
                selected = score_a if score_a.confidence_score >= score_b.confidence_score else score_b
                consensus = selected.model_copy(update={
                    "id": f"score_consensus_{args.video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                    "source": "consensus",
                    "model_name": "fallback_selector",
                    "model_version": selected.model_version,
                })
            store.save_score(consensus)
        except Exception as exc:
            console.print(f"[yellow]GPT-4o failed, falling back to Claude-only consensus: {exc}[/yellow]")
            consensus = score_a.model_copy(update={
                "id": f"score_consensus_{args.video_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "source": "consensus",
                "model_name": "fallback_claude",
                "model_version": score_a.model_version,
            })
            store.save_score(consensus)
    else:
        consensus = score_a

    # Display comparison table
    table = Table(title=f"Scoring Results — {args.video_id}")
    table.add_column("Metric", style="bold")
    table.add_column("Claude", justify="right")
    if not args.skip_gpt and score_b is not None:
        table.add_column("GPT-4o", justify="right")
    table.add_column("Consensus", justify="right", style="bold green")

    rows = [
        ("Time (s)", f"{score_a.completion_time_seconds:.0f}",
         f"{consensus.completion_time_seconds:.0f}"),
        ("FLS Score", f"{score_a.estimated_fls_score:.0f}",
         f"{consensus.estimated_fls_score:.0f}"),
        ("Confidence", f"{score_a.confidence_score:.2f}",
         f"{consensus.confidence_score:.2f}"),
        ("Penalties", f"{score_a.estimated_penalties:.1f}",
         f"{consensus.estimated_penalties:.1f}"),
    ]

    for row in rows:
        if args.skip_gpt or score_b is None:
            table.add_row(row[0], row[1], row[2])
        else:
            table.add_row(row[0], row[1], f"{score_b.completion_time_seconds:.0f}" if row[0] == "Time (s)" else
                         f"{score_b.estimated_fls_score:.0f}" if row[0] == "FLS Score" else
                         f"{score_b.confidence_score:.2f}" if row[0] == "Confidence" else
                         f"{score_b.estimated_penalties:.1f}", row[2])

    console.print(table)

    # Generate feedback
    console.print("\n[bold]Generating coaching feedback...[/bold]")
    from src.feedback.generator import generate_feedback
    previous = [s for s in all_scores if s.scored_at < consensus.scored_at]
    report = generate_feedback(consensus, previous, store.get_trainee_profile())
    store.save_feedback(report)
    console.print(f"  Headline: {report.headline}")
    if report.top_priorities:
        console.print(f"  Top priority: {report.top_priorities[0].description}")

    # Update trainee profile
    store.rebuild_trainee_profile()
    console.print("\n[green]Done. Score and feedback saved to memory/[/green]")


if __name__ == "__main__":
    main()
