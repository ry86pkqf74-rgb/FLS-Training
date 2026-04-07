#!/usr/bin/env python3
"""020_score_frontier.py — Score a video with Claude + GPT-4o, then critique.

Usage:
    python scripts/020_score_frontier.py --video-id abc123 --video /path/to/video.mov
    python scripts/020_score_frontier.py --video-id abc123 --video /path/to/video.mov --models claude
    python scripts/020_score_frontier.py --video-id abc123 --video /path/to/video.mov --skip-critique
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.frame_extractor import (
    extract_all_frames_for_scoring,
    frames_to_b64_list,
    frames_timestamps,
)
from src.ingest.video_metadata import ingest_video
from src.memory.learning_log import LearningLog
from src.memory.memory_store import MemoryStore
from src.scoring.ensemble_scorer import critique_and_resolve
from src.scoring.frontier_scorer import score_with_claude, score_with_gpt4o
from src.scoring.schema import FLSTask, VideoMetadata

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
console = Console()

TASK_MAP = {
    "1": FLSTask.PEG_TRANSFER,
    "2": FLSTask.PATTERN_CUT,
    "3": FLSTask.CLIP_APPLY,
    "4": FLSTask.EXTRACORPOREAL,
    "5": FLSTask.INTRACORPOREAL,
}


def _print_score_summary(label: str, score):
    table = Table(title=f"{label} Score")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("FLS Score", f"{score.estimated_fls_score:.1f}")
    table.add_row("Completion Time", f"{score.completion_time_seconds:.1f}s")
    table.add_row("Penalties", f"{score.estimated_penalties:.1f}")
    table.add_row("Confidence", f"{score.confidence_score:.2f}")
    table.add_row("API Cost", f"${score.api_cost_usd:.4f}")
    table.add_row("Latency", f"{score.latency_seconds:.1f}s")

    if score.knot_assessments:
        for ka in score.knot_assessments:
            secure = "✓" if ka.appears_secure else "✗"
            switched = "✓" if ka.hand_switched else ("✗" if ka.hand_switched is False else "—")
            table.add_row(f"  Throw {ka.throw_number}", f"secure={secure} hand_switch={switched} hand={ka.hand_used.value}")

    if score.drain_assessment:
        da = score.drain_assessment
        table.add_row("Drain", f"gap={da.gap_visible} avulsed={da.drain_avulsed} closure={da.slit_closure_quality}")

    console.print(table)

    if score.strengths:
        console.print(Panel("\n".join(f"• {s}" for s in score.strengths), title="Strengths", border_style="green"))
    if score.improvement_suggestions:
        console.print(Panel("\n".join(f"• {s}" for s in score.improvement_suggestions), title="Improvements", border_style="yellow"))


def main():
    parser = argparse.ArgumentParser(description="Score an FLS video with frontier models")
    parser.add_argument("--video-id", required=True, help="Video ID from ingestion")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--task", default="5", choices=["1", "2", "3", "4", "5"])
    parser.add_argument("--models", default="claude,gpt4o",
                        help="Comma-separated: claude, gpt4o")
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--n-final", type=int, default=3)
    parser.add_argument("--prompt-version", default="v001")
    parser.add_argument("--skip-critique", action="store_true",
                        help="Skip critique agent, just score independently")
    parser.add_argument("--db", default="data/fls_training.duckdb")
    args = parser.parse_args()

    store = MemoryStore(args.db)
    log = LearningLog()
    models = [m.strip() for m in args.models.split(",")]

    # Check video exists in store (or ingest on the fly)
    video_record = store.get_video(args.video_id)
    if not video_record:
        console.print(f"[yellow]Video {args.video_id} not in store, ingesting...[/yellow]")
        meta = ingest_video(args.video, TASK_MAP[args.task])
        meta.id = args.video_id  # preserve the provided ID
        store.insert_video(meta)
        log.log_video_ingested(meta.id, meta.filename, TASK_MAP[args.task].value)
    else:
        meta = VideoMetadata(
            id=video_record["id"],
            filename=video_record["filename"],
            task=FLSTask(video_record["task"]),
            duration_seconds=video_record["duration_seconds"],
            resolution=video_record["resolution"],
            fps=video_record["fps"],
            file_hash=video_record.get("file_hash", ""),
        )

    # Extract frames
    console.print(f"\n[bold]Extracting frames from {args.video}...[/bold]")
    uniform_frames, final_frames = extract_all_frames_for_scoring(
        args.video, n_uniform=args.n_frames, n_final=args.n_final
    )
    all_frames = uniform_frames + final_frames
    b64s = frames_to_b64_list(all_frames)
    timestamps = frames_timestamps(all_frames)
    console.print(f"Extracted {len(uniform_frames)} uniform + {len(final_frames)} final = {len(all_frames)} frames\n")

    scores = {}

    # Score with Claude
    if "claude" in models:
        console.print("[bold blue]Scoring with Claude (Teacher A)...[/bold blue]")
        score_a = score_with_claude(b64s, timestamps, meta, args.prompt_version)
        scores["claude"] = score_a
        store.insert_score(score_a)
        log.save_score(score_a)
        _print_score_summary("Teacher A (Claude)", score_a)

    # Score with GPT-4o
    if "gpt4o" in models:
        console.print("\n[bold green]Scoring with GPT-4o (Teacher B)...[/bold green]")
        score_b = score_with_gpt4o(b64s, timestamps, meta, args.prompt_version)
        scores["gpt4o"] = score_b
        store.insert_score(score_b)
        log.save_score(score_b)
        _print_score_summary("Teacher B (GPT-4o)", score_b)

    # Critique (if both teachers scored)
    if "claude" in scores and "gpt4o" in scores and not args.skip_critique:
        console.print("\n[bold magenta]Running Critique Agent...[/bold magenta]")
        critique = critique_and_resolve(scores["claude"], scores["gpt4o"], meta.id)
        store.insert_critique(critique)
        log.save_critique(critique)

        console.print(f"\n[bold]Agreement: {critique.agreement_score:.2f}[/bold]")
        if critique.divergences:
            div_table = Table(title="Divergences Resolved")
            div_table.add_column("Field", style="cyan")
            div_table.add_column("Teacher A", style="blue")
            div_table.add_column("Teacher B", style="green")
            div_table.add_column("Resolution", style="magenta")
            for d in critique.divergences:
                div_table.add_row(d.field, d.teacher_a_value, d.teacher_b_value, d.resolution)
            console.print(div_table)

        if critique.consensus_score:
            _print_score_summary("CONSENSUS", critique.consensus_score)
            store.insert_score(critique.consensus_score)
            log.save_score(critique.consensus_score)

    # Summary
    console.print(f"\n[bold]Total API cost this run: ${sum(s.api_cost_usd for s in scores.values()):.4f}[/bold]")
    stats = store.get_stats()
    console.print(f"Cumulative: {stats['total_videos']} videos, {stats['total_scores']} scores, ${stats['total_api_cost']:.4f} total spend")

    store.close()


if __name__ == "__main__":
    main()
