#!/usr/bin/env python3
"""Generate or regenerate coaching feedback for a video."""
import argparse
import json
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from src.feedback.generator import generate_feedback
from src.memory.memory_store import MemoryStore

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Generate feedback report")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--export-md", help="Export as markdown file")
    args = parser.parse_args()

    store = MemoryStore(args.base_dir)
    scores = store.get_scores_for_video(args.video_id)

    if not scores:
        console.print(f"[red]No scores found for {args.video_id}[/red]")
        return

    # Use consensus or latest score
    score = next((s for s in scores if s.source == "consensus"), scores[-1])
    all_scores = store.get_all_scores()
    previous = [s for s in sorted(all_scores, key=lambda x: x.scored_at) if s.scored_at < score.scored_at]
    profile = store.get_trainee_profile()

    report = generate_feedback(score, previous, profile)
    store.save_feedback(report)

    # Display
    console.print(Panel(f"[bold]{report.headline}[/bold]\n\n"
                       f"FLS Score: {report.fls_score:.0f}  |  Time: {report.completion_time:.0f}s  |  "
                       f"Attempt #{report.attempt_number}",
                       title=f"Feedback — {args.video_id}"))

    if report.top_priorities:
        console.print("\n[bold]Top priorities:[/bold]")
        for p in report.top_priorities:
            console.print(f"  {p.rank}. [{p.priority.value}] {p.description}")

    if report.progression_insights:
        console.print("\n[bold]Progression insights:[/bold]")
        for ins in report.progression_insights:
            console.print(f"  • {ins.description}")

    if report.strengths:
        console.print("\n[bold]Strengths:[/bold]")
        for s in report.strengths:
            console.print(f"  ✓ {s}")

    if report.fatigue_risk != "none":
        console.print(f"\n[yellow]Fatigue: {report.fatigue_risk} — {report.fatigue_evidence}[/yellow]")

    console.print(f"\n[bold]Next session:[/bold] {report.next_session_plan}")
    console.print(f"[dim]{report.distance_to_proficiency}[/dim]")

    # Export
    if args.export_md:
        md = _to_markdown(report)
        with open(args.export_md, "w") as f:
            f.write(md)
        console.print(f"\n[green]Exported to {args.export_md}[/green]")


def _to_markdown(report) -> str:
    lines = [
        f"# Feedback Report — {report.video_id}",
        f"\n**{report.headline}**\n",
        f"- FLS Score: {report.fls_score:.0f}",
        f"- Completion Time: {report.completion_time:.0f}s",
        f"- Attempt: #{report.attempt_number}",
        f"\n## Top Priorities\n",
    ]
    for p in report.top_priorities:
        lines.append(f"{p.rank}. **[{p.priority.value}]** {p.description}")

    lines.append(f"\n## Phase Coaching\n")
    for pc in report.phase_coaching:
        status_emoji = {"improving": "↗", "plateau": "→", "regressing": "↘"}.get(pc.status, "•")
        lines.append(f"- **{pc.phase}**: {pc.duration_seconds:.0f}s (target: {pc.benchmark_seconds:.0f}s) {status_emoji} {pc.coaching_note}")

    lines.append(f"\n## Progression Insights\n")
    for ins in report.progression_insights:
        lines.append(f"- {ins.description}")
        if ins.recommendation:
            lines.append(f"  - *Recommendation*: {ins.recommendation}")

    lines.append(f"\n## Strengths\n")
    for s in report.strengths:
        lines.append(f"- {s}")

    lines.append(f"\n## Next Session Plan\n")
    lines.append(report.next_session_plan)

    lines.append(f"\n---\n*{report.distance_to_proficiency}*")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
