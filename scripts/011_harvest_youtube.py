#!/usr/bin/env python3
"""
FLS-Training: YouTube Video Harvester
======================================
Downloads FLS Task 5 videos from YouTube, extracts metadata,
classifies tier (A/B/C), and feeds into the scoring pipeline.

Usage:
  # Auto-discover via default queries
  python scripts/011_harvest_youtube.py --search

  # Custom queries inline
  python scripts/011_harvest_youtube.py --queries "FLS task 5" "laparoscopic suturing"

  # Queries from a file
  python scripts/011_harvest_youtube.py --queries-file queries.txt

  # Download from a list of specific URLs
  python scripts/011_harvest_youtube.py --urls urls.txt

  # Single URL
  python scripts/011_harvest_youtube.py --url "https://www.youtube.com/watch?v=XXXXX"

  # Dry run — show what would be downloaded
  python scripts/011_harvest_youtube.py --search --dry-run

  # Download + ingest into pipeline
  python scripts/011_harvest_youtube.py --search --score

Requirements:
  pip install yt-dlp  (typer and rich already in pyproject.toml)
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table


# ============================================================
# CONFIGURATION
# ============================================================

DOWNLOAD_DIR = Path.home() / "fls_harvested_videos"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
YT_DLP_CMD = [sys.executable, "-m", "yt_dlp"]

# Harvest log — absolute path anchored to repo root
HARVEST_LOG = REPO_ROOT / "harvest_log.jsonl"

DEFAULT_QUERIES: list[str] = [
    "FLS intracorporeal suture",
    "FLS task 5",
    "laparoscopic suturing FLS box",
    "FLS skills test suturing",
    "intracorporeal knot tying FLS trainer",
    "FLS proficiency suture",
]

MIN_DURATION_SEC = 30
MAX_DURATION_SEC = 600     # Hard FLS ceiling — no buffer
MAX_RESOLUTION = "720"
MAX_VIDEOS_PER_QUERY = 20

# Reject videos that are clearly a different FLS task
EXCLUDE_TITLE_KEYWORDS: list[str] = [
    "peg transfer",
    "pattern cutting",
    "pattern cut",
    "clip applying",
    "ligating loop",
    "extracorporeal",
    "task 1", "task 2", "task 3", "task 4",
    "da vinci", "robotic",
    "cholecystectomy", "appendectomy", "hernia repair",
]

# Presence of any of these boosts confidence it IS Task 5 intracorporeal suturing
TASK5_KEYWORDS: list[str] = [
    "intracorporeal", "task 5", "suture", "knot tying",
    "penrose drain", "needle driver", "fls box",
    "surgeon knot", "single throw", "double throw",
]

# Tier A: authoritative expert / professional demonstration
TIER_A_SIGNALS: list[str] = [
    "expert", "attending", "staff surgeon", "faculty", "sages",
    "demonstration", "demo", "tutorial", "board certified", "professor",
]

# Tier B: labeled trainee — identifiable performance context (score / level)
TIER_B_SIGNALS: list[str] = [
    "pgy", "resident", "student", "ms3", "ms4", "intern",
    "beginner", "first attempt", "score", "seconds", "time:",
]


console = Console()
app = typer.Typer(
    name="harvest",
    help="FLS-Training YouTube harvester — discovers, filters, and ingests Task 5 videos.",
    add_completion=False,
)


# ============================================================
# CORE HELPERS
# ============================================================

def log_entry(entry: dict, base_dir: Path = REPO_ROOT) -> None:
    log_path = base_dir / "harvest_log.jsonl"
    with log_path.open("a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def get_already_downloaded(base_dir: Path = REPO_ROOT) -> set[str]:
    log_path = base_dir / "harvest_log.jsonl"
    ids: set[str] = set()
    if not log_path.exists():
        return ids
    for line in log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
            if "youtube_id" in entry:
                ids.add(entry["youtube_id"])
        except json.JSONDecodeError:
            continue
    return ids


def classify_tier(title: str, description: str) -> str:
    """Classify video into tier A / B / C.

    A — expert/attending/SAGES demonstration (authoritative ground truth)
    B — labeled trainee with identifiable performance context
    C — unlabeled (useful but lower confidence)
    """
    text = ((title or "") + " " + (description or "")).lower()
    if any(sig in text for sig in TIER_A_SIGNALS):
        return "A"
    if any(sig in text for sig in TIER_B_SIGNALS):
        return "B"
    return "C"


def extract_time_from_metadata(title: str, description: str) -> Optional[float]:
    """Try to parse a completion time (seconds) from title or description."""
    text = (title or "") + " " + (description or "")
    patterns = [
        r'(\d{2,3})\s*(?:seconds|sec|s)\b',
        r'(?:time|completed?)\s*[:=]?\s*(\d{2,3})\b',
        r'\b(\d{1,2}):(\d{2})\b',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            groups = m.groups()
            if len(groups) == 2 and groups[1]:
                val = int(groups[0]) * 60 + int(groups[1])
            else:
                val = int(groups[0])
            if 30 <= val <= 600:
                return float(val)
    return None


def is_likely_task5(title: str, description: str) -> bool:
    """Return True if the video is plausibly FLS Task 5 intracorporeal suturing."""
    text = ((title or "") + " " + (description or "")).lower()
    # Hard-reject non-Task-5 FLS tasks by title keyword
    for kw in EXCLUDE_TITLE_KEYWORDS:
        if kw in text:
            return False
    # Require at least one positive Task-5 keyword
    return any(kw in text for kw in TASK5_KEYWORDS)


def search_youtube(query: str, max_results: int = MAX_VIDEOS_PER_QUERY) -> list[dict]:
    cmd = [
        *YT_DLP_CMD,
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    videos: list[dict] = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        try:
            info = json.loads(line)
            videos.append({
                "youtube_id": info.get("id", ""),
                "title": info.get("title", ""),
                "url": info.get("url") or f"https://www.youtube.com/watch?v={info.get('id', '')}",
                "duration": info.get("duration"),
                "channel": info.get("channel") or info.get("uploader", ""),
                "description": info.get("description", ""),
                "upload_date": info.get("upload_date", ""),
                "view_count": info.get("view_count", 0),
            })
        except json.JSONDecodeError:
            continue
    return videos


def download_video(url: str, output_dir: Path) -> Optional[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    ffmpeg_available = shutil.which("ffmpeg") is not None
    if ffmpeg_available:
        fmt = (
            f"bestvideo[height<={MAX_RESOLUTION}][vcodec!=none]+"
            f"bestaudio[acodec!=none]/best[height<={MAX_RESOLUTION}][vcodec!=none]"
        )
    else:
        fmt = (
            f"best[ext=mp4][height<={MAX_RESOLUTION}][vcodec!=none][acodec!=none]/"
            f"best[ext=mp4][vcodec!=none][acodec!=none]/"
            f"best[vcodec!=none][acodec!=none]"
        )

    cmd = [
        *YT_DLP_CMD,
        url,
        "--output", output_template,
        "--format", fmt,
        "--write-info-json",
        "--no-playlist",
        "--socket-timeout", "30",
        "--retries", "3",
        "--quiet",
        "--print-json",
    ]
    if ffmpeg_available:
        cmd.extend(["--merge-output-format", "mp4"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            console.print(f"  [red]ERROR[/red] downloading: {result.stderr[:200]}")
            return None

        info = json.loads(result.stdout.strip().split("\n")[-1])

        filepath: Optional[Path] = None
        for ext in ("mp4", "mkv", "webm"):
            candidate = output_dir / f"{info['id']}.{ext}"
            if candidate.exists():
                filepath = candidate
                break
        if not filepath:
            for f in output_dir.glob(f"{info['id']}.*"):
                if f.suffix != ".json":
                    filepath = f
                    break
        if not filepath:
            console.print(f"  [red]ERROR[/red]: can't find downloaded file for {info['id']}")
            return None

        file_hash = hashlib.md5(filepath.read_bytes()[:1024 * 1024]).hexdigest()[:16]
        return {
            "youtube_id": info.get("id", ""),
            "title": info.get("title", ""),
            "channel": info.get("channel") or info.get("uploader", ""),
            "description": info.get("description", "")[:500],
            "upload_date": info.get("upload_date", ""),
            "duration": info.get("duration"),
            "resolution": f"{info.get('width', '?')}x{info.get('height', '?')}",
            "fps": info.get("fps"),
            "filepath": str(filepath),
            "file_hash": file_hash,
            "url": url,
        }
    except subprocess.TimeoutExpired:
        console.print(f"  [red]TIMEOUT[/red] downloading {url}")
        return None
    except (json.JSONDecodeError, KeyError) as exc:
        console.print(f"  [red]ERROR[/red] parsing download result: {exc}")
        return None


def ingest_video(filepath: str, video_id: str, base_dir: Path = REPO_ROOT) -> Optional[dict]:
    """Feed a downloaded video into 010_ingest_video.py → 020_score_frontier.py → 026_auto_validate.py."""
    ingest_script = base_dir / "scripts/010_ingest_video.py"
    score_script = base_dir / "scripts/020_score_frontier.py"
    validate_script = base_dir / "scripts/026_auto_validate.py"

    if not ingest_script.exists():
        console.print(f"  [yellow]SKIP[/yellow] ingest: {ingest_script} not found")
        return None

    console.print(f"  Ingesting [cyan]{video_id}[/cyan]...")
    r = subprocess.run(
        [sys.executable, str(ingest_script),
         "--base-dir", str(base_dir),
         "--video", filepath,
         "--video-id", video_id,
         "--task", "5"],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        console.print(f"  [red]ERROR[/red] ingesting: {r.stderr[:200]}")
        return None

    ingested_id = video_id
    for line in r.stdout.split("\n"):
        if line.startswith("ID:"):
            ingested_id = line.split()[-1]
            break

    if score_script.exists():
        console.print(f"  Scoring [cyan]{ingested_id}[/cyan]...")
        r = subprocess.run(
            [sys.executable, str(score_script),
             "--base-dir", str(base_dir),
             "--video-id", ingested_id,
             "--video", filepath],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            console.print(f"  [red]ERROR[/red] scoring: {r.stderr[:200]}")
            return {"ingested_id": ingested_id, "scored": False}

    if validate_script.exists():
        subprocess.run(
            [sys.executable, str(validate_script),
             "--base-dir", str(base_dir),
             "--video-id", ingested_id],
            capture_output=True, text=True, timeout=60,
        )

    return {"ingested_id": ingested_id, "scored": True}


# ============================================================
# SHARED DOWNLOAD LOOP
# ============================================================

def _download_candidates(
    candidates: list[dict],
    *,
    dry_run: bool,
    score: bool,
    output_dir: Path,
    base_dir: Path,
    queries: Optional[list[str]] = None,
) -> dict:
    stats = {"candidates": len(candidates), "downloaded": 0, "scored": 0, "errors": 0}

    if dry_run:
        table = Table(title=f"Dry-run — {len(candidates)} candidates", show_lines=False)
        table.add_column("#", style="dim", width=4)
        table.add_column("Tier", style="bold", width=4)
        table.add_column("Title", no_wrap=False, max_width=60)
        table.add_column("Duration", justify="right", width=8)
        table.add_column("Channel", max_width=24)
        for i, v in enumerate(candidates, 1):
            dur = v.get("duration")
            dur_str = f"{dur}s" if dur else "?"
            table.add_row(str(i), v.get("tier", "?"), v.get("title", "")[:60], dur_str, v.get("channel", ""))
        console.print(table)
        console.print(f"\n[dim]Stats: {json.dumps(stats)}[/dim]")
        return stats

    already = get_already_downloaded(base_dir)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading…", total=len(candidates))

        for v in candidates:
            yt_id = v["youtube_id"]
            title_short = v.get("title", "")[:50]
            progress.update(task, description=f"[cyan]{title_short}[/cyan]")

            if yt_id in already:
                progress.advance(task)
                continue

            meta = download_video(v["url"], output_dir)
            if meta is None:
                stats["errors"] += 1
                log_entry({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "download_failed",
                    "youtube_id": yt_id,
                    "url": v["url"],
                    "title": v.get("title", ""),
                }, base_dir)
                progress.advance(task)
                continue

            stats["downloaded"] += 1
            already.add(yt_id)

            entry: dict = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "downloaded",
                **meta,
                "tier": v.get("tier", "C"),
                "extracted_time": v.get("extracted_time"),
                "task5_confidence": (
                    "high"
                    if sum(1 for kw in TASK5_KEYWORDS
                           if kw in (v.get("title", "") + " " + v.get("description", "")).lower()) >= 2
                    else "medium"
                ),
            }
            if queries:
                entry["search_queries"] = queries
            log_entry(entry, base_dir)
            progress.advance(task)

            if score:
                vid = f"yt_{yt_id}"
                result = ingest_video(meta["filepath"], vid, base_dir)
                if result and result.get("scored"):
                    stats["scored"] += 1
                    log_entry({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "event": "scored",
                        "youtube_id": yt_id,
                        **result,
                    }, base_dir)

    return stats


# ============================================================
# TYPER CLI
# ============================================================

@app.command()
def main(
    # Mode (mutually exclusive via flag convention)
    search: bool = typer.Option(False, "--search", help="Auto-discover via search queries"),
    url: Optional[str] = typer.Option(None, "--url", help="Download a single YouTube URL"),
    urls: Optional[Path] = typer.Option(None, "--urls", help="File with one YouTube URL per line"),
    # Query options (only relevant with --search)
    queries: Optional[list[str]] = typer.Option(None, "--queries", help="Search query strings (repeatable)"),
    queries_file: Optional[Path] = typer.Option(None, "--queries-file", help="File with one search query per line"),
    # Behaviour flags
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be downloaded, don't download"),
    score: bool = typer.Option(False, "--score", help="Auto-feed valid videos into 010_ingest_video.py"),
    # I/O
    output: Optional[Path] = typer.Option(None, "--output", help="Download directory (default: ~/fls_harvested_videos)"),
    base_dir: Path = typer.Option(REPO_ROOT, "--base-dir", help="Repo root (for harvest_log.jsonl)"),
    category: Optional[str] = typer.Option(None, "--category", help="(ignored — kept for CLI compat)"),
) -> None:
    """FLS-Training YouTube harvester.

    Discover, filter, and optionally ingest Task 5 intracorporeal suturing videos.
    """
    output_dir = (output or DOWNLOAD_DIR).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_flags = [search, url is not None, urls is not None]
    if sum(mode_flags) != 1:
        console.print("[red]Provide exactly one of --search, --url, or --urls[/red]")
        raise typer.Exit(1)

    # ── single URL ──────────────────────────────────────────
    if url is not None:
        yt_match = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
        yt_id = yt_match.group(1) if yt_match else "unknown"

        if dry_run:
            console.print(f"[dim]DRY RUN[/dim] would download: {url}")
            return

        meta = download_video(url, output_dir)
        if meta is None:
            raise typer.Exit(1)

        title = meta.get("title", "")
        desc = meta.get("description", "")
        entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "downloaded",
            **meta,
            "tier": classify_tier(title, desc),
            "extracted_time": extract_time_from_metadata(title, desc),
            "source": "single_url",
        }
        log_entry(entry, base_dir)

        console.print(f"\n[green]Downloaded:[/green] {meta['filepath']}")
        console.print(f"  Duration:   {meta.get('duration', '?')}s")
        console.print(f"  Resolution: {meta.get('resolution', '?')}")
        console.print(f"  Tier:       {entry['tier']}")

        if score:
            ingest_video(meta["filepath"], f"yt_{yt_id}", base_dir)
        return

    # ── URL list ─────────────────────────────────────────────
    if urls is not None:
        raw = urls.read_text().strip().splitlines()
        url_list = [u.strip() for u in raw if u.strip() and not u.startswith("#")]
        console.print(f"[bold]Processing {len(url_list)} URLs from {urls}[/bold]\n")
        already = get_already_downloaded(base_dir)
        candidates: list[dict] = []
        for u in url_list:
            m = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', u)
            yt_id = m.group(1) if m else None
            if yt_id and yt_id in already:
                continue
            candidates.append({
                "youtube_id": yt_id or u,
                "title": u,
                "url": u,
                "duration": None,
                "channel": "",
                "description": "",
                "tier": "C",
                "extracted_time": None,
            })
        stats = _download_candidates(candidates, dry_run=dry_run, score=score,
                                     output_dir=output_dir, base_dir=base_dir)
        console.print(f"\n[bold green]Done.[/bold green] {json.dumps(stats)}")
        return

    # ── search mode ───────────────────────────────────────────
    active_queries: list[str]
    if queries:
        active_queries = list(queries)
    elif queries_file is not None and queries_file.exists():
        active_queries = [
            q.strip() for q in queries_file.read_text().splitlines()
            if q.strip() and not q.startswith("#")
        ]
    elif (Path.cwd() / "queries.txt").exists():
        active_queries = [
            q.strip() for q in (Path.cwd() / "queries.txt").read_text().splitlines()
            if q.strip() and not q.startswith("#")
        ]
    else:
        active_queries = list(DEFAULT_QUERIES)

    already = get_already_downloaded(base_dir)
    console.print(f"[bold]FLS YouTube Harvester[/bold] — {len(active_queries)} queries, "
                  f"{len(already)} already logged\n")

    all_candidates: list[dict] = []
    seen_ids: set[str] = set()
    stats_search = {"filtered_duration": 0, "filtered_task5": 0, "already_had": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching…", total=len(active_queries))
        for q in active_queries:
            progress.update(task, description=f"Searching: [cyan]{q[:60]}[/cyan]")
            videos = search_youtube(q)
            for v in videos:
                yt_id = v["youtube_id"]
                if not yt_id or yt_id in seen_ids:
                    continue
                if yt_id in already:
                    stats_search["already_had"] += 1
                    continue
                dur = v.get("duration") or 0
                if dur < MIN_DURATION_SEC or dur > MAX_DURATION_SEC:
                    stats_search["filtered_duration"] += 1
                    continue
                if not is_likely_task5(v["title"], v["description"]):
                    stats_search["filtered_task5"] += 1
                    continue
                seen_ids.add(yt_id)
                v["tier"] = classify_tier(v["title"], v["description"])
                v["extracted_time"] = extract_time_from_metadata(v["title"], v["description"])
                all_candidates.append(v)
            progress.advance(task)

    console.print(
        f"\nFound [bold]{len(all_candidates)}[/bold] new candidates  "
        f"([dim]filtered: duration={stats_search['filtered_duration']}, "
        f"task5={stats_search['filtered_task5']}, "
        f"already={stats_search['already_had']}[/dim])\n"
    )

    stats = _download_candidates(
        all_candidates, dry_run=dry_run, score=score,
        output_dir=output_dir, base_dir=base_dir, queries=active_queries,
    )
    console.print(f"\n[bold green]Harvest complete.[/bold green] {json.dumps(stats)}")


if __name__ == "__main__":
    app()

