#!/usr/bin/env python3
"""
FLS-Training: YouTube Video Harvester
======================================
Downloads FLS Task 5 videos from YouTube, extracts metadata,
and feeds them into the scoring pipeline automatically.

Usage:
  # Harvest from search queries (auto-discovers videos)
  python scripts/011_harvest_youtube.py --search

  # Harvest from a file of URLs (one per line)
  python scripts/011_harvest_youtube.py --urls urls.txt

  # Harvest a single video
  python scripts/011_harvest_youtube.py --url "https://www.youtube.com/watch?v=XXXXX"

  # Harvest + immediately score (end-to-end)
  python scripts/011_harvest_youtube.py --urls urls.txt --score

  # Dry run (just show what would be downloaded)
  python scripts/011_harvest_youtube.py --search --dry-run

Requirements:
  pip install yt-dlp
"""

import argparse
import json
import os
import re
import subprocess
import sys
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

# Where downloaded videos go
DOWNLOAD_DIR = Path.home() / "fls_harvested_videos"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
YT_DLP_CMD = [sys.executable, "-m", "yt_dlp"]

# Harvest log (append-only JSONL tracking everything downloaded)
HARVEST_LOG = Path("harvest_log.jsonl")

# Search queries to find FLS Task 5 videos
SEARCH_QUERIES = [
    "FLS intracorporeal suture task 5",
    "FLS intracorporeal knot tying",
    "laparoscopic suturing FLS box trainer",
    "FLS skills test suturing knot",
    "FLS proficiency intracorporeal suture",
    "fundamentals laparoscopic surgery suture task",
    "FLS task 5 practice resident",
    "FLS exam suturing technique",
    "intracorporeal suturing box trainer practice",
    "laparoscopic knot tying penrose drain",
    "FLS suturing PGY resident training",
    "SAGES FLS skills demonstration",
]

# Filters
MIN_DURATION_SEC = 30       # Skip clips shorter than 30s
MAX_DURATION_SEC = 660      # FLS max is 600s, allow 60s buffer for intro/outro
MAX_RESOLUTION = "720"      # Don't waste bandwidth on 4K
MAX_VIDEOS_PER_QUERY = 20   # Cap per search query

# Keywords that indicate NOT Task 5 (skip these)
EXCLUDE_KEYWORDS = [
    "peg transfer", "pattern cut", "ligating loop", "extracorporeal",
    "clip applying", "task 1", "task 2", "task 3", "task 4",
    "da vinci", "robotic", "real surgery", "cholecystectomy",
    "appendectomy", "hernia repair", "operating room",
]

# Keywords that boost confidence this IS Task 5
TASK5_KEYWORDS = [
    "intracorporeal", "task 5", "suture", "knot tying",
    "penrose drain", "needle driver", "FLS box",
    "surgeon knot", "single throw", "double throw",
]

# Skill level signals in title/description
SKILL_SIGNALS = {
    "expert": ["expert", "attending", "staff surgeon", "faculty", "demonstration", "demo", "tutorial"],
    "advanced": ["pgy-4", "pgy-5", "pgy4", "pgy5", "senior resident", "chief resident", "fellow"],
    "intermediate": ["pgy-2", "pgy-3", "pgy2", "pgy3", "resident"],
    "novice": ["pgy-1", "pgy1", "intern", "medical student", "ms3", "ms4", "beginner", "first attempt"],
}


# ============================================================
# CORE FUNCTIONS
# ============================================================

def log_entry(entry: dict):
    """Append an entry to the harvest log."""
    with open(HARVEST_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def get_already_downloaded() -> set:
    """Return set of YouTube video IDs already in the harvest log."""
    ids = set()
    if HARVEST_LOG.exists():
        for line in HARVEST_LOG.read_text().splitlines():
            try:
                entry = json.loads(line)
                if "youtube_id" in entry:
                    ids.add(entry["youtube_id"])
            except json.JSONDecodeError:
                continue
    return ids


def extract_skill_level(title: str, description: str) -> str:
    """Guess skill level from video metadata."""
    text = ((title or "") + " " + (description or "")).lower()
    for level, keywords in SKILL_SIGNALS.items():
        if any(kw in text for kw in keywords):
            return level
    return "unknown"


def extract_time_from_metadata(title: str, description: str) -> float | None:
    """Try to extract a completion time from title or description."""
    text = (title or "") + " " + (description or "")
    # Patterns like "142 seconds", "2:22", "time: 135s"
    patterns = [
        r'(\d{2,3})\s*(?:seconds|sec|s)\b',           # "142 seconds" or "142s"
        r'(?:time|completed?)\s*[:=]?\s*(\d{2,3})\b',  # "time: 142" or "completed 142"
        r'\b(\d{1,2}):(\d{2})\b',                       # "2:22" format
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            groups = m.groups()
            if len(groups) == 2 and groups[1]:  # mm:ss format
                return int(groups[0]) * 60 + int(groups[1])
            elif groups[0]:
                val = int(groups[0])
                if 30 <= val <= 600:  # plausible FLS time
                    return float(val)
    return None


def is_likely_task5(title: str, description: str) -> bool:
    """Check if a video is likely FLS Task 5 based on metadata."""
    text = ((title or "") + " " + (description or "")).lower()

    # Reject if it matches exclusion keywords
    for kw in EXCLUDE_KEYWORDS:
        if kw in text:
            return False

    # Accept if it matches Task 5 keywords
    score = sum(1 for kw in TASK5_KEYWORDS if kw in text)
    return score >= 1  # At least one Task 5 keyword


def search_youtube(query: str, max_results: int = MAX_VIDEOS_PER_QUERY) -> list[dict]:
    """Search YouTube and return video metadata (without downloading)."""
    cmd = [
        *YT_DLP_CMD,
        f"ytsearch{max_results}:{query}",
        "--dump-json",
        "--flat-playlist",
        "--no-download",
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    videos = []
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


def download_video(url: str, output_dir: Path) -> dict | None:
    """Download a single video and return its metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")

    ffmpeg_available = shutil.which("ffmpeg") is not None
    if ffmpeg_available:
        format_selector = (
            f"bestvideo[height<={MAX_RESOLUTION}][vcodec!=none]+"
            f"bestaudio[acodec!=none]/best[height<={MAX_RESOLUTION}][vcodec!=none]"
        )
    else:
        # Without ffmpeg, prefer single-file MP4 video+audio assets that OpenCV can read.
        format_selector = (
            f"best[ext=mp4][height<={MAX_RESOLUTION}][vcodec!=none][acodec!=none]/"
            f"best[ext=mp4][vcodec!=none][acodec!=none]/"
            f"best[vcodec!=none][acodec!=none]"
        )

    cmd = [
        *YT_DLP_CMD,
        url,
        "--output", output_template,
        "--format", format_selector,
        "--write-info-json",          # Save metadata JSON alongside video
        "--no-playlist",              # Single video only
        "--socket-timeout", "30",
        "--retries", "3",
        "--quiet",
        "--print-json",              # Print final metadata as JSON
    ]

    if ffmpeg_available:
        cmd.extend(["--merge-output-format", "mp4"])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ERROR downloading: {result.stderr[:200]}")
            return None

        info = json.loads(result.stdout.strip().split("\n")[-1])

        filepath = None
        for ext in ["mp4", "mkv", "webm"]:
            candidate = output_dir / f"{info['id']}.{ext}"
            if candidate.exists():
                filepath = candidate
                break

        if not filepath:
            # Try to find any file matching the ID
            for f in output_dir.glob(f"{info['id']}.*"):
                if f.suffix != ".json":
                    filepath = f
                    break

        if not filepath:
            print(f"  ERROR: Downloaded but can't find file for {info['id']}")
            return None

        # Compute file hash
        file_hash = hashlib.md5(filepath.read_bytes()[:1024*1024]).hexdigest()[:16]

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
        print(f"  TIMEOUT downloading {url}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  ERROR parsing download result: {e}")
        return None


def score_video(filepath: str, video_id: str) -> dict | None:
    """Run the FLS scoring pipeline on a downloaded video.

    Calls:
      1. scripts/010_ingest_video.py  -- registers the video
      2. scripts/020_score_frontier.py -- dual-teacher scoring
      3. scripts/026_auto_validate.py  -- auto-validation

    Returns scoring summary or None if pipeline not found.
    """
    # Check if pipeline scripts exist
    ingest_script = REPO_ROOT / "scripts/010_ingest_video.py"
    score_script = REPO_ROOT / "scripts/020_score_frontier.py"
    validate_script = REPO_ROOT / "scripts/026_auto_validate.py"

    if not ingest_script.exists():
        print(f"  SKIP scoring: {ingest_script} not found (push framework code first)")
        return None

    # 1. Ingest
    print(f"  Ingesting {video_id}...")
    result = subprocess.run(
            [
                sys.executable,
                str(ingest_script),
                "--base-dir", str(REPO_ROOT),
                "--video", filepath,
                "--video-id", video_id,
                "--task", "5",
            ],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"  ERROR ingesting: {result.stderr[:200]}")
        return None

    # Extract the video ID from ingest output
    ingested_id = video_id
    for line in result.stdout.split("\n"):
        if "ID" in line:
            parts = line.split()
            if len(parts) >= 2:
                ingested_id = parts[-1]
                break

    # 2. Score
    if score_script.exists():
        print(f"  Scoring {ingested_id}...")
        result = subprocess.run(
            [sys.executable, str(score_script), "--base-dir", str(REPO_ROOT), "--video-id", ingested_id, "--video", filepath],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"  ERROR scoring: {result.stderr[:200]}")
            return {"ingested_id": ingested_id, "scored": False, "error": result.stderr[:200]}

    # 3. Auto-validate
    if validate_script.exists():
        print(f"  Validating {ingested_id}...")
        subprocess.run(
            [sys.executable, str(validate_script), "--base-dir", str(REPO_ROOT), "--video-id", ingested_id],
            capture_output=True, text=True, timeout=60
        )

    return {"ingested_id": ingested_id, "scored": True}


# ============================================================
# MAIN WORKFLOWS
# ============================================================

def harvest_from_search(dry_run: bool = False, score: bool = False):
    """Discover and download FLS Task 5 videos from YouTube search."""
    already_downloaded = get_already_downloaded()
    stats = {"searched": 0, "candidates": 0, "filtered_out": 0,
             "already_had": 0, "downloaded": 0, "scored": 0, "errors": 0}

    print(f"=== FLS YouTube Harvester ===")
    print(f"Already have {len(already_downloaded)} videos in harvest log")
    print(f"Running {len(SEARCH_QUERIES)} search queries...\n")

    all_candidates = []

    for query in SEARCH_QUERIES:
        print(f"Searching: '{query}'...")
        videos = search_youtube(query)
        stats["searched"] += 1
        print(f"  Found {len(videos)} results")

        for v in videos:
            yt_id = v["youtube_id"]
            if not yt_id:
                continue

            # Skip already downloaded
            if yt_id in already_downloaded:
                stats["already_had"] += 1
                continue

            # Skip based on duration
            dur = v.get("duration") or 0
            if dur < MIN_DURATION_SEC or dur > MAX_DURATION_SEC:
                stats["filtered_out"] += 1
                continue

            # Skip if not likely Task 5
            if not is_likely_task5(v["title"], v["description"]):
                stats["filtered_out"] += 1
                continue

            # Deduplicate across queries
            if yt_id not in {c["youtube_id"] for c in all_candidates}:
                v["skill_level"] = extract_skill_level(v["title"], v["description"])
                v["extracted_time"] = extract_time_from_metadata(v["title"], v["description"])
                all_candidates.append(v)
                stats["candidates"] += 1

    print(f"\n=== Found {stats['candidates']} new candidates ===\n")

    if dry_run:
        print("DRY RUN — would download these videos:\n")
        for i, v in enumerate(all_candidates, 1):
            skill = v["skill_level"]
            time_str = f" [{v['extracted_time']:.0f}s]" if v["extracted_time"] else ""
            print(f"  {i:3d}. [{skill:12s}] {v['title'][:70]}{time_str}")
            print(f"       {v['url']}  ({v.get('duration', '?')}s, {v['channel']})")
        print(f"\nStats: {json.dumps(stats, indent=2)}")
        return

    # Download each candidate
    for i, v in enumerate(all_candidates, 1):
        yt_id = v["youtube_id"]
        print(f"\n[{i}/{len(all_candidates)}] Downloading: {v['title'][:60]}...")

        meta = download_video(v["url"], DOWNLOAD_DIR)
        if meta is None:
            stats["errors"] += 1
            log_entry({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": "download_failed",
                "youtube_id": yt_id,
                "url": v["url"],
                "title": v["title"],
            })
            continue

        stats["downloaded"] += 1
        already_downloaded.add(yt_id)

        # Merge search metadata with download metadata
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "downloaded",
            **meta,
            "skill_level": v["skill_level"],
            "extracted_time": v["extracted_time"],
            "task5_confidence": "high" if sum(1 for kw in TASK5_KEYWORDS if kw in (v["title"] + " " + v["description"]).lower()) >= 2 else "medium",
        }
        log_entry(entry)
        print(f"  OK: {meta['filepath']} ({meta.get('duration', '?')}s, {meta.get('resolution', '?')})")

        # Score if requested
        if score:
            video_id = f"yt_{yt_id}"
            result = score_video(meta["filepath"], video_id)
            if result and result.get("scored"):
                stats["scored"] += 1
                log_entry({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event": "scored",
                    "youtube_id": yt_id,
                    **result,
                })

    print(f"\n=== Harvest Complete ===")
    print(json.dumps(stats, indent=2))


def harvest_from_urls(url_file: str, score: bool = False):
    """Download specific URLs from a file (one URL per line)."""
    already_downloaded = get_already_downloaded()
    urls = Path(url_file).read_text().strip().splitlines()
    urls = [u.strip() for u in urls if u.strip() and not u.startswith("#")]

    print(f"=== Processing {len(urls)} URLs from {url_file} ===\n")

    stats = {"total": len(urls), "downloaded": 0, "skipped": 0, "errors": 0, "scored": 0}

    for i, url in enumerate(urls, 1):
        # Extract YouTube ID from URL
        yt_id = None
        m = re.search(r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
        if m:
            yt_id = m.group(1)

        if yt_id and yt_id in already_downloaded:
            print(f"[{i}/{len(urls)}] SKIP (already have): {url}")
            stats["skipped"] += 1
            continue

        print(f"[{i}/{len(urls)}] Downloading: {url}")
        meta = download_video(url, DOWNLOAD_DIR)
        if meta is None:
            stats["errors"] += 1
            continue

        stats["downloaded"] += 1
        if yt_id:
            already_downloaded.add(yt_id)

        # Get title/desc for classification
        title = meta.get("title", "")
        desc = meta.get("description", "")

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "downloaded",
            **meta,
            "skill_level": extract_skill_level(title, desc),
            "extracted_time": extract_time_from_metadata(title, desc),
            "source": "url_list",
            "source_file": url_file,
        }
        log_entry(entry)
        print(f"  OK: {meta['filepath']} ({meta.get('duration', '?')}s)")

        if score:
            video_id = f"yt_{meta.get('youtube_id', f'manual_{i}')}"
            result = score_video(meta["filepath"], video_id)
            if result and result.get("scored"):
                stats["scored"] += 1

    print(f"\n=== URL Harvest Complete ===")
    print(json.dumps(stats, indent=2))


def harvest_single(url: str, score: bool = False):
    """Download and optionally score a single video."""
    print(f"Downloading: {url}")
    meta = download_video(url, DOWNLOAD_DIR)
    if meta is None:
        print("Failed to download.")
        sys.exit(1)

    title = meta.get("title", "")
    desc = meta.get("description", "")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "downloaded",
        **meta,
        "skill_level": extract_skill_level(title, desc),
        "extracted_time": extract_time_from_metadata(title, desc),
        "source": "single_url",
    }
    log_entry(entry)

    print(f"\nDownloaded: {meta['filepath']}")
    print(f"  Duration:    {meta.get('duration', '?')}s")
    print(f"  Resolution:  {meta.get('resolution', '?')}")
    print(f"  Channel:     {meta.get('channel', '?')}")
    print(f"  Skill level: {entry['skill_level']}")
    if entry["extracted_time"]:
        print(f"  Extracted time: {entry['extracted_time']:.0f}s")

    if score:
        video_id = f"yt_{meta.get('youtube_id', 'single')}"
        score_video(meta["filepath"], video_id)


# ============================================================
# CLI
# ============================================================

def main():
    global DOWNLOAD_DIR

    download_dir = DOWNLOAD_DIR

    parser = argparse.ArgumentParser(
        description="FLS-Training YouTube Video Harvester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover & download FLS videos from YouTube search
  python scripts/011_harvest_youtube.py --search

  # Preview what would be downloaded (no actual downloads)
  python scripts/011_harvest_youtube.py --search --dry-run

  # Download from a list of specific URLs
  python scripts/011_harvest_youtube.py --urls my_videos.txt

  # Download a single video
  python scripts/011_harvest_youtube.py --url "https://youtube.com/watch?v=..."

  # Download AND score through the full pipeline
  python scripts/011_harvest_youtube.py --urls my_videos.txt --score

  # Change download directory
  python scripts/011_harvest_youtube.py --search --output ~/my_fls_videos
        """
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--search", action="store_true",
                      help="Auto-discover FLS Task 5 videos via YouTube search")
    mode.add_argument("--urls", type=str,
                      help="Path to a text file with one YouTube URL per line")
    mode.add_argument("--url", type=str,
                      help="Single YouTube URL to download")

    parser.add_argument("--score", action="store_true",
                        help="Also run the scoring pipeline on each downloaded video")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be downloaded without actually downloading")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Download directory (default: {DOWNLOAD_DIR})")

    args = parser.parse_args()

    if args.output:
        download_dir = Path(args.output).expanduser()

    download_dir.mkdir(parents=True, exist_ok=True)

    DOWNLOAD_DIR = download_dir

    if args.search:
        harvest_from_search(dry_run=args.dry_run, score=args.score)
    elif args.urls:
        harvest_from_urls(args.urls, score=args.score)
    elif args.url:
        harvest_single(args.url, score=args.score)


if __name__ == "__main__":
    main()
