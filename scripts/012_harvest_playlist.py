#!/usr/bin/env python3
"""
FLS-Training: Playlist Harvester & Classifier
===============================================
Downloads an entire YouTube playlist, then classifies each video as:
  - fls_task5:     Standard FLS setup (penrose drain, foam block, FLS box)
  - intracorporeal_general: Intracorporeal knot tying but NOT standard FLS setup
  - non_relevant:  Not intracorporeal knot tying at all

The classification uses metadata first (title/description keywords),
then optionally uses the VLM (Claude) to classify from a single frame
for ambiguous cases.

Usage:
  # Step 1: Download entire playlist
  python scripts/012_harvest_playlist.py --download \
    "https://www.youtube.com/playlist?list=PLHU3jJlMwNt1oVDG87l3i3n76-zTZ6jbh"

  # Step 2: Classify all downloaded videos (metadata-only, fast)
  python scripts/012_harvest_playlist.py --classify

  # Step 3: Classify ambiguous ones with VLM (costs API calls)
  python scripts/012_harvest_playlist.py --classify --use-vlm

  # Step 4: Score only the FLS-style ones through the full pipeline
  python scripts/012_harvest_playlist.py --score --category fls_task5

  # Step 5: Score the general intracorporeal ones with adapted prompt
  python scripts/012_harvest_playlist.py --score --category intracorporeal_general

  # Or do everything at once:
  python scripts/012_harvest_playlist.py --download --classify --score \
    "https://www.youtube.com/playlist?list=PLHU3jJlMwNt1oVDG87l3i3n76-zTZ6jbh"
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DOWNLOAD_DIR = Path.home() / "fls_harvested_videos"
CLASSIFICATION_LOG = Path("playlist_classification.jsonl")

# Keywords for metadata-based classification
FLS_KEYWORDS = [
    "fls", "fundamentals of laparoscopic", "penrose drain",
    "foam block", "suture block", "fls box", "fls trainer",
    "task 5", "task five", "fls exam", "fls skills",
    "fls proficiency", "fls test",
]

INTRACORPOREAL_KEYWORDS = [
    "intracorporeal", "knot tying", "knot-tying", "suturing",
    "laparoscopic suture", "laparoscopic knot",
    "instrument tie", "surgeon knot", "square knot",
    "box trainer", "sim lab", "simulation",
    "needle driver", "needle holder",
]

EXCLUDE_KEYWORDS = [
    "extracorporeal", "da vinci", "robotic", "real surgery",
    "cholecystectomy", "appendectomy", "operating room",
    "hernia", "patient", "open surgery",
]


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_from_metadata(title: str, description: str) -> dict:
    """Classify a video based on title and description keywords.

    Returns:
        {
            "category": "fls_task5" | "intracorporeal_general" | "non_relevant" | "ambiguous",
            "confidence": "high" | "medium" | "low",
            "signals": [list of matched keywords]
        }
    """
    text = (title + " " + description).lower()

    # Check exclusions first
    exclude_hits = [kw for kw in EXCLUDE_KEYWORDS if kw in text]
    if len(exclude_hits) >= 2:
        return {
            "category": "non_relevant",
            "confidence": "high",
            "signals": exclude_hits,
            "reason": "Multiple exclusion keywords matched",
        }

    # Check FLS-specific keywords
    fls_hits = [kw for kw in FLS_KEYWORDS if kw in text]
    ic_hits = [kw for kw in INTRACORPOREAL_KEYWORDS if kw in text]

    if len(fls_hits) >= 2:
        return {
            "category": "fls_task5",
            "confidence": "high",
            "signals": fls_hits,
            "reason": "Strong FLS keyword match",
        }
    elif len(fls_hits) == 1 and len(ic_hits) >= 1:
        return {
            "category": "fls_task5",
            "confidence": "medium",
            "signals": fls_hits + ic_hits,
            "reason": "FLS keyword + intracorporeal keyword",
        }
    elif len(ic_hits) >= 2:
        return {
            "category": "intracorporeal_general",
            "confidence": "high" if len(ic_hits) >= 3 else "medium",
            "signals": ic_hits,
            "reason": "Intracorporeal keywords, no FLS-specific signals",
        }
    elif len(ic_hits) == 1:
        return {
            "category": "ambiguous",
            "confidence": "low",
            "signals": ic_hits,
            "reason": "Only one intracorporeal keyword — needs VLM or manual check",
        }
    else:
        return {
            "category": "non_relevant",
            "confidence": "medium",
            "signals": [],
            "reason": "No relevant keywords found",
        }


def classify_with_vlm(video_path: str) -> dict:
    """Use Claude to classify a video from its first frame.

    Extracts 1 frame at 25% through the video and asks Claude
    to identify if it's FLS-style or general intracorporeal.
    """
    import tempfile

    # Extract a single frame at 25% through the video
    # First get duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    try:
        duration = float(result.stdout.strip())
        timestamp = duration * 0.25
    except (ValueError, AttributeError):
        timestamp = 10.0  # fallback

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        frame_path = tmp.name

    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
         "-vframes", "1", "-q:v", "2", frame_path],
        capture_output=True, timeout=30
    )

    if not os.path.exists(frame_path) or os.path.getsize(frame_path) == 0:
        os.unlink(frame_path)
        return {
            "category": "ambiguous",
            "confidence": "low",
            "reason": "Could not extract frame for VLM classification",
        }

    # Call Claude with the frame
    # (This would use the Anthropic API — placeholder for actual implementation)
    # The prompt would be:
    vlm_prompt = """Look at this single frame from a laparoscopic training video.
Classify it as one of:

1. "fls_task5" — Standard FLS (Fundamentals of Laparoscopic Surgery) setup:
   - FLS trainer box visible
   - Penrose drain on a foam/suture block
   - Marked targets on the drain
   - Standard FLS instruments (needle drivers, scissors)

2. "intracorporeal_general" — Intracorporeal knot tying but NOT standard FLS:
   - Box trainer or sim lab but different setup
   - Different tissue model (not penrose drain on foam block)
   - Real tissue (cadaver/animal lab)
   - Different instruments or technique demonstration

3. "non_relevant" — Not intracorporeal knot tying at all

Respond with ONLY a JSON object:
{"category": "fls_task5|intracorporeal_general|non_relevant", "confidence": "high|medium|low", "reason": "brief explanation"}
"""

    print(f"  [VLM classification would go here — frame saved at {frame_path}]")
    print(f"  [Prompt: classify frame as fls_task5 / intracorporeal_general / non_relevant]")

    os.unlink(frame_path)

    # Placeholder return — in production, parse Claude's response
    return {
        "category": "ambiguous",
        "confidence": "low",
        "reason": "VLM classification not yet implemented — implement with Anthropic API call",
    }


# ============================================================
# DOWNLOAD
# ============================================================

def download_playlist(playlist_url: str):
    """Download all videos from a YouTube playlist."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Downloading playlist to {DOWNLOAD_DIR} ===")
    print(f"URL: {playlist_url}\n")

    cmd = [
        "yt-dlp",
        playlist_url,
        "--output", str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "--format", "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "--merge-output-format", "mp4",
        "--write-info-json",       # Metadata sidecar for each video
        "--no-overwrites",         # Skip already-downloaded
        "--socket-timeout", "30",
        "--retries", "3",
        "--progress",
    ]

    print("Running yt-dlp...\n")
    result = subprocess.run(cmd, timeout=3600)  # 1 hour max

    if result.returncode != 0:
        print(f"\nWARNING: yt-dlp exited with code {result.returncode}")
        print("Some videos may have failed — check output above.")
    else:
        print("\nDownload complete.")

    # Count what we got
    mp4s = list(DOWNLOAD_DIR.glob("*.mp4"))
    jsons = list(DOWNLOAD_DIR.glob("*.info.json"))
    print(f"\nFiles in {DOWNLOAD_DIR}:")
    print(f"  Videos: {len(mp4s)}")
    print(f"  Metadata files: {len(jsons)}")


def classify_all(use_vlm: bool = False):
    """Classify all downloaded videos and write results."""
    print(f"=== Classifying videos in {DOWNLOAD_DIR} ===\n")

    # Find all info.json files (written by yt-dlp)
    json_files = sorted(DOWNLOAD_DIR.glob("*.info.json"))

    if not json_files:
        print("No .info.json files found. Run --download first.")
        return

    results = {"fls_task5": [], "intracorporeal_general": [], "ambiguous": [], "non_relevant": []}

    for jf in json_files:
        try:
            info = json.loads(jf.read_text())
        except json.JSONDecodeError:
            continue

        yt_id = info.get("id", jf.stem.replace(".info", ""))
        title = info.get("title", "")
        desc = info.get("description", "")
        duration = info.get("duration", 0)

        # Find the corresponding video file
        video_path = None
        for ext in ["mp4", "mkv", "webm"]:
            candidate = DOWNLOAD_DIR / f"{yt_id}.{ext}"
            if candidate.exists():
                video_path = str(candidate)
                break

        # Classify from metadata
        classification = classify_from_metadata(title, desc)

        # If ambiguous and VLM requested, try VLM
        if classification["category"] == "ambiguous" and use_vlm and video_path:
            print(f"  VLM classifying: {title[:60]}...")
            vlm_result = classify_with_vlm(video_path)
            if vlm_result["category"] != "ambiguous":
                classification = vlm_result
                classification["method"] = "vlm"
            else:
                classification["method"] = "metadata_ambiguous"
        else:
            classification["method"] = "metadata"

        entry = {
            "youtube_id": yt_id,
            "title": title,
            "channel": info.get("channel") or info.get("uploader", ""),
            "duration_seconds": duration,
            "video_path": video_path,
            **classification,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        results[classification["category"]].append(entry)

        # Log
        with open(CLASSIFICATION_LOG, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        cat = classification["category"]
        conf = classification["confidence"]
        print(f"  [{cat:25s}] ({conf:6s}) {title[:65]}")

    print(f"\n=== Classification Summary ===")
    print(f"  FLS Task 5:              {len(results['fls_task5'])}")
    print(f"  Intracorporeal General:  {len(results['intracorporeal_general'])}")
    print(f"  Ambiguous:               {len(results['ambiguous'])}")
    print(f"  Non-relevant:            {len(results['non_relevant'])}")
    print(f"  TOTAL:                   {sum(len(v) for v in results.values())}")

    if results["ambiguous"]:
        print(f"\n  Ambiguous videos (review manually or re-run with --use-vlm):")
        for v in results["ambiguous"]:
            print(f"    - {v['title'][:70]}")
            print(f"      https://youtube.com/watch?v={v['youtube_id']}")


def score_category(category: str):
    """Score all videos of a given category through the pipeline."""
    if not CLASSIFICATION_LOG.exists():
        print("No classification log found. Run --classify first.")
        return

    entries = []
    for line in CLASSIFICATION_LOG.read_text().splitlines():
        try:
            entry = json.loads(line)
            if entry.get("category") == category:
                entries.append(entry)
        except json.JSONDecodeError:
            continue

    # Deduplicate by youtube_id (take latest classification)
    by_id = {}
    for e in entries:
        by_id[e["youtube_id"]] = e
    entries = list(by_id.values())

    print(f"=== Scoring {len(entries)} '{category}' videos ===\n")

    if category == "fls_task5":
        task_arg = "5"
        print("Using standard FLS Task 5 scoring prompt.\n")
    elif category == "intracorporeal_general":
        task_arg = "5_general"
        print("Using adapted intracorporeal scoring prompt.")
        print("NOTE: These videos won't have penrose drain / marked targets.")
        print("Scoring focuses on: knot quality, hand technique, economy of motion.\n")
    else:
        print(f"Cannot score category '{category}'. Use 'fls_task5' or 'intracorporeal_general'.")
        return

    ingest_script = Path("scripts/010_ingest_video.py")
    score_script = Path("scripts/020_score_frontier.py")

    if not ingest_script.exists():
        print(f"Pipeline not found at {ingest_script}. Push framework code to repo first.")
        return

    scored = 0
    errors = 0
    for i, entry in enumerate(entries, 1):
        vpath = entry.get("video_path")
        if not vpath or not Path(vpath).exists():
            print(f"[{i}/{len(entries)}] SKIP (file not found): {entry['title'][:50]}")
            errors += 1
            continue

        yt_id = entry["youtube_id"]
        video_id = f"yt_{yt_id}"
        print(f"[{i}/{len(entries)}] {entry['title'][:60]}")

        # Ingest
        result = subprocess.run(
            ["python", str(ingest_script), "--video", vpath, "--task", task_arg],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print(f"  ERROR ingesting: {result.stderr[:100]}")
            errors += 1
            continue

        # Score
        if score_script.exists():
            result = subprocess.run(
                ["python", str(score_script), "--video-id", video_id, "--video", vpath],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                print(f"  ERROR scoring: {result.stderr[:100]}")
                errors += 1
                continue

        scored += 1
        print(f"  OK (scored)")

    print(f"\n=== Scoring Complete ===")
    print(f"  Scored: {scored}")
    print(f"  Errors: {errors}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FLS-Training Playlist Harvester & Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. --download <playlist_url>     Download all videos
  2. --classify                     Classify each as FLS / general / non-relevant
  3. --classify --use-vlm           Re-classify ambiguous ones with Claude
  4. --score --category fls_task5   Score the FLS-style videos
  5. --score --category intracorporeal_general   Score the rest
        """
    )

    parser.add_argument("--download", action="store_true",
                        help="Download all videos from the playlist")
    parser.add_argument("--classify", action="store_true",
                        help="Classify downloaded videos by type")
    parser.add_argument("--score", action="store_true",
                        help="Score videos of a given category")
    parser.add_argument("--category", type=str, default="fls_task5",
                        choices=["fls_task5", "intracorporeal_general"],
                        help="Which category to score (default: fls_task5)")
    parser.add_argument("--use-vlm", action="store_true",
                        help="Use Claude VLM to classify ambiguous videos")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Download directory (default: {DOWNLOAD_DIR})")
    parser.add_argument("url", nargs="?", default=None,
                        help="Playlist URL (required for --download)")

    args = parser.parse_args()

    if args.output:
        global DOWNLOAD_DIR
        DOWNLOAD_DIR = Path(args.output)

    if args.download:
        if not args.url:
            print("ERROR: Provide a playlist URL for --download")
            sys.exit(1)
        download_playlist(args.url)

    if args.classify:
        classify_all(use_vlm=args.use_vlm)

    if args.score:
        score_category(args.category)

    if not any([args.download, args.classify, args.score]):
        parser.print_help()


if __name__ == "__main__":
    main()
