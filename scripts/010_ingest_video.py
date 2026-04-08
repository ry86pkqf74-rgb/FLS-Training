#!/usr/bin/env python3
"""Ingest a video for scoring. Extracts frames and registers in memory."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from src.ingest.frame_extractor import extract_frames, frames_to_base64
from src.scoring.schema import FLSTask, VideoRecord
from src.memory.memory_store import MemoryStore


def main():
    parser = argparse.ArgumentParser(description="Ingest an FLS video")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--video-id", help="Explicit video ID to use instead of deriving from filename")
    parser.add_argument("--task", default="5", help="FLS task number or variant (default: 5)")
    parser.add_argument("--note", default="", help="Recording note")
    parser.add_argument("--base-dir", default=".", help="Repo base directory")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return

    print(f"Extracting frames from {video_path.name}...")
    frames, metadata = extract_frames(video_path)

    video_id = args.video_id or video_path.stem.lower().replace(" ", "_").replace("-", "_")

    task_value = str(args.task)
    if task_value == "5_general":
        task_name = FLSTask.TASK5_INTRACORPOREAL
        note = f"{args.note} [category=intracorporeal_general]".strip()
    else:
        task_name = FLSTask(f"task{task_value}_intracorporeal_suture")
        note = args.note

    record = VideoRecord(
        video_id=video_id,
        filename=video_path.name,
        task=task_name,
        duration_seconds=metadata["duration_seconds"],
        resolution=metadata["resolution"],
        fps=metadata["fps"],
        file_hash=metadata["file_hash"],
        recording_note=note,
        frame_count_extracted=metadata["frame_count"],
    )

    # Save frames as base64 cache for scoring
    frames_dir = Path(args.base_dir) / "memory" / "frames" / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    b64_frames = frames_to_base64(frames)
    frames_cache = {
        "video_id": video_id,
        "frame_count": len(b64_frames),
        "frames_b64": b64_frames,
    }
    (frames_dir / "frames.json").write_text(json.dumps(frames_cache))

    store = MemoryStore(args.base_dir)
    store._append_ledger("video_ingested", record.model_dump(mode="json"))

    print(f"\nIngested: {video_id}")
    print(f"  Duration: {metadata['duration_seconds']}s")
    print(f"  Frames extracted: {metadata['frame_count']}")
    print(f"  Resolution: {metadata['resolution']}")
    print(f"  Hash: {metadata['file_hash']}")
    print(f"\nID: {video_id}")


if __name__ == "__main__":
    main()
