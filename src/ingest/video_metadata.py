"""Video metadata extraction and registration."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from src.ingest.frame_extractor import get_video_info
from src.scoring.schema import FLSTask, VideoMetadata

logger = logging.getLogger(__name__)


def compute_file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file for deduplication."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def ingest_video(video_path: str | Path, task: FLSTask) -> VideoMetadata:
    """Extract metadata from a video and return a VideoMetadata object."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    info = get_video_info(path)
    file_hash = compute_file_hash(path)

    meta = VideoMetadata(
        filename=path.name,
        task=task,
        duration_seconds=info["duration_seconds"],
        resolution=info["resolution"],
        fps=info["fps"],
        file_hash=file_hash,
    )

    logger.info(f"Ingested video {meta.id}: {path.name} ({info['duration_seconds']}s)")
    return meta
