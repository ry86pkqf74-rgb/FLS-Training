"""Extract frames from FLS training videos for VLM analysis.

Strategies:
  - uniform: evenly-spaced frames across video duration
  - final_state: last N frames showing completed result
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    frame_number: int
    timestamp_seconds: float
    image_bytes: bytes = field(repr=False)

    @property
    def image_b64(self) -> str:
        return base64.b64encode(self.image_bytes).decode("utf-8")


def get_video_info(video_path: str | Path) -> dict:
    """Return video metadata: duration, fps, resolution, total_frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    return {
        "duration_seconds": round(duration, 2),
        "fps": round(fps, 2),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "total_frames": total_frames,
    }


def _frame_to_jpeg(
    frame_bgr: np.ndarray,
    max_dimension: int = 1280,
    quality: int = 85,
) -> bytes:
    """Convert a BGR OpenCV frame to JPEG bytes, resizing if needed."""
    h, w = frame_bgr.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        frame_bgr = cv2.resize(
            frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def extract_uniform_frames(
    video_path: str | Path,
    n_frames: int = 20,
    max_dimension: int = 1280,
    quality: int = 85,
) -> list[ExtractedFrame]:
    """Extract n evenly-spaced frames across the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if n_frames >= total_frames:
        indices = list(range(total_frames))
    else:
        indices = [int(i * total_frames / n_frames) for i in range(n_frames)]

    frames: list[ExtractedFrame] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp = idx / fps if fps > 0 else 0.0
        jpeg_bytes = _frame_to_jpeg(frame, max_dimension, quality)
        frames.append(ExtractedFrame(
            frame_number=len(frames) + 1,
            timestamp_seconds=round(timestamp, 2),
            image_bytes=jpeg_bytes,
        ))

    cap.release()
    logger.info(f"Extracted {len(frames)} uniform frames from {video_path}")
    return frames


def extract_final_state_frames(
    video_path: str | Path,
    n_frames: int = 3,
    max_dimension: int = 1280,
    quality: int = 85,
) -> list[ExtractedFrame]:
    """Extract the last N frames showing the completed knot/suture."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_idx = max(0, total_frames - n_frames)
    indices = list(range(start_idx, total_frames))

    frames: list[ExtractedFrame] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        timestamp = idx / fps if fps > 0 else 0.0
        jpeg_bytes = _frame_to_jpeg(frame, max_dimension, quality)
        frames.append(ExtractedFrame(
            frame_number=len(frames) + 1,
            timestamp_seconds=round(timestamp, 2),
            image_bytes=jpeg_bytes,
        ))

    cap.release()
    logger.info(f"Extracted {len(frames)} final-state frames from {video_path}")
    return frames


def extract_all_frames_for_scoring(
    video_path: str | Path,
    n_uniform: int = 20,
    n_final: int = 3,
    max_dimension: int = 1280,
) -> tuple[list[ExtractedFrame], list[ExtractedFrame]]:
    """Extract both uniform sample frames and final-state frames.

    Returns (uniform_frames, final_frames).
    """
    uniform = extract_uniform_frames(video_path, n_uniform, max_dimension)
    final = extract_final_state_frames(video_path, n_final, max_dimension)
    return uniform, final


def frames_to_b64_list(frames: list[ExtractedFrame]) -> list[str]:
    """Convert frames to base64 strings for API payloads."""
    return [f.image_b64 for f in frames]


def frames_timestamps(frames: list[ExtractedFrame]) -> list[float]:
    """Get timestamp list for prompt template."""
    return [f.timestamp_seconds for f in frames]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingest.frame_extractor <video_path> [n_frames]")
        sys.exit(1)

    path = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    info = get_video_info(path)
    print(f"Video info: {info}")

    uniform, final = extract_all_frames_for_scoring(path, n_uniform=n)
    print(f"Extracted {len(uniform)} uniform + {len(final)} final frames")
    print(f"Timestamps: {frames_timestamps(uniform + final)}")
