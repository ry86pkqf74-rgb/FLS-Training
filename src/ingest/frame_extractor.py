"""Extract frames from FLS training videos for VLM analysis.

Strategies:
  - uniform: evenly-spaced frames across video duration
  - motion: select frames at high-motion moments (optical flow peaks)
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


def extract_motion_keyframes(
    video_path: str | Path,
    n_frames: int = 20,
    sample_every_n: int = 5,
    max_dimension: int = 1280,
    quality: int = 85,
) -> list[ExtractedFrame]:
    """Extract frames at high-motion moments using optical flow magnitude.

    Computes dense optical flow between sampled frame pairs, ranks by motion
    magnitude, and selects the top-N most informative moments. This captures
    phase transitions (needle load → placement, hand switches, cutting) that
    uniform sampling may miss.

    Args:
        video_path: Path to video file.
        n_frames: Number of keyframes to return.
        sample_every_n: Compute flow every N frames (lower = more precise, slower).
        max_dimension: Resize frames for API payload.
        quality: JPEG quality.

    Returns:
        List of ExtractedFrame sorted by timestamp.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < n_frames * 2:
        logger.info("Video too short for motion analysis, falling back to uniform")
        cap.release()
        return extract_uniform_frames(video_path, n_frames, max_dimension, quality)

    # Pass 1: compute optical flow magnitude at sampled intervals
    flow_scores: list[tuple[int, float]] = []  # (frame_idx, motion_magnitude)
    prev_gray = None

    sample_indices = list(range(0, total_frames, sample_every_n))
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Downscale for fast flow computation
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=1, winsize=15,
                iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_scores.append((idx, float(np.mean(magnitude))))

        prev_gray = gray

    if not flow_scores:
        logger.warning("No flow computed, falling back to uniform")
        cap.release()
        return extract_uniform_frames(video_path, n_frames, max_dimension, quality)

    # Pass 2: select top-N by motion, but ensure temporal spread
    # Sort by motion magnitude (descending)
    flow_scores.sort(key=lambda x: x[1], reverse=True)

    # Greedily pick frames that are at least min_gap apart
    min_gap = max(total_frames // (n_frames * 2), sample_every_n)
    selected_indices: list[int] = []
    for idx, score in flow_scores:
        if len(selected_indices) >= n_frames:
            break
        if all(abs(idx - s) >= min_gap for s in selected_indices):
            selected_indices.append(idx)

    # If we didn't get enough (e.g., very static video), fill with uniform
    if len(selected_indices) < n_frames:
        uniform_fill = [
            int(i * total_frames / n_frames)
            for i in range(n_frames)
        ]
        for idx in uniform_fill:
            if len(selected_indices) >= n_frames:
                break
            if all(abs(idx - s) >= min_gap for s in selected_indices):
                selected_indices.append(idx)

    selected_indices.sort()  # chronological order

    # Pass 3: extract the actual frames at full quality
    frames: list[ExtractedFrame] = []
    for idx in selected_indices:
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
    logger.info(
        f"Extracted {len(frames)} motion keyframes from {video_path} "
        f"(analyzed {len(flow_scores)} flow samples)"
    )
    return frames


def extract_all_frames_for_scoring(
    video_path: str | Path,
    n_uniform: int = 20,
    n_final: int = 3,
    max_dimension: int = 1280,
    strategy: str = "uniform",
) -> tuple[list[ExtractedFrame], list[ExtractedFrame]]:
    """Extract both sample frames and final-state frames.

    Args:
        strategy: "uniform" (default, fast) or "motion" (optical-flow keyframes).

    Returns (sample_frames, final_frames).
    """
    if strategy == "motion":
        sample = extract_motion_keyframes(video_path, n_uniform, max_dimension=max_dimension)
    else:
        sample = extract_uniform_frames(video_path, n_uniform, max_dimension)
    final = extract_final_state_frames(video_path, n_final, max_dimension)
    return sample, final


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
