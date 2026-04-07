"""Extract frames from FLS training videos for scoring."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def extract_frames(
    video_path: str | Path,
    n_uniform: int = 20,
    n_final: int = 3,
    output_size: tuple[int, int] = (1280, 720),
) -> tuple[list[Image.Image], dict]:
    """Extract uniform + final-state frames from a video.

    Returns:
        frames: List of PIL Images
        metadata: Dict with duration, fps, resolution, frame_indices, file_hash
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    # Compute file hash
    hasher = hashlib.blake2b(digest_size=8)
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    file_hash = hasher.hexdigest()

    # Calculate frame indices
    uniform_indices = np.linspace(0, total_frames - 1, n_uniform, dtype=int).tolist()

    # Final-state frames: last 10% of video
    final_start = int(total_frames * 0.9)
    final_indices = np.linspace(final_start, total_frames - 1, n_final, dtype=int).tolist()

    all_indices = sorted(set(uniform_indices + final_indices))

    frames = []
    for idx in all_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        if pil_img.size != output_size:
            pil_img = pil_img.resize(output_size, Image.LANCZOS)
        frames.append(pil_img)

    cap.release()

    metadata = {
        "duration_seconds": round(duration, 1),
        "fps": int(fps),
        "resolution": f"{width}x{height}",
        "total_frames": total_frames,
        "extracted_indices": all_indices,
        "frame_count": len(frames),
        "file_hash": file_hash,
    }

    return frames, metadata


def frames_to_base64(frames: list[Image.Image], max_size: int = 1024) -> list[str]:
    """Convert PIL frames to base64 strings for API calls."""
    import base64
    import io

    encoded = []
    for frame in frames:
        # Resize if needed
        if max(frame.size) > max_size:
            ratio = max_size / max(frame.size)
            new_size = (int(frame.width * ratio), int(frame.height * ratio))
            frame = frame.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=85)
        encoded.append(base64.b64encode(buf.getvalue()).decode())

    return encoded
