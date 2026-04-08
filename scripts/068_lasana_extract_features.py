#!/usr/bin/env python3
"""
LASANA frame extraction + DINOv2 feature caching.

Two modes (run separately, on different boxes):

  --frames-only      CPU pod or laptop. Decode HEVC stereo bitstreams to
                     1 fps left-channel JPEGs. Outputs ~8 GB.

  --features-only    Small GPU pod (L40S / A10 / T4). Run DINOv2-base over
                     extracted frames in batched mode. Outputs ~1.5 GB of
                     .npy files (one per video_id, shape [N, 768]).

  --all              Run both phases sequentially. Use only on a single
                     box that has both ffmpeg and a GPU. Not recommended
                     for cost reasons — the CPU phase wastes GPU time.

Layout assumed:
    ${LASANA_DIR}/<video_id>/video.hevc                  ← input (W6 layout)
    ${LASANA_DIR}/<task>/<trial>.hevc                    ← input (legacy tolerated)
    ${OUT_DIR}/frames/<video_id>/frame_NNNN.jpg          ← phase 1 output
    ${OUT_DIR}/features/<video_id>.npy                   ← phase 2 output
  ${OUT_DIR}/manifest.csv                              ← bookkeeping

Usage:
  # Phase 1 on CPU pod
  python scripts/068_lasana_extract_features.py --frames-only \
      --lasana-dir /workspace/lasana \
      --out-dir   /workspace/lasana_processed \
      --fps 1

  # Phase 2 on small GPU
  python scripts/068_lasana_extract_features.py --features-only \
      --out-dir /workspace/lasana_processed \
      --batch-size 64

  # Test on 2 trials end-to-end (laptop)
  python scripts/068_lasana_extract_features.py --all \
      --lasana-dir ./data/external/lasana \
      --out-dir   /tmp/lasana_test \
      --max-trials 2
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path

LOG = logging.getLogger("lasana_extract")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Phase 1: HEVC stereo → 1 fps left-channel JPEGs (CPU)
# ---------------------------------------------------------------------------

def find_bitstreams(lasana_dir: Path) -> list[Path]:
    """Walk LASANA_DIR for .h265 / .hevc files. Returns list of input paths."""
    exts = (".h265", ".hevc", ".mp4", ".mkv")
    out: list[Path] = []
    for ext in exts:
        out.extend(sorted(lasana_dir.rglob(f"*{ext}")))
    return out


def video_id_from_relpath(rel: Path) -> str:
    """Resolve a stable video_id from a bitstream path relative to LASANA_DIR."""
    stem = rel.stem.lower()
    if stem == "video" and rel.parent != Path("."):
        return rel.parent.name
    return "_".join(rel.with_suffix("").parts)


def decode_one_trial(
    bitstream: Path,
    out_dir: Path,
    fps: float,
    left_only: bool = True,
) -> tuple[int, int]:
    """
    Decode one LASANA trial to JPEG frames using ffmpeg.

    Returns (n_frames, exit_code). LASANA stereo is side-by-side; if
    left_only=True we crop to the left half via ffmpeg crop filter.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%04d.jpg")

    vf_parts = [f"fps={fps}"]
    if left_only:
        # Side-by-side stereo: keep left half. crop=w:h:x:y, w=iw/2.
        vf_parts.append("crop=in_w/2:in_h:0:0")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(bitstream),
        "-vf", vf,
        "-q:v", "3",         # JPEG quality (1=best, 31=worst); 3 is high
        "-f", "image2",
        pattern,
    ]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        LOG.warning("ffmpeg failed for %s: %s", bitstream.name, res.stderr.decode()[:200])
        return (0, res.returncode)

    n = len(list(out_dir.glob("frame_*.jpg")))
    return (n, 0)


def phase1_frames(args) -> Path:
    """Phase 1: decode all LASANA bitstreams to JPEGs. Returns manifest path."""
    lasana_dir = Path(args.lasana_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_root = out_dir / "frames"
    manifest_path = out_dir / "manifest.csv"

    if not lasana_dir.exists():
        LOG.error("LASANA dir %s does not exist", lasana_dir)
        sys.exit(2)

    bitstreams = find_bitstreams(lasana_dir)
    if args.max_trials:
        bitstreams = bitstreams[: args.max_trials]
    LOG.info("Found %d bitstreams under %s", len(bitstreams), lasana_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, bs in enumerate(bitstreams, 1):
        rel = bs.relative_to(lasana_dir)
        video_id = video_id_from_relpath(rel)
        out_subdir = frames_root / video_id
        if out_subdir.exists() and any(out_subdir.glob("frame_*.jpg")):
            n = len(list(out_subdir.glob("frame_*.jpg")))
            LOG.info("[%d/%d] %s SKIP (already %d frames)", i, len(bitstreams), video_id, n)
        else:
            n, rc = decode_one_trial(bs, out_subdir, fps=args.fps, left_only=True)
            LOG.info("[%d/%d] %s -> %d frames (rc=%d)", i, len(bitstreams), video_id, n, rc)
        rows.append({
            "video_id": video_id,
            "source": "lasana",
            "bitstream": str(rel),
            "n_frames": n,
            "frames_dir": str(out_subdir.relative_to(out_dir)),
        })

    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["video_id", "source", "bitstream", "n_frames", "frames_dir"])
        w.writeheader()
        w.writerows(rows)
    LOG.info("Wrote manifest with %d rows -> %s", len(rows), manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Phase 2: JPEGs → DINOv2 features (small GPU)
# ---------------------------------------------------------------------------

def phase2_features(args) -> None:
    """Phase 2: run DINOv2-base over all frames, write one .npy per video_id."""
    out_dir = Path(args.out_dir).expanduser().resolve()
    frames_root = out_dir / "frames"
    feats_root = out_dir / "features"
    manifest_path = out_dir / "manifest.csv"

    if not frames_root.exists():
        LOG.error("frames dir %s missing — run --frames-only first", frames_root)
        sys.exit(2)
    if not manifest_path.exists():
        LOG.error("manifest %s missing — run --frames-only first", manifest_path)
        sys.exit(2)

    # Imports here so phase 1 doesn't need torch installed
    try:
        import numpy as np
        import torch
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except ImportError as e:
        LOG.error("Phase 2 needs torch + torchvision + numpy + Pillow: %s", e)
        sys.exit(2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Device: %s", device)
    if device.type == "cpu":
        LOG.warning("Running DINOv2 on CPU — this will be SLOW. Use a GPU pod.")

    LOG.info("Loading DINOv2-base from torch hub (downloads ~340 MB on first run)...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
    model = model.to(device).eval()

    # DINOv2 expects ImageNet-normalized 224x224 (or any multiple of 14)
    tfm = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    class FrameDir(Dataset):
        def __init__(self, paths: list[Path]):
            self.paths = paths

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, i: int):
            img = Image.open(self.paths[i]).convert("RGB")
            return tfm(img)

    feats_root.mkdir(parents=True, exist_ok=True)

    with manifest_path.open() as f:
        rows = list(csv.DictReader(f))
    if args.max_trials:
        rows = rows[: args.max_trials]

    for i, row in enumerate(rows, 1):
        video_id = row["video_id"]
        frames_dir = out_dir / row["frames_dir"]
        out_path = feats_root / f"{video_id}.npy"
        if out_path.exists() and not args.overwrite:
            LOG.info("[%d/%d] %s SKIP (cached)", i, len(rows), video_id)
            continue

        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        if not frame_paths:
            LOG.warning("[%d/%d] %s no frames", i, len(rows), video_id)
            continue

        ds = FrameDir(frame_paths)
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        feats: list = []
        with torch.inference_mode():
            for batch in dl:
                batch = batch.to(device, non_blocking=True)
                # DINOv2 .forward returns the CLS token embedding (768-d for vit_b14)
                out = model(batch)
                feats.append(out.cpu().numpy())

        import numpy as np  # already imported above; redundant for type hints
        arr = np.concatenate(feats, axis=0)
        np.save(out_path, arr)
        LOG.info("[%d/%d] %s -> %s shape=%s", i, len(rows), video_id, out_path.name, arr.shape)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LASANA frame + feature extraction")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--frames-only", action="store_true",
                      help="Phase 1: HEVC → JPEG frames (CPU)")
    mode.add_argument("--features-only", action="store_true",
                      help="Phase 2: JPEG → DINOv2 features (small GPU)")
    mode.add_argument("--all", action="store_true",
                      help="Run both phases (testing only — wastes GPU time at scale)")

    p.add_argument("--lasana-dir", default="./data/external/lasana",
                   help="Root directory containing LASANA HEVC bitstreams")
    p.add_argument("--out-dir", default="./data/external/lasana_processed",
                   help="Where to write frames/, features/, manifest.csv")
    p.add_argument("--fps", type=float, default=1.0,
                   help="Frame extraction rate (default 1 fps)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Phase 2 GPU batch size")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Phase 2 dataloader workers")
    p.add_argument("--max-trials", type=int, default=0,
                   help="Limit to first N trials (for testing). 0 = all.")
    p.add_argument("--overwrite", action="store_true",
                   help="Phase 2: re-compute features even if .npy exists")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames_only or args.all:
        phase1_frames(args)
    if args.features_only or args.all:
        phase2_features(args)


if __name__ == "__main__":
    main()
