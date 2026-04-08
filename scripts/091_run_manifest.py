#!/usr/bin/env python3
"""Generate a run manifest at training launch time.

Records commit SHA, GPU, dataset version, config hash, vision mode, and
other reproducibility metadata into memory/training_runs/.

Called automatically by deploy/runpod_launch.sh. Can also be invoked
manually for offline inspection.

Usage:
    python scripts/091_run_manifest.py \
        --config /tmp/finetune_runtime.yaml \
        --dataset-path data/training/2026-04-07_v2
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_REPO_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
        )
        return result.returncode != 0
    except Exception:
        return True


def _gpu_info() -> str:
    try:
        return (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "none"


def _file_hash(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _dataset_stats(dataset_path: Path) -> dict:
    stats: dict = {}
    train = dataset_path / "train.jsonl"
    val = dataset_path / "val.jsonl"
    if train.is_file():
        stats["train_examples"] = sum(1 for _ in train.open())
    if val.is_file():
        stats["val_examples"] = sum(1 for _ in val.open())
    manifest = dataset_path / "manifest.yaml"
    if manifest.is_file():
        stats["manifest_hash"] = _file_hash(manifest)
    return stats


def generate_manifest(
    config_path: str | Path,
    dataset_path: str | Path,
    extra: dict | None = None,
) -> dict:
    config_path = Path(config_path)
    dataset_path = Path(dataset_path)

    # Load config for metadata extraction
    config: dict = {}
    if config_path.is_file():
        import yaml

        config = yaml.safe_load(config_path.read_text()) or {}

    manifest = {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "gpu": _gpu_info(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "config_path": str(config_path),
        "config_hash": _file_hash(config_path),
        "config_family": config.get("config_family", "unknown"),
        "base_model": config.get("base_model", "unknown"),
        "framework": config.get("framework", "unknown"),
        "vision_mode": config.get("require_vision", False),
        "quantization": config.get("quantization", "unknown"),
        "lora_r": config.get("lora_r"),
        "num_epochs": config.get("num_epochs"),
        "learning_rate": config.get("learning_rate"),
        "batch_size": config.get("batch_size"),
        "gradient_accumulation": config.get("gradient_accumulation"),
        "dataset_path": str(dataset_path),
        "dataset_stats": _dataset_stats(dataset_path),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict, out_dir: Path | None = None) -> Path:
    if out_dir is None:
        out_dir = _REPO_ROOT / "memory" / "training_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{manifest['run_id']}_manifest.json"
    out_path = out_dir / filename
    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate training run manifest")
    parser.add_argument("--config", required=True, help="Path to runtime config YAML")
    parser.add_argument("--dataset-path", required=True, help="Path to training dataset")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: memory/training_runs/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    manifest = generate_manifest(args.config, args.dataset_path)
    path = write_manifest(manifest, out_dir)
    print(f"Run manifest written to {path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
