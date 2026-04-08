#!/usr/bin/env python3
"""Write a reproducibility manifest at the START of a training run.

Writes <output_dir>/run_manifest.json capturing the full environment
snapshot (git state, GPU, config hash, dataset fingerprint, etc.) so
every checkpoint is traceable back to the exact conditions that produced it.

Usage:
    python scripts/041_write_run_manifest.py \\
        --config /tmp/finetune_runtime.yaml \\
        --dataset-path training/data/v2

The resolved output directory is printed to stdout on success so the
caller (runpod_launch.sh) can capture it:

    OUTPUT_DIR=$(python scripts/041_write_run_manifest.py --config ...)

If the config does not already contain an output_dir key, this script
patches the config file with the auto-generated directory so that
src.training.finetune_vlm lands its checkpoints in the same place the
manifest was written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(output)
    except Exception:
        return False


def _gpu_info() -> tuple[str, float]:
    """Return (gpu_name, gpu_memory_gb).  Falls back to nvidia-smi if torch unavailable."""
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            memory_gb = props.total_memory / (1024 ** 3)
            return name, round(memory_gb, 2)
    except Exception:
        pass

    try:
        name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().split("\n")[0]
        mem_str = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().split("\n")[0]
        # mem_str is e.g. "24576 MiB"
        mem_mb = int(mem_str.split()[0])
        return name, round(mem_mb / 1024, 2)
    except Exception:
        pass

    return "unknown", 0.0


def _torch_version() -> str:
    try:
        import torch  # noqa: PLC0415
        return torch.__version__
    except ImportError:
        return "unknown"


# ---------------------------------------------------------------------------
# Dataset fingerprinting
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _dataset_version(dataset_path: Path) -> dict:
    """Return line count + sha256 of the first 1000 lines of train.jsonl."""
    train_file = dataset_path / "train.jsonl"
    if not train_file.exists():
        return {"line_count": 0, "first_1000_sha256": None}
    lines = train_file.read_text(encoding="utf-8").splitlines()
    line_count = len(lines)
    sample = "\n".join(lines[:1000]).encode("utf-8")
    return {
        "line_count": line_count,
        "first_1000_sha256": _sha256_bytes(sample),
    }


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write run_manifest.json at training start",
    )
    parser.add_argument("--config", required=True, help="Path to YAML training config")
    parser.add_argument(
        "--dataset-path",
        help="Override dataset_path from config (e.g. training/data/v2)",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output_dir from config",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    with config_path.open() as fh:
        config = yaml.safe_load(fh)

    now = datetime.now(timezone.utc)
    run_id = f"fls_task5_{now.strftime('%Y%m%d_%H%M%S')}"

    # --- resolve dataset path ---
    dataset_path_str = args.dataset_path or config.get("dataset_path", "")
    dataset_path = Path(dataset_path_str).resolve() if dataset_path_str else None

    # --- resolve output directory ---
    output_dir_str = (
        args.output_dir
        or config.get("output_dir")
        or f"memory/model_checkpoints/{now.strftime('%Y%m%d_%H%M%S')}_unsloth"
    )
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If the config did not specify output_dir, patch it so finetune_vlm.py
    # writes checkpoints into the same directory the manifest was created in.
    if not config.get("output_dir"):
        config["output_dir"] = str(output_dir)
        with config_path.open("w") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)

    # --- collect fields ---
    gpu_name, gpu_memory_gb = _gpu_info()

    dataset_version = _dataset_version(dataset_path) if dataset_path else {}
    num_train = _count_lines(dataset_path / "train.jsonl") if dataset_path else 0
    num_eval = _count_lines(dataset_path / "val.jsonl") if dataset_path else 0

    manifest: dict = {
        "run_id": run_id,
        "git_commit_sha": _git_commit_sha(),
        "git_dirty": _git_dirty(),
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory_gb,
        "config_path": str(config_path),
        "config_hash": _file_sha256(config_path),
        "dataset_path": str(dataset_path) if dataset_path else None,
        "dataset_version": dataset_version,
        "num_train_samples": num_train,
        "num_eval_samples": num_eval,
        "base_model": config.get("base_model", "unknown"),
        "framework": config.get("framework", "unknown"),
        "vision_mode": bool(config.get("require_vision", False)),
        "quantization": str(config.get("quantization", "none")),
        "resume_checkpoint": config.get("resume_from_checkpoint"),
        "started_at": now.isoformat(),
        "server_hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": _torch_version(),
    }

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Print the resolved output_dir for shell-script capture
    print(str(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
