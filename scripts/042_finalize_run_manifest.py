#!/usr/bin/env python3
"""Append final metrics to a run manifest after training completes.

Called by runpod_launch.sh immediately after src.training.finetune_vlm
exits, whether successfully or not.  Reads trainer_state.json from the
checkpoint directory to extract the final train/eval losses and the best
checkpoint path.

Usage:
    python scripts/042_finalize_run_manifest.py \\
        --output-dir memory/model_checkpoints/fls_task5_20260408_120000 \\
        --exit-status 0

Fields appended to run_manifest.json:
    finished_at            ISO-8601 UTC timestamp
    final_train_loss       float or null
    final_eval_loss        float or null
    best_checkpoint_path   str or null
    total_training_seconds int or null
    exit_status            int
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# trainer_state.json parsing
# ---------------------------------------------------------------------------

def _find_trainer_state(output_dir: Path) -> Path | None:
    """Return the most relevant trainer_state.json, searching output_dir then checkpoint subdirs."""
    candidate = output_dir / "trainer_state.json"
    if candidate.exists():
        return candidate
    # Fall back to the highest-numbered checkpoint subdir
    checkpoints = sorted(
        output_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[-1]) if p.parent.name.split("-")[-1].isdigit() else 0,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def _read_trainer_state(output_dir: Path) -> dict:
    state_path = _find_trainer_state(output_dir)
    if state_path is None:
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _extract_losses(
    state: dict,
) -> tuple[float | None, float | None, str | None]:
    """Scan log_history in reverse to find the last train_loss and eval_loss entries."""
    log_history = state.get("log_history", [])
    final_train_loss: float | None = None
    final_eval_loss: float | None = None

    for entry in reversed(log_history):
        if final_train_loss is None and "train_loss" in entry:
            final_train_loss = float(entry["train_loss"])
        if final_eval_loss is None and "eval_loss" in entry:
            final_eval_loss = float(entry["eval_loss"])
        if final_train_loss is not None and final_eval_loss is not None:
            break

    best_checkpoint = state.get("best_model_checkpoint")
    return final_train_loss, final_eval_loss, best_checkpoint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Finalize run_manifest.json after training completes",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Training output directory (must contain run_manifest.json)",
    )
    parser.add_argument(
        "--exit-status",
        type=int,
        default=0,
        help="Exit code returned by the training process (0 = success)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "run_manifest.json"

    if not manifest_path.exists():
        print(
            f"WARNING: {manifest_path} not found; cannot finalize manifest.",
            file=sys.stderr,
        )
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    finished_at = datetime.now(timezone.utc)
    manifest["finished_at"] = finished_at.isoformat()

    # Compute elapsed seconds from started_at if available
    started_at_str = manifest.get("started_at")
    if started_at_str:
        try:
            start = datetime.fromisoformat(started_at_str)
            manifest["total_training_seconds"] = round(
                (finished_at - start).total_seconds()
            )
        except Exception:
            manifest["total_training_seconds"] = None
    else:
        manifest["total_training_seconds"] = None

    # Extract loss metrics from HuggingFace trainer_state.json
    state = _read_trainer_state(output_dir)
    final_train_loss, final_eval_loss, best_checkpoint = _extract_losses(state)

    manifest["final_train_loss"] = final_train_loss
    manifest["final_eval_loss"] = final_eval_loss
    manifest["best_checkpoint_path"] = best_checkpoint
    manifest["exit_status"] = args.exit_status

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Run manifest finalized: {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
