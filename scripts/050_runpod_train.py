#!/usr/bin/env python3
"""DEPRECATED stub — do not invoke.

Quarantined 2026-04-08 (hardening sprint). The original launcher called
the Florence-2 trainer and has been moved to
`archive/deprecated/050_runpod_train_florence2.py`.

Canonical training launch path is `deploy/runpod_launch.sh`, which calls
`python -m src.training.finetune_vlm --config <runtime.yaml>`. That
pipeline is the one the proven April 2026 deploy uses.
"""
from __future__ import annotations

import sys


MESSAGE = (
    "scripts/050_runpod_train.py was quarantined on 2026-04-08.\n"
    "It launched Florence-2 via src.training.runpod_trainer — the wrong\n"
    "backbone, the wrong data path, and the wrong schema defaults.\n\n"
    "Canonical launch path:\n"
    "  bash deploy/runpod_launch.sh "
    "<dataset_path> src/configs/finetune_lasana_v4.yaml\n\n"
    "Before launching anything on a paid GPU, confirm you have:\n"
    "  1. prepared data via  python scripts/040_prepare_training_data.py "
    "--include-coach-feedback --ver v4\n"
    "  2. (for vision mode) extracted frames via "
    "scripts/068_lasana_extract_features.py --frames-only\n"
    "  3. run  bash scripts/runpod_smoke_test.sh  and seen every check PASS\n"
    "  4. cleared the v4 gate in src/configs/finetune_lasana_v4.yaml\n"
    "  5. reviewed abort thresholds in docs/LASANA_DEPLOYMENT_PLAN.md\n"
)


def main() -> int:
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
