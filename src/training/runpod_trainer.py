"""DEPRECATED — Florence-2 training loop, do not import or launch.

Quarantined 2026-04-08 (hardening sprint). The original file was moved to
`archive/deprecated/runpod_trainer.py`.

Florence-2 was never the intended student model for FLS. The real
trainer is `src.training.finetune_vlm`, launched via
`deploy/runpod_launch.sh`. This stub exists so that a stale
`from src.training.runpod_trainer import ...` statement fails loudly
instead of silently re-initialising the wrong backbone on a paid GPU.
"""

from __future__ import annotations


_DEPRECATION_MESSAGE = (
    "src.training.runpod_trainer was quarantined on 2026-04-08. It trained "
    "microsoft/Florence-2-large on flattened-text examples, which is the "
    "wrong backbone and the wrong training signal. Use "
    "`deploy/runpod_launch.sh` → `src.training.finetune_vlm` instead. See "
    "archive/deprecated/README.md for context."
)


def run_full_training(*_args, **_kwargs):  # pragma: no cover - deprecation guard
    raise RuntimeError(_DEPRECATION_MESSAGE)


class TrainingConfig:  # pragma: no cover - deprecation guard
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(_DEPRECATION_MESSAGE)


def __getattr__(name: str):  # pragma: no cover - deprecation guard
    raise AttributeError(_DEPRECATION_MESSAGE)
