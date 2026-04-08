"""DEPRECATED — do not import from this module.

The original `src/training/data_prep.py` flattened teacher frame narrations
into plaintext strings and wrote them as training examples, which turned
VLM fine-tuning into text-only imitation of teacher prose. It was
quarantined on 2026-04-08 (hardening sprint) and moved to
`archive/deprecated/data_prep_text_flatten.py` for forensic reference.

Replacement: `src.training.prepare_dataset.prepare_dataset`, driven by
`scripts/040_prepare_training_data.py`. That path emits chat-message
records and, when a frames directory is provided, real image content
blocks that Qwen2.5-VL can actually consume.

This stub exists only so that any lingering `from src.training.data_prep
import prepare_training_data` fails with a clear pointer instead of a
confusing AttributeError.
"""

from __future__ import annotations


_DEPRECATION_MESSAGE = (
    "src.training.data_prep.prepare_training_data was quarantined on 2026-04-08. "
    "It flattened frame_analyses to plaintext and would fine-tune a text-only "
    "language model on a supposedly-vision task. Use "
    "src.training.prepare_dataset.prepare_dataset (via "
    "scripts/040_prepare_training_data.py) instead. See "
    "archive/deprecated/README.md for context."
)


def prepare_training_data(*_args, **_kwargs):  # pragma: no cover - deprecation guard
    raise RuntimeError(_DEPRECATION_MESSAGE)


def __getattr__(name: str):  # pragma: no cover - deprecation guard
    raise AttributeError(_DEPRECATION_MESSAGE)
