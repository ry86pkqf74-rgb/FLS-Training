# Deprecated training code — do NOT import, do NOT launch

Quarantined 2026-04-08 as part of the hardening sprint triggered by the
2026-04-08 pipeline review. These files are kept for historical reference
and for future forensic debugging; they must not be imported or launched
from live code.

## `runpod_trainer.py`

Old Florence-2 training loop. Default `base_model` was
`microsoft/Florence-2-large`. Florence-2 was never the intended student
model for FLS — Qwen2.5-VL-7B-Instruct is. A misclick that launched this
script on a paid GPU would have burned budget fine-tuning the wrong
backbone on a schema-mismatched dataset.

Replacement: `src/training/finetune_vlm.py`.

## `data_prep_text_flatten.py` (was `src/training/data_prep.py`)

Old training-data builder. Flattened `frame_analyses` into natural-language
strings of the form `"Frame N (phase): description"` and fed them as
plaintext to a tokenizer. This reduced training to imitating teacher
narration — the student model never saw a pixel. Combined with
`runpod_trainer.py`, it fine-tuned a 7B language model pretending to be a
VLM.

Replacement: `src/training/prepare_dataset.py` (writes chat-message
records with image content blocks when `--frames-dir` is set, and merges
trainee/source-domain metadata so eval splits are resident-aware).

## `050_runpod_train_florence2.py` (was `scripts/050_runpod_train.py`)

Launcher for `runpod_trainer.py`. Default `--ver v1` + default
`--base-model microsoft/Florence-2-large`. Both defaults were landmines.

Replacement: `deploy/runpod_launch.sh` (the canonical launcher used by
the proven April 2026 deploy path). It calls
`python -m src.training.finetune_vlm --config <runtime.yaml>`.

## Before un-archiving anything

Re-read `docs/LASANA_DEPLOYMENT_PLAN.md` and the baseline report at
`memory/baselines/2026-04-08_teacher_mae_baseline.md`. Any resurrection of
these files must be justified against the MAE noise-floor numbers
(teacher-vs-teacher ≈21.6).
