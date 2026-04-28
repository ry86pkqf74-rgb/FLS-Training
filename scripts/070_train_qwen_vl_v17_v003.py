#!/usr/bin/env python3
"""
v17 (v003) Multimodal LoRA fine-tuning for Qwen2.5-VL-7B-Instruct.

This is a continuation of scripts/059_train_qwen_vl_v16_multimodal.py with two
v003-specific differences:

1. Resumes from the v16 LoRA adapter (instead of training from scratch) so the
   model retains its existing scoring competence and only learns the new v003
   schema fields (formula_applied, critical_errors, severity, cannot_determine,
   confidence_rationale, task_specific_assessments).
2. Reads the v003-shape JSONL produced by ``scripts/030c_prep_v003_multimodal.py``
   or ``scripts/040_prepare_training_data.py --v003-target-schema``.

Run on RunPod (single high-VRAM GPU, e.g. H100 SXM 80GB):

    cd /workspace/FLS-Training
    python scripts/070_train_qwen_vl_v17_v003.py \\
        --data-dir /workspace/v003_multimodal \\
        --frames-dir /workspace/v003_frames \\
        --base-adapter /workspace/v16_lora_output/final_adapter \\
        --output-dir /workspace/v17_lora_output

Pass ``--no-resume`` to train from base Qwen2.5-VL weights (not recommended on
the first iteration — losing v16 progress is unnecessary for a schema upgrade).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch  # type: ignore
from transformers import (  # type: ignore
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # type: ignore

# Reuse the dataset / collator from the v16 trainer so we don't duplicate logic.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib.util  # noqa: E402

_v16_path = ROOT / "scripts" / "059_train_qwen_vl_v16_multimodal.py"
_spec = importlib.util.spec_from_file_location("v16_train", _v16_path)
assert _spec and _spec.loader, f"Cannot load v16 trainer from {_v16_path}"
_v16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v16)

FLSMultimodalDataset = _v16.FLSMultimodalDataset
MultimodalDataCollator = _v16.MultimodalDataCollator

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
# Bumped from the v16 default (4096) — v003 examples carry an explicit task
# rubric in the user message + a JSON v003 target in the assistant turn, so
# the combined token count comfortably exceeds 4k once image tokens are added.
MAX_LENGTH = 8192
IMAGE_MIN_PIXELS = _v16.IMAGE_MIN_PIXELS
IMAGE_MAX_PIXELS = _v16.IMAGE_MAX_PIXELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/workspace/v003_multimodal")
    parser.add_argument("--frames-dir", default="/workspace/v003_frames")
    parser.add_argument("--output-dir", default="/workspace/v17_lora_output")
    parser.add_argument(
        "--base-adapter",
        default="/workspace/v16_lora_output/final_adapter",
        help="Path to the v16 LoRA adapter to resume from (set to empty string with --no-resume to skip).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Train from base Qwen2.5-VL weights instead of resuming from --base-adapter.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Lower than v16's 1e-4 because we're resuming.")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[{datetime.now()}] v17 (v003) multimodal LoRA training")
    print(f"  Data dir:     {args.data_dir}")
    print(f"  Frames dir:   {args.frames_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Resume from:  {'<none>' if args.no_resume else args.base_adapter}")

    # ── Processor ──
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        min_pixels=IMAGE_MIN_PIXELS,
        max_pixels=IMAGE_MAX_PIXELS,
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    # ── Datasets ──
    train_jsonl = os.path.join(args.data_dir, "train.jsonl")
    val_jsonl = os.path.join(args.data_dir, "val.jsonl")
    print(f"\n[{datetime.now()}] Loading datasets...")
    train_dataset = FLSMultimodalDataset(train_jsonl, processor, args.frames_dir, max_length=MAX_LENGTH)
    val_dataset = FLSMultimodalDataset(val_jsonl, processor, args.frames_dir, max_length=MAX_LENGTH)
    print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    # ── Base model (4-bit) ──
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"\n[{datetime.now()}] Loading base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── LoRA: resume from v16 OR fresh ──
    if not args.no_resume and args.base_adapter and Path(args.base_adapter).exists():
        print(f"\n[{datetime.now()}] Resuming from v16 adapter: {args.base_adapter}")
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
        # Make all LoRA params trainable for the v003 fine-tune.
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    else:
        print(f"\n[{datetime.now()}] Fresh LoRA (no resume)")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # ── Trainer config ──
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        seed=42,
        dataloader_pin_memory=True,
    )

    data_collator = MultimodalDataCollator(processor=processor, max_length=MAX_LENGTH)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # ── Train ──
    print(f"\n[{datetime.now()}] Starting v17 training...")
    print(f"  Effective batch: {args.batch_size * args.grad_accum}")
    print(f"  LR: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    result = trainer.train()
    print(f"\n[{datetime.now()}] Training complete. Train loss: {result.training_loss:.4f}")

    # ── Save adapter ──
    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.save_model(adapter_dir)
    processor.tokenizer.save_pretrained(adapter_dir)
    print(f"[{datetime.now()}] Saved adapter to {adapter_dir}")

    # ── Eval ──
    eval_result = trainer.evaluate()
    print(f"[{datetime.now()}] Eval loss: {eval_result['eval_loss']:.4f}")

    metrics = {
        "version": "v17_v003",
        "base_model": MODEL_ID,
        "resumed_from": (None if args.no_resume else args.base_adapter),
        "train_loss": result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "train_steps": result.global_step,
        "epochs": args.epochs,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.learning_rate,
        "completed_at": datetime.now().isoformat(),
    }
    with open(os.path.join(adapter_dir, "training_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"v17 (v003) Multimodal LoRA training COMPLETE")
    print(f"Adapter: {adapter_dir}")
    print(f"Train loss: {result.training_loss:.4f}")
    print(f"Eval loss:  {eval_result['eval_loss']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
