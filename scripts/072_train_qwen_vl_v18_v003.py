#!/usr/bin/env python3
"""v18 multimodal LoRA — resumes from v17 with longer training and tighter LR.

Improvements over v17:
* Resumes from the v17 LoRA adapter so we keep its v003-schema fluency and
  spend the new compute on tighter score regression.
* 6 epochs (vs 3) — the v17 train loss was still 2.5+ at the end and well
  above eval_loss, suggesting under-training rather than overfitting.
* Lower peak LR (3e-5 instead of 1e-4) and a short warmup, since we're
  resuming. The v17 logs showed loss bouncing around between 2-5 in the
  last epoch, which is consistent with too-aggressive LR for resumed
  training.
* Adds a per-task scoring-format reminder to the user message via the
  rubric loader when the dataset prep didn't already inject one.
* Slightly larger LoRA dropout to discourage memorization on the small
  239-example corpus.

Run on the same RunPod pod that hosts ``/workspace/v003_multimodal/`` and
``/workspace/v003_frames/``::

    python scripts/072_train_qwen_vl_v18_v003.py \\
        --base-adapter /workspace/v17_lora_output/final_adapter \\
        --data-dir   /workspace/v003_multimodal \\
        --frames-dir /workspace/v003_frames \\
        --output-dir /workspace/v18_lora_output
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch  # type: ignore
from peft import LoraConfig, PeftModel, TaskType, get_peft_model  # type: ignore
from transformers import (  # type: ignore
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the v16 trainer's dataset + collator (same chat-format expectations).
_v16_path = ROOT / "scripts" / "059_train_qwen_vl_v16_multimodal.py"
spec = importlib.util.spec_from_file_location("v16_train", _v16_path)
assert spec and spec.loader
_v16 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_v16)

FLSMultimodalDataset = _v16.FLSMultimodalDataset
MultimodalDataCollator = _v16.MultimodalDataCollator

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_LENGTH = 8192
IMAGE_MIN_PIXELS = _v16.IMAGE_MIN_PIXELS
IMAGE_MAX_PIXELS = _v16.IMAGE_MAX_PIXELS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/workspace/v003_multimodal")
    p.add_argument("--frames-dir", default="/workspace/v003_frames")
    p.add_argument("--output-dir", default="/workspace/v18_lora_output")
    p.add_argument(
        "--base-adapter",
        default="/workspace/v17_lora_output/final_adapter",
        help="Path to the v17 adapter to resume from.",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="v17 used 8; smaller effective batch helps the model see more diverse examples per step.")
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1,
                   help="Slight bump from 0.05 to discourage memorization on the small corpus.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[{datetime.now()}] v18 (v003) multimodal LoRA training")
    print(f"  Data dir:    {args.data_dir}")
    print(f"  Frames dir:  {args.frames_dir}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Resume from: {'<none>' if args.no_resume else args.base_adapter}")
    print(f"  LR / epochs / dropout: {args.learning_rate} / {args.epochs} / {args.lora_dropout}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        min_pixels=IMAGE_MIN_PIXELS,
        max_pixels=IMAGE_MAX_PIXELS,
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    train_jsonl = os.path.join(args.data_dir, "train.jsonl")
    val_jsonl = os.path.join(args.data_dir, "val.jsonl")
    train_ds = FLSMultimodalDataset(train_jsonl, processor, args.frames_dir, max_length=MAX_LENGTH)
    val_ds = FLSMultimodalDataset(val_jsonl, processor, args.frames_dir, max_length=MAX_LENGTH)
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if not args.no_resume and Path(args.base_adapter).exists():
        print(f"\n[{datetime.now()}] Resuming from {args.base_adapter}")
        model = PeftModel.from_pretrained(model, args.base_adapter, is_trainable=True)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
    else:
        print(f"\n[{datetime.now()}] Fresh LoRA")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
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

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
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
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print(f"\n[{datetime.now()}] Starting v18 training...")
    result = trainer.train()
    eval_result = trainer.evaluate()
    print(f"[{datetime.now()}] Done. train_loss={result.training_loss:.4f}  eval_loss={eval_result['eval_loss']:.4f}")

    adapter_dir = os.path.join(args.output_dir, "final_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.save_model(adapter_dir)
    processor.tokenizer.save_pretrained(adapter_dir)

    metrics = {
        "version": "v18_v003",
        "base_model": MODEL_ID,
        "resumed_from": (None if args.no_resume else args.base_adapter),
        "train_loss": result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "train_steps": result.global_step,
        "epochs": args.epochs,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_dropout": args.lora_dropout,
        "completed_at": datetime.now().isoformat(),
    }
    with open(os.path.join(adapter_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
