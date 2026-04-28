#!/usr/bin/env python3
"""Train the v003 report-generation LoRA.

This is a text-only LoRA on top of a small instruction-tuned base model. The
training pairs are produced by ``scripts/060_generate_report_v3_labels.py`` and
flattened to chat format by ``scripts/062_prepare_lora_report_v3_dataset.py``.

The model learns the mapping::

    (score_json, rubric_json, resident_level, previous_attempts_summary,
     experimental_metrics) -> v003 report JSON + rendered markdown

Run on RunPod after v17 multimodal training completes (so the same pod can
host both adapters in sequence). On an H100 the run completes in roughly
30–60 minutes for a few thousand examples.

    python scripts/063_train_report_lora_v3.py \\
        --dataset-jsonl /workspace/v003_report_lora/train.jsonl \\
        --val-jsonl /workspace/v003_report_lora/val.jsonl \\
        --output-dir /workspace/report_lora_v003 \\
        --base-model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --human-reviewed
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch  # type: ignore
from datasets import Dataset  # type: ignore
from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace/report_lora_v003"),
    )
    parser.add_argument(
        "--base-model",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Base text model for the report LoRA.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--human-reviewed",
        action="store_true",
        help=(
            "Required acknowledgement that the v003 labels passed validation "
            "and a human inspected a sample. Refuses to train without it."
        ),
    )
    return parser.parse_args()


def _content_to_str(content: Any) -> str:
    """Flatten the chat ``content`` (which may be a dict or list) to a string."""
    if isinstance(content, str):
        return content
    return json.dumps(content, default=str)


def _format_example(example: dict, tokenizer) -> dict:
    """Convert a chat-format example into a tokenized prompt+completion."""
    messages = example["messages"]
    formatted_messages = [
        {"role": msg["role"], "content": _content_to_str(msg["content"])}
        for msg in messages
    ]
    text = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    if not args.human_reviewed:
        raise SystemExit(
            "Refusing to train report LoRA v3 until labels pass validation and "
            "human review. Re-run scripts/061_validate_report_v3_labels.py first, "
            "spot-check a handful of examples, then re-run with --human-reviewed."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now()}] v003 report LoRA training")
    print(f"  Base model: {args.base_model}")
    print(f"  Train:      {args.dataset_jsonl}")
    print(f"  Val:        {args.val_jsonl}")
    print(f"  Output:     {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_rows = _load_jsonl(args.dataset_jsonl)
    if args.val_jsonl and args.val_jsonl.exists():
        val_rows = _load_jsonl(args.val_jsonl)
    else:
        # 10% holdout if no val provided.
        split = max(1, len(train_rows) // 10)
        val_rows = train_rows[-split:]
        train_rows = train_rows[:-split]
    print(f"  Train rows: {len(train_rows)}  Val rows: {len(val_rows)}")

    train_ds = Dataset.from_list([_format_example(r, tokenizer) for r in train_rows])
    val_ds = Dataset.from_list([_format_example(r, tokenizer) for r in val_rows])

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=["text"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

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

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
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
        dataloader_num_workers=2,
        seed=42,
    )

    from transformers import DataCollatorForLanguageModeling  # type: ignore

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    print(f"\n[{datetime.now()}] Starting training...")
    result = trainer.train()
    eval_result = trainer.evaluate()
    print(f"\n[{datetime.now()}] Done. train_loss={result.training_loss:.4f} eval_loss={eval_result['eval_loss']:.4f}")

    adapter_dir = args.output_dir / "final_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "version": "report_lora_v003",
        "base_model": args.base_model,
        "train_loss": result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "train_steps": result.global_step,
        "epochs": args.epochs,
        "train_examples": len(train_ds),
        "val_examples": len(val_ds),
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.learning_rate,
        "completed_at": datetime.now().isoformat(),
    }
    (adapter_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()
