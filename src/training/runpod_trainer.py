"""Legacy Florence-based RunPod training script for the FLS student model.

This module is retained so older experiment artifacts remain reproducible,
but it is not the primary training path anymore. The supported RunPod flow
uses `src.training.finetune_vlm` with `src/configs/finetune_task5_v2.yaml`.

This script runs ON the RunPod A100 pod. It:
1. Loads training data from the cloned repo
2. Fine-tunes Florence-2-large with LoRA (scoring + coaching heads)
3. Saves adapters and logs back to the repo directory
4. Can be run via: python -m src.training.runpod_trainer

After training completes, git push results back to GitHub.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Configuration for a training run."""
    # Model
    base_model: str = "microsoft/Florence-2-large"
    adapter_type: str = "lora"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "down_proj", "up_proj",
    ])

    # Training
    # Defaults tuned for the v2 corpus (~150-250 examples after dedupe).
    # 5 epochs at 2e-4 was overfitting territory; 3 epochs at 1e-4 with
    # effective batch 16 (2 * grad_accum 8) gives stabler convergence
    # and lets early stopping pick the best checkpoint.
    num_epochs: int = 3
    learning_rate: float = 1e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 2048
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    early_stopping_patience: int = 1

    # Data
    scoring_train_path: str = "training/data/scoring_train_v1.jsonl"
    scoring_val_path: str = "training/data/scoring_val_v1.jsonl"
    coaching_train_path: str = "training/data/coaching_train_v1.jsonl"
    coaching_val_path: str = "training/data/coaching_val_v1.jsonl"

    # Output
    output_dir: str = "models/student_v1"
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 25

    # Tracking
    use_wandb: bool = True
    wandb_project: str = "fls-training"
    run_name: str = ""

    def __post_init__(self):
        if not self.run_name:
            self.run_name = f"fls_student_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def train_scoring_head(config: TrainingConfig) -> dict:
    """Fine-tune the scoring head: frames → structured FLS score."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoProcessor, TrainingArguments, Trainer,
            EarlyStoppingCallback,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
    except ImportError as e:
        print(f"Missing training dependency: {e}")
        print("Install with: pip install torch transformers peft datasets accelerate bitsandbytes")
        return {"error": str(e)}

    print(f"\n{'='*60}")
    print(f"  FLS Student Model — Scoring Head Training")
    print(f"  Base model: {config.base_model}")
    print(f"  LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"  Epochs: {config.num_epochs}, LR: {config.learning_rate}")
    print(f"{'='*60}\n")

    start_time = time.time()
    output_dir = Path(config.output_dir) / "scoring"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(config.base_model, trust_remote_code=True)

    # Apply LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"Loading scoring dataset from {config.scoring_train_path}...")
    train_ds = load_dataset("json", data_files=config.scoring_train_path, split="train")
    val_ds = load_dataset("json", data_files=config.scoring_val_path, split="train")
    print(f"  Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    # Tokenize
    def tokenize(examples):
        texts = [
            f"<s>[INST] {inst}\n\n{inp} [/INST] {out}</s>"
            for inst, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]
        tokenized = processor.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=config.max_seq_length,
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    # Training args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config.seed,
        data_seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name=f"{config.run_name}_scoring",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    print("Starting training...")
    result = trainer.train()

    # Save adapter
    model.save_pretrained(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "processor"))

    elapsed = time.time() - start_time
    run_meta = {
        "head": "scoring",
        "run_name": config.run_name,
        "base_model": config.base_model,
        "elapsed_seconds": round(elapsed),
        "train_loss": result.training_loss,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "config": {
            "lora_r": config.lora_r,
            "epochs": config.num_epochs,
            "lr": config.learning_rate,
            "batch_size": config.batch_size,
        },
        "completed_at": datetime.utcnow().isoformat(),
    }

    meta_path = output_dir / "run_meta.json"
    meta_path.write_text(json.dumps(run_meta, indent=2))
    print(f"\nScoring head training complete in {elapsed/60:.1f} minutes")
    print(f"Adapter saved to: {output_dir / 'adapter'}")

    return run_meta


def train_coaching_head(config: TrainingConfig) -> dict:
    """Fine-tune the coaching head: frames + history → feedback report."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoProcessor, TrainingArguments, Trainer,
            EarlyStoppingCallback,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
    except ImportError as e:
        return {"error": str(e)}

    print(f"\n{'='*60}")
    print(f"  FLS Student Model — Coaching Head Training")
    print(f"{'='*60}\n")

    start_time = time.time()
    output_dir = Path(config.output_dir) / "coaching"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model (fresh copy for separate LoRA)
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(config.base_model, trust_remote_code=True)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    print(f"Loading coaching dataset from {config.coaching_train_path}...")
    train_ds = load_dataset("json", data_files=config.coaching_train_path, split="train")
    val_ds = load_dataset("json", data_files=config.coaching_val_path, split="train")
    print(f"  Train: {len(train_ds)} examples, Val: {len(val_ds)} examples")

    def tokenize(examples):
        texts = [
            f"<s>[INST] {inst}\n\n{inp} [/INST] {out}</s>"
            for inst, inp, out in zip(
                examples["instruction"], examples["input"], examples["output"]
            )
        ]
        tokenized = processor.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=config.max_seq_length, return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=config.seed,
        data_seed=config.seed,
        report_to="wandb" if config.use_wandb else "none",
        run_name=f"{config.run_name}_coaching",
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    result = trainer.train()

    model.save_pretrained(str(output_dir / "adapter"))
    processor.save_pretrained(str(output_dir / "processor"))

    elapsed = time.time() - start_time
    run_meta = {
        "head": "coaching",
        "run_name": config.run_name,
        "base_model": config.base_model,
        "elapsed_seconds": round(elapsed),
        "train_loss": result.training_loss,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "completed_at": datetime.utcnow().isoformat(),
    }
    (output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    print(f"\nCoaching head training complete in {elapsed/60:.1f} minutes")

    return run_meta


def run_full_training(config: TrainingConfig | None = None) -> dict:
    """Run complete training cycle: both heads."""
    if config is None:
        config = TrainingConfig()

    print(f"\n{'='*60}")
    print(f"  FLS STUDENT MODEL — FULL TRAINING CYCLE")
    print(f"  Run: {config.run_name}")
    print(f"{'='*60}\n")

    scoring_result = train_scoring_head(config)
    coaching_result = train_coaching_head(config)

    # Save combined run info
    run_dir = Path("training/runs") / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "run_name": config.run_name,
        "scoring": scoring_result,
        "coaching": coaching_result,
        "completed_at": datetime.utcnow().isoformat(),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(combined, indent=2))

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Scoring:  {scoring_result.get('elapsed_seconds', 0)/60:.1f} min")
    print(f"  Coaching: {coaching_result.get('elapsed_seconds', 0)/60:.1f} min")
    print(f"  Output:   {config.output_dir}")
    print(f"\n  Next: git add -A && git commit && git push")
    print(f"{'='*60}\n")

    return combined


if __name__ == "__main__":
    run_full_training()
