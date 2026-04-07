"""Fine-tune Qwen2.5-VL-7B-Instruct on FLS scoring data.

This script runs on a GPU server (RunPod A100 80GB recommended).
It uses LoRA for parameter-efficient fine-tuning.

Usage:
    python -m src.training.finetune_vlm --config configs/finetune_task5_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def finetune(config: dict) -> dict:
    """Run fine-tuning. Requires GPU environment with training dependencies.

    Returns eval metrics dict.
    """
    # Import training dependencies (only available on GPU server)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "Training dependencies not installed. "
            "Install with: pip install fls-training[training]"
        )
        sys.exit(1)

    base_model = config.get("base_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    dataset_path = config["dataset_path"]
    output_dir = config.get("output_dir", f"memory/model_checkpoints/{datetime.now(timezone.utc).strftime('%Y-%m-%d')}_run")

    logger.info(f"Loading base model: {base_model}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_dir}")

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 8),
        learning_rate=config.get("learning_rate", 2e-5),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        bf16=config.get("bf16", True),
        logging_steps=config.get("logging_steps", 10),
        save_strategy="epoch",
        evaluation_strategy="epoch",
        report_to="none",  # or "wandb"
    )

    # Load model with LoRA
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": str(Path(dataset_path) / "train.jsonl"),
        "validation": str(Path(dataset_path) / "val.jsonl"),
    })

    # TODO: Implement proper data collator for VLM with image inputs
    # This requires Qwen2.5-VL-specific preprocessing
    logger.warning(
        "Image-based training data collation not yet implemented. "
        "This script provides the training scaffold — implement "
        "Qwen2.5-VL-specific image processing in a data collator."
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Eval metrics
    eval_results = trainer.evaluate()
    metrics_path = Path(output_dir) / "eval_results.json"
    with open(metrics_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"Training complete. Checkpoint saved to {output_dir}")
    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune VLM on FLS scoring data")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    metrics = finetune(config)
    print(f"Eval metrics: {json.dumps(metrics, indent=2)}")
