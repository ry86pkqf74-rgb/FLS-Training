#!/usr/bin/env python3
"""Train a DPO adapter on teacher preference pairs.

The DPO stage starts from an SFT checkpoint and nudges the model toward
responses that are closer to the teacher-consensus target. This is useful
when the corpus is small but high-quality pairwise judgments already exist.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DPO adapter for FLS scoring/coaching")
    parser.add_argument("--config", help="Optional YAML config file")
    parser.add_argument("--sft-checkpoint", help="Merged or base SFT checkpoint path")
    parser.add_argument("--dataset-path", default="data/training/dpo_v1", help="Directory with train.jsonl and val.jsonl")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Max DPO epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--loss-type", default="sigmoid", help="DPO loss type")
    parser.add_argument("--max-length", type=int, default=4096, help="Sequence length")
    parser.add_argument("--max-prompt-length", type=int, default=2048, help="Prompt truncation length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/config without training")
    return parser.parse_args()


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    file_config = load_config(args.config)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    output_dir = args.output_dir or file_config.get("output_dir") or f"memory/model_checkpoints/{timestamp}_dpo"
    settings = {
        "sft_checkpoint": args.sft_checkpoint or file_config.get("sft_checkpoint"),
        "dataset_path": args.dataset_path or file_config.get("dataset_path", "data/training/dpo_v1"),
        "output_dir": output_dir,
        "epochs": min(int(args.epochs or file_config.get("epochs", 3)), 3),
        "learning_rate": float(args.learning_rate or file_config.get("learning_rate", 5e-6)),
        "batch_size": int(args.batch_size or file_config.get("batch_size", 1)),
        "grad_accum": int(args.grad_accum or file_config.get("grad_accum", 8)),
        "beta": float(args.beta or file_config.get("beta", 0.1)),
        "loss_type": str(args.loss_type or file_config.get("loss_type", "sigmoid")),
        "max_length": int(args.max_length or file_config.get("max_length", 4096)),
        "max_prompt_length": int(args.max_prompt_length or file_config.get("max_prompt_length", 2048)),
        "seed": int(args.seed or file_config.get("seed", 42)),
        "lora_r": int(file_config.get("lora_r", 16)),
        "lora_alpha": int(file_config.get("lora_alpha", 32)),
        "lora_dropout": float(file_config.get("lora_dropout", 0.05)),
        "lora_target_modules": file_config.get("lora_target_modules", DEFAULT_TARGET_MODULES),
        "wandb": bool(args.wandb or file_config.get("wandb", False)),
        "wandb_project": file_config.get("wandb_project", "fls-training"),
        "run_name": file_config.get("run_name") or Path(output_dir).name,
        "dry_run": bool(args.dry_run),
    }
    if not settings["sft_checkpoint"]:
        raise ValueError("--sft-checkpoint is required unless provided by --config")
    return settings


def validate_dataset(dataset_dir: str | Path) -> dict[str, int]:
    dataset_path = Path(dataset_dir)
    counts = {}
    for split in ("train", "val"):
        path = dataset_path / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing DPO split: {path}")
        count = 0
        with open(path) as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                missing = [key for key in ("prompt", "chosen", "rejected") if not payload.get(key)]
                if missing:
                    raise ValueError(f"{path}:{line_number} missing required fields: {missing}")
                count += 1
        counts[split] = count
    return counts


def load_training_model(checkpoint_path: str | Path):
    import torch
    from transformers import AutoModelForCausalLM
    from transformers import Qwen2_5_VLForConditionalGeneration

    try:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as exc:
        logger.warning("Falling back to causal LM load for DPO model: %s", exc)

    return AutoModelForCausalLM.from_pretrained(
        str(checkpoint_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )


def maybe_log_final_metrics(use_wandb: bool, metrics: dict[str, Any]) -> None:
    if not use_wandb:
        return
    try:
        import wandb
    except ImportError:
        logger.warning("wandb requested but package is not installed")
        return
    if wandb.run is not None:
        wandb.log({f"final/{key}": value for key, value in metrics.items()})
        wandb.finish()


def configure_wandb_env(settings: dict[str, Any]) -> None:
    if not settings["wandb"]:
        return
    os.environ.setdefault("WANDB_PROJECT", settings["wandb_project"])
    os.environ.setdefault("WANDB_NAME", settings["run_name"])


def train(settings: dict[str, Any]) -> dict[str, Any]:
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, EarlyStoppingCallback
    from trl import DPOConfig, DPOTrainer

    dataset_path = Path(settings["dataset_path"])
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_wandb_env(settings)

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(dataset_path / "train.jsonl"),
            "validation": str(dataset_path / "val.jsonl"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(settings["sft_checkpoint"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_training_model(settings["sft_checkpoint"])
    peft_config = LoraConfig(
        r=settings["lora_r"],
        lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"],
        target_modules=settings["lora_target_modules"],
        task_type="CAUSAL_LM",
    )

    args = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["grad_accum"],
        learning_rate=settings["learning_rate"],
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if settings["wandb"] else "none",
        run_name=settings["run_name"],
        bf16=True,
        seed=settings["seed"],
        beta=settings["beta"],
        loss_type=settings["loss_type"],
        max_length=settings["max_length"],
        max_prompt_length=settings["max_prompt_length"],
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    (output_dir / "eval_results.json").write_text(json.dumps(eval_metrics, indent=2))
    (output_dir / "training_config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
    maybe_log_final_metrics(settings["wandb"], eval_metrics)
    return eval_metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    settings = resolve_settings(args)
    split_counts = validate_dataset(settings["dataset_path"])

    print(json.dumps({"config": settings, "dataset": split_counts}, indent=2))
    if settings["dry_run"]:
        return

    metrics = train(settings)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()