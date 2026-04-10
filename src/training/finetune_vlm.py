"""Fine-tune Qwen2.5-VL-7B-Instruct on FLS scoring data.

Supports two backends:
  - unsloth (preferred): 2-5x faster, lower VRAM via QLoRA 4-bit
  - hf_trainer (fallback): standard HuggingFace Trainer + PEFT

This script runs on a GPU server (RunPod RTX 4090 spot recommended).

Usage:
    python -m src.training.finetune_vlm --config src/configs/finetune_task5_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _example_to_messages(example: dict) -> list[dict]:
    """Normalize either chat-style or instruction-style data into chat messages."""
    messages = example.get("messages")
    if messages:
        return messages

    instruction = str(example.get("instruction") or "").strip()
    input_text = str(example.get("input") or "").strip()
    output_text = str(example.get("output") or "").strip()

    user_parts = [part for part in [instruction, input_text] if part]
    if not user_parts:
        raise KeyError("Expected either 'messages' or instruction-style fields")

    normalized = [{"role": "user", "content": "\n\n".join(user_parts)}]
    if output_text:
        normalized.append({"role": "assistant", "content": output_text})
    return normalized


def _example_has_images(example: dict) -> bool:
    """Does any message in this example carry an image content block?"""
    messages = example.get("messages") or []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    return True
    return False


def _load_pil_images_from_messages(messages: list[dict]):
    """Extract and open every image referenced by a chat-message record.

    Returns a list of PIL.Image objects in the order they appear across
    user messages. Raises FileNotFoundError if any referenced path is
    missing — we'd rather fail fast at training startup than feed a
    partial image list to the processor and silently misalign examples.
    """
    from PIL import Image

    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "image":
                continue
            path = block.get("image") or block.get("image_url") or block.get("path")
            if not path:
                continue
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"Image path missing at train time: {path}. "
                    "Confirm the frames directory is rsync'd onto the GPU pod "
                    "and that the paths in train.jsonl resolve on this host."
                )
            images.append(Image.open(path).convert("RGB"))
    return images


def _format_chat_examples(batch: dict, tokenizer) -> list[str]:
    """Render chat-style training examples into plain text for SFT trainers."""
    if "messages" in batch:
        if batch["messages"] and isinstance(batch["messages"][0], dict):
            return [
                tokenizer.apply_chat_template(
                    batch["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            ]

        rendered = []
        for messages in batch["messages"]:
            rendered.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
        return rendered

    if isinstance(batch.get("instruction"), list):
        examples = []
        total = len(batch["instruction"])
        for index in range(total):
            example = {key: value[index] for key, value in batch.items() if isinstance(value, list)}
            examples.append(example)
    else:
        examples = [batch]

    rendered = []
    for example in examples:
        rendered.append(
            tokenizer.apply_chat_template(
                _example_to_messages(example),
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return rendered


def _tokenize_hf_example(example: dict, tokenizer, max_seq_length: int) -> dict:
    """Tokenize one chat-style example for the fallback HF Trainer path."""
    rendered = tokenizer.apply_chat_template(
        _example_to_messages(example),
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(
        rendered,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    tokens["labels"] = list(tokens["input_ids"])
    return tokens


def _should_export_merged(config: dict) -> bool:
    return bool(config.get("export_merged_16bit", True))


def _use_wandb(config: dict) -> bool:
    return bool(config.get("use_wandb", False))


def _wandb_project(config: dict) -> str:
    return str(config.get("wandb_project", "fls-training"))


def _run_name(config: dict, output_dir: str) -> str:
    return str(config.get("run_name") or Path(output_dir).name)


def _maybe_finish_wandb(config: dict, metrics: dict) -> None:
    if not _use_wandb(config):
        return
    try:
        import wandb
    except ImportError:
        logger.warning("wandb requested but not installed")
        return
    if wandb.run is not None:
        wandb.log({f"final/{key}": value for key, value in metrics.items()})
        wandb.finish()


def _configure_wandb_env(config: dict, output_dir: str) -> None:
    if not _use_wandb(config):
        return
    os.environ.setdefault("WANDB_PROJECT", _wandb_project(config))
    os.environ.setdefault("WANDB_NAME", _run_name(config, output_dir))


def _finetune_unsloth(config: dict) -> dict:
    """Fine-tune using Unsloth (fast QLoRA path)."""
    try:
        from unsloth import FastVisionModel
        import torch
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
    except ImportError:
        logger.error(
            "Unsloth deps not installed. On your GPU server run:\n"
            "  pip install unsloth trl datasets"
        )
        sys.exit(1)

    base_model = config.get("base_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    dataset_path = config["dataset_path"]
    output_dir = config.get("output_dir") or (
        f"memory/model_checkpoints/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}_unsloth"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _configure_wandb_env(config, output_dir)

    use_4bit = config.get("quantization") == "4bit"
    use_grad_ckpt = config.get("gradient_checkpointing", False)

    logger.info(f"[unsloth] Loading {base_model} ({'4-bit' if use_4bit else 'bf16'})...")
    model, tokenizer = FastVisionModel.from_pretrained(
        base_model,
        load_in_4bit=use_4bit,
        use_gradient_checkpointing="unsloth" if use_grad_ckpt else False,
    )

    # Cap image resolution to control vision token count per image
    max_px = config.get("max_pixels", 1024 * 28 * 28)  # default ~1M pixels
    if hasattr(tokenizer, "image_processor"):
        tokenizer.image_processor.max_pixels = max_px
        logger.info(f"[vision] Set image_processor.max_pixels = {max_px}")

    model = FastVisionModel.get_peft_model(
        model,
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get(
            "lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        ),
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    )
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={
        "train": str(Path(dataset_path) / "train.jsonl"),
        "validation": str(Path(dataset_path) / "val.jsonl"),
    })

    # Detect whether this dataset contains real image content blocks. If
    # it does, we switch to Unsloth's vision data collator which uses the
    # processor to encode images alongside text. If it does not (legacy
    # text-only path), we fall back to the old SFTTrainer formatting_func
    # path. This is the lever that guarantees a "vision" config with
    # empty image paths cannot silently train as text-only.
    first_example = dataset["train"][0] if len(dataset["train"]) else {}
    has_images = _example_has_images(first_example)
    logger.info("Dataset vision mode: %s", has_images)
    if config.get("require_vision", False) and not has_images:
        raise RuntimeError(
            "require_vision=True in config, but the first training example "
            "has no image blocks. Re-run scripts/040_prepare_training_data.py "
            "with --frames-dir pointing at extracted frames."
        )

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.get("num_epochs", 2),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 4),
        learning_rate=config.get("learning_rate", 1e-4),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        bf16=config.get("bf16", True),
        logging_steps=config.get("logging_steps", 10),
        save_strategy=config.get("save_strategy", "epoch"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit"),
        eval_strategy=config.get("eval_strategy", "epoch"),
        report_to="none",
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        packing=config.get("packing", False),
        dataset_text_field="",
        remove_unused_columns=False,
        max_seq_length=config.get("max_seq_length", 2048),
        run_name=_run_name(config, output_dir),
    )

    if has_images:
        # Unsloth ships UnslothVisionDataCollator, which expects a list of
        # examples each carrying "messages" with image blocks. It uses
        # the FastVisionModel processor (tokenizer arg) to encode both
        # vision and text into the batch tensors the Qwen2.5-VL backbone
        # expects.
        try:
            from unsloth.trainer import UnslothVisionDataCollator
        except ImportError as exc:  # pragma: no cover - runtime
            raise RuntimeError(
                "Vision training requires a recent Unsloth that exposes "
                "UnslothVisionDataCollator. Upgrade with:\n"
                "  pip install -U 'unsloth @ git+https://github.com/unslothai/unsloth.git'"
            ) from exc

        FastVisionModel.for_training(model)  # flip into train-ready state
        vision_collator = UnslothVisionDataCollator(model, tokenizer)

        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            data_collator=vision_collator,
        )
    else:
        trainer = SFTTrainer(
            model=model, args=sft_config,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            formatting_func=lambda batch: _format_chat_examples(batch, tokenizer),
        )

    logger.info("Starting Unsloth QLoRA training...")
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if _should_export_merged(config):
        merged_dir = str(Path(output_dir) / "merged_16bit")
        try:
            model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        except Exception as exc:
            logger.warning("Merged 16-bit export failed; keeping adapter checkpoint only: %s", exc)

    eval_results = trainer.evaluate()
    with open(Path(output_dir) / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    with open(Path(output_dir) / "training_config.yaml", "w") as f:
        yaml.dump(config, f)

    _maybe_finish_wandb(config, eval_results)

    logger.info(f"[unsloth] Done. Checkpoint: {output_dir}")
    return eval_results


def _finetune_hf(config: dict) -> dict:
    """Fallback: HuggingFace Trainer + PEFT with 4-bit quantization."""
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer,
            BitsAndBytesConfig, TrainingArguments, Trainer,
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install torch transformers peft bitsandbytes datasets")
        sys.exit(1)

    base_model = config.get("base_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    dataset_path = config["dataset_path"]
    output_dir = config.get("output_dir") or (
        f"memory/model_checkpoints/{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}_hf"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _configure_wandb_env(config, output_dir)

    quant_config = None
    if config.get("quantization") == "4bit":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quant_config,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2" if config.get("flash_attention_2") else None,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_r", 16), lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if config.get("gradient_checkpointing"):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={
        "train": str(Path(dataset_path) / "train.jsonl"),
        "validation": str(Path(dataset_path) / "val.jsonl"),
    })
    max_seq_length = min(config.get("max_seq_length", 4096), 2048)
    tokenized_dataset = dataset.map(
        lambda example: _tokenize_hf_example(example, tokenizer, max_seq_length),
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_epochs", 2),
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 4),
        learning_rate=config.get("learning_rate", 1e-4),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        bf16=config.get("bf16", True),
        logging_steps=config.get("logging_steps", 10),
        save_strategy=config.get("save_strategy", "epoch"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit"),
        evaluation_strategy=config.get("eval_strategy", "epoch"),
        dataloader_num_workers=config.get("dataloader_num_workers", 0),
        report_to="wandb" if _use_wandb(config) else "none",
        run_name=_run_name(config, output_dir),
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=tokenized_dataset["train"], eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )
    trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    eval_results = trainer.evaluate()
    with open(Path(output_dir) / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    with open(Path(output_dir) / "training_config.yaml", "w") as f:
        yaml.dump(config, f)

    _maybe_finish_wandb(config, eval_results)

    logger.info(f"[hf_trainer] Done. Checkpoint: {output_dir}")
    return eval_results


def finetune(config: dict) -> dict:
    framework = config.get("framework", "unsloth")
    if framework == "unsloth":
        return _finetune_unsloth(config)
    elif framework == "hf_trainer":
        return _finetune_hf(config)
    else:
        raise ValueError(f"Unknown framework: {framework}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune VLM on FLS scoring data")
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = load_config(args.config)
    config.setdefault("use_wandb", False)
    config.setdefault("wandb_project", "fls-training")
    if args.wandb:
        config["use_wandb"] = True
    if _use_wandb(config):
        config.setdefault("run_name", Path(config.get("output_dir") or "finetune_vlm").name)
    metrics = finetune(config)
    print(f"Eval metrics: {json.dumps(metrics, indent=2)}")
