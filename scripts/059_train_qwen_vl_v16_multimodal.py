#!/usr/bin/env python3
"""
v16 Multimodal LoRA fine-tuning for Qwen2.5-VL-7B-Instruct
All 6 FLS tasks, multimodal dataset with 8 sampled frames per video.
Uses processor for image+text inputs, custom data collator, label masking.

Run on RunPod H100: python3 /workspace/train_v16_lora.py
"""

import json
import os
import torch
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from qwen_vl_utils import process_vision_info

# ── Paths ──
DATA_DIR = "/workspace/v16_multimodal"
FRAMES_DIR = "/workspace/v16_frames"
OUTPUT_DIR = "/workspace/v16_lora_output"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config ──
MAX_LENGTH = 4096  # tokens — 8 images × ~256 tokens + text
IMAGE_MIN_PIXELS = 128 * 28 * 28  # ~100K pixels per image (reduced for training)
IMAGE_MAX_PIXELS = 256 * 28 * 28  # ~200K pixels per image


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def remap_image_paths(messages, frames_dir):
    """Remap file:// image paths to the local frames directory on RunPod."""
    for msg in messages:
        if isinstance(msg.get("content"), list):
            for item in msg["content"]:
                if item.get("type") == "image" and "image" in item:
                    # Original: file:///data/fls/frames/VIDEO_ID/frame_NNN.jpg
                    # or: file:///data/fls/lasana_processed/frames/SUBJECT/frame_NNNN.jpg
                    orig_path = item["image"].replace("file://", "")
                    # Extract just the last two path components: dir/filename
                    parts = orig_path.split("/")
                    # Use video_dir/frame_file as relative path
                    relative = f"{parts[-2]}/{parts[-1]}"
                    local_path = os.path.join(frames_dir, relative)
                    if os.path.exists(local_path):
                        item["image"] = f"file://{local_path}"
                    else:
                        # Try just the filename in a flat structure
                        flat_name = f"{parts[-2]}_{parts[-1]}"
                        flat_path = os.path.join(frames_dir, flat_name)
                        if os.path.exists(flat_path):
                            item["image"] = f"file://{flat_path}"
                        else:
                            # Mark as missing — will skip this image
                            item["_missing"] = True
    return messages


def filter_missing_images(messages):
    """Remove image entries that couldn't be found locally."""
    for msg in messages:
        if isinstance(msg.get("content"), list):
            msg["content"] = [
                item for item in msg["content"]
                if not item.get("_missing", False)
            ]
    return messages


class FLSMultimodalDataset(torch.utils.data.Dataset):
    """Dataset that processes multimodal examples for Qwen2.5-VL training."""

    def __init__(self, jsonl_path, processor, frames_dir, max_length=MAX_LENGTH):
        self.raw_examples = load_jsonl(jsonl_path)
        self.processor = processor
        self.frames_dir = frames_dir
        self.max_length = max_length

        # Pre-validate: filter examples where we can load at least some images
        self.examples = []
        skipped = 0
        for ex in self.raw_examples:
            messages = remap_image_paths(
                json.loads(json.dumps(ex["messages"])),  # deep copy
                self.frames_dir,
            )
            messages = filter_missing_images(messages)
            self.examples.append({"messages": messages, "meta": ex})
        print(f"  Dataset: {len(self.examples)} examples loaded, {skipped} skipped")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        messages = ex["messages"]

        # Apply chat template WITHOUT generation prompt (training mode)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Extract images using qwen_vl_utils
        try:
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception:
            # If image loading fails, fall back to text-only
            image_inputs, video_inputs = None, None

        # Process with the multimodal processor
        if image_inputs:
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
        else:
            inputs = self.processor.tokenizer(
                text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )

        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Create labels: mask everything except assistant response
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # Find assistant token positions — mask everything before the last assistant turn
        # Qwen2.5 uses: <|im_start|>assistant\n ... <|im_end|>
        # We want to only compute loss on assistant tokens
        im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # Find the last occurrence of <|im_start|>assistant pattern
        # Simple approach: find all <|im_start|> positions, the last one before <|im_end|> is assistant
        input_ids_list = input_ids.tolist()

        # Find assistant start: look for the pattern after the last <|im_start|>
        assistant_start = None
        for i in range(len(input_ids_list) - 1, -1, -1):
            if input_ids_list[i] == im_start_id:
                # Check if next tokens spell "assistant"
                assistant_start = i
                break

        if assistant_start is not None:
            # Mask everything up to and including the "assistant\n" header
            # Find the newline after "assistant"
            header_end = assistant_start
            for j in range(assistant_start, min(assistant_start + 10, len(input_ids_list))):
                # Look for the newline token after "assistant"
                token_str = self.processor.tokenizer.decode([input_ids_list[j]])
                if "\n" in token_str and j > assistant_start:
                    header_end = j + 1
                    break

            # Mask [0, header_end) with -100
            labels[:header_end] = -100
        else:
            # Can't find assistant — mask everything (won't contribute to loss)
            labels[:] = -100

        inputs["labels"] = labels
        return inputs


@dataclass
class MultimodalDataCollator:
    """Collate multimodal examples with proper padding."""

    processor: Any
    max_length: int = MAX_LENGTH

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # Separate different tensor types
        batch = {}

        # Pad input_ids, attention_mask, labels
        pad_token_id = self.processor.tokenizer.pad_token_id or 0
        max_len = min(
            max(f["input_ids"].shape[0] for f in features),
            self.max_length,
        )

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for f in features:
            ids = f["input_ids"][:max_len]
            mask = f.get("attention_mask", torch.ones_like(ids))[:max_len]
            labs = f["labels"][:max_len]

            # Pad to max_len
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=ids.dtype)])
                mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
            labels_list.append(labs)

        batch["input_ids"] = torch.stack(input_ids_list)
        batch["attention_mask"] = torch.stack(attention_mask_list)
        batch["labels"] = torch.stack(labels_list)

        # Handle pixel_values and image_grid_thw (variable size per example)
        if "pixel_values" in features[0]:
            # Concatenate all pixel values
            all_pixels = []
            all_grid_thw = []
            for f in features:
                if "pixel_values" in f and f["pixel_values"] is not None:
                    pv = f["pixel_values"]
                    if pv.dim() == 1:
                        pv = pv.unsqueeze(0)
                    all_pixels.append(pv)
                if "image_grid_thw" in f and f["image_grid_thw"] is not None:
                    gt = f["image_grid_thw"]
                    if gt.dim() == 1:
                        gt = gt.unsqueeze(0)
                    all_grid_thw.append(gt)

            if all_pixels:
                batch["pixel_values"] = torch.cat(all_pixels, dim=0)
            if all_grid_thw:
                batch["image_grid_thw"] = torch.cat(all_grid_thw, dim=0)

        return batch


def main():
    print(f"[{datetime.now()}] Starting v16 multimodal LoRA training")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Frames dir: {FRAMES_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")

    # ── Load processor ──
    print(f"\n[{datetime.now()}] Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        min_pixels=IMAGE_MIN_PIXELS,
        max_pixels=IMAGE_MAX_PIXELS,
    )
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    # ── Load datasets ──
    print(f"\n[{datetime.now()}] Loading datasets...")
    train_dataset = FLSMultimodalDataset(
        os.path.join(DATA_DIR, "train.jsonl"),
        processor,
        FRAMES_DIR,
        max_length=MAX_LENGTH,
    )
    val_dataset = FLSMultimodalDataset(
        os.path.join(DATA_DIR, "val.jsonl"),
        processor,
        FRAMES_DIR,
        max_length=MAX_LENGTH,
    )
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Load model with 4-bit quantization ──
    print(f"\n[{datetime.now()}] Loading model {MODEL_ID}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # ── LoRA config ──
    print(f"\n[{datetime.now()}] Applying LoRA adapter...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # Enable input gradients for gradient checkpointing with quantized model
    model.enable_input_require_grads()

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training config ──
    print(f"\n[{datetime.now()}] Setting up training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,  # smaller due to images
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # effective batch = 16
        learning_rate=1e-4,  # slightly lower than v15 for multimodal
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        dataloader_num_workers=4,
        remove_unused_columns=False,  # IMPORTANT: we have custom columns
        seed=42,
        dataloader_pin_memory=True,
    )

    data_collator = MultimodalDataCollator(
        processor=processor,
        max_length=MAX_LENGTH,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # ── Train ──
    print(f"\n[{datetime.now()}] Starting training...")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Batch size: {training_args.per_device_train_batch_size}")
    print(f"  Grad accum: {training_args.gradient_accumulation_steps}")
    print(f"  Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  LR: {training_args.learning_rate}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Image pixels: {IMAGE_MIN_PIXELS}-{IMAGE_MAX_PIXELS}")
    print(f"  LoRA r={lora_config.r}, alpha={lora_config.lora_alpha}")

    result = trainer.train()

    print(f"\n[{datetime.now()}] Training complete!")
    print(f"  Train loss: {result.training_loss:.4f}")
    print(f"  Train steps: {result.global_step}")

    # ── Save ──
    print(f"\n[{datetime.now()}] Saving adapter...")
    adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    trainer.save_model(adapter_dir)
    processor.tokenizer.save_pretrained(adapter_dir)

    # ── Final eval ──
    print(f"\n[{datetime.now()}] Running final eval...")
    eval_result = trainer.evaluate()
    print(f"  Eval loss: {eval_result['eval_loss']:.4f}")

    # Save metrics
    metrics = {
        "version": "v16",
        "model": MODEL_ID,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "train_loss": result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "train_steps": result.global_step,
        "epochs": int(training_args.num_train_epochs),
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "max_length": MAX_LENGTH,
        "image_min_pixels": IMAGE_MIN_PIXELS,
        "image_max_pixels": IMAGE_MAX_PIXELS,
        "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        "completed_at": datetime.now().isoformat(),
    }

    with open(os.path.join(adapter_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"v16 Multimodal LoRA training COMPLETE")
    print(f"Adapter: {adapter_dir}")
    print(f"Train loss: {result.training_loss:.4f}")
    print(f"Eval loss:  {eval_result['eval_loss']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
