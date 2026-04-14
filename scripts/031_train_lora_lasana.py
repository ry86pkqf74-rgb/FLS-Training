#!/usr/bin/env python3
"""FLS Pre-training v2 — uses PEFT directly instead of Unsloth to avoid compat issues."""
import json, sys, os, random
from pathlib import Path
from datetime import datetime

print(f"=== FLS Pre-training v2 — {datetime.now()} ===")

import torch
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f}GB)")
print(f"PyTorch: {torch.__version__}")

# Install deps
os.system("pip install -q peft trl datasets bitsandbytes accelerate transformers 2>&1 | tail -5")

# Load data
def load_jsonl(path):
    return [json.loads(l) for l in open(path)]

lasana_train = load_jsonl("/workspace/lasana_train.jsonl")
lasana_val = load_jsonl("/workspace/lasana_val.jsonl")
yt_train = load_jsonl("/workspace/yt_train.jsonl")
yt_val = load_jsonl("/workspace/yt_val.jsonl")

print(f"LASANA: {len(lasana_train)} train, {len(lasana_val)} val")
print(f"YouTube: {len(yt_train)} train, {len(yt_val)} val")

SYSTEM_PROMPT = (
    "You are an expert FLS (Fundamentals of Laparoscopic Surgery) proctor AI. "
    "Given video frames of a surgical training performance, analyze the technique "
    "and output a structured JSON scoring result including estimated_fls_score, "
    "completion_time_seconds, confidence, score_components, technique_summary, "
    "strengths, and improvement_suggestions."
)

def to_conversation(ex):
    target = ex.get("target", ex)
    task = ex.get("task_id", "unknown")
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Score this FLS {task} video performance. Return ONLY valid JSON."},
            {"role": "assistant", "content": json.dumps(target, indent=2, default=str)},
        ]
    }

train_convos = [to_conversation(ex) for ex in lasana_train] + [to_conversation(ex) for ex in yt_train]
val_convos = [to_conversation(ex) for ex in lasana_val] + [to_conversation(ex) for ex in yt_val]
random.seed(42)
random.shuffle(train_convos)

print(f"Combined: {len(train_convos)} train, {len(val_convos)} val")

# Load model with 4-bit quantization
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("\nLoading Qwen2.5-7B-Instruct (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {trainable:,} trainable / {total_params:,} total ({100*trainable/total_params:.2f}%)")

from trl import SFTTrainer, SFTConfig
from datasets import Dataset

train_ds = Dataset.from_list(train_convos)
val_ds = Dataset.from_list(val_convos)

def formatting_func(example):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)

total_steps = len(train_convos) * 3 // (4 * 4)
print(f"\n=== Starting training — {datetime.now()} ===")
print(f"Epochs: 3 | Batch: 4x4=16 | Steps: ~{total_steps} | LR: 2e-4")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=SFTConfig(
        output_dir="/workspace/checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        bf16=True,
        max_seq_length=2048,
        report_to="none",
        seed=42,
        gradient_checkpointing=True,
    ),
    formatting_func=formatting_func,
)

result = trainer.train()

print(f"\n=== Training complete — {datetime.now()} ===")
print(f"Train loss: {result.training_loss:.4f}")

model.save_pretrained("/workspace/checkpoints/final")
tokenizer.save_pretrained("/workspace/checkpoints/final")

manifest = {
    "run_id": f"fls_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "base_model": "Qwen2.5-7B-Instruct",
    "adapter": "LoRA r=16 (PEFT)",
    "total_train": len(train_convos),
    "total_val": len(val_convos),
    "epochs": 3,
    "train_loss": result.training_loss,
    "gpu": gpu,
    "completed_at": datetime.now().isoformat(),
}
with open("/workspace/checkpoints/run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
