#!/usr/bin/env python3
"""FLS Round 2 training — warm-start from LASANA-pretrained adapter, continue on
LASANA + YouTube SFT v1 combined.

Inputs (expected at /workspace):
  - lasana_train.jsonl, lasana_val.jsonl   (944 / 121 examples)
  - yt_train.jsonl, yt_val.jsonl           (full batch-2 YouTube SFT, ~85-115 train)
  - adapter_init/                          (round 1 adapter dir: adapter_model.safetensors + adapter_config.json)

Outputs:
  - /workspace/checkpoints_r2/final/       (round 2 adapter)
  - /workspace/checkpoints_r2/run_manifest.json
"""
import json, os, random
from pathlib import Path
from datetime import datetime

print(f"=== FLS Round 2 (LASANA + YouTube v1) — {datetime.now()} ===")

import torch
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f}GB)  |  PyTorch: {torch.__version__}")

os.system("pip install -q peft trl datasets bitsandbytes accelerate transformers 2>&1 | tail -5")

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

WS = Path("/workspace")
lasana_train = load_jsonl(WS / "lasana_train.jsonl")
lasana_val   = load_jsonl(WS / "lasana_val.jsonl")
yt_train     = load_jsonl(WS / "yt_train.jsonl")
yt_val       = load_jsonl(WS / "yt_val.jsonl")
print(f"LASANA:  {len(lasana_train)} train / {len(lasana_val)} val")
print(f"YouTube: {len(yt_train)} train / {len(yt_val)} val")

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

train_convos = [to_conversation(e) for e in lasana_train] + [to_conversation(e) for e in yt_train]
val_convos   = [to_conversation(e) for e in lasana_val]   + [to_conversation(e) for e in yt_val]
random.seed(42); random.shuffle(train_convos)
print(f"Combined: {len(train_convos)} train / {len(val_convos)} val")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

print("\nLoading Qwen2.5-7B-Instruct (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
base = prepare_model_for_kbit_training(base)

ADAPTER_INIT = WS / "adapter_init"
if ADAPTER_INIT.exists():
    print(f"Warm-starting from adapter at {ADAPTER_INIT}")
    model = PeftModel.from_pretrained(base, str(ADAPTER_INIT), is_trainable=True)
else:
    print("WARNING: adapter_init/ not found — training fresh LoRA.")
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_config)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

from trl import SFTTrainer, SFTConfig
from datasets import Dataset

train_ds = Dataset.from_list(train_convos)
val_ds = Dataset.from_list(val_convos)

def formatting_func(example):
    return tokenizer.apply_chat_template(example["messages"], tokenize=False)

# Lower LR on warm-start to preserve learned priors.
WARM_START = ADAPTER_INIT.exists()
lr = 1e-4 if WARM_START else 2e-4
epochs = 2 if WARM_START else 3

print(f"\n=== Round 2 training — {datetime.now()} ===")
print(f"Warm-start: {WARM_START} | Epochs: {epochs} | Batch: 4x4=16 | LR: {lr}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    args=SFTConfig(
        output_dir="/workspace/checkpoints_r2",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=lr,
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

out = Path("/workspace/checkpoints_r2/final")
model.save_pretrained(str(out))
tokenizer.save_pretrained(str(out))

manifest = {
    "run_id": f"fls_round2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "base_model": "Qwen2.5-7B-Instruct",
    "adapter": "LoRA r=16 (PEFT) — warm-start" if WARM_START else "LoRA r=16 (PEFT) — fresh",
    "init_from": str(ADAPTER_INIT) if WARM_START else None,
    "lasana_train": len(lasana_train),
    "lasana_val": len(lasana_val),
    "yt_train": len(yt_train),
    "yt_val": len(yt_val),
    "total_train": len(train_convos),
    "total_val": len(val_convos),
    "epochs": epochs,
    "lr": lr,
    "train_loss": result.training_loss,
    "gpu": gpu,
    "completed_at": datetime.now().isoformat(),
}
with open("/workspace/checkpoints_r2/run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(json.dumps(manifest, indent=2))
