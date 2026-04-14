#!/usr/bin/env python3
"""FLS Round 2 (revised) — fresh LoRA on Qwen2.5-VL-7B-Instruct, YouTube SFT only.

Why this supersedes 033_train_lora_round2.py:
  The round 1 adapter (memory/model_checkpoints/lasana_pretrain) is mode-collapsed:
  only 3 unique predicted FLS values across 205 held-out examples. Root cause:
    (a) Base was text-only Qwen2.5-7B-Instruct — image paths fed as tokens.
    (b) LASANA targets are z-score labels (1–5), schema-incompatible with v002 FLS.
  See READINESS_2026-04-14.md §4 for full analysis.

  This script fixes both: loads Qwen2.5-VL and real frames, trains only on the
  YouTube SFT v002-schema data produced by dual-teacher consensus.

Inputs (expected at /workspace):
  yt_train.jsonl, yt_val.jsonl, yt_test.jsonl     (from 030_prep_sft_data.py)
  frames/                                          (resolved frame dirs — see below)

Each jsonl row has schema:
  {
    "video_id": "yt_<id>",
    "task_id": "task{1..5}_*",
    "label_type": "consensus" | "single_high_conf",
    "consensus_fls": <float>,
    "consensus_conf": <float>,
    "target": { <v002-schema scoring JSON> },
    "frames": [<relative or absolute jpg paths>]  # optional; falls back to video_id
  }

Outputs:
  /workspace/checkpoints_vl/final/                  Round 2 VL adapter
  /workspace/checkpoints_vl/run_manifest.json       Hyperparams + metrics
"""
import json, os, random, sys
from pathlib import Path
from datetime import datetime

print(f"=== FLS Round 2 (Qwen2.5-VL, YouTube-only) — {datetime.now()} ===")

import torch
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f}GB)  |  PyTorch: {torch.__version__}")

os.system("pip install -q peft trl datasets bitsandbytes accelerate transformers qwen-vl-utils pillow 2>&1 | tail -5")

WS = Path("/workspace")
FRAMES_ROOT = WS / "frames"  # e.g. /workspace/frames/yt_<id>/frame_*.jpg
MAX_FRAMES = 8               # per example; balance between signal and seq-length

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

yt_train = load_jsonl(WS / "yt_train.jsonl")
yt_val   = load_jsonl(WS / "yt_val.jsonl")
yt_test_p = WS / "yt_test.jsonl"
yt_test  = load_jsonl(yt_test_p) if yt_test_p.exists() else []
print(f"YouTube: {len(yt_train)} train / {len(yt_val)} val / {len(yt_test)} test")
if len(yt_train) < 30:
    print("WARNING: fewer than 30 train examples — results will be noisy.")

SYSTEM_PROMPT = (
    "You are an expert FLS (Fundamentals of Laparoscopic Surgery) proctor AI. "
    "Given sampled video frames of an FLS task performance, analyze the technique "
    "and output a single strict-JSON ScoringResult matching the v002 universal "
    "scoring schema. Include: video_classification, task_id, frame_analyses, "
    "phase_timings, penalties, task_specific_assessments, score_components "
    "(with max_score, time_used, total_penalties, total_fls_score, formula_applied), "
    "confidence, confidence_rationale, technique_summary, strengths, "
    "improvement_suggestions. Output ONLY valid JSON — no prose, no markdown fences."
)

def resolve_frames(ex):
    """Return up to MAX_FRAMES absolute image paths for this example."""
    frames = ex.get("frames") or []
    if frames:
        paths = [str((WS / f).resolve()) if not os.path.isabs(f) else f for f in frames]
    else:
        d = FRAMES_ROOT / ex["video_id"]
        if d.exists():
            paths = sorted(str(p) for p in d.glob("*.jpg"))
        else:
            paths = []
    if len(paths) > MAX_FRAMES:
        step = len(paths) / MAX_FRAMES
        paths = [paths[int(i * step)] for i in range(MAX_FRAMES)]
    return paths

def to_vl_conversation(ex):
    frames = resolve_frames(ex)
    task = ex.get("task_id", "unknown")
    target = ex.get("target") or {}
    user_content = [{"type": "image", "image": p} for p in frames]
    user_content.append({
        "type": "text",
        "text": f"Score this FLS {task} performance. Return ONLY valid JSON per the v002 schema.",
    })
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": [{"type": "text",
                "text": json.dumps(target, indent=2, default=str)}]},
        ],
        "frame_count": len(frames),
    }

# Filter examples with no resolvable frames
def prep(split):
    out = [to_vl_conversation(ex) for ex in split]
    usable = [c for c in out if c["frame_count"] > 0]
    dropped = len(out) - len(usable)
    if dropped:
        print(f"  dropped {dropped} examples with no resolvable frames")
    return [{"messages": c["messages"]} for c in usable]

print("\nResolving frames...")
train_convos = prep(yt_train)
val_convos   = prep(yt_val)
random.seed(42); random.shuffle(train_convos)
print(f"Usable: {len(train_convos)} train / {len(val_convos)} val")

if len(train_convos) == 0:
    print("FATAL: no usable training examples. Ensure /workspace/frames/ is populated.")
    sys.exit(1)

from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("\nLoading Qwen2.5-VL-7B-Instruct (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # flash-attn NOT installed on RunPod image
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = prepare_model_for_kbit_training(model)

# Train LoRA only on LLM-side projections; keep vision tower frozen
# (cheaper and avoids destabilizing visual features on a small dataset)
lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

# Build datasets using the processor's chat template for VL inputs
from datasets import Dataset

def format_example(ex):
    """Apply Qwen2.5-VL chat template → pixel_values + input_ids."""
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(ex["messages"])
    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt",
    )
    return {k: v[0] for k, v in batch.items()}

train_ds = Dataset.from_list(train_convos)
val_ds = Dataset.from_list(val_convos)

# Custom collator — pre-tokenized, pad by pixel_values/input_ids separately
from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    items = [format_example(ex) for ex in batch]
    input_ids = pad_sequence([it["input_ids"] for it in items],
                             batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = pad_sequence([it["attention_mask"] for it in items],
                                  batch_first=True, padding_value=0)
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if "pixel_values" in items[0]:
        out["pixel_values"] = torch.cat([it["pixel_values"].unsqueeze(0) if it["pixel_values"].ndim == 3
                                         else it["pixel_values"] for it in items], dim=0)
    if "image_grid_thw" in items[0]:
        out["image_grid_thw"] = torch.cat([it["image_grid_thw"] for it in items], dim=0)
    return out

from transformers import Trainer, TrainingArguments

BATCH = 1
GRAD_ACCUM = 8
EPOCHS = 3
LR = 2e-4

args = TrainingArguments(
    output_dir="/workspace/checkpoints_vl",
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    bf16=True,
    report_to="none",
    seed=42,
    gradient_checkpointing=True,
    remove_unused_columns=False,
    dataloader_num_workers=2,
)

print(f"\n=== Training — {datetime.now()} ===")
print(f"Epochs={EPOCHS}  Batch={BATCH}x{GRAD_ACCUM}  LR={LR}  MaxFrames={MAX_FRAMES}")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate,
)

result = trainer.train()

print(f"\n=== Training complete — {datetime.now()} ===")
print(f"Train loss: {result.training_loss:.4f}")

out = Path("/workspace/checkpoints_vl/final")
model.save_pretrained(str(out))
processor.save_pretrained(str(out))

manifest = {
    "run_id": f"fls_round2_vl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "base_model": MODEL_ID,
    "adapter": "LoRA r=16 (fresh, no warm-start)",
    "data_source": "youtube_sft_v1 (dual-teacher consensus, v002 schema)",
    "yt_train": len(train_convos),
    "yt_val": len(val_convos),
    "yt_test": len(yt_test),
    "epochs": EPOCHS,
    "lr": LR,
    "max_frames_per_example": MAX_FRAMES,
    "train_loss": result.training_loss,
    "gpu": gpu,
    "completed_at": datetime.now().isoformat(),
    "supersedes": "031_train_lora_lasana.py (mode-collapsed)",
}
with open("/workspace/checkpoints_vl/run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
