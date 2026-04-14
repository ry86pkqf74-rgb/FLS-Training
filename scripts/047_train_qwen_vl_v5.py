#!/usr/bin/env python3
"""FLS Round 2 v4 — Qwen2.5-VL LoRA trained on merged v002 pool (~1000 examples).

Supersedes 043_train_qwen_vl_v3.py (which used only 45 examples). The core
finding across v1→v3 was the same bottleneck: not enough schema-consistent
training data. v4 unlocks that by training on the merged output of
044_expand_sft_pool.py, which combines:
  - LASANA rescore (~940 videos, /data/fls/scored/lasana_v002/)
  - YouTube v4 pool (103 rows, old-schema converted to v002)
  - Gold set (17 rows, old-schema converted to v002)

Changes from v3:
  * Default data paths point at youtube_sft_v2/ (the merged pool)
  * EPOCHS back down to 5 (with ~800 training examples × 5 = 4000 optimizer
    steps at BATCH=1,GA=4, which is 6-7x what v3 had)
  * Everything else preserved from v3: cosine_with_min_lr (min 10%), schema-
    anchored system prompt, preflight schema check, skip-bad-schema filter.

Inputs expected on pod (at WS = /workspace):
  yt_train.jsonl, yt_val.jsonl, yt_test.jsonl  — emitted by 044
  frames/              — YT frame dirs (optional; falls through to "ex[frames]")
  lasana_frames/       — symlink or bindmount to /data/fls/lasana_processed/frames

Launch:
  python3 scripts/045_train_qwen_vl_v4.py
"""
import json, os, random, sys
from pathlib import Path
from datetime import datetime

print(f"=== FLS Round 2 v4 (Qwen2.5-VL, merged pool) — {datetime.now()} ===")

import torch
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f}GB)  |  PyTorch: {torch.__version__}")

os.system("pip install -q peft trl datasets bitsandbytes accelerate transformers qwen-vl-utils pillow 2>&1 | tail -5")

WS = Path("/workspace")
FRAMES_YT      = WS / "frames"
FRAMES_LASANA  = WS / "lasana_frames"
MAX_FRAMES     = 8
CHECKPOINT_DIR = "/workspace/checkpoints_vl_v5"

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

yt_train = load_jsonl(WS / "yt_train.jsonl")
yt_val   = load_jsonl(WS / "yt_val.jsonl")
yt_test_p = WS / "yt_test.jsonl"
yt_test  = load_jsonl(yt_test_p) if yt_test_p.exists() else []
print(f"Pool (raw): {len(yt_train)} train / {len(yt_val)} val / {len(yt_test)} test")

# Preflight: every row MUST have score_components as dict.
for split_name, split in [("train", yt_train), ("val", yt_val)]:
    bad = sum(1 for r in split if not isinstance(
        (r.get("target") or {}).get("score_components"), dict))
    if bad:
        print(f"FATAL: {split_name} has {bad} rows with non-dict score_components")
        sys.exit(1)

SCHEMA_EXAMPLE = {
    "task_id": "task1_peg_transfer",
    "task_name": "Peg Transfer",
    "completion_time_seconds": 48.0,
    "score_components": {
        "max_score": 300, "time_used": 48.0, "total_penalties": 5.0,
        "total_fls_score": 247.0,
        "formula_applied": "300 - 48.0 - 5.0 = 247.0"
    },
    "confidence": 0.85,
    "technique_summary": "…",
    "strengths": ["…"],
    "improvement_suggestions": ["…"],
    "penalties": [],
    "estimated_fls_score": 247.0
}

SYSTEM_PROMPT = (
    "You are an expert FLS (Fundamentals of Laparoscopic Surgery) proctor AI. "
    "Given sampled video frames of an FLS task performance, analyze the technique "
    "and output a single strict-JSON ScoringResult matching the v002 universal "
    "scoring schema. Output ONLY valid JSON — no prose, no markdown fences.\n\n"
    "CRITICAL schema notes:\n"
    "  - score_components MUST be an OBJECT (not an array) with keys: "
    "max_score, time_used, total_penalties, total_fls_score, formula_applied.\n"
    "  - estimated_fls_score is a top-level numeric field equal to total_fls_score.\n"
    "  - task_id is one of: task1_peg_transfer, task2_pattern_cut, task3_endoloop, "
    "task4_extracorporeal_knot, task5_intracorporeal_suturing.\n\n"
    "Exact shape example (values are illustrative):\n" +
    json.dumps(SCHEMA_EXAMPLE, indent=2)
)

def resolve_frames(ex):
    frames = ex.get("frames") or []
    paths = []
    for f in frames:
        if not f: continue
        # absolute paths (e.g. from 044 output): use as-is if exists
        if os.path.isabs(f) and os.path.exists(f):
            paths.append(f); continue
        # relative: try WS, FRAMES_YT, FRAMES_LASANA
        for root in (WS, FRAMES_YT, FRAMES_LASANA):
            p = str((root / f).resolve())
            if os.path.exists(p): paths.append(p); break
    if not paths:
        vid = ex.get("video_id", "")
        for candidate in (FRAMES_LASANA / vid, FRAMES_YT / vid,
                          FRAMES_YT / vid.replace("yt_", "", 1)):
            if candidate.exists():
                paths = sorted(str(p) for p in candidate.glob("*.jpg"))
                break
    if len(paths) > MAX_FRAMES:
        step = len(paths) / MAX_FRAMES
        paths = [paths[int(i * step)] for i in range(MAX_FRAMES)]
    return paths

def to_vl_conversation(ex):
    task = ex.get("task_id") or ""
    if not task or task == "unclassified": return None
    frames = resolve_frames(ex)
    target = ex.get("target") or {}
    sc = target.get("score_components")
    if not isinstance(sc, dict): return None
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

def prep(split, name):
    dropped_schema = dropped_frames = 0
    out = []
    for ex in split:
        c = to_vl_conversation(ex)
        if c is None: dropped_schema += 1; continue
        if c["frame_count"] == 0: dropped_frames += 1; continue
        out.append({"messages": c["messages"]})
    print(f"  [{name}] kept {len(out)}; dropped {dropped_schema} (bad schema) + "
          f"{dropped_frames} (no frames)")
    return out

print("\nResolving frames + filtering schema...")
train_convos = prep(yt_train, "train")
val_convos   = prep(yt_val,   "val")
random.seed(42); random.shuffle(train_convos)
print(f"Usable: {len(train_convos)} train / {len(val_convos)} val")

if len(train_convos) == 0:
    print("FATAL: no usable training examples."); sys.exit(1)

from transformers import AutoProcessor, BitsAndBytesConfig
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("\nLoading Qwen2.5-VL-7B-Instruct (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
model = VLModel.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto",
    torch_dtype=torch.bfloat16, attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

from torch.utils.data import Dataset as _TorchDataset

class ConvoDataset(_TorchDataset):
    def __init__(self, convos): self.convos = convos
    def __len__(self): return len(self.convos)
    def __getitem__(self, i): return self.convos[i]

def format_example(ex):
    import copy as _copy
    from PIL import Image
    msgs_for_template = _copy.deepcopy(ex["messages"])
    text = processor.apply_chat_template(msgs_for_template, tokenize=False, add_generation_prompt=False)
    image_inputs = []
    MAX_DIM = 448
    for m in ex["messages"]:
        c = m.get("content")
        if isinstance(c, list):
            for item in c:
                if isinstance(item, dict) and item.get("type") == "image":
                    p = item.get("image")
                    if p and isinstance(p, str):
                        img = Image.open(p).convert("RGB")
                        img.thumbnail((MAX_DIM, MAX_DIM))
                        image_inputs.append(img)
    batch = processor(text=[text], images=image_inputs, videos=None,
                      padding=True, truncation=False, return_tensors="pt")
    out = {}
    for k, v in batch.items():
        out[k] = v[0] if k in ("input_ids", "attention_mask") else v
    return out

train_ds = ConvoDataset(train_convos); val_ds = ConvoDataset(val_convos)

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

BATCH = 1; GRAD_ACCUM = 4; EPOCHS = 5; LR = 3e-4
LR_KWARGS = {"lr_scheduler_type": "cosine_with_min_lr",
             "lr_scheduler_kwargs": {"min_lr_rate": 0.1}}

args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS, learning_rate=LR, warmup_ratio=0.05,
    logging_steps=20, eval_strategy="steps", eval_steps=200,
    save_strategy="steps", save_steps=400, save_total_limit=3,
    bf16=True, report_to="none", seed=42,
    gradient_checkpointing=True, remove_unused_columns=False,
    dataloader_num_workers=0, **LR_KWARGS,
)

print(f"\n=== Training v4 — {datetime.now()} ===")
print(f"Epochs={EPOCHS}  Batch={BATCH}x{GRAD_ACCUM}  LR={LR} (min 10%)  "
      f"Examples={len(train_convos)}  MaxFrames={MAX_FRAMES}")

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collate)
result = trainer.train()

print(f"\n=== Training complete — {datetime.now()} ===")
print(f"Train loss: {result.training_loss:.4f}")

out = Path(CHECKPOINT_DIR) / "final"
model.save_pretrained(str(out)); processor.save_pretrained(str(out))

manifest = {
    "run_id": f"fls_round2_vl_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "base_model": MODEL_ID, "adapter": "LoRA r=16 (fresh)",
    "data_source": "youtube_sft_v2 = v4 + gold + LASANA rescore (all v002 schema)",
    "yt_train": len(train_convos), "yt_val": len(val_convos), "yt_test": len(yt_test),
    "epochs": EPOCHS, "lr": LR, "lr_schedule": "cosine_with_min_lr (min 10%)",
    "max_frames_per_example": MAX_FRAMES,
    "train_loss": result.training_loss, "gpu": gpu,
    "completed_at": datetime.now().isoformat(),
    "supersedes": "043_train_qwen_vl_v3.py (v3 had only 45 examples)",
}
with open(Path(CHECKPOINT_DIR) / "run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
