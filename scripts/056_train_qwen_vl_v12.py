#!/usr/bin/env python3
"""FLS Round 2 v12 — Qwen2.5-VL LoRA + visual.merger unfrozen.

Changes from v11:
  - Adds V32-V37 intracorporeal suturing gold examples (+6 task5 rows)
  - Adds 11 rings-of-rings task6 examples (new task, pgy4 pre-practice)
  - SYSTEM_PROMPT updated to include task6_rings_needle_manipulation
  - dataset: yt_train_v12.jsonl / yt_val_v12.jsonl / yt_test_v12.jsonl
    (430 train / 53 val / 52 test vs 417/51/50 in v11)

Inherited from v11 (unchanged):
  - visual.merger module unfrozen → 139M trainable params (1.66%)
  - assistant-only loss masking
  - LoRA r=32, EPOCHS=5, LR=1e-4 cosine, BATCH=1x4
"""
import json, os, random, sys
from pathlib import Path
from datetime import datetime

print(f"=== FLS Round 2 v12 (Qwen2.5-VL + task6) — {datetime.now()} ===")

import torch
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f}GB)  |  PyTorch: {torch.__version__}")

os.system("pip install -q peft trl datasets bitsandbytes accelerate transformers qwen-vl-utils pillow 2>&1 | tail -3")

WS             = Path("/workspace")
FRAMES_YT      = WS / "frames"
FRAMES_LASANA  = WS / "lasana_frames"
MAX_FRAMES     = 8
MIN_FRAMES     = 4
CHECKPOINT_DIR = "/workspace/checkpoints_vl_v12"

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

yt_train = load_jsonl(WS / "yt_train_v12.jsonl")
yt_val   = load_jsonl(WS / "yt_val_v12.jsonl")
yt_test_p = WS / "yt_test_v12.jsonl"
yt_test  = load_jsonl(yt_test_p) if yt_test_p.exists() else []
print(f"Pool (raw): {len(yt_train)} train / {len(yt_val)} val / {len(yt_test)} test")

for split_name, split in [("train", yt_train), ("val", yt_val)]:
    bad = sum(1 for r in split if not isinstance(
        (r.get("target") or {}).get("score_components"), dict))
    if bad:
        print(f"FATAL: {split_name} has {bad} rows with non-dict score_components")
        sys.exit(1)

from collections import Counter
task_dist = Counter(r.get("task_id","?") for r in yt_train)
print(f"Task distribution (train): {dict(task_dist)}")

SCHEMA_EXAMPLE = {
    "task_id": "task5_intracorporeal_suturing",
    "task_name": "Intracorporeal Suture with Knot Tying",
    "completion_time_seconds": 104.0,
    "score_components": {
        "max_score": 600, "time_used": 104.0, "total_penalties": 1.0,
        "total_fls_score": 495.0,
        "formula_applied": "600 - 104.0 - 1.0 = 495.0"
    },
    "confidence": 0.72,
    "technique_summary": "Competent performance with clean knot sequence.",
    "strengths": ["Consistent hand switching", "Clean needle passage"],
    "improvement_suggestions": ["Reduce needle loading time"],
    "penalties": [],
    "estimated_fls_score": 495.0
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
    "task4_extracorporeal_knot, task5_intracorporeal_suturing, "
    "task6_rings_needle_manipulation.\n\n"
    "For task6_rings_needle_manipulation: Score = 315 - completion_time - (20 * rings_missed). "
    "Auto-fail (score=0) if needle exits field of view or block is dislodged.\n\n"
    "Exact shape example (values are illustrative):\n" +
    json.dumps(SCHEMA_EXAMPLE, indent=2)
)


def resolve_frames(ex):
    frames = ex.get("frames") or []
    paths = []
    for f in frames:
        if not f: continue
        if os.path.isabs(f) and os.path.exists(f):
            paths.append(f); continue
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
        if c["frame_count"] < MIN_FRAMES: dropped_frames += 1; continue
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

# Pixel-variance sanity: frames must differ
print("\n[pixel-sanity] Inspecting pixel variance across two examples...")
_f0 = resolve_frames(yt_train[0]); _f1 = resolve_frames(yt_train[1])
if _f0 and _f1:
    from PIL import Image; import numpy as np
    _arr0 = np.array(Image.open(_f0[0]).convert("RGB")).mean()
    _arr1 = np.array(Image.open(_f1[0]).convert("RGB")).mean()
    print(f"  mean|pv0-pv1|={abs(_arr0-_arr1):.4f}  (>1e-3 means frames are distinct)")

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
    r=32, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ── Unfreeze visual.merger (same as v8–v11) ────────────────────────────
merger_params = []
for name, param in model.named_parameters():
    if "visual.merger" in name:
        param.requires_grad_(True)
        param.data = param.data.to(torch.bfloat16)
        merger_params.append(name)
print(f"\n[v8] Trainable visual.merger params: "
      f"{sum(p.numel() for n,p in model.named_parameters() if 'visual.merger' in n):,}  "
      f"dtypes: {set(str(p.dtype) for n,p in model.named_parameters() if 'visual.merger' in n)}")
if merger_params:
    print("[v8] Sample trainable merger param names:")
    for n in merger_params[:5]: print(f"      {n}")

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

from torch.utils.data import Dataset as _TorchDataset

class ConvoDataset(_TorchDataset):
    def __init__(self, convos): self.convos = convos
    def __len__(self): return len(self.convos)
    def __getitem__(self, i): return self.convos[i]


def _load_images(messages):
    from PIL import Image
    images = []
    MAX_DIM = 448
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for item in c:
                if isinstance(item, dict) and item.get("type") == "image":
                    p = item.get("image")
                    if p and isinstance(p, str):
                        img = Image.open(p).convert("RGB")
                        img.thumbnail((MAX_DIM, MAX_DIM))
                        images.append(img)
    return images


def format_example(ex):
    msgs = ex["messages"]
    prompt_msgs = msgs[:-1]
    images = _load_images(msgs)
    full_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    full_batch = processor(text=[full_text], images=images, videos=None,
                           padding=False, truncation=False, return_tensors="pt")
    prompt_batch = processor(text=[prompt_text], images=images, videos=None,
                             padding=False, truncation=False, return_tensors="pt")
    prompt_len = int(prompt_batch["input_ids"].shape[1])
    out = {k: (v[0] if k in ("input_ids", "attention_mask") else v)
           for k, v in full_batch.items()}
    out["prompt_len"] = prompt_len
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
    for i, it in enumerate(items):
        labels[i, :it["prompt_len"]] = -100
    labels[labels == processor.tokenizer.pad_token_id] = -100
    out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if "pixel_values" in items[0]:
        out["pixel_values"] = torch.cat([it["pixel_values"].unsqueeze(0) if it["pixel_values"].ndim == 3
                                         else it["pixel_values"] for it in items], dim=0)
    if "image_grid_thw" in items[0]:
        out["image_grid_thw"] = torch.cat([it["image_grid_thw"] for it in items], dim=0)
    return out

print("\n[mask-sanity] Inspecting first training example...")
_s = format_example(train_convos[0])
_total = int(_s["input_ids"].shape[0])
_sup = _total - _s["prompt_len"]
print(f"  total_tokens={_total}  prompt_tokens={_s['prompt_len']}  supervised_tokens={_sup}")
if _sup < 20:
    print("FATAL: fewer than 20 supervised tokens — mask is wrong."); sys.exit(1)

from transformers import Trainer, TrainingArguments

BATCH = 1; GRAD_ACCUM = 4; EPOCHS = 5; LR = 1e-4
LR_KWARGS = {"lr_scheduler_type": "cosine"}

n_steps = (len(train_convos) // (BATCH * GRAD_ACCUM)) * EPOCHS
print(f"\n=== Training v12 — {datetime.now()} ===")
print(f"Epochs={EPOCHS}  Batch={BATCH}x{GRAD_ACCUM}  LR={LR}  "
      f"Examples={len(train_convos)}  MaxFrames={MAX_FRAMES}  LossMask=assistant-only  Merger=unfrozen")
print(f"Total steps: {n_steps}")

args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=BATCH, per_device_eval_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS, learning_rate=LR, warmup_ratio=0.05, max_grad_norm=1.0,
    logging_steps=10, eval_strategy="steps", eval_steps=200,
    save_strategy="steps", save_steps=400, save_total_limit=3,
    bf16=True, report_to="none", seed=42,
    gradient_checkpointing=True, remove_unused_columns=False,
    dataloader_num_workers=0, **LR_KWARGS,
)

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collate)
result = trainer.train()

print(f"\n=== Training complete — {datetime.now()} ===")
print(f"Train loss: {result.training_loss:.4f}")

out = Path(CHECKPOINT_DIR) / "final"
model.save_pretrained(str(out)); processor.save_pretrained(str(out))

manifest = {
    "run_id": f"fls_round2_vl_v12_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "base_model": MODEL_ID, "adapter": "LoRA r=32 + visual.merger unfrozen",
    "data_source": "youtube_sft_v12 = v11_base(417) + V32-V37(6 task5) + rings_task6(11)",
    "yt_train": len(train_convos), "yt_val": len(val_convos), "yt_test": len(yt_test),
    "epochs": EPOCHS, "lr": LR, "lr_schedule": "cosine",
    "max_frames_per_example": MAX_FRAMES,
    "loss_masking": "assistant-only (labels[:prompt_len] = -100 + pad masked)",
    "new_in_v12": {
        "task5_new_examples": "V32-V37 intracorporeal suturing gold scores",
        "task6_new_task": "rings_needle_manipulation (11 pgy4 pre-practice videos, score=315-t-20*missed)",
    },
    "train_loss": result.training_loss, "gpu": gpu,
    "completed_at": datetime.now().isoformat(),
    "supersedes": "v11 (417 train, no task6 data)",
}
with open(Path(CHECKPOINT_DIR) / "run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest saved. Adapter at {out}")
