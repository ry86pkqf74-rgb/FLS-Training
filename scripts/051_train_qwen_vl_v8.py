#!/usr/bin/env python3
"""FLS Round 2 v8 — Qwen2.5-VL LoRA + vision-merger unfrozen.

v7 recap: loss-masking fix (assistant-only labels) landed — training loss
descended cleanly from 10 → 0.15. But the eval gate still failed with 0.00%
unique_prediction_ratio (mode collapse): every test video produced a
training-set JSON row echo, independent of the input frames. See
memory/model_checkpoints/round2_vl_v7/DIAGNOSIS.md.

Root cause diagnosed in v7 DIAGNOSIS: LoRA target_modules covered only the
LM transformer blocks (q/k/v/o/gate/up/down_proj). The vision→LM merger
MLP was frozen, so the adapted LM received visual features from the
pretrained projection unchanged and had no gradient pressure to condition
on them; it learned the highest-likelihood continuation given the textual
prompt alone (memorized training-row boilerplate).

v8 primary change: unfreeze `visual.merger` (the two-layer MLP that
projects ViT features into the LM input token space). Keep LM LoRA
unchanged. Add a preflight assert that at least one merger param has
requires_grad=True so we fail fast if the module name is wrong.

Kept from v7:
  LR=1e-4 cosine, LoRA r=32, grad_clip=1.0, EPOCHS=5, MIN_FRAMES=4,
  assistant-only loss mask, preflight supervised-token check.
"""
import json, os, random, sys
from pathlib import Path
from datetime import datetime

print(f"=== FLS Round 2 v8 (Qwen2.5-VL, LoRA + visual.merger unfrozen) — {datetime.now()} ===")

import torch
import torch.nn as _nn
# nn.Module.set_submodule was added in torch 2.5; the pod ships torch 2.4.
# Newer `transformers` bnb 4-bit skip-modules path calls it — monkey-patch
# a compatible shim so we don't need to force a torch upgrade.
if not hasattr(_nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: _nn.Module):
        if target == "":
            raise ValueError("set_submodule: empty target")
        parts = target.split(".")
        parent = self.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else self
        setattr(parent, parts[-1], module)
    _nn.Module.set_submodule = _set_submodule
    print("[compat] Installed nn.Module.set_submodule shim (torch<2.5).")
# peft 0.19 iterates all dtype names (including float8_e8m0fnu added in
# torch 2.6) via getattr(torch, name). On torch 2.4 this raises AttributeError.
# Attach sentinel dtypes so getattr succeeds and the iteration just skips them.
for _dt in ("float8_e8m0fnu", "float8_e4m3fnuz", "float8_e5m2fnuz"):
    if not hasattr(torch, _dt):
        setattr(torch, _dt, None)
gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {gpu} ({vram:.1f}GB)  |  PyTorch: {torch.__version__}")

os.system("pip install -q peft trl datasets bitsandbytes accelerate transformers qwen-vl-utils pillow 2>&1 | tail -5")

WS = Path("/workspace")
FRAMES_YT      = WS / "frames"
FRAMES_LASANA  = WS / "lasana_frames"
MAX_FRAMES     = 8
MIN_FRAMES     = 4
CHECKPOINT_DIR = "/workspace/checkpoints_vl_v8"

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

yt_train = load_jsonl(WS / "yt_train.jsonl")
yt_val   = load_jsonl(WS / "yt_val.jsonl")
yt_test_p = WS / "yt_test.jsonl"
yt_test  = load_jsonl(yt_test_p) if yt_test_p.exists() else []
print(f"Pool (raw): {len(yt_train)} train / {len(yt_val)} val / {len(yt_test)} test")

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

from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    from transformers import Qwen2VLForConditionalGeneration as VLModel
from peft import LoraConfig, get_peft_model

print("\nLoading Qwen2.5-VL-7B-Instruct (full bf16, no 4-bit)...")
# v8.1: dropping 4-bit quantization. Rationale: llm_int8_skip_modules=["visual"]
# did NOT keep visual.merger.mlp.[0,2] as nn.Linear in this transformers build
# — bnb still replaced them with Linear4bit (uint8 weight), which makes
# PEFT's modules_to_save=["merger"] crash in requires_grad_ on uint8 tensors.
# Qwen2.5-VL-7B in bf16 (~14 GB) + LoRA adapters (~100 MB) + trainable merger
# (~45 M params * 4B grad * 4B Adam states ≈ 540 MB) + activations with
# gradient checkpointing fits comfortably in the H100's 80 GB. No quantization
# = no skip-list games; merger stays plain nn.Linear and PEFT works as
# designed.
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
model = VLModel.from_pretrained(
    MODEL_ID, device_map="auto",
    torch_dtype=torch.bfloat16, attn_implementation="sdpa",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    # === v8: fully train the vision→LM merger ===
    # Qwen2.5-VL routes ViT features through `visual.merger.*` (a small RMSNorm
    # + 2-layer MLP) to produce the visual token sequence that gets
    # concatenated into the LM input. v5/v6/v7 left this frozen, which is why
    # the LM learned to ignore visual features. PEFT's modules_to_save makes a
    # full trainable copy of matching modules and persists it inside the
    # adapter's safetensors, so the existing eval path (042_eval_vl_adapter.py)
    # picks up merger weights automatically via PeftModel.from_pretrained. We
    # match on "merger" — the only module in the model with that suffix.
    modules_to_save=["merger"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Verify PEFT set up modules_to_save correctly: at least one `visual.merger`
# param should now have requires_grad=True, be on GPU, and NOT be 4-bit
# quantized (else llm_int8_skip_modules didn't match and gradients will be
# silently zero). Fail fast here, not 3.5h into training.
merger_trainable = 0
merger_names = []
merger_sample_dtypes = set()
merger_quantized = []
for name, p in model.named_parameters():
    if "visual.merger" in name and p.requires_grad:
        merger_trainable += p.numel()
        merger_sample_dtypes.add(str(p.dtype))
        if hasattr(p, "quant_state") or p.dtype == torch.uint8:
            merger_quantized.append(name)
        if len(merger_names) < 6: merger_names.append(name)
if merger_trainable == 0:
    print("FATAL: no trainable 'visual.merger' params after get_peft_model.")
    print("       modules_to_save likely did not match. Candidate names (first 40):")
    for n, _ in list(model.named_parameters())[:40]:
        print("       ", n)
    sys.exit(1)
if merger_quantized:
    print("FATAL: visual.merger params are 4-bit quantized — cannot receive gradients.")
    print(f"       Offending params: {merger_quantized[:5]}")
    print("       Fix: ensure llm_int8_skip_modules=[\"visual\"] in BitsAndBytesConfig.")
    sys.exit(1)
print(f"\n[v8] Trainable visual.merger params: {merger_trainable:,}  dtypes: {merger_sample_dtypes}")
print("[v8] Sample trainable merger param names:")
for n in merger_names: print("     ", n)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
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

# Preflight 1: supervised-token count (carry over from v7)
print("\n[mask-sanity] Inspecting first training example...")
_s = format_example(train_convos[0])
_total = int(_s["input_ids"].shape[0])
_sup = _total - _s["prompt_len"]
print(f"  total_tokens={_total}  prompt_tokens={_s['prompt_len']}  supervised_tokens={_sup}")
if _sup < 20:
    print("FATAL: fewer than 20 supervised tokens — mask is wrong."); sys.exit(1)

# Preflight 2: pixel-variance sanity. If frames aren't actually reaching the
# model, v8 would mode-collapse for the same reason as v7 regardless of
# merger training. Confirm pixel_values varies across two training examples.
print("[pixel-sanity] Inspecting pixel variance across two examples...")
_s0 = format_example(train_convos[0])
_s1 = format_example(train_convos[1])
if "pixel_values" in _s0 and "pixel_values" in _s1:
    _pv0 = _s0["pixel_values"].flatten()[:10000].float()
    _pv1 = _s1["pixel_values"].flatten()[:10000].float()
    _diff = (_pv0 - _pv1).abs().mean().item()
    print(f"  mean|pv0-pv1|={_diff:.4f}  (>1e-3 means frames are distinct)")
    if _diff < 1e-3:
        print("FATAL: pixel_values nearly identical across examples — collator issue.")
        sys.exit(1)
else:
    print("  WARNING: no pixel_values in formatted example (frames missing?).")

from transformers import Trainer, TrainingArguments

BATCH = 1; GRAD_ACCUM = 4; EPOCHS = 5; LR = 1e-4
LR_KWARGS = {"lr_scheduler_type": "cosine"}

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

print(f"\n=== Training v8 — {datetime.now()} ===")
print(f"Epochs={EPOCHS}  Batch={BATCH}x{GRAD_ACCUM}  LR={LR}  "
      f"Examples={len(train_convos)}  MaxFrames={MAX_FRAMES}  "
      f"LossMask=assistant-only  Merger=unfrozen")

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=val_ds, data_collator=collate)
result = trainer.train()

print(f"\n=== Training complete — {datetime.now()} ===")
print(f"Train loss: {result.training_loss:.4f}")

out = Path(CHECKPOINT_DIR) / "final"
# PEFT's save_pretrained persists LoRA weights AND any modules_to_save
# submodules (our merger) inside adapter_model.safetensors. Eval loading via
# PeftModel.from_pretrained restores both automatically — no extra steps.
model.save_pretrained(str(out))
processor.save_pretrained(str(out))

manifest = {
    "run_id": f"fls_round2_vl_v8_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "base_model": MODEL_ID, "adapter": "LoRA r=32 (LM) + visual.merger unfrozen",
    "data_source": "youtube_sft_v2 = v4 + gold + LASANA rescore (all v002 schema)",
    "yt_train": len(train_convos), "yt_val": len(val_convos), "yt_test": len(yt_test),
    "epochs": EPOCHS, "lr": LR, "lr_schedule": "cosine",
    "max_frames_per_example": MAX_FRAMES,
    "loss_masking": "assistant-only (labels[:prompt_len] = -100 + pad masked)",
    "merger_params_trainable": merger_trainable,
    "train_loss": result.training_loss, "gpu": gpu,
    "completed_at": datetime.now().isoformat(),
    "supersedes": "050_train_qwen_vl_v7.py (v7 passed loss fix but mode-collapsed "
                  "because visual.merger was frozen — LM had no gradient pressure "
                  "to condition on visual features).",
}
with open(Path(CHECKPOINT_DIR) / "run_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))
