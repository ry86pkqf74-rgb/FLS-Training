#!/bin/bash
set -euo pipefail

# ============================================================================
# RunPod LASANA Pre-training Launch Script
# 
# Run this ON the RunPod pod after SSH'ing in.
# Expects: A100 80GB or similar, PyTorch template
#
# Data source: Contabo (207.244.235.10) has 1,270 LASANA trials with frames
# Training data: /data/fls/training/lasana_pretrain_v2/ on Contabo
# ============================================================================

echo "=== LASANA Pre-training Setup — $(date) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'checking...')"

WORK="/workspace/FLS-Training"
DATA="/workspace/lasana_data"

# Step 1: Clone repo
echo "[1/5] Cloning FLS-Training repo..."
if [ -d "$WORK/.git" ]; then
    cd "$WORK" && git fetch origin main && git reset --hard origin/main
else
    git clone https://github.com/ry86pkqf74-rgb/FLS-Training.git "$WORK"
fi
cd "$WORK"

# Step 2: Install dependencies
echo "[2/5] Installing dependencies..."
pip install -q -e "." 2>&1 | tail -3
pip install -q unsloth "unsloth[cu124-torch250]" 2>&1 | tail -3 || \
    pip install -q "unsloth @ git+https://github.com/unslothai/unsloth.git" 2>&1 | tail -3
pip install -q anthropic pyyaml rich typer 2>&1 | tail -3
echo "  Torch: $(python3 -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")')"

# Step 3: Pull LASANA training data from Contabo
echo "[3/5] Pulling LASANA data from Contabo..."
mkdir -p "$DATA"

# Pull the pre-built JSONL training files
rsync -avz --progress \
    -e "ssh -o StrictHostKeyChecking=no" \
    root@207.244.235.10:/data/fls/training/lasana_pretrain_v2/ \
    "$DATA/" 2>&1 | tail -5

echo "  Training files:"
wc -l "$DATA"/*.jsonl 2>/dev/null

# Step 4: Pull a sample of frames for validation (not all 275K — just 20 per trial for the val set)
echo "[4/5] Pulling validation frames (subset)..."
mkdir -p "$DATA/frames"

# Read val set video IDs and pull their frames
python3 << 'PYEOF'
import json, subprocess
from pathlib import Path

val_path = Path("/workspace/lasana_data/val.jsonl")
if not val_path.exists():
    print("No val.jsonl found!"); exit(1)

vids = set()
with open(val_path) as f:
    for line in f:
        ex = json.loads(line)
        vids.add(ex["video_id"])

print(f"Val set: {len(vids)} videos, pulling frames...")
for i, vid in enumerate(sorted(vids)):
    if i % 20 == 0:
        print(f"  {i}/{len(vids)}...")
    dest = Path(f"/workspace/lasana_data/frames/{vid}")
    dest.mkdir(parents=True, exist_ok=True)
    # Pull just 20 uniformly sampled frames per trial
    subprocess.run([
        "rsync", "-a", "--include=*.jpg", "--exclude=*",
        "-e", "ssh -o StrictHostKeyChecking=no",
        f"root@207.244.235.10:/data/fls/lasana_processed/frames/{vid}/",
        str(dest) + "/"
    ], capture_output=True, timeout=60)

print(f"Done. Frames pulled for {len(vids)} validation videos.")
PYEOF

# Step 5: Launch pre-training
echo "[5/5] Launching pre-training..."
echo "  Train examples: $(wc -l < $DATA/train.jsonl)"
echo "  Val examples: $(wc -l < $DATA/val.jsonl)"
echo "  Test examples: $(wc -l < $DATA/test.jsonl)"

python3 << 'TRAINEOF'
import json, torch, os, sys
from pathlib import Path
from datetime import datetime

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Load training data
DATA = Path("/workspace/lasana_data")
train_examples = [json.loads(l) for l in open(DATA / "train.jsonl")]
val_examples = [json.loads(l) for l in open(DATA / "val.jsonl")]

print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

# Build conversation format for Qwen2.5-VL
# Each example: system prompt + frames → target JSON
def build_conversation(example):
    target = example["target"]
    task = example.get("task_id", "unknown")
    grs = example.get("metadata", {}).get("grs_zscore", 0)
    
    system = (
        "You are an expert FLS surgical skills scorer. "
        "Given video frames of a surgical training performance, "
        "output a JSON scoring result with estimated_fls_score, "
        "completion_time_seconds, confidence, technique_summary, "
        "strengths, and improvement_suggestions."
    )
    
    user = f"Score this FLS {task} video performance. Return ONLY valid JSON."
    
    assistant = json.dumps({
        "task_id": target.get("task_id", task),
        "estimated_fls_score": target.get("estimated_fls_score", 0),
        "completion_time_seconds": target.get("completion_time_seconds", 0),
        "confidence": target.get("confidence", 0.5),
        "score_components": target.get("score_components", {}),
        "technique_summary": target.get("technique_summary", ""),
        "strengths": target.get("strengths", []),
        "improvement_suggestions": target.get("improvement_suggestions", []),
        "ground_truth": target.get("ground_truth", {}),
    }, indent=2)
    
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "video_id": example["video_id"],
    }

# Convert to training format
train_convos = [build_conversation(ex) for ex in train_examples]
val_convos = [build_conversation(ex) for ex in val_examples]

# Write formatted training data
out_dir = Path("/workspace/lasana_formatted")
out_dir.mkdir(exist_ok=True)
with open(out_dir / "train.jsonl", "w") as f:
    for c in train_convos:
        f.write(json.dumps(c) + "\n")
with open(out_dir / "val.jsonl", "w") as f:
    for c in val_convos:
        f.write(json.dumps(c) + "\n")

print(f"Formatted: {len(train_convos)} train, {len(val_convos)} val")
print(f"Output: {out_dir}")

# Now run actual training with Unsloth
try:
    from unsloth import FastVisionModel
    
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=42,
    )
    
    print(f"Model loaded. Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training with text-only (no vision for pre-training, vision comes in fine-tuning)
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    
    train_ds = Dataset.from_list(train_convos)
    val_ds = Dataset.from_list(val_convos)
    
    def formatting_func(example):
        return tokenizer.apply_chat_template(example["messages"], tokenize=False)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir="/workspace/lasana_checkpoints",
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            max_seq_length=2048,
            report_to="none",
            seed=42,
        ),
        formatting_func=formatting_func,
    )
    
    print(f"\n=== Starting LASANA pre-training — {datetime.now()} ===")
    print(f"Epochs: 3 | Batch: 4×4=16 | LR: 2e-4 | Steps: ~{len(train_convos)*3//16}")
    
    result = trainer.train()
    
    print(f"\n=== Training complete — {datetime.now()} ===")
    print(f"Train loss: {result.training_loss:.4f}")
    
    # Save
    model.save_pretrained("/workspace/lasana_checkpoints/final")
    tokenizer.save_pretrained("/workspace/lasana_checkpoints/final")
    
    # Save manifest
    manifest = {
        "run_id": f"lasana_pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "base_model": "Qwen2.5-VL-7B-Instruct",
        "dataset": "lasana_pretrain_v2",
        "train_examples": len(train_convos),
        "val_examples": len(val_convos),
        "epochs": 3,
        "train_loss": result.training_loss,
        "completed_at": datetime.now().isoformat(),
    }
    with open("/workspace/lasana_checkpoints/run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print("Checkpoint saved to /workspace/lasana_checkpoints/final")
    
except ImportError as e:
    print(f"Unsloth not available: {e}")
    print("Saving formatted data only — run training manually")
except Exception as e:
    print(f"Training error: {e}")
    import traceback
    traceback.print_exc()

TRAINEOF

echo "=== LASANA Pre-training Setup Complete — $(date) ==="
