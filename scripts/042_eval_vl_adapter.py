#!/usr/bin/env python3
"""Evaluate a Qwen2.5-VL LoRA adapter on a held-out FLS test set.

Metrics:
  - valid_json_rate
  - task_id_accuracy
  - mae_fls_score
  - per_task MAE
  - unique_prediction_ratio  (GATE: must be > 0.5 or flag as mode-collapsed)
  - classification_accuracy (video_classification field)

Exit code 1 if the diversity gate fails — so CI / runbooks can block bad adapters.

Usage:
  python 042_eval_vl_adapter.py \\
    --adapter /workspace/checkpoints_vl/final \\
    --test    /workspace/yt_test.jsonl \\
    --out     /workspace/checkpoints_vl/eval_results.json
"""
import argparse, json, re, sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--test", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_frames", type=int, default=8)
ap.add_argument("--diversity_gate", type=float, default=0.5,
                help="Minimum unique_pred_ratio — below this, exit 1")
args = ap.parse_args()

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

print(f"=== Eval {args.adapter} on {args.test} — {datetime.now()} ===")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
base = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=bnb, device_map="auto",
    torch_dtype=torch.bfloat16, attn_implementation="sdpa")
model = PeftModel.from_pretrained(base, args.adapter)
model.eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

SYSTEM_PROMPT = (
    "You are an expert FLS (Fundamentals of Laparoscopic Surgery) proctor AI. "
    "Given sampled video frames of an FLS task performance, analyze the technique "
    "and output a single strict-JSON ScoringResult matching the v002 universal scoring schema. "
    "Output ONLY valid JSON — no prose, no markdown fences."
)

def sample_frames(paths, n):
    if len(paths) <= n: return paths
    step = len(paths) / n
    return [paths[int(i*step)] for i in range(n)]

def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except: return None

def get_fls(d):
    if not isinstance(d, dict): return None
    for k in ("estimated_fls_score", "fls_score"):
        v = d.get(k)
        try:
            if v is not None: return float(v)
        except: pass
    sc = d.get("score_components") or {}
    if isinstance(sc, dict):
        v = sc.get("total_fls_score")
        try:
            if v is not None: return float(v)
        except: pass
    return None

def expected_fls(ex):
    v = ex.get("consensus_fls")
    if v is not None: return float(v)
    return get_fls(ex.get("target") or {})

test = [json.loads(l) for l in open(args.test)]
print(f"Test examples: {len(test)}")

from qwen_vl_utils import process_vision_info

results = []
valid = 0
task_correct = 0
cls_correct = 0
abs_errors = []
per_task_err = defaultdict(list)
preds_fls = []
preds_task = []

for i, ex in enumerate(test):
    frames = ex.get("frames") or []
    if not frames:
        d = Path("/workspace/frames") / ex["video_id"]
        frames = sorted(str(p) for p in d.glob("*.jpg")) if d.exists() else []
    frames = sample_frames(frames, args.max_frames)
    if not frames:
        results.append({"video_id": ex["video_id"], "error": "no_frames"})
        continue

    messages = [
        {"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]},
        {"role":"user","content":[*[{"type":"image","image":p} for p in frames],
                                  {"type":"text","text":f"Score this FLS {ex.get('task_id','unknown')} performance. JSON only."}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inp = processor(text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=1400, do_sample=False, temperature=0.0)
    gen = processor.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]
    parsed = extract_json(gen)

    exp_task = ex.get("task_id")
    exp_fls  = expected_fls(ex)
    pred_fls = get_fls(parsed) if parsed else None
    pred_task = (parsed or {}).get("task_id")
    pred_cls  = (parsed or {}).get("video_classification")

    if parsed is not None: valid += 1
    if pred_task == exp_task: task_correct += 1
    if pred_cls == ex.get("video_classification", "performance"): cls_correct += 1
    if pred_fls is not None and exp_fls is not None:
        err = abs(pred_fls - exp_fls)
        abs_errors.append(err)
        per_task_err[exp_task].append(err)
        preds_fls.append(round(pred_fls, 1))
    if pred_task: preds_task.append(pred_task)

    results.append({
        "video_id": ex["video_id"], "expected_task": exp_task, "predicted_task": pred_task,
        "expected_fls": exp_fls, "predicted_fls": pred_fls,
        "json_valid": parsed is not None,
        "raw_head": gen[:300],
    })
    if (i+1) % 5 == 0: print(f"  {i+1}/{len(test)}...")

n = len(test)
mae = sum(abs_errors)/len(abs_errors) if abs_errors else None
unique_ratio = (len(set(preds_fls)) / len(preds_fls)) if preds_fls else 0.0

summary = {
    "adapter": args.adapter,
    "n_examples": n,
    "valid_json_rate": valid/n if n else 0,
    "task_id_accuracy": task_correct/n if n else 0,
    "classification_accuracy": cls_correct/n if n else 0,
    "mae_fls_score": mae,
    "mae_n": len(abs_errors),
    "unique_prediction_ratio": unique_ratio,
    "unique_fls_values": len(set(preds_fls)),
    "per_task_mae": {t: sum(e)/len(e) for t, e in per_task_err.items()},
    "completed_at": datetime.now().isoformat(),
}

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"summary": summary, "results": results}, f, indent=2)

print(json.dumps(summary, indent=2))

# --- Diversity gate ---
print("\n=== Gates ===")
print(f"  valid_json_rate:   {summary['valid_json_rate']:.2%}  (gate: > 0.9)")
print(f"  unique_pred_ratio: {unique_ratio:.2%}  (gate: > {args.diversity_gate:.0%})")
failed = []
if summary["valid_json_rate"] < 0.9: failed.append("valid_json_rate")
if unique_ratio < args.diversity_gate: failed.append("unique_prediction_ratio (mode collapse)")
if failed:
    print(f"\nFAILED gates: {failed}")
    sys.exit(1)
print("PASSED all gates.")
