#!/usr/bin/env python3
"""Evaluate LASANA-pretrained adapter on the held-out test set.

Loads the base Qwen2.5-7B-Instruct model with the trained LoRA adapter,
runs inference on all 205 LASANA test examples, and computes:
- JSON validity rate (did the model emit parseable JSON?)
- Mean Absolute Error on estimated_fls_score vs ground-truth FLS score
- Classification agreement (predicted vs expected task_id)

Writes results to /workspace/eval_results.json.
"""
import json, time, sys
from pathlib import Path
from datetime import datetime, timezone
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = (
    "You are an expert FLS (Fundamentals of Laparoscopic Surgery) proctor AI. "
    "Given video frames of a surgical training performance, analyze the technique "
    "and output a structured JSON scoring result including estimated_fls_score, "
    "completion_time_seconds, confidence, score_components, technique_summary, "
    "strengths, and improvement_suggestions."
)

def load_test(path):
    return [json.loads(l) for l in open(path)]

def main():
    print(f"=== FLS eval — {datetime.now(timezone.utc).isoformat()} ===")
    test = load_test("/workspace/lasana_test.jsonl")
    print(f"Loaded {len(test)} test examples")

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True)
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                                 quantization_config=bnb,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tok.pad_token = tok.eos_token
    print("Attaching adapter...")
    model = PeftModel.from_pretrained(base, "/workspace/checkpoints/final")
    model.eval()

    results = []
    valid_json = 0
    mae_scores = []
    task_match = 0

    t0 = time.time()
    for i, ex in enumerate(test):
        target = ex.get("target", {})
        task = ex.get("task_id", "unknown")
        gt_score = target.get("estimated_fls_score")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Score this FLS {task} video performance. Return ONLY valid JSON."},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512,
                                  do_sample=False, temperature=1.0,
                                  pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        parsed = None
        try:
            parsed = json.loads(gen.strip())
            valid_json += 1
        except Exception:
            # try to extract JSON block
            try:
                s = gen.find("{")
                e = gen.rfind("}")
                if s >= 0 and e > s:
                    parsed = json.loads(gen[s:e+1])
                    valid_json += 1
            except Exception:
                pass

        pred_score = None
        pred_task = None
        if parsed:
            sc = parsed.get("score_components") or {}
            pred_score = sc.get("total_fls_score") or parsed.get("estimated_fls_score")
            pred_task = parsed.get("task_id")
            if pred_task == task:
                task_match += 1
            if isinstance(pred_score, (int, float)) and isinstance(gt_score, (int, float)):
                mae_scores.append(abs(pred_score - gt_score))

        results.append({
            "video_id": ex.get("video_id"),
            "expected_task": task,
            "expected_fls": gt_score,
            "predicted_task": pred_task,
            "predicted_fls": pred_score,
            "json_valid": parsed is not None,
            "raw_output_head": gen[:300],
        })

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (len(test) - i - 1)
            print(f"  [{i+1}/{len(test)}] valid_json={valid_json} "
                  f"mae_n={len(mae_scores)} task_match={task_match} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    summary = {
        "run_id": "fls_eval_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "adapter": "/workspace/checkpoints/final",
        "n_examples": len(test),
        "valid_json_rate": valid_json / len(test),
        "task_id_accuracy": task_match / len(test),
        "mae_fls_score_mean": sum(mae_scores)/len(mae_scores) if mae_scores else None,
        "mae_fls_score_n": len(mae_scores),
        "wall_seconds": time.time() - t0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    Path("/workspace/eval_results.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2, default=str)
    )
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
