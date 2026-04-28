#!/usr/bin/env python3
"""Evaluate a v003 multimodal scoring adapter on the held-out test set.

Concrete metrics (vs the teacher target on each test example):

* score_mae          — mean abs error of estimated_fls_score
* score_rmse         — RMSE of estimated_fls_score
* normalized_score_mae — score_mae divided by max_score per task
* time_mae           — mean abs error of completion_time_seconds
* task_accuracy      — fraction where predicted task_id matches target
* schema_compliance  — fraction with all required v003 fields populated
* critical_precision — true critical-error mentions / predicted critical-error mentions
* critical_recall    — true critical-error mentions / target critical-error mentions
* per_task_score_mae — score_mae bucketed by task_id

Run on a GPU pod::

    python scripts/071_eval_v003_adapter.py \\
        --adapter /workspace/v17_lora_output/final_adapter \\
        --test-jsonl /workspace/v003_multimodal/test.jsonl \\
        --output /workspace/v17_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch  # type: ignore
from peft import PeftModel  # type: ignore
from transformers import (  # type: ignore
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qwen_vl_utils import process_vision_info  # type: ignore

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
REQUIRED_V003_FIELDS = (
    "task_id",
    "max_score",
    "completion_time_seconds",
    "score_components",
    "estimated_fls_score",
    "critical_errors",
    "cannot_determine",
    "confidence_score",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", required=True)
    p.add_argument("--test-jsonl", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--limit", type=int, default=0, help="Eval at most N examples (0 = all).")
    return p.parse_args()


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None or v == "":
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of free-form text."""
    text = text.strip()
    if not text:
        return None
    # If the model emits prose before/after, find the outermost {...}.
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try a permissive cleanup pass for the most common failure mode:
        # smart quotes / trailing commas.
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return None


def _critical_error_set(parsed: dict) -> set[str]:
    """Collect the *types* of critical errors a parsed scoring JSON declares."""
    errors = parsed.get("critical_errors") or []
    out = set()
    if isinstance(errors, list):
        for e in errors:
            if isinstance(e, dict):
                t = (e.get("type") or "").strip().lower()
                if t and (e.get("present") is not False):
                    out.add(t)
            elif isinstance(e, str):
                out.add(e.strip().lower())
    # Also count high-severity penalties as critical proxies, so v17's
    # output (which over-uses penalties + under-uses critical_errors) still
    # gets credit for surfacing major issues.
    for p in parsed.get("penalties") or []:
        if isinstance(p, dict) and (p.get("severity") in {"major", "critical", "auto_fail"}):
            out.add((p.get("type") or "").strip().lower())
    return {t for t in out if t}


def _schema_compliance(parsed: dict) -> dict[str, bool]:
    return {f: f in parsed for f in REQUIRED_V003_FIELDS}


def main() -> None:
    args = parse_args()

    rows = []
    for line in Path(args.test_jsonl).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"Eval examples: {len(rows)}")

    # Load model + adapter (4-bit so the eval matches the train regime).
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        min_pixels=128 * 28 * 28,
        max_pixels=256 * 28 * 28,
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    per_task_score_diffs: dict[str, list[float]] = defaultdict(list)
    per_task_normalized_diffs: dict[str, list[float]] = defaultdict(list)
    score_diffs: list[float] = []
    time_diffs: list[float] = []
    task_correct = 0
    schema_hits = defaultdict(int)
    crit_tp = crit_fp = crit_fn = 0
    raw_outputs: list[dict] = []

    for i, row in enumerate(rows):
        target_assistant = row["messages"][-1]["content"]
        target_obj = json.loads(target_assistant) if isinstance(target_assistant, str) else target_assistant
        prompt_messages = row["messages"][:-1]

        text = processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        try:
            image_inputs, video_inputs = process_vision_info(prompt_messages)
        except Exception:
            image_inputs, video_inputs = None, None

        if image_inputs:
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(model.device)
        else:
            inputs = processor.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=args.max_length
            ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        gen = processor.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        parsed = _extract_json(gen) or {}

        # Schema compliance.
        for k, present in _schema_compliance(parsed).items():
            if present:
                schema_hits[k] += 1

        # Score MAE.
        target_score = _safe_float(target_obj.get("estimated_fls_score"))
        target_max = _safe_float(target_obj.get("max_score") or target_obj.get("score_components", {}).get("max_score"), default=300)
        target_time = _safe_float(target_obj.get("completion_time_seconds"))
        target_task = (target_obj.get("task_id") or "").strip().lower()

        pred_score = _safe_float(parsed.get("estimated_fls_score") or parsed.get("score_components", {}).get("total_fls_score"))
        pred_time = _safe_float(parsed.get("completion_time_seconds"))
        pred_task = (parsed.get("task_id") or "").strip().lower()

        score_diff = abs(pred_score - target_score)
        score_diffs.append(score_diff)
        per_task_score_diffs[target_task].append(score_diff)
        if target_max > 0:
            per_task_normalized_diffs[target_task].append(score_diff / target_max)

        time_diffs.append(abs(pred_time - target_time))
        if pred_task and pred_task == target_task:
            task_correct += 1

        # Critical-error precision/recall.
        target_crits = _critical_error_set(target_obj)
        pred_crits = _critical_error_set(parsed)
        crit_tp += len(target_crits & pred_crits)
        crit_fp += len(pred_crits - target_crits)
        crit_fn += len(target_crits - pred_crits)

        raw_outputs.append({
            "video_id": row.get("metadata", {}).get("video_id"),
            "task_id_target": target_task,
            "task_id_pred": pred_task,
            "score_target": target_score,
            "score_pred": pred_score,
            "max_score": target_max,
            "score_abs_err": score_diff,
            "score_normalized_err": (score_diff / target_max) if target_max else None,
            "time_target": target_time,
            "time_pred": pred_time,
            "json_parsed": bool(parsed),
            "target_critical": sorted(target_crits),
            "pred_critical": sorted(pred_crits),
            "raw_output_first_400": gen[:400],
        })

        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(rows)}] score_mae_running={sum(score_diffs)/len(score_diffs):.1f}")

    n = max(1, len(rows))
    score_mae = sum(score_diffs) / n
    score_rmse = math.sqrt(sum(d * d for d in score_diffs) / n)
    time_mae = sum(time_diffs) / n
    norm_diffs_all = [d for diffs in per_task_normalized_diffs.values() for d in diffs]
    norm_mae = sum(norm_diffs_all) / max(1, len(norm_diffs_all))

    crit_precision = crit_tp / max(1, crit_tp + crit_fp)
    crit_recall = crit_tp / max(1, crit_tp + crit_fn)
    crit_f1 = (2 * crit_precision * crit_recall / (crit_precision + crit_recall)) if (crit_precision + crit_recall) else 0.0

    per_task_metrics = {
        t: {
            "n": len(diffs),
            "score_mae": sum(diffs) / max(1, len(diffs)),
            "normalized_mae": (
                sum(per_task_normalized_diffs[t]) / max(1, len(per_task_normalized_diffs[t]))
            ),
        }
        for t, diffs in per_task_score_diffs.items()
    }

    summary = {
        "adapter": args.adapter,
        "test_jsonl": args.test_jsonl,
        "n_examples": n,
        "score_mae": score_mae,
        "score_rmse": score_rmse,
        "normalized_score_mae": norm_mae,
        "time_mae_seconds": time_mae,
        "task_accuracy": task_correct / n,
        "schema_compliance": {k: v / n for k, v in schema_hits.items()},
        "critical_error": {
            "precision": crit_precision,
            "recall": crit_recall,
            "f1": crit_f1,
            "tp": crit_tp,
            "fp": crit_fp,
            "fn": crit_fn,
        },
        "per_task": per_task_metrics,
    }

    Path(args.output).write_text(json.dumps(summary, indent=2))
    Path(args.output + ".raw.jsonl").write_text(
        "\n".join(json.dumps(r) for r in raw_outputs) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
