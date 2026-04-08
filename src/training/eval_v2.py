"""Shared inference and evaluation helpers for checkpoint validation.

The v2 evaluation path deliberately measures behavior instead of only loss:
it runs held-out prompts through a checkpoint, parses the generated scoring
JSON, compares against teacher/consensus targets, and saves qualitative
artifacts for manual review.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.training.schema_adapter import (
    CANONICAL_PENALTY_KEYS as PENALTY_KEYS,
    get_penalty_labels,
    get_phase_presence,
    get_total_score,
)

logger = logging.getLogger(__name__)

CANONICAL_PHASES = [
    "idle",
    "needle_load",
    "suture_placement",
    "first_throw",
    "second_throw",
    "third_throw",
    "suture_cut",
    "completion",
]


@dataclass
class ModelBundle:
    model: Any
    handler: Any
    model_type: str


def resolve_latest_test_split(base_dir: str | Path = ".") -> Path:
    base = Path(base_dir)
    candidates = sorted((base / "data" / "training").glob("*/test.jsonl"))
    if not candidates:
        raise FileNotFoundError("No held-out test split found under data/training/*/test.jsonl")
    return candidates[-1]


def load_examples(data_path: str | Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    path = Path(data_path)
    examples: list[dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if max_examples is not None and len(examples) >= max_examples:
                break
    return examples


def extract_prompt_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    messages = []
    for message in example.get("messages", []):
        if message.get("role") == "assistant":
            continue
        messages.append(message)
    return messages


def extract_video_id(example: dict[str, Any]) -> str:
    metadata = example.get("metadata") or {}
    if metadata.get("video_id"):
        return str(metadata["video_id"])

    for message in example.get("messages", []):
        content = message.get("content")
        if isinstance(content, str) and '"video_id"' in content:
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                continue
            if payload.get("video_id"):
                return str(payload["video_id"])

    return f"unknown_{abs(hash(json.dumps(example, sort_keys=True))) % 10_000_000}"


def parse_assistant_target(example: dict[str, Any]) -> dict[str, Any]:
    for message in example.get("messages", []):
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", "")).strip()
        if not content:
            return {}
        return parse_prediction_text(content)
    return {}


def parse_prediction_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if cleaned.startswith("json"):
        cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def load_model_bundle(model_path: str | Path) -> ModelBundle:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        from transformers import Qwen2_5_VLForConditionalGeneration
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Install torch and transformers to run checkpoint evaluation") from exc

    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {model_dir}")

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        handler = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        return ModelBundle(model=model, handler=handler, model_type="vlm")
    except Exception as exc:
        logger.warning("Falling back to causal LM load for %s: %s", model_dir, exc)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    handler = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if handler.pad_token is None:
        handler.pad_token = handler.eos_token
    return ModelBundle(model=model, handler=handler, model_type="causal")


def _handler_decode(handler: Any, token_ids) -> str:
    if hasattr(handler, "decode"):
        return handler.decode(token_ids, skip_special_tokens=True)
    tokenizer = getattr(handler, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    raise AttributeError("Handler does not expose a decode method")


def generate_text(
    bundle: ModelBundle,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 2048,
) -> str:
    import torch

    handler = bundle.handler
    if not hasattr(handler, "apply_chat_template"):
        raise AttributeError("Tokenizer/processor must support apply_chat_template")

    prompt_text = handler.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if bundle.model_type == "vlm":
        inputs = handler(text=prompt_text, return_tensors="pt")
    else:
        inputs = handler(prompt_text, return_tensors="pt")

    device = getattr(bundle.model, "device", None)
    if device is None:
        device = next(bundle.model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = bundle.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    return _handler_decode(handler, output_ids[0][input_len:])


def _normalize_teacher_source(source: str) -> str:
    lowered = source.lower()
    if "gpt" in lowered:
        return "teacher_gpt4o"
    if "claude" in lowered:
        return "teacher_claude"
    return lowered


def load_teacher_scores(
    base_dir: str | Path,
    video_ids: set[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    base = Path(base_dir)
    scores_dir = base / "memory" / "scores"
    latest: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}

    for path in sorted(scores_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        video_id = payload.get("video_id")
        if not video_id or (video_ids is not None and video_id not in video_ids):
            continue
        source = _normalize_teacher_source(str(payload.get("source", "")))
        if source not in {"teacher_claude", "teacher_gpt4o"}:
            continue
        timestamp = str(payload.get("scored_at") or payload.get("timestamp") or "")
        key = (video_id, source)
        previous = latest.get(key)
        if previous is None or timestamp >= previous[0]:
            latest[key] = (timestamp, payload)

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for (video_id, source), (_, payload) in latest.items():
        grouped.setdefault(video_id, {})[source] = payload
    return grouped


def build_coaching_messages(
    coach_system_prompt: str,
    video_id: str,
    consensus_payload: dict[str, Any],
    teacher_scores: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    teacher_scores = teacher_scores or {}
    user_parts = [
        f"Video ID: {video_id}",
        "Consensus Score JSON:",
        json.dumps(consensus_payload, indent=2),
    ]

    claude = teacher_scores.get("teacher_claude")
    if claude and claude.get("frame_analyses"):
        user_parts.extend([
            "Teacher A Frame Analyses:",
            json.dumps(claude.get("frame_analyses"), indent=2),
        ])

    gpt = teacher_scores.get("teacher_gpt4o")
    if gpt and gpt.get("frame_analyses"):
        user_parts.extend([
            "Teacher B Frame Analyses:",
            json.dumps(gpt.get("frame_analyses"), indent=2),
        ])

    user_parts.append("Trainee History: unavailable")
    user_parts.append("Generate progression-aware coaching feedback.")
    return [
        {"role": "system", "content": coach_system_prompt},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# Phase/penalty/score helpers are now defined in src.training.schema_adapter
# to keep train-time and eval-time code in lockstep. Local thin wrappers remain
# so the existing call sites in this file stay unchanged.

def derive_phase_presence(payload: dict[str, Any]) -> set[str]:
    return get_phase_presence(payload)


def derive_penalty_labels(payload: dict[str, Any]) -> set[str]:
    return get_penalty_labels(payload)


def _get_score(payload: dict[str, Any]) -> float:
    # Reads v002 score_components.total_fls_score first, falls back to v001
    # estimated_fls_score. Using the adapter is mandatory — do NOT inline
    # payload.get('estimated_fls_score') here because it silently reads zero
    # from fresh v002 records and poisons the MAE.
    return get_total_score(payload)


def _get_phase_accuracy(predicted: dict[str, Any], target: dict[str, Any]) -> float:
    predicted_phases = derive_phase_presence(predicted)
    target_phases = derive_phase_presence(target)
    matches = 0
    for phase in CANONICAL_PHASES:
        if (phase in predicted_phases) == (phase in target_phases):
            matches += 1
    return matches / len(CANONICAL_PHASES)


def _safe_pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(right) < 2:
        return None
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if np.std(left_array) == 0 or np.std(right_array) == 0:
        return None
    return float(np.corrcoef(left_array, right_array)[0, 1])


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "examples": 0,
            "parse_rate": 0.0,
            "score_mae_consensus": None,
            "score_mae_claude": None,
            "score_mae_gpt4o": None,
            "pearson_r_claude": None,
            "pearson_r_gpt4o": None,
            "phase_accuracy": None,
            "penalty_precision": None,
            "penalty_recall": None,
            "penalty_f1": None,
        }

    parse_count = 0
    consensus_errors: list[float] = []
    claude_errors: list[float] = []
    gpt_errors: list[float] = []
    predicted_scores: list[float] = []
    predicted_scores_for_claude: list[float] = []
    predicted_scores_for_gpt: list[float] = []
    claude_scores: list[float] = []
    gpt_scores: list[float] = []
    phase_accuracies: list[float] = []

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for record in records:
        predicted = record.get("parsed_output") or {}
        target = record.get("target_output") or {}
        teachers = record.get("teacher_scores") or {}

        if record.get("parse_success"):
            parse_count += 1

        predicted_score = _get_score(predicted)
        consensus_score = _get_score(target)
        predicted_scores.append(predicted_score)
        consensus_errors.append(abs(predicted_score - consensus_score))
        phase_accuracies.append(_get_phase_accuracy(predicted, target))

        predicted_penalties = derive_penalty_labels(predicted)
        target_penalties = derive_penalty_labels(target)
        for penalty in PENALTY_KEYS:
            predicted_has = penalty in predicted_penalties
            target_has = penalty in target_penalties
            if predicted_has and target_has:
                true_positive += 1
            elif predicted_has and not target_has:
                false_positive += 1
            elif not predicted_has and target_has:
                false_negative += 1

        claude = teachers.get("teacher_claude")
        if claude:
            teacher_score = _get_score(claude)
            predicted_scores_for_claude.append(predicted_score)
            claude_scores.append(teacher_score)
            claude_errors.append(abs(predicted_score - teacher_score))

        gpt = teachers.get("teacher_gpt4o")
        if gpt:
            teacher_score = _get_score(gpt)
            predicted_scores_for_gpt.append(predicted_score)
            gpt_scores.append(teacher_score)
            gpt_errors.append(abs(predicted_score - teacher_score))

    precision = None
    recall = None
    f1 = None
    if true_positive or false_positive or false_negative:
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
    else:
        precision = 1.0
        recall = 1.0
        f1 = 1.0

    return {
        "examples": len(records),
        "parse_rate": round(parse_count / len(records), 4),
        "score_mae_consensus": round(float(np.mean(consensus_errors)), 4),
        "score_mae_claude": round(float(np.mean(claude_errors)), 4) if claude_errors else None,
        "score_mae_gpt4o": round(float(np.mean(gpt_errors)), 4) if gpt_errors else None,
        "pearson_r_claude": None if not claude_scores else _safe_pearson(predicted_scores_for_claude, claude_scores),
        "pearson_r_gpt4o": None if not gpt_scores else _safe_pearson(predicted_scores_for_gpt, gpt_scores),
        "phase_accuracy": round(float(np.mean(phase_accuracies)), 4),
        "penalty_precision": round(float(precision), 4) if precision is not None else None,
        "penalty_recall": round(float(recall), 4) if recall is not None else None,
        "penalty_f1": round(float(f1), 4) if f1 is not None else None,
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_path: str | Path,
    base_dir: str | Path = ".",
    max_examples: int | None = None,
    qualitative_dir: str | Path = "memory/prompt_evals",
    qualitative_samples: int = 5,
) -> dict[str, Any]:
    base = Path(base_dir)
    examples = load_examples(data_path, max_examples=max_examples)
    if not examples:
        raise ValueError(f"No evaluation examples found in {data_path}")

    video_ids = {extract_video_id(example) for example in examples}
    teacher_scores = load_teacher_scores(base, video_ids=video_ids)
    coach_prompt_path = base / "prompts" / "v001_task5_coach.md"
    coach_prompt = coach_prompt_path.read_text() if coach_prompt_path.exists() else ""

    bundle = load_model_bundle(checkpoint_path)
    records: list[dict[str, Any]] = []
    coaching_samples: list[dict[str, Any]] = []

    for index, example in enumerate(examples):
        video_id = extract_video_id(example)
        prompt_messages = extract_prompt_messages(example)
        raw_output = generate_text(bundle, prompt_messages)
        parse_success = True
        try:
            parsed_output = parse_prediction_text(raw_output)
        except json.JSONDecodeError:
            parse_success = False
            parsed_output = {}

        target_output = parse_assistant_target(example)
        record = {
            "video_id": video_id,
            "parse_success": parse_success,
            "raw_output": raw_output,
            "parsed_output": parsed_output,
            "target_output": target_output,
            "teacher_scores": teacher_scores.get(video_id, {}),
        }
        records.append(record)

        if coach_prompt and index < qualitative_samples:
            coach_messages = build_coaching_messages(
                coach_system_prompt=coach_prompt,
                video_id=video_id,
                consensus_payload=target_output,
                teacher_scores=teacher_scores.get(video_id, {}),
            )
            coaching_raw = generate_text(bundle, coach_messages)
            coaching_samples.append(
                {
                    "video_id": video_id,
                    "messages": coach_messages,
                    "raw_output": coaching_raw,
                }
            )

    metrics = summarize_records(records)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    checkpoint_name = Path(checkpoint_path).name.replace("/", "_")
    qualitative_path = Path(qualitative_dir)
    qualitative_path.mkdir(parents=True, exist_ok=True)
    artifact_path = qualitative_path / f"{timestamp}_{checkpoint_name}_eval.json"
    artifact_payload = {
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "records": records,
        "coaching_outputs": coaching_samples,
    }
    artifact_path.write_text(json.dumps(artifact_payload, indent=2))

    return {
        "checkpoint": str(checkpoint_path),
        "data_path": str(data_path),
        "metrics": metrics,
        "records": records,
        "qualitative_path": str(artifact_path),
    }