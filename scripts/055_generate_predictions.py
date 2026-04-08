#!/usr/bin/env python3
"""Generate student predictions from a fine-tuned checkpoint.

Loads the merged model, runs inference on test/val data,
and writes prediction JSONs that scripts/060_evaluate_student.py can consume.

Usage:
    # Evaluate v1 checkpoint on v1 test data
    python scripts/055_generate_predictions.py \
        --model memory/model_checkpoints/20260407_1619_unsloth/merged_16bit \
        --data data/training/2026-04-07_v1/test.jsonl \
        --output memory/predictions/v1_on_test

    # Evaluate v2 checkpoint on v2 val data
    python scripts/055_generate_predictions.py \
        --model memory/model_checkpoints/v2_diverse/merged_16bit \
        --data training/data/scoring_val_v2.jsonl \
        --output memory/predictions/v2_on_val
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_model(model_path: str):
    """Load the merged fine-tuned model for inference."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        import torch
    except ImportError:
        logger.error("pip install torch transformers")
        sys.exit(1)

    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"Model path not found: {model_path}")
        sys.exit(1)

    logger.info(f"Loading model from {model_path}...")

    # Try loading as a vision-language model first
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        logger.info("Loaded as Qwen2.5-VL model")
        return model, processor, "vlm"
    except Exception as e:
        logger.warning(f"VLM load failed ({e}), trying causal LM...")

    # Fallback to standard causal LM
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    logger.info("Loaded as causal LM")
    return model, tokenizer, "causal"


def generate_prediction(model, tokenizer_or_processor, example: dict, model_type: str) -> str:
    """Run inference on a single example and return the generated text."""
    import torch

    messages = example.get("messages", [])
    if not messages:
        return ""

    # Build prompt from all messages except the last assistant response
    prompt_messages = []
    expected_output = ""
    for msg in messages:
        if msg["role"] == "assistant":
            expected_output = msg.get("content", "")
        else:
            prompt_messages.append(msg)

    if model_type == "vlm":
        text = tokenizer_or_processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer_or_processor(text=text, return_tensors="pt").to(model.device)
    else:
        text = tokenizer_or_processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer_or_processor(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=False,
        )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    generated = tokenizer_or_processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )
    return generated


def extract_video_id(example: dict) -> str:
    """Try to extract video_id from the example messages."""
    for msg in example.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str) and "video_id" in content:
            try:
                data = json.loads(content)
                if "video_id" in data:
                    return data["video_id"]
            except json.JSONDecodeError:
                pass
    return f"unknown_{hash(json.dumps(example.get('messages', [])[:1]))}"


def main():
    parser = argparse.ArgumentParser(description="Generate student predictions")
    parser.add_argument("--model", required=True, help="Path to merged model checkpoint")
    parser.add_argument("--data", required=True, help="Path to test/val JSONL file")
    parser.add_argument("--output", required=True, help="Output directory for prediction JSONs")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load model
    model, tokenizer, model_type = load_model(args.model)

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if args.max_examples:
        examples = examples[:args.max_examples]

    logger.info(f"Generating predictions for {len(examples)} examples...")

    # Generate predictions
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, example in enumerate(examples):
        video_id = extract_video_id(example)
        logger.info(f"[{i+1}/{len(examples)}] {video_id}")

        generated_text = generate_prediction(model, tokenizer, example, model_type)

        # Try to parse as JSON (the model should output ScoringResult JSON)
        try:
            # Strip markdown fences if present
            clean = generated_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()

            prediction = json.loads(clean)
        except json.JSONDecodeError:
            logger.warning(f"  Could not parse output as JSON, saving raw text")
            prediction = {
                "raw_output": generated_text,
                "parse_error": True,
            }

        # Ensure video_id is set
        prediction["video_id"] = video_id
        prediction["source"] = "student"

        # Write prediction
        out_file = output_dir / f"{video_id}_student.json"
        with open(out_file, "w") as f:
            json.dump(prediction, f, indent=2)

    logger.info(f"Wrote {len(examples)} predictions to {output_dir}/")
    logger.info(f"Evaluate with: python scripts/060_evaluate_student.py --student-scores {output_dir}")


if __name__ == "__main__":
    main()
