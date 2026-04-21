#!/usr/bin/env python3
"""
FLS Video Scoring Demo v2 — Two-stage pipeline
Accepts video upload OR text-only input.
Extracts 8 frames → single-pass multimodal inference through Qwen2.5-VL + v15 LoRA.
CPU float16 inference on S6.
"""

import gradio as gr
import torch
import json
import subprocess
import tempfile
import os
import shutil
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "/opt/fls/adapters/v15"
N_FRAMES = 8

TASK_DESCRIPTIONS = {
    "task1_peg_transfer": "Peg Transfer",
    "task2_pattern_cut": "Pattern Cutting (Circle Cut)",
    "task3_endoloop": "Endoloop (Ligating Loop)",
    "task4_extracorporeal_knot": "Extracorporeal Knot Tying",
    "task5_intracorporeal_suturing": "Intracorporeal Suturing",
    "task6_rings_of_rings": "Rings of Rings",
}

SYSTEM_PROMPT = (
    "You are an expert FLS (Fundamentals of Laparoscopic Surgery) "
    "surgical skills evaluator. You analyze surgical training videos "
    "and provide detailed, structured scoring evaluations following "
    "the FLS scoring rubric. Your evaluations include estimated FLS scores, "
    "score components, technique assessment, strengths, and improvement suggestions."
)

# ── Model loading ──
print("Loading base model (CPU, float16)... this may take a few minutes.")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
print("Loading v15 LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = processor.tokenizer
print("Model loaded! Starting Gradio...")


def extract_frames(video_path, n_frames=N_FRAMES):
    """Extract N evenly-spaced frames from a video using ffmpeg."""
    # Get duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True,
    )
    try:
        duration = float(json.loads(probe.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError):
        # Fallback: try to get duration from streams
        probe2 = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True,
        )
        try:
            streams = json.loads(probe2.stdout)["streams"]
            duration = float(streams[0].get("duration", 60))
        except Exception:
            duration = 60.0  # fallback to 60s

    tmpdir = tempfile.mkdtemp(prefix="fls_frames_")
    frames = []
    for i in range(n_frames):
        t = duration * (i + 0.5) / n_frames
        frame_path = os.path.join(tmpdir, f"frame_{i:03d}.jpg")
        subprocess.run(
            [
                "ffmpeg", "-ss", f"{t:.2f}", "-i", video_path,
                "-vframes", "1", "-q:v", "2",
                "-vf", "scale=448:448:force_original_aspect_ratio=decrease,pad=448:448:(ow-iw)/2:(oh-ih)/2",
                frame_path, "-y", "-loglevel", "quiet",
            ],
        )
        if os.path.exists(frame_path):
            frames.append(frame_path)
    return tmpdir, frames


def evaluate_fls(task_id, completion_time, additional_context, video_file):
    """Run FLS evaluation — multimodal if video provided, text-only otherwise."""
    task_name = TASK_DESCRIPTIONS.get(task_id, task_id)
    tmpdir = None

    try:
        # Build user message content
        content = []
        frame_count = 0

        if video_file is not None:
            # Extract frames from uploaded video
            tmpdir, frame_paths = extract_frames(video_file, N_FRAMES)
            frame_count = len(frame_paths)

            if frame_count > 0:
                # Add each frame as an image
                for fp in frame_paths:
                    content.append({"type": "image", "image": f"file://{fp}"})
                content.append({
                    "type": "text",
                    "text": f"Above are {frame_count} evenly-sampled frames from an FLS {task_name} performance video.",
                })

        # Build text prompt
        prompt = f"Please evaluate this FLS {task_name} performance."
        if frame_count > 0:
            prompt += f" The video contains {frame_count} sampled frames."
        if completion_time and completion_time > 0:
            prompt += f" Completion time: {int(completion_time)} seconds."
        if additional_context and additional_context.strip():
            prompt += f" {additional_context.strip()}"
        prompt += (
            "\n\nProvide a structured JSON evaluation with estimated_fls_score, "
            "score_components, technique_summary, strengths, and improvement_suggestions."
        )
        content.append({"type": "text", "text": prompt})

        # Build messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        # Process with Qwen2.5-VL processor
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if frame_count > 0:
            # Multimodal path
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )
        else:
            # Text-only path
            inputs = tokenizer(text, return_tensors="pt")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Parse response
        try:
            parsed = json.loads(response)
            formatted = json.dumps(parsed, indent=2)
            score = parsed.get("estimated_fls_score", "N/A")
            summary = parsed.get("technique_summary", "")
            mode = f"[{'Video' if frame_count > 0 else 'Text-only'} mode, {frame_count} frames]"
            return formatted, f"FLS Score: {score}  {mode}", summary
        except json.JSONDecodeError:
            mode = f"[{'Video' if frame_count > 0 else 'Text-only'} mode]"
            return response, f"Score: see raw output  {mode}", ""

    finally:
        # Clean up temp frames
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


# ── Gradio UI ──
with gr.Blocks(title="FLS Video Scoring - v15") as demo:
    gr.Markdown("# FLS Surgical Skills Evaluator (v15)")
    gr.Markdown(
        "Upload an FLS task video for multimodal scoring, or use text-only mode.\n\n"
        "**Qwen2.5-VL-7B + v15 LoRA** · 1,433 training examples · All 6 FLS tasks · CPU inference"
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload FLS Video (optional)")
            task_dropdown = gr.Dropdown(
                choices=list(TASK_DESCRIPTIONS.keys()),
                value="task2_pattern_cut",
                label="FLS Task",
            )
            time_input = gr.Number(label="Completion Time (seconds)", value=0)
            context_input = gr.Textbox(
                label="Additional Context (optional)",
                placeholder="e.g., dropped peg twice, cut outside the line",
                lines=3,
            )
            evaluate_btn = gr.Button("Evaluate", variant="primary", size="lg")

            gr.Markdown(
                "**Tips:**\n"
                "- Upload a video for vision-based scoring (slower, ~2-5 min on CPU)\n"
                "- Or leave video empty for instant text-only scoring\n"
                "- Completion time of 0 = unknown/not provided"
            )

        with gr.Column(scale=1):
            score_display = gr.Textbox(label="Score", interactive=False)
            summary_display = gr.Textbox(
                label="Technique Summary", interactive=False, lines=3
            )
            json_output = gr.Code(label="Full Evaluation (JSON)", language="json")

    evaluate_btn.click(
        fn=evaluate_fls,
        inputs=[task_dropdown, time_input, context_input, video_input],
        outputs=[json_output, score_display, summary_display],
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
