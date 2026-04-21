#!/usr/bin/env python3
"""
FLS Video Scoring Platform v3
- Email login for residents
- SQLite results storage with full history
- Personal dashboard with score trends
- Admin export view for program directors
- Qwen2.5-VL-7B + v16 multimodal LoRA
- CPU float16 inference on S6
"""

import gradio as gr
import torch
import json
import subprocess
import tempfile
import os
import shutil
import sqlite3
import hashlib
import csv
import io
from datetime import datetime
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ── Configuration ──
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "/opt/fls/adapters/v16"
N_FRAMES = 8
DB_PATH = "/opt/fls/results.db"
ADMIN_PASSWORD = "fls_admin_2026"  # Change this!

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


# ── Database Setup ──
def init_db():
    """Initialize SQLite database for results storage."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS residents (
            email TEXT PRIMARY KEY,
            name TEXT,
            pgy_level TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            task_id TEXT NOT NULL,
            task_name TEXT NOT NULL,
            completion_time_sec REAL,
            estimated_fls_score REAL,
            score_components TEXT,
            technique_summary TEXT,
            strengths TEXT,
            improvement_suggestions TEXT,
            full_response TEXT,
            mode TEXT,
            frame_count INTEGER DEFAULT 0,
            additional_context TEXT,
            evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES residents(email)
        )
    """)
    conn.commit()
    conn.close()


def register_resident(email, name, pgy_level):
    """Register or update a resident."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO residents (email, name, pgy_level)
        VALUES (?, ?, ?)
        ON CONFLICT(email) DO UPDATE SET name=?, pgy_level=?
    """, (email, name, pgy_level, name, pgy_level))
    conn.commit()
    conn.close()


def save_evaluation(email, task_id, task_name, completion_time, score,
                    score_components, technique_summary, strengths,
                    improvements, full_response, mode, frame_count, context):
    """Save an evaluation result to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO evaluations
        (email, task_id, task_name, completion_time_sec, estimated_fls_score,
         score_components, technique_summary, strengths, improvement_suggestions,
         full_response, mode, frame_count, additional_context)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        email, task_id, task_name, completion_time, score,
        json.dumps(score_components) if score_components else None,
        technique_summary, json.dumps(strengths) if strengths else None,
        json.dumps(improvements) if improvements else None,
        full_response, mode, frame_count, context
    ))
    conn.commit()
    conn.close()


def get_resident_history(email):
    """Get all evaluations for a resident."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT task_name, estimated_fls_score, completion_time_sec,
               technique_summary, mode, evaluated_at
        FROM evaluations
        WHERE email = ?
        ORDER BY evaluated_at DESC
    """, (email,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_resident_stats(email):
    """Get summary statistics for a resident."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Overall stats
    c.execute("""
        SELECT COUNT(*), AVG(estimated_fls_score), MAX(estimated_fls_score),
               MIN(estimated_fls_score)
        FROM evaluations WHERE email = ? AND estimated_fls_score IS NOT NULL
    """, (email,))
    total, avg_score, max_score, min_score = c.fetchone()

    # Per-task averages
    c.execute("""
        SELECT task_name, COUNT(*), AVG(estimated_fls_score), MAX(estimated_fls_score)
        FROM evaluations
        WHERE email = ? AND estimated_fls_score IS NOT NULL
        GROUP BY task_name
        ORDER BY task_name
    """, (email,))
    task_stats = c.fetchall()

    # Recent trend (last 10)
    c.execute("""
        SELECT estimated_fls_score, evaluated_at
        FROM evaluations
        WHERE email = ? AND estimated_fls_score IS NOT NULL
        ORDER BY evaluated_at DESC LIMIT 10
    """, (email,))
    recent = c.fetchall()

    conn.close()
    return {
        "total_evaluations": total or 0,
        "avg_score": round(avg_score, 1) if avg_score else 0,
        "max_score": max_score or 0,
        "min_score": min_score or 0,
        "task_stats": task_stats,
        "recent_scores": recent,
    }


def export_all_results():
    """Export all results as CSV for admin."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT r.name, r.email, r.pgy_level,
               e.task_name, e.estimated_fls_score, e.completion_time_sec,
               e.technique_summary, e.mode, e.evaluated_at
        FROM evaluations e
        JOIN residents r ON e.email = r.email
        ORDER BY e.evaluated_at DESC
    """)
    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Name", "Email", "PGY Level", "Task", "FLS Score",
        "Time (sec)", "Technique Summary", "Mode", "Date"
    ])
    for row in rows:
        writer.writerow(row)
    return output.getvalue()


# ── Model Loading ──
print("Loading base model (CPU, float16)... this may take a few minutes.")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
print("Loading v16 LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = processor.tokenizer
print("Model loaded!")

# Initialize database
init_db()
print("Database ready. Starting Gradio...")


# ── Core Functions ──
def extract_frames(video_path, n_frames=N_FRAMES):
    """Extract N evenly-spaced frames from a video using ffmpeg."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True,
    )
    try:
        duration = float(json.loads(probe.stdout)["format"]["duration"])
    except (json.JSONDecodeError, KeyError):
        probe2 = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
            capture_output=True, text=True,
        )
        try:
            streams = json.loads(probe2.stdout)["streams"]
            duration = float(streams[0].get("duration", 60))
        except Exception:
            duration = 60.0

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


def evaluate_fls(email, task_id, completion_time, additional_context, video_file):
    """Run FLS evaluation and save results."""
    if not email or "@" not in email:
        return "", "⚠️ Please log in first (enter your email above)", "", ""

    task_name = TASK_DESCRIPTIONS.get(task_id, task_id)
    tmpdir = None

    try:
        content = []
        frame_count = 0

        if video_file is not None:
            tmpdir, frame_paths = extract_frames(video_file, N_FRAMES)
            frame_count = len(frame_paths)

            if frame_count > 0:
                for fp in frame_paths:
                    content.append({"type": "image", "image": f"file://{fp}"})
                content.append({
                    "type": "text",
                    "text": f"Above are {frame_count} evenly-sampled frames from an FLS {task_name} performance video.",
                })

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

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if frame_count > 0:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            )
        else:
            inputs = tokenizer(text, return_tensors="pt")

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

        # Parse and save
        mode = "video" if frame_count > 0 else "text-only"
        try:
            parsed = json.loads(response)
            formatted = json.dumps(parsed, indent=2)
            score = parsed.get("estimated_fls_score", None)
            score_components = parsed.get("score_components", None)
            technique_summary = parsed.get("technique_summary", "")
            strengths = parsed.get("strengths", [])
            improvements = parsed.get("improvement_suggestions", [])

            # Save to database
            save_evaluation(
                email, task_id, task_name, completion_time, score,
                score_components, technique_summary, strengths,
                improvements, formatted, mode, frame_count, additional_context
            )

            score_text = f"🎯 FLS Score: {score}  [{mode}, {frame_count} frames]"
            status = f"✅ Result saved for {email}"
            return formatted, score_text, technique_summary, status
        except json.JSONDecodeError:
            save_evaluation(
                email, task_id, task_name, completion_time, None,
                None, "", None, None, response, mode, frame_count, additional_context
            )
            return response, f"Score: see raw output [{mode}]", "", "✅ Saved (raw)"

    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def login_resident(email, name, pgy_level):
    """Handle resident login/registration."""
    if not email or "@" not in email:
        return "⚠️ Please enter a valid email address", ""
    if not name:
        return "⚠️ Please enter your name", ""

    register_resident(email, name, pgy_level)
    stats = get_resident_stats(email)

    welcome = f"✅ Welcome, {name} ({pgy_level})!\n"
    if stats["total_evaluations"] > 0:
        welcome += f"You have {stats['total_evaluations']} evaluations on record.\n"
        welcome += f"Average score: {stats['avg_score']} | Best: {stats['max_score']}"
    else:
        welcome += "This is your first session. Upload a video to get started!"

    return welcome, email


def get_dashboard(email):
    """Generate the personal dashboard for a resident."""
    if not email or "@" not in email:
        return "Please log in first.", "", ""

    stats = get_resident_stats(email)
    history = get_resident_history(email)

    if stats["total_evaluations"] == 0:
        return "No evaluations yet. Upload a video to get started!", "", ""

    # Summary card
    summary = f"""## Your Performance Summary

**Total Evaluations:** {stats['total_evaluations']}
**Average Score:** {stats['avg_score']}
**Best Score:** {stats['max_score']} | **Lowest:** {stats['min_score']}

### Scores by Task
"""
    for task_name, count, avg, best in stats["task_stats"]:
        summary += f"- **{task_name}**: {count} attempts, avg {avg:.1f}, best {best}\n"

    # Trend
    if stats["recent_scores"]:
        trend = "### Recent Scores (newest first)\n"
        for score, date in stats["recent_scores"]:
            trend += f"- {date[:10]}: **{score}**\n"
    else:
        trend = ""

    # History table
    history_text = "### Full History\n\n| Date | Task | Score | Time | Mode |\n|------|------|-------|------|------|\n"
    for task_name, score, time_sec, summary_text, mode, date in history[:20]:
        time_str = f"{int(time_sec)}s" if time_sec else "—"
        score_str = str(score) if score else "—"
        history_text += f"| {date[:16]} | {task_name} | {score_str} | {time_str} | {mode} |\n"

    return summary + trend, history_text, ""


def admin_export(password):
    """Admin function to export all data."""
    if password != ADMIN_PASSWORD:
        return "❌ Invalid admin password", None

    csv_data = export_all_results()

    # Write to temp file for download
    export_path = "/opt/fls/exports"
    os.makedirs(export_path, exist_ok=True)
    filename = f"fls_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(export_path, filename)
    with open(filepath, "w") as f:
        f.write(csv_data)

    # Count stats
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM residents")
    n_residents = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM evaluations")
    n_evals = c.fetchone()[0]
    c.execute("""
        SELECT r.name, r.pgy_level, COUNT(*) as n,
               ROUND(AVG(e.estimated_fls_score), 1) as avg_score
        FROM evaluations e JOIN residents r ON e.email = r.email
        GROUP BY e.email ORDER BY avg_score DESC
    """)
    leaderboard = c.fetchall()
    conn.close()

    report = f"## Admin Report\n\n"
    report += f"**{n_residents} residents** | **{n_evals} total evaluations**\n\n"
    report += "### Leaderboard\n\n| Name | PGY | Attempts | Avg Score |\n|------|-----|----------|----------|\n"
    for name, pgy, n, avg in leaderboard:
        report += f"| {name} | {pgy} | {n} | {avg} |\n"
    report += f"\n\n📁 CSV exported to: `{filepath}`"

    return report, filepath


# ── Gradio UI ──
with gr.Blocks(title="FLS Scoring Platform", theme=gr.themes.Soft()) as demo:
    # State
    current_email = gr.State("")

    gr.Markdown("# 🏥 FLS Surgical Skills Scoring Platform")
    gr.Markdown(
        "**Qwen2.5-VL-7B + v16 Multimodal LoRA** · "
        "Upload FLS task videos for AI-powered scoring with personalized tracking."
    )

    with gr.Tabs():
        # ── Tab 1: Login ──
        with gr.Tab("👤 Login"):
            gr.Markdown("### Sign in to track your progress")
            with gr.Row():
                with gr.Column():
                    login_email = gr.Textbox(label="Email", placeholder="resident@emory.edu")
                    login_name = gr.Textbox(label="Full Name", placeholder="Jane Smith")
                    login_pgy = gr.Dropdown(
                        choices=["PGY-1", "PGY-2", "PGY-3", "PGY-4", "PGY-5", "Fellow", "Attending"],
                        value="PGY-1",
                        label="Training Level"
                    )
                    login_btn = gr.Button("Sign In", variant="primary")
                with gr.Column():
                    login_status = gr.Markdown("")

            login_btn.click(
                fn=login_resident,
                inputs=[login_email, login_name, login_pgy],
                outputs=[login_status, current_email],
            )

        # ── Tab 2: Evaluate ──
        with gr.Tab("🎬 Evaluate Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    eval_email_display = gr.Textbox(
                        label="Logged in as", interactive=False,
                        placeholder="Sign in on the Login tab first"
                    )
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
                        lines=2,
                    )
                    evaluate_btn = gr.Button("🔍 Evaluate", variant="primary", size="lg")
                    gr.Markdown(
                        "**Tips:**\n"
                        "- Upload a video for vision-based scoring (~2-5 min)\n"
                        "- Leave video empty for instant text-only scoring\n"
                        "- Results are automatically saved to your profile"
                    )

                with gr.Column(scale=1):
                    score_display = gr.Textbox(label="Score", interactive=False)
                    summary_display = gr.Textbox(label="Technique Summary", interactive=False, lines=3)
                    save_status = gr.Textbox(label="Status", interactive=False)
                    json_output = gr.Code(label="Full Evaluation (JSON)", language="json")

            # Update email display when login state changes
            current_email.change(fn=lambda x: x, inputs=current_email, outputs=eval_email_display)

            evaluate_btn.click(
                fn=evaluate_fls,
                inputs=[current_email, task_dropdown, time_input, context_input, video_input],
                outputs=[json_output, score_display, summary_display, save_status],
            )

        # ── Tab 3: My Dashboard ──
        with gr.Tab("📊 My Dashboard"):
            gr.Markdown("### Your Performance History")
            dashboard_btn = gr.Button("🔄 Refresh Dashboard", variant="secondary")
            dashboard_summary = gr.Markdown("")
            dashboard_history = gr.Markdown("")
            dashboard_msg = gr.Textbox(visible=False)

            dashboard_btn.click(
                fn=get_dashboard,
                inputs=[current_email],
                outputs=[dashboard_summary, dashboard_history, dashboard_msg],
            )

        # ── Tab 4: Admin ──
        with gr.Tab("🔒 Admin"):
            gr.Markdown("### Program Director Export")
            gr.Markdown("Enter the admin password to view aggregate data and export results.")
            admin_pw = gr.Textbox(label="Admin Password", type="password")
            admin_btn = gr.Button("Export Results", variant="secondary")
            admin_report = gr.Markdown("")
            admin_file = gr.File(label="Download CSV")

            admin_btn.click(
                fn=admin_export,
                inputs=[admin_pw],
                outputs=[admin_report, admin_file],
            )

    gr.Markdown(
        "---\n"
        "*FLS Scoring Platform v3 · Qwen2.5-VL-7B + v16 LoRA · "
        "Emory General Surgery Research*"
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
