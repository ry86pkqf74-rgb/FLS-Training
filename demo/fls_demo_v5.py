#!/usr/bin/env python3
"""
FLS Video Scoring Platform v5
- Two-pass evaluation: model scoring → narrative generation
- Rich detailed feedback with strengths and improvement areas
- Markdown report format (.md) with download support
- Email reports to residents after evaluation
- Per-resident report history with video name and date/time
- Admin can browse and view any resident's full reports
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
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
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
REPORTS_DIR = "/opt/fls/reports"
ADMIN_PASSWORD = "fls_admin_2026"

# Email configuration (Gmail SMTP — use App Password, not account password)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = ""  # e.g., "fls-reports@gmail.com" — set to enable email
SMTP_PASS = ""  # Gmail App Password — set to enable email
EMAIL_FROM_NAME = "FLS Scoring Platform"

TASK_DESCRIPTIONS = {
    "task1_peg_transfer": "Peg Transfer",
    "task2_pattern_cut": "Pattern Cutting (Circle Cut)",
    "task3_endoloop": "Endoloop (Ligating Loop)",
    "task4_extracorporeal_knot": "Extracorporeal Knot Tying",
    "task5_intracorporeal_suturing": "Intracorporeal Suturing",
    "task6_rings_of_rings": "Rings of Rings",
}

TASK_MAX_SCORES = {
    "task1_peg_transfer": 300,
    "task2_pattern_cut": 300,
    "task3_endoloop": 180,
    "task4_extracorporeal_knot": 420,
    "task5_intracorporeal_suturing": 600,
    "task6_rings_of_rings": 315,
}

TASK_SCORING_NOTES = {
    "task1_peg_transfer": "Score = 300 - (completion_time_penalty + dropped_pegs_penalty). Each dropped peg or out-of-field peg costs points. Proficient cutoff: ~180+.",
    "task2_pattern_cut": "Score = 300 - (completion_time_penalty + cutting_accuracy_penalty). Penalties for cutting outside the marked circle. Proficient cutoff: ~200+.",
    "task3_endoloop": "Score = 300 - (completion_time_penalty + placement_accuracy_penalty). Loop must be placed within 3mm of the marked line. Proficient cutoff: ~175+.",
    "task4_extracorporeal_knot": "Score = 300 - (completion_time_penalty + knot_security_penalty). Knot must be square and secure without excessive tension. Proficient cutoff: ~175+.",
    "task5_intracorporeal_suturing": "Score = 600 - (completion_time) - (penalties). Max 600. Penalties for poor needle placement, tissue damage, knot insecurity. Proficient cutoff: ~340+.",
    "task6_rings_of_rings": "Score = 315 - (completion_time) - (20 × rings_missed). Max 315. Each missed ring = 20 point penalty. Proficient cutoff: ~180+.",
}

GRS_DIMENSIONS = {
    "bimanual_dexterity": {
        "name": "Bimanual Dexterity",
        "description": "Coordinated use of both hands with instruments",
        "low": "Hands frequently do not coordinate; one hand dominates while the other is idle or counterproductive. Instruments collide or interfere with each other.",
        "mid": "Generally coordinated hand movements with occasional hesitation or one-handed tendencies. Some tasks completed bimanually but not consistently.",
        "high": "Excellent bilateral coordination throughout. Both hands work in concert with fluid, complementary movements. Instruments used simultaneously and purposefully.",
        "improve_tips": "Practice mirror exercises and ambidextrous drills. Focus on giving your non-dominant hand active roles during each step rather than just stabilizing.",
    },
    "depth_perception": {
        "name": "Depth Perception",
        "description": "Accurate spatial awareness in the 2D laparoscopic view",
        "low": "Repeatedly misjudges distances; instruments overshoot or undershoot targets. Multiple attempts needed to grasp objects or place sutures.",
        "mid": "Generally accurate depth judgment with occasional over- or under-reaching. Corrects quickly but initial approach is sometimes off-target.",
        "high": "Consistently accurate instrument placement on first approach. Excellent spatial awareness translating the 2D monitor image to 3D operative field.",
        "improve_tips": "Practice targeted grasping drills at varying depths. Work on approaching targets from consistent angles to build reliable depth cues from the 2D display.",
    },
    "efficiency": {
        "name": "Efficiency of Movement",
        "description": "Economy of motion and minimal wasted movement",
        "low": "Excessive, unnecessary movements. Instruments travel far from the operative field. Many wasted motions and repeated attempts at the same maneuver.",
        "mid": "Mostly efficient with some unnecessary movements. Occasional wandering of instruments or repeated adjustments that add time without improving outcome.",
        "high": "Highly efficient, economical movements. Every motion is purposeful and directed. Minimal instrument travel outside the immediate operative area.",
        "improve_tips": "Focus on planning each movement before executing. Reduce the 'search and peck' pattern by keeping instruments close to the working area and minimizing repositioning.",
    },
    "tissue_handling": {
        "name": "Tissue Handling / Material Respect",
        "description": "Appropriate force and care with tissues and materials",
        "low": "Excessive force on tissues/materials causing visible damage. Rough grasping, tearing, or inappropriate tension on suture material.",
        "mid": "Generally appropriate force with occasional rough handling. Some unnecessary tension or slight tissue/material damage that does not critically affect outcome.",
        "high": "Consistently gentle and appropriate tissue/material handling. Proper grasping force, appropriate traction, and careful manipulation throughout.",
        "improve_tips": "Practice with more delicate materials to develop better force sensitivity. Focus on grasping only what is needed and using traction rather than pulling force.",
    },
}

SYSTEM_PROMPT = (
    "You are an expert FLS (Fundamentals of Laparoscopic Surgery) "
    "surgical skills evaluator. You analyze surgical training videos "
    "and provide detailed, structured scoring evaluations following "
    "the FLS scoring rubric. Your evaluations include estimated FLS scores, "
    "score components, technique assessment, strengths, and improvement suggestions."
)

NARRATIVE_SYSTEM_PROMPT = (
    "You are a senior attending surgeon and expert FLS examiner providing "
    "detailed, educational feedback to a surgical resident after reviewing "
    "their FLS task performance. Your feedback should be thorough, specific, "
    "encouraging, and actionable — like a mentor debriefing after a skills lab session. "
    "Write in clear, professional prose. Address the resident directly using 'you/your'."
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
            video_filename TEXT,
            completion_time_sec REAL,
            estimated_fls_score REAL,
            score_components TEXT,
            technique_summary TEXT,
            strengths TEXT,
            improvement_suggestions TEXT,
            full_response TEXT,
            narrative_report TEXT,
            report_md_path TEXT,
            mode TEXT,
            frame_count INTEGER DEFAULT 0,
            additional_context TEXT,
            evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (email) REFERENCES residents(email)
        )
    """)
    # Migration: add new columns if upgrading from v3/v4
    for col, coltype in [
        ("narrative_report", "TEXT"),
        ("video_filename", "TEXT"),
        ("report_md_path", "TEXT"),
    ]:
        try:
            c.execute(f"ALTER TABLE evaluations ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # Column already exists
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


def save_evaluation(email, task_id, task_name, video_filename, completion_time,
                    score, score_components, technique_summary, strengths,
                    improvements, full_response, narrative_report, report_md_path,
                    mode, frame_count, context):
    """Save an evaluation result to the database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO evaluations
        (email, task_id, task_name, video_filename, completion_time_sec,
         estimated_fls_score, score_components, technique_summary, strengths,
         improvement_suggestions, full_response, narrative_report, report_md_path,
         mode, frame_count, additional_context)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        email, task_id, task_name, video_filename, completion_time, score,
        json.dumps(score_components) if score_components else None,
        technique_summary, json.dumps(strengths) if strengths else None,
        json.dumps(improvements) if improvements else None,
        full_response, narrative_report, report_md_path,
        mode, frame_count, context
    ))
    eval_id = c.lastrowid
    conn.commit()
    conn.close()
    return eval_id


def save_report_md(email, eval_id, task_name, video_filename, narrative_report):
    """Save the markdown report to disk and return the file path."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    # Create per-resident directory
    safe_email = email.replace("@", "_at_").replace(".", "_")
    resident_dir = os.path.join(REPORTS_DIR, safe_email)
    os.makedirs(resident_dir, exist_ok=True)

    # Build filename from video name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if video_filename:
        # Clean video filename for use in report filename
        base_name = os.path.splitext(os.path.basename(video_filename))[0]
        base_name = re.sub(r'[^\w\-]', '_', base_name)[:50]
        md_filename = f"{base_name}_{timestamp}.md"
    else:
        safe_task = task_name.replace(" ", "_").replace("(", "").replace(")", "")
        md_filename = f"{safe_task}_text_only_{timestamp}.md"

    md_path = os.path.join(resident_dir, md_filename)

    # Build full markdown document with header
    header = f"""# FLS Performance Report

**Resident:** {email}
**Task:** {task_name}
**Video:** {video_filename or 'Text-only assessment'}
**Date:** {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

---

"""
    full_md = header + narrative_report

    with open(md_path, "w") as f:
        f.write(full_md)

    # Update the database with the file path
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE evaluations SET report_md_path = ? WHERE id = ?", (md_path, eval_id))
    conn.commit()
    conn.close()

    return md_path


def send_report_email(email, task_name, video_filename, score, narrative_report, md_path):
    """Email the markdown report to the resident."""
    if not SMTP_USER or not SMTP_PASS:
        return False, "Email not configured (SMTP credentials not set)"

    try:
        msg = MIMEMultipart()
        msg["From"] = f"{EMAIL_FROM_NAME} <{SMTP_USER}>"
        msg["To"] = email
        msg["Subject"] = f"FLS Report: {task_name} — Score {score}"

        # Plain text body with key info
        body = f"""Hi,

Your FLS evaluation report is attached.

Task: {task_name}
Video: {video_filename or 'Text-only assessment'}
Score: {score}
Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

The full detailed report is attached as a Markdown (.md) file.
You can open it in any text editor, or view it rendered in VS Code, GitHub, etc.

— FLS Scoring Platform
"""
        msg.attach(MIMEText(body, "plain"))

        # Attach .md file
        if md_path and os.path.exists(md_path):
            with open(md_path, "rb") as f:
                attachment = MIMEBase("application", "octet-stream")
                attachment.set_payload(f.read())
                encoders.encode_base64(attachment)
                attachment.add_header(
                    "Content-Disposition",
                    f"attachment; filename={os.path.basename(md_path)}"
                )
                msg.attach(attachment)

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)

        return True, "Email sent"
    except Exception as e:
        return False, f"Email failed: {str(e)[:100]}"


# ── Report Retrieval ──

def get_resident_reports(email):
    """Get list of all reports for a resident (for My Reports tab)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT id, task_name, video_filename, estimated_fls_score,
               mode, evaluated_at, report_md_path
        FROM evaluations
        WHERE email = ?
        ORDER BY evaluated_at DESC
    """, (email,))
    rows = c.fetchall()
    conn.close()
    return rows


def get_report_by_id(eval_id):
    """Get full report for a specific evaluation."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT e.id, e.task_name, e.video_filename, e.estimated_fls_score,
               e.narrative_report, e.full_response, e.mode, e.frame_count,
               e.completion_time_sec, e.evaluated_at, e.report_md_path,
               r.name, r.email, r.pgy_level
        FROM evaluations e
        JOIN residents r ON e.email = r.email
        WHERE e.id = ?
    """, (eval_id,))
    row = c.fetchone()
    conn.close()
    return row


def get_all_residents_admin():
    """Get list of all residents with evaluation counts (for admin)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT r.email, r.name, r.pgy_level, COUNT(e.id) as eval_count,
               ROUND(AVG(e.estimated_fls_score), 1) as avg_score,
               MAX(e.evaluated_at) as last_eval
        FROM residents r
        LEFT JOIN evaluations e ON r.email = e.email
        GROUP BY r.email
        ORDER BY r.name
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def get_resident_reports_admin(email):
    """Get all reports for a specific resident (admin view)."""
    return get_resident_reports(email)


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

    c.execute("""
        SELECT COUNT(*), AVG(estimated_fls_score), MAX(estimated_fls_score),
               MIN(estimated_fls_score)
        FROM evaluations WHERE email = ? AND estimated_fls_score IS NOT NULL
    """, (email,))
    total, avg_score, max_score, min_score = c.fetchone()

    c.execute("""
        SELECT task_name, COUNT(*), AVG(estimated_fls_score), MAX(estimated_fls_score)
        FROM evaluations
        WHERE email = ? AND estimated_fls_score IS NOT NULL
        GROUP BY task_name
        ORDER BY task_name
    """, (email,))
    task_stats = c.fetchall()

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
               e.task_name, e.video_filename, e.estimated_fls_score,
               e.completion_time_sec, e.technique_summary, e.mode, e.evaluated_at
        FROM evaluations e
        JOIN residents r ON e.email = r.email
        ORDER BY e.evaluated_at DESC
    """)
    rows = c.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Name", "Email", "PGY Level", "Task", "Video File", "FLS Score",
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

# Initialize database and reports directory
init_db()
os.makedirs(REPORTS_DIR, exist_ok=True)
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


def generate_narrative(task_id, task_name, score_data, completion_time,
                       additional_context, frame_count):
    """Generate a detailed narrative evaluation from the model's score output."""
    max_score = TASK_MAX_SCORES.get(task_id, 300)
    scoring_notes = TASK_SCORING_NOTES.get(task_id, "")

    fls_score = score_data.get("estimated_fls_score", "N/A")
    grs_zscore = None
    sub_scores = {}
    errors = {}
    technique = score_data.get("technique_summary", "")

    sc = score_data.get("score_components", {})
    if isinstance(sc, dict):
        grs_zscore = sc.get("grs_zscore")

    gt = score_data.get("ground_truth", {})
    if isinstance(gt, dict):
        grs_zscore = grs_zscore or gt.get("grs_zscore")
        sub_scores = gt.get("sub_scores", {})
        errors = gt.get("errors", score_data.get("errors", {}))

    if not errors:
        errors = score_data.get("errors", {})
    if not sub_scores:
        sub_scores = score_data.get("sub_scores", {})

    if isinstance(fls_score, (int, float)):
        pct = fls_score / max_score * 100
        if pct >= 75:
            perf_level = "excellent"
        elif pct >= 55:
            perf_level = "proficient"
        elif pct >= 35:
            perf_level = "developing"
        else:
            perf_level = "needs significant improvement"
    else:
        pct = 0
        perf_level = "not determined"

    score_breakdown = f"Estimated FLS Score: {fls_score}"
    if max_score:
        score_breakdown += f" out of {max_score} ({pct:.0f}%)"
    if grs_zscore is not None:
        score_breakdown += f"\nGlobal Rating Scale z-score: {grs_zscore}"

    sub_text = ""
    if sub_scores:
        sub_text = "\nComponent z-scores:\n"
        for dim, val in sub_scores.items():
            dim_name = GRS_DIMENSIONS.get(dim, {}).get("name", dim.replace("_", " ").title())
            sub_text += f"  - {dim_name}: {val}\n"

    error_text = ""
    if errors and isinstance(errors, dict):
        flagged = [k.replace("_", " ") for k, v in errors.items() if v]
        not_flagged = [k.replace("_", " ") for k, v in errors.items() if not v]
        if flagged:
            error_text = f"\nErrors observed: {', '.join(flagged)}"
        if not_flagged:
            error_text += f"\nNo issues with: {', '.join(not_flagged)}"

    narrative_prompt = f"""Based on the following FLS evaluation data, write a detailed performance report for a surgical resident.

TASK: {task_name}
SCORING: {scoring_notes}

RESULTS:
{score_breakdown}
{sub_text}
{error_text}

Completion time: {int(completion_time) if completion_time else 'Not recorded'} seconds
Mode: {'Video analysis ({} frames)'.format(frame_count) if frame_count > 0 else 'Text-only assessment'}
{f'Additional context from resident: {additional_context}' if additional_context else ''}

Performance level: {perf_level}

Please write a comprehensive evaluation report with these sections:

## Overall Assessment
A 2-3 sentence summary of the performance including the score, what it means, and overall impression.

## Score Breakdown
Explain what the FLS score of {fls_score}/{max_score} means. Break down the components and what contributed to this score. Reference the completion time and any penalties.

## What You Did Well
Identify 3-4 specific strengths based on the sub-scores and error analysis. Be specific and encouraging. Reference the GRS dimensions where scores were strong.

## Areas for Improvement
Identify 2-3 specific areas where the resident can improve. For each area:
- What was observed
- Why it matters clinically
- A specific drill or practice technique to improve

## Practice Recommendations
Give 2-3 concrete next steps the resident should take in their next practice session. Be specific about what to focus on and how to structure practice.

Write in a warm but professional tone, as an experienced mentor would. Be specific, not generic."""

    messages = [
        {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
        {"role": "user", "content": narrative_prompt},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
        )

    narrative = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return narrative


def build_fallback_narrative(task_id, task_name, score_data, completion_time,
                             additional_context, frame_count):
    """Build a structured narrative from score data without a second model call."""
    max_score = TASK_MAX_SCORES.get(task_id, 300)
    scoring_notes = TASK_SCORING_NOTES.get(task_id, "")
    fls_score = score_data.get("estimated_fls_score", "N/A")

    sub_scores = {}
    errors = {}
    gt = score_data.get("ground_truth", {})
    if isinstance(gt, dict):
        sub_scores = gt.get("sub_scores", {})
        errors = gt.get("errors", {})
    if not sub_scores:
        sub_scores = score_data.get("sub_scores", {})
    if not errors:
        errors = score_data.get("errors", {})

    pct = (fls_score / max_score * 100) if isinstance(fls_score, (int, float)) and max_score else 0

    if pct >= 75:
        overall_msg = f"Excellent performance on {task_name}. Your score of **{fls_score}/{max_score} ({pct:.0f}%)** places you well above the proficiency threshold. This demonstrates strong fundamental laparoscopic skills."
    elif pct >= 55:
        overall_msg = f"Proficient performance on {task_name}. Your score of **{fls_score}/{max_score} ({pct:.0f}%)** meets the proficiency benchmark. With targeted practice on the areas identified below, you can push into the excellent range."
    elif pct >= 35:
        overall_msg = f"Developing performance on {task_name}. Your score of **{fls_score}/{max_score} ({pct:.0f}%)** shows a solid foundation but falls below the proficiency cutoff. The good news is the specific areas below are very trainable with focused practice."
    else:
        overall_msg = f"Your {task_name} score of **{fls_score}/{max_score} ({pct:.0f}%)** indicates this is an area that would benefit from significant additional practice. Don't be discouraged — FLS skills improve dramatically with deliberate practice."

    report = f"## Overall Assessment\n\n{overall_msg}\n\n"

    # Score breakdown
    report += f"## Score Breakdown\n\n"
    report += f"**FLS Score: {fls_score} / {max_score}**\n\n"
    report += f"{scoring_notes}\n\n"
    if completion_time and completion_time > 0:
        report += f"Your completion time was **{int(completion_time)} seconds**. "
        if "task5" in task_id:
            report += f"For intracorporeal suturing, the time directly reduces your score (600 - time - penalties = score).\n\n"
        elif "task6" in task_id:
            report += f"For this task, time directly reduces your score from the maximum of 315.\n\n"
        else:
            report += f"Faster completion times reduce the time penalty component of your score.\n\n"

    if sub_scores:
        report += "**Component Scores (GRS z-scores, higher is better):**\n\n"
        sorted_dims = sorted(sub_scores.items(), key=lambda x: float(x[1]) if x[1] else 0, reverse=True)
        for dim, val in sorted_dims:
            dim_info = GRS_DIMENSIONS.get(dim, {})
            dim_name = dim_info.get("name", dim.replace("_", " ").title())
            val_f = float(val) if val else 0
            if val_f >= 0.9:
                indicator = "Strong"
            elif val_f >= 0.5:
                indicator = "Adequate"
            else:
                indicator = "Needs work"
            report += f"- **{dim_name}**: z={val} ({indicator})\n"
        report += "\n"

    if errors and isinstance(errors, dict):
        flagged = [(k, v) for k, v in errors.items() if v]
        clean = [(k, v) for k, v in errors.items() if not v]
        if flagged:
            report += "**Errors Identified:**\n\n"
            for err_name, _ in flagged:
                report += f"- {err_name.replace('_', ' ').title()}\n"
            report += "\n"
        if clean:
            report += "**Clean Areas (no errors):**\n\n"
            for err_name, _ in clean:
                report += f"- {err_name.replace('_', ' ').title()}\n"
            report += "\n"

    # What you did well
    report += "## What You Did Well\n\n"
    strengths_found = 0
    if sub_scores:
        sorted_dims = sorted(sub_scores.items(), key=lambda x: float(x[1]) if x[1] else 0, reverse=True)
        for dim, val in sorted_dims:
            val_f = float(val) if val else 0
            if val_f >= 0.7 and strengths_found < 3:
                dim_info = GRS_DIMENSIONS.get(dim, {})
                dim_name = dim_info.get("name", dim.replace("_", " ").title())
                high_desc = dim_info.get("high", f"Strong performance in {dim_name.lower()}.")
                report += f"**{dim_name} (z={val}):** {high_desc}\n\n"
                strengths_found += 1

    if errors and isinstance(errors, dict):
        clean_errors = [k for k, v in errors.items() if not v]
        if clean_errors and strengths_found < 4:
            report += f"**Error-free execution:** You completed the task without issues in: {', '.join(e.replace('_', ' ') for e in clean_errors)}. This shows good attention to technique fundamentals.\n\n"
            strengths_found += 1

    if strengths_found == 0:
        report += "You completed the task, which is itself a meaningful step. Each attempt builds muscle memory and spatial awareness that will compound over time.\n\n"

    # Areas for improvement
    report += "## Areas for Improvement\n\n"
    improvements_found = 0
    if sub_scores:
        sorted_dims = sorted(sub_scores.items(), key=lambda x: float(x[1]) if x[1] else 0)
        for dim, val in sorted_dims:
            val_f = float(val) if val else 0
            if val_f < 0.85 and improvements_found < 3:
                dim_info = GRS_DIMENSIONS.get(dim, {})
                dim_name = dim_info.get("name", dim.replace("_", " ").title())
                if val_f < 0.5:
                    level_desc = dim_info.get("low", f"This area needs focused attention.")
                else:
                    level_desc = dim_info.get("mid", f"This area is adequate but has room for growth.")
                improve_tips = dim_info.get("improve_tips", "Practice targeted drills focusing on this dimension.")
                report += f"**{dim_name} (z={val}):** {level_desc}\n\n"
                report += f"*How to improve:* {improve_tips}\n\n"
                improvements_found += 1

    if errors and isinstance(errors, dict):
        flagged = [k for k, v in errors.items() if v]
        for err in flagged[:2]:
            err_name = err.replace("_", " ").title()
            report += f"**{err_name}:** This error was flagged during your performance. In a clinical setting, this could lead to complications. Focus on the specific step where this occurred and practice it in isolation.\n\n"
            improvements_found += 1

    if improvements_found == 0:
        report += "Your scores are strong across all dimensions. To continue improving, focus on reducing completion time while maintaining your technique quality.\n\n"

    # Practice recommendations
    report += "## Practice Recommendations\n\n"
    report += f"1. **Targeted repetition:** Perform 3-5 repetitions of {task_name} in your next practice session, focusing specifically on "
    if sub_scores:
        weakest = sorted(sub_scores.items(), key=lambda x: float(x[1]) if x[1] else 0)
        if weakest:
            weakest_name = GRS_DIMENSIONS.get(weakest[0][0], {}).get("name", weakest[0][0].replace("_", " "))
            report += f"{weakest_name.lower()}.\n\n"
        else:
            report += "maintaining consistency.\n\n"
    else:
        report += "smooth, deliberate movements.\n\n"

    report += f"2. **Time challenge:** After warming up, try to complete the task 10-15% faster than your current time of {int(completion_time) if completion_time else '—'} seconds while keeping technique clean. Speed and precision should develop together.\n\n"

    report += f"3. **Self-assessment:** Before your next evaluation, record yourself and review the video. Compare your instrument movements to expert demonstration videos. This builds the self-awareness that accelerates improvement.\n\n"

    if frame_count > 0:
        report += f"\n---\n*This evaluation was based on video analysis ({frame_count} frames extracted) using the Qwen2.5-VL multimodal model with FLS-specialized LoRA fine-tuning.*\n"
    else:
        report += f"\n---\n*This was a text-only evaluation. For more detailed and accurate feedback, upload a video of your performance.*\n"

    return report


def evaluate_fls(email, task_id, completion_time, additional_context, video_file):
    """Run FLS evaluation: score with LoRA model, then generate narrative."""
    if not email or "@" not in email:
        return "Please log in first (enter your email on the Login tab)", "", "", ""

    task_name = TASK_DESCRIPTIONS.get(task_id, task_id)
    tmpdir = None

    # Get original video filename
    video_filename = None
    if video_file is not None:
        video_filename = os.path.basename(video_file)

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

        # ── Pass 1: Get structured scores from the LoRA-tuned model ──
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

        # Parse the model's JSON scores
        mode = "video" if frame_count > 0 else "text-only"
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    parsed = None
            else:
                parsed = None

        if parsed:
            score = parsed.get("estimated_fls_score", None)
            score_components = parsed.get("score_components", None)
            technique_summary = parsed.get("technique_summary", "")
            strengths = parsed.get("strengths", [])
            improvements = parsed.get("improvement_suggestions", [])
            formatted_json = json.dumps(parsed, indent=2)

            # ── Pass 2: Generate detailed narrative ──
            try:
                narrative = generate_narrative(
                    task_id, task_name, parsed, completion_time,
                    additional_context, frame_count
                )
                if len(narrative) < 200 or "##" not in narrative:
                    narrative = build_fallback_narrative(
                        task_id, task_name, parsed, completion_time,
                        additional_context, frame_count
                    )
            except Exception as e:
                print(f"Narrative generation failed: {e}, using fallback")
                narrative = build_fallback_narrative(
                    task_id, task_name, parsed, completion_time,
                    additional_context, frame_count
                )

            # Save to database
            eval_id = save_evaluation(
                email, task_id, task_name, video_filename, completion_time,
                score, score_components, technique_summary, strengths,
                improvements, formatted_json, narrative, None,
                mode, frame_count, additional_context
            )

            # Save .md report to disk
            md_path = save_report_md(email, eval_id, task_name, video_filename, narrative)

            # Email the report
            email_status = ""
            if SMTP_USER and SMTP_PASS:
                ok, msg = send_report_email(
                    email, task_name, video_filename, score, narrative, md_path
                )
                email_status = f" | {'Email sent' if ok else msg}"
            else:
                email_status = " | Email not configured"

            max_score = TASK_MAX_SCORES.get(task_id, 300)
            pct = (score / max_score * 100) if isinstance(score, (int, float)) and max_score else 0
            score_text = f"FLS Score: {score} / {max_score} ({pct:.0f}%)  [{mode}, {frame_count} frames]"
            status = f"Saved for {email}{email_status}"
            return narrative, score_text, formatted_json, status
        else:
            eval_id = save_evaluation(
                email, task_id, task_name, video_filename, completion_time,
                None, None, "", None, None, response, response, None,
                mode, frame_count, additional_context
            )
            md_path = save_report_md(email, eval_id, task_name, video_filename, response)
            return response, f"Score: see report [{mode}]", response, "Saved (raw)"

    finally:
        if tmpdir and os.path.exists(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


def login_resident(email, name, pgy_level):
    """Handle resident login/registration."""
    if not email or "@" not in email:
        return "Please enter a valid email address", ""
    if not name:
        return "Please enter your name", ""

    register_resident(email, name, pgy_level)
    stats = get_resident_stats(email)

    welcome = f"### Welcome, {name} ({pgy_level})!\n"
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

    summary = f"""## Your Performance Summary

**Total Evaluations:** {stats['total_evaluations']}
**Average Score:** {stats['avg_score']}
**Best Score:** {stats['max_score']} | **Lowest:** {stats['min_score']}

### Scores by Task
"""
    for task_name, count, avg, best in stats["task_stats"]:
        summary += f"- **{task_name}**: {count} attempts, avg {avg:.1f}, best {best}\n"

    if stats["recent_scores"]:
        trend = "\n### Recent Scores (newest first)\n"
        for score, date in stats["recent_scores"]:
            trend += f"- {date[:10]}: **{score}**\n"
    else:
        trend = ""

    history_text = "\n### Full History\n\n| Date | Task | Score | Time | Mode |\n|------|------|-------|------|------|\n"
    for task_name, score, time_sec, summary_text, mode, date in history[:20]:
        time_str = f"{int(time_sec)}s" if time_sec else "—"
        score_str = str(score) if score else "—"
        history_text += f"| {date[:16]} | {task_name} | {score_str} | {time_str} | {mode} |\n"

    return summary + trend, history_text, ""


# ── My Reports Tab Functions ──

def load_reports_list(email):
    """Load the list of reports for the My Reports tab dropdown."""
    if not email or "@" not in email:
        return gr.update(choices=[], value=None), "Please log in first."

    reports = get_resident_reports(email)
    if not reports:
        return gr.update(choices=[], value=None), "No reports yet. Evaluate a video to generate your first report!"

    choices = []
    for eval_id, task_name, video_fn, score, mode, eval_date, md_path in reports:
        video_display = video_fn if video_fn else "Text-only"
        score_display = f"{score}" if score else "N/A"
        label = f"[{eval_date[:16]}] {task_name} — {video_display} (Score: {score_display})"
        choices.append((label, str(eval_id)))

    return gr.update(choices=choices, value=choices[0][1] if choices else None), f"Found {len(choices)} report(s)."


def view_report(eval_id_str):
    """View a specific report by evaluation ID."""
    if not eval_id_str:
        return "Select a report from the list above.", None

    try:
        eval_id = int(eval_id_str)
    except (ValueError, TypeError):
        return "Invalid report selection.", None

    row = get_report_by_id(eval_id)
    if not row:
        return "Report not found.", None

    (eid, task_name, video_fn, score, narrative, full_resp,
     mode, frame_count, comp_time, eval_date, md_path,
     res_name, res_email, pgy_level) = row

    if narrative:
        report_display = narrative
    else:
        report_display = f"*No detailed report available for this evaluation.*\n\nRaw output:\n```\n{full_resp}\n```"

    # Return the md file path for download
    download_file = md_path if md_path and os.path.exists(md_path) else None

    return report_display, download_file


# ── Admin Tab Functions ──

def admin_login(password):
    """Authenticate admin and return resident list."""
    if password != ADMIN_PASSWORD:
        return "Invalid admin password", gr.update(choices=[], value=None), "", ""

    residents = get_all_residents_admin()
    if not residents:
        return "No residents registered yet.", gr.update(choices=[], value=None), "", ""

    summary = f"## Admin Dashboard\n\n**{len(residents)} residents registered**\n\n"
    summary += "| Name | PGY | Evaluations | Avg Score | Last Activity |\n"
    summary += "|------|-----|-------------|-----------|---------------|\n"

    choices = []
    for email, name, pgy, count, avg, last in residents:
        avg_str = str(avg) if avg else "—"
        last_str = last[:16] if last else "—"
        summary += f"| {name} | {pgy} | {count} | {avg_str} | {last_str} |\n"
        choices.append((f"{name} ({email}) — {count} evals", email))

    return summary, gr.update(choices=choices, value=choices[0][1] if choices else None), "", ""


def admin_load_resident_reports(email):
    """Load a specific resident's reports for admin viewing."""
    if not email:
        return gr.update(choices=[], value=None), ""

    reports = get_resident_reports(email)
    if not reports:
        return gr.update(choices=[], value=None), f"No reports for {email}"

    choices = []
    for eval_id, task_name, video_fn, score, mode, eval_date, md_path in reports:
        video_display = video_fn if video_fn else "Text-only"
        score_display = f"{score}" if score else "N/A"
        label = f"[{eval_date[:16]}] {task_name} — {video_display} (Score: {score_display})"
        choices.append((label, str(eval_id)))

    return gr.update(choices=choices, value=choices[0][1] if choices else None), f"{len(choices)} reports for {email}"


def admin_view_report(eval_id_str):
    """View a specific report in admin mode."""
    if not eval_id_str:
        return "Select a report from the list above.", None

    try:
        eval_id = int(eval_id_str)
    except (ValueError, TypeError):
        return "Invalid report selection.", None

    row = get_report_by_id(eval_id)
    if not row:
        return "Report not found.", None

    (eid, task_name, video_fn, score, narrative, full_resp,
     mode, frame_count, comp_time, eval_date, md_path,
     res_name, res_email, pgy_level) = row

    # Build admin header
    header = f"**Resident:** {res_name} ({res_email}) — {pgy_level}\n"
    header += f"**Task:** {task_name} | **Score:** {score} | **Mode:** {mode}\n"
    header += f"**Video:** {video_fn or 'Text-only'} | **Date:** {eval_date}\n\n---\n\n"

    if narrative:
        report_display = header + narrative
    else:
        report_display = header + f"*No detailed report available.*\n\nRaw output:\n```\n{full_resp}\n```"

    download_file = md_path if md_path and os.path.exists(md_path) else None
    return report_display, download_file


def admin_export(password):
    """Admin function to export all data as CSV."""
    if password != ADMIN_PASSWORD:
        return "Invalid admin password", None

    csv_data = export_all_results()

    export_path = "/opt/fls/exports"
    os.makedirs(export_path, exist_ok=True)
    filename = f"fls_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join(export_path, filename)
    with open(filepath, "w") as f:
        f.write(csv_data)

    return f"CSV exported: {filename}", filepath


# ── Gradio UI ──
with gr.Blocks(title="FLS Scoring Platform", theme=gr.themes.Soft()) as demo:
    # State
    current_email = gr.State("")

    gr.Markdown("# FLS Surgical Skills Scoring Platform")
    gr.Markdown(
        "**Qwen2.5-VL-7B + v16 Multimodal LoRA** · "
        "Upload FLS task videos for AI-powered scoring with detailed performance feedback."
    )

    with gr.Tabs():
        # ── Tab 1: Login ──
        with gr.Tab("Login"):
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
        with gr.Tab("Evaluate Video"):
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
                    evaluate_btn = gr.Button("Evaluate", variant="primary", size="lg")
                    gr.Markdown(
                        "**Tips:**\n"
                        "- Upload a video for vision-based scoring (~3-7 min with narrative)\n"
                        "- Leave video empty for instant text-only scoring\n"
                        "- Results are saved to your profile and emailed to you"
                    )

                with gr.Column(scale=2):
                    score_display = gr.Textbox(label="Score", interactive=False)
                    save_status = gr.Textbox(label="Status", interactive=False)
                    report_output = gr.Markdown(
                        label="Performance Report",
                        value="*Submit a video or text evaluation to see your detailed report here.*"
                    )
                    with gr.Accordion("Raw Model Output (JSON)", open=False):
                        json_output = gr.Code(label="Raw Scores", language="json")

            current_email.change(fn=lambda x: x, inputs=current_email, outputs=eval_email_display)

            evaluate_btn.click(
                fn=evaluate_fls,
                inputs=[current_email, task_dropdown, time_input, context_input, video_input],
                outputs=[report_output, score_display, json_output, save_status],
            )

        # ── Tab 3: My Reports ──
        with gr.Tab("My Reports"):
            gr.Markdown("### Your Evaluation Reports")
            gr.Markdown("View detailed reports from all your past evaluations. Each report includes the video assessed, date/time, and full performance analysis.")
            reports_refresh_btn = gr.Button("Load My Reports", variant="secondary")
            reports_status = gr.Markdown("")
            reports_dropdown = gr.Dropdown(
                label="Select a report to view",
                choices=[],
                interactive=True,
            )
            report_view = gr.Markdown("")
            report_download = gr.File(label="Download Report (.md)")

            reports_refresh_btn.click(
                fn=load_reports_list,
                inputs=[current_email],
                outputs=[reports_dropdown, reports_status],
            )
            reports_dropdown.change(
                fn=view_report,
                inputs=[reports_dropdown],
                outputs=[report_view, report_download],
            )

        # ── Tab 4: My Dashboard ──
        with gr.Tab("Dashboard"):
            gr.Markdown("### Your Performance History")
            dashboard_btn = gr.Button("Refresh Dashboard", variant="secondary")
            dashboard_summary = gr.Markdown("")
            dashboard_history = gr.Markdown("")
            dashboard_msg = gr.Textbox(visible=False)

            dashboard_btn.click(
                fn=get_dashboard,
                inputs=[current_email],
                outputs=[dashboard_summary, dashboard_history, dashboard_msg],
            )

        # ── Tab 5: Admin ──
        with gr.Tab("Admin"):
            gr.Markdown("### Program Director View")
            gr.Markdown("Enter the admin password to view all residents and their reports.")

            with gr.Row():
                admin_pw = gr.Textbox(label="Admin Password", type="password", scale=2)
                admin_login_btn = gr.Button("Authenticate", variant="primary", scale=1)

            admin_summary = gr.Markdown("")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Select Resident")
                    admin_resident_dropdown = gr.Dropdown(
                        label="Resident",
                        choices=[],
                        interactive=True,
                    )
                    admin_resident_status = gr.Markdown("")

                    gr.Markdown("#### Select Report")
                    admin_report_dropdown = gr.Dropdown(
                        label="Report",
                        choices=[],
                        interactive=True,
                    )

                with gr.Column(scale=2):
                    admin_report_view = gr.Markdown("")
                    admin_report_download = gr.File(label="Download Report (.md)")

            with gr.Accordion("Export All Data", open=False):
                admin_export_btn = gr.Button("Export CSV", variant="secondary")
                admin_export_status = gr.Markdown("")
                admin_export_file = gr.File(label="Download CSV")

            # Admin event handlers
            admin_login_btn.click(
                fn=admin_login,
                inputs=[admin_pw],
                outputs=[admin_summary, admin_resident_dropdown, admin_resident_status, admin_report_view],
            )
            admin_resident_dropdown.change(
                fn=admin_load_resident_reports,
                inputs=[admin_resident_dropdown],
                outputs=[admin_report_dropdown, admin_resident_status],
            )
            admin_report_dropdown.change(
                fn=admin_view_report,
                inputs=[admin_report_dropdown],
                outputs=[admin_report_view, admin_report_download],
            )
            admin_export_btn.click(
                fn=admin_export,
                inputs=[admin_pw],
                outputs=[admin_export_status, admin_export_file],
            )

    gr.Markdown(
        "---\n"
        "*FLS Scoring Platform v5 · Qwen2.5-VL-7B + v16 LoRA · "
        "Emory General Surgery Research*"
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
