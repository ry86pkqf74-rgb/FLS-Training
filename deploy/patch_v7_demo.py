"""Apply the remaining v7 patches that the inline ssh heredoc choked on.

Run on the Contabo box (or via ``ssh ... 'python3 -' < deploy/patch_v7_demo.py``).
Idempotent: re-running a second time is a safe no-op once the strings have
been replaced.
"""
from pathlib import Path

V003_DISCLAIMER_LINE = (
    'V003_DISCLAIMER = "This is AI-assisted training feedback, not an '
    'official FLS certification result.\\n\\n"\n\n'
)

OLD_LADDER = (
    "    if pct >= 75:\n"
    "        overall_msg = f\"Excellent performance on {task_name}. Your score of "
    "**{fls_score}/{max_score} ({pct:.0f}%)** places you well above the proficiency "
    "threshold. This demonstrates strong fundamental laparoscopic skills.\"\n"
    "    elif pct >= 55:\n"
    "        overall_msg = f\"Proficient performance on {task_name}. Your score of "
    "**{fls_score}/{max_score} ({pct:.0f}%)** meets the proficiency benchmark. With "
    "targeted practice on the areas identified below, you can push into the excellent range.\"\n"
    "    elif pct >= 35:\n"
    "        overall_msg = f\"Developing performance on {task_name}. Your score of "
    "**{fls_score}/{max_score} ({pct:.0f}%)** shows a solid foundation but falls below "
    "the proficiency cutoff. The good news is the specific areas below are very trainable "
    "with focused practice.\"\n"
    "    else:\n"
    "        overall_msg = f\"Your {task_name} score of **{fls_score}/{max_score} "
    "({pct:.0f}%)** indicates this is an area that would benefit from significant "
    "additional practice. Don't be discouraged — FLS skills improve dramatically "
    "with deliberate practice.\""
)

NEW_LADDER = (
    "    # v003 gating: only label readiness; never claim certification proficiency.\n"
    "    if pct >= 75:\n"
    "        overall_msg = f\"Strong training-score performance on {task_name}. "
    "Training score **{fls_score}/{max_score} ({pct:.0f}%)** meets the local training "
    "target for this task. {V003_DISCLAIMER.rstrip()}\"\n"
    "    elif pct >= 55:\n"
    "        overall_msg = f\"On-track training-score performance on {task_name}. "
    "Training score **{fls_score}/{max_score} ({pct:.0f}%)**. With targeted practice on "
    "the areas below, you can push into the strong range. {V003_DISCLAIMER.rstrip()}\"\n"
    "    elif pct >= 35:\n"
    "        overall_msg = f\"Borderline training-score performance on {task_name}. "
    "Training score **{fls_score}/{max_score} ({pct:.0f}%)**. Specific areas below are "
    "trainable with focused practice. {V003_DISCLAIMER.rstrip()}\"\n"
    "    else:\n"
    "        overall_msg = f\"Training score **{fls_score}/{max_score} ({pct:.0f}%)** "
    "for {task_name} indicates this is an area that would benefit from focused remediation "
    "before formal FLS readiness. {V003_DISCLAIMER.rstrip()}\""
)

PATCHES = [
    ("FLS Video Scoring Platform v5", "FLS Video Scoring Platform v7 (v003)"),
    ("def build_fallback_narrative(", V003_DISCLAIMER_LINE + "def build_fallback_narrative("),
    (OLD_LADDER, NEW_LADDER),
]


def main() -> None:
    p = Path("/opt/fls/fls_demo_v7.py")
    src = p.read_text()
    applied = 0
    missing = []
    for old, new in PATCHES:
        if old in src:
            src = src.replace(old, new, 1)
            applied += 1
        elif new[:60] in src:
            # Already patched.
            applied += 1
        else:
            missing.append(old[:80])
    p.write_text(src)
    print(f"applied {applied}/{len(PATCHES)} patches")
    for m in missing:
        print("MISSING:", m)


if __name__ == "__main__":
    main()
