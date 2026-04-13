#!/usr/bin/env python3
"""026_auto_validate.py — DuckDB-backed auto-validation of dual-teacher scores.

For each video, pulls both teacher scores (teacher_claude and teacher_gpt4o)
from the DuckDB database at $FLS_DB_PATH and applies self-consistency +
time-anchor rules:

    ACCEPTED    |claude_fls − gpt4o_fls| ≤ 25
                AND |claude_time − gpt4o_time| ≤ 15s
                AND both confidence > 0.40
    QUARANTINED score divergence 25–50  OR one confidence < 0.40
    REJECTED    divergence > 50         OR score outside time-anchor band

Time-anchor band (using video duration T from the videos table):
    floor   = 600 − T − 20
    ceiling = 600 − T

Writes validation_status, validation_reason, validated_at to a
`validations` table and prints a Rich table summary.

Usage:
    python scripts/026_auto_validate.py --all
    python scripts/026_auto_validate.py --video-id V31
    python scripts/026_auto_validate.py --all --revalidate
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb
import typer
from rich.console import Console
from rich.table import Table

# ----------------------------------------------------------------------------
# Validation thresholds (from docs/EXECUTION_PLAN.md)
# ----------------------------------------------------------------------------

MAX_SCORE_DIVERGENCE = 25         # FLS pts — accept ceiling
QUARANTINE_SCORE_DIVERGENCE = 50  # FLS pts — reject threshold
MAX_TIME_DIVERGENCE = 15          # seconds between teacher completion times
MIN_CONFIDENCE = 0.40             # minimum teacher confidence

MAX_REASONABLE_PENALTY = 20       # ceiling penalties for floor of time-anchor
MIN_REASONABLE_PENALTY = 0        # floor penalties for ceiling of time-anchor

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "fls_training.duckdb"
DEFAULT_SCORES_DIR = Path(__file__).resolve().parent.parent / "memory" / "scores"
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent.parent / "memory" / "validation_results.jsonl"

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


# ----------------------------------------------------------------------------
# DB helpers
# ----------------------------------------------------------------------------

def _db_path() -> Path:
    env = os.environ.get("FLS_DB_PATH")
    return Path(env) if env else DEFAULT_DB_PATH


def _connect(path: Path) -> duckdb.DuckDBPyConnection:
    if not path.exists():
        console.print(f"[red]DuckDB not found at {path}.[/red]")
        console.print("Hint: set FLS_DB_PATH or run scripts/035_backfill_duckdb.py first.")
        raise typer.Exit(code=2)
    return duckdb.connect(str(path))


def _ensure_validations_table(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS validations (
            video_id           VARCHAR PRIMARY KEY,
            validation_status  VARCHAR,
            validation_reason  VARCHAR,
            claude_fls         DOUBLE,
            gpt4o_fls          DOUBLE,
            score_delta        DOUBLE,
            claude_time        DOUBLE,
            gpt4o_time         DOUBLE,
            time_delta         DOUBLE,
            claude_confidence  DOUBLE,
            gpt4o_confidence   DOUBLE,
            duration_seconds   DOUBLE,
            floor              DOUBLE,
            ceiling            DOUBLE,
            validated_at       TIMESTAMP
        )
        """
    )


def _score_columns(conn: duckdb.DuckDBPyConnection) -> tuple[str, str, str]:
    """Return (fls_col, time_col, conf_col) for whichever schema this DB uses.

    The repo currently has two flavours of the scores table — one with
    `fls_score / completion_time / confidence` and one with
    `estimated_fls_score / completion_time_seconds / confidence_score`.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info('scores')").fetchall()}
    fls = "fls_score" if "fls_score" in cols else "estimated_fls_score"
    t = "completion_time" if "completion_time" in cols else "completion_time_seconds"
    c = "confidence" if "confidence" in cols else "confidence_score"
    return fls, t, c


def _fetch_latest_teacher_scores(
    conn: duckdb.DuckDBPyConnection,
    video_id: str,
    fls_col: str,
    time_col: str,
    conf_col: str,
) -> dict[str, dict]:
    """Return {'teacher_claude': {...}, 'teacher_gpt4o': {...}} — latest per source."""
    rows = conn.execute(
        f"""
        SELECT source, {fls_col}, {time_col}, {conf_col}, scored_at
        FROM scores
        WHERE video_id = ?
          AND source IN ('teacher_claude', 'teacher_gpt4o', 'teacher_gpt')
        ORDER BY scored_at DESC
        """,
        [video_id],
    ).fetchall()

    latest: dict[str, dict] = {}
    for source, fls, t, conf, scored_at in rows:
        # normalise teacher_gpt → teacher_gpt4o
        key = "teacher_gpt4o" if source in ("teacher_gpt", "teacher_gpt4o") else source
        if key in latest:
            continue  # already have newer row (ORDER BY DESC)
        latest[key] = {
            "fls": float(fls or 0),
            "time": float(t or 0),
            "confidence": float(conf or 0),
            "scored_at": scored_at,
        }
    return latest


def _fetch_video_duration(conn: duckdb.DuckDBPyConnection, video_id: str) -> Optional[float]:
    row = conn.execute(
        "SELECT duration_seconds FROM videos WHERE id = ?",
        [video_id],
    ).fetchone()
    if row and row[0]:
        return float(row[0])
    return None


def _candidate_video_ids(
    conn: duckdb.DuckDBPyConnection, revalidate: bool
) -> list[str]:
    if revalidate:
        return [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT video_id FROM scores ORDER BY video_id"
            ).fetchall()
        ]
    return [
        r[0]
        for r in conn.execute(
            """
            SELECT DISTINCT s.video_id
            FROM scores s
            LEFT JOIN validations v ON v.video_id = s.video_id
            WHERE v.video_id IS NULL
            ORDER BY s.video_id
            """
        ).fetchall()
    ]


# ----------------------------------------------------------------------------
# Validation logic
# ----------------------------------------------------------------------------

def _time_anchor(duration_seconds: float) -> tuple[float, float]:
    """floor = 600 − T − 20, ceiling = 600 − T."""
    ceiling = 600.0 - duration_seconds
    floor = ceiling - MAX_REASONABLE_PENALTY
    return floor, ceiling


def validate_video(
    teachers: dict[str, dict],
    duration_seconds: Optional[float],
) -> dict:
    """Apply accept/quarantine/reject rules. Returns a dict suitable for insert/display."""
    claude = teachers.get("teacher_claude")
    gpt = teachers.get("teacher_gpt4o")

    result: dict = {
        "claude_fls": (claude or {}).get("fls", 0.0),
        "gpt4o_fls": (gpt or {}).get("fls", 0.0),
        "claude_time": (claude or {}).get("time", 0.0),
        "gpt4o_time": (gpt or {}).get("time", 0.0),
        "claude_confidence": (claude or {}).get("confidence", 0.0),
        "gpt4o_confidence": (gpt or {}).get("confidence", 0.0),
        "duration_seconds": duration_seconds or 0.0,
        "score_delta": None,
        "time_delta": None,
        "floor": None,
        "ceiling": None,
    }

    if claude is None or gpt is None:
        missing = [
            name
            for name, val in (("teacher_claude", claude), ("teacher_gpt4o", gpt))
            if val is None
        ]
        result["status"] = "REJECTED"
        result["reason"] = f"missing teacher scores: {', '.join(missing)}"
        return result

    score_delta = abs(claude["fls"] - gpt["fls"])
    time_delta = abs(claude["time"] - gpt["time"])
    result["score_delta"] = score_delta
    result["time_delta"] = time_delta

    reasons: list[str] = []

    # --- hard REJECT conditions ---------------------------------------------
    if score_delta > QUARANTINE_SCORE_DIVERGENCE:
        reasons.append(
            f"score divergence {score_delta:.0f} > {QUARANTINE_SCORE_DIVERGENCE}"
        )

    if duration_seconds and duration_seconds > 0:
        floor, ceiling = _time_anchor(duration_seconds)
        result["floor"] = floor
        result["ceiling"] = ceiling
        if not (floor <= claude["fls"] <= ceiling):
            reasons.append(
                f"claude {claude['fls']:.0f} outside time-anchor "
                f"[{floor:.0f}, {ceiling:.0f}]"
            )
        if not (floor <= gpt["fls"] <= ceiling):
            reasons.append(
                f"gpt4o {gpt['fls']:.0f} outside time-anchor "
                f"[{floor:.0f}, {ceiling:.0f}]"
            )

    if any(
        "time-anchor" in r or f"> {QUARANTINE_SCORE_DIVERGENCE}" in r for r in reasons
    ):
        result["status"] = "REJECTED"
        result["reason"] = "; ".join(reasons)
        return result

    # --- QUARANTINE conditions ----------------------------------------------
    if score_delta > MAX_SCORE_DIVERGENCE:
        reasons.append(
            f"score divergence {score_delta:.0f} > {MAX_SCORE_DIVERGENCE}"
        )
    if time_delta > MAX_TIME_DIVERGENCE:
        reasons.append(f"time divergence {time_delta:.0f}s > {MAX_TIME_DIVERGENCE}s")
    if claude["confidence"] < MIN_CONFIDENCE:
        reasons.append(f"claude confidence {claude['confidence']:.2f} < {MIN_CONFIDENCE}")
    if gpt["confidence"] < MIN_CONFIDENCE:
        reasons.append(f"gpt4o confidence {gpt['confidence']:.2f} < {MIN_CONFIDENCE}")

    if not reasons:
        result["status"] = "ACCEPTED"
        result["reason"] = "all checks passed"
    else:
        result["status"] = "QUARANTINED"
        result["reason"] = "; ".join(reasons)
    return result


def validate_pair_only(teachers: dict[str, dict]) -> dict[str, Any]:
    """Apply the pairwise file-based validation requested for score JSON files."""
    claude = teachers.get("teacher_claude")
    gpt = teachers.get("teacher_gpt4o")

    result: dict[str, Any] = {
        "claude_fls": (claude or {}).get("fls"),
        "gpt4o_fls": (gpt or {}).get("fls"),
        "claude_time": (claude or {}).get("time"),
        "gpt4o_time": (gpt or {}).get("time"),
        "claude_confidence": (claude or {}).get("confidence"),
        "gpt4o_confidence": (gpt or {}).get("confidence"),
        "score_delta": None,
        "time_delta": None,
    }

    if claude is None or gpt is None:
        missing = [
            name
            for name, value in (("teacher_claude", claude), ("teacher_gpt4o", gpt))
            if value is None
        ]
        result["status"] = "REJECTED"
        result["reason"] = f"missing teacher scores: {', '.join(missing)}"
        return result

    score_delta = abs(claude["fls"] - gpt["fls"])
    time_delta = abs(claude["time"] - gpt["time"])
    result["score_delta"] = score_delta
    result["time_delta"] = time_delta

    if (
        score_delta <= MAX_SCORE_DIVERGENCE
        and time_delta <= MAX_TIME_DIVERGENCE
        and claude["confidence"] > MIN_CONFIDENCE
        and gpt["confidence"] > MIN_CONFIDENCE
    ):
        result["status"] = "ACCEPTED"
        result["reason"] = "pairwise thresholds passed"
        return result

    reasons: list[str] = []
    if score_delta > MAX_SCORE_DIVERGENCE:
        reasons.append(f"score delta {score_delta:.1f} > {MAX_SCORE_DIVERGENCE}")
    if time_delta > MAX_TIME_DIVERGENCE:
        reasons.append(f"time delta {time_delta:.1f}s > {MAX_TIME_DIVERGENCE}s")
    if claude["confidence"] <= MIN_CONFIDENCE:
        reasons.append(f"claude confidence {claude['confidence']:.2f} <= {MIN_CONFIDENCE}")
    if gpt["confidence"] <= MIN_CONFIDENCE:
        reasons.append(f"gpt4o confidence {gpt['confidence']:.2f} <= {MIN_CONFIDENCE}")

    result["status"] = "QUARANTINED"
    result["reason"] = "; ".join(reasons)
    return result


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_score_payload(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    source = payload.get("source")
    if source not in {"teacher_claude", "teacher_gpt", "teacher_gpt4o", "teacher_haiku"}:
        return None

    video_id = payload.get("video_id")
    if not video_id:
        return None

    scored_at = payload.get("scored_at") or ""
    normalized_source = "teacher_gpt4o" if source in {"teacher_gpt", "teacher_gpt4o", "teacher_haiku"} else source
    return {
        "path": str(path),
        "video_id": video_id,
        "source": normalized_source,
        "scored_at": scored_at,
        "fls": _coerce_float(
            payload.get("estimated_fls_score")
            or payload.get("fls_score")
            or (payload.get("score_components") or {}).get("total_fls_score")
        ),
        "time": _coerce_float(
            payload.get("completion_time_seconds")
            or payload.get("completion_time")
        ),
        "confidence": _coerce_float(
            payload.get("confidence_score")
            or payload.get("confidence")
        ),
        "model_name": payload.get("model_name"),
        "id": payload.get("id"),
    }


def _collect_latest_file_scores(scores_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Recursively scan scores_dir for all teacher score JSONs.

    Handles both naming conventions:
      - Root-level:  score_claude_*.json / score_gpt_*.json
      - Dated dirs:  2026-04-07/V15_video_claude-sonnet-4_*.json
                     2026-04-08/abc123_gpt-4o_*.json

    Skips consensus files and quarantine subdirectories.
    """
    by_video: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(scores_dir.rglob("*.json")):
        # Skip consensus files, quarantine dirs, and non-teacher files
        fname = path.name.lower()
        if "consensus" in fname:
            continue
        if "_quarantine" in str(path):
            continue

        payload = _load_score_payload(path)
        if payload is None:
            continue

        video_scores = by_video.setdefault(payload["video_id"], {})
        current = video_scores.get(payload["source"])
        current_key = (current or {}).get("scored_at", "")
        next_key = payload.get("scored_at", "")
        if current is None or next_key >= current_key:
            video_scores[payload["source"]] = payload
    return by_video


def _validate_score_directory(scores_dir: Path, output_path: Path) -> dict[str, int]:
    if not scores_dir.exists():
        console.print(f"[red]Scores directory not found: {scores_dir}[/red]")
        raise typer.Exit(code=2)

    latest_scores = _collect_latest_file_scores(scores_dir)
    counts = {"ACCEPTED": 0, "QUARANTINED": 0, "REJECTED": 0}
    generated_at = datetime.now(timezone.utc).isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for video_id in sorted(latest_scores):
            teachers = latest_scores[video_id]
            result = validate_pair_only(teachers)
            counts[result["status"]] += 1

            record = {
                "type": "validation_result",
                "video_id": video_id,
                "status": result["status"],
                "reason": result["reason"],
                "score_delta": result["score_delta"],
                "time_delta": result["time_delta"],
                "claude_score": result["claude_fls"],
                "gpt4o_score": result["gpt4o_fls"],
                "claude_time_seconds": result["claude_time"],
                "gpt4o_time_seconds": result["gpt4o_time"],
                "claude_confidence": result["claude_confidence"],
                "gpt4o_confidence": result["gpt4o_confidence"],
                "claude_score_path": (teachers.get("teacher_claude") or {}).get("path"),
                "gpt4o_score_path": (teachers.get("teacher_gpt4o") or {}).get("path"),
                "generated_at": generated_at,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        summary = {
            "type": "summary",
            "scores_dir": str(scores_dir),
            "output_path": str(output_path),
            "generated_at": generated_at,
            "counts": counts,
            "total_videos": sum(counts.values()),
            "training_ready_videos": counts["ACCEPTED"],
        }
        handle.write(json.dumps(summary, sort_keys=True) + "\n")

    console.print(
        f"[bold]Summary[/bold]  "
        f"[green]ACCEPTED {counts['ACCEPTED']}[/green]  "
        f"[yellow]QUARANTINED {counts['QUARANTINED']}[/yellow]  "
        f"[red]REJECTED {counts['REJECTED']}[/red]"
    )
    console.print(f"Wrote [cyan]{output_path}[/cyan]")
    return counts


# ----------------------------------------------------------------------------
# Persistence + display
# ----------------------------------------------------------------------------

def _upsert_validation(
    conn: duckdb.DuckDBPyConnection, video_id: str, result: dict
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO validations VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            video_id,
            result["status"],
            result["reason"],
            result["claude_fls"],
            result["gpt4o_fls"],
            result["score_delta"],
            result["claude_time"],
            result["gpt4o_time"],
            result["time_delta"],
            result["claude_confidence"],
            result["gpt4o_confidence"],
            result["duration_seconds"],
            result["floor"],
            result["ceiling"],
            datetime.now(timezone.utc),
        ],
    )


STATUS_STYLES = {
    "ACCEPTED": "green",
    "QUARANTINED": "yellow",
    "REJECTED": "red",
}


def _render_table(rows: list[tuple[str, dict]]) -> None:
    table = Table(title="FLS Auto-Validation Results")
    table.add_column("video_id", style="cyan", no_wrap=True)
    table.add_column("claude_score", justify="right")
    table.add_column("gpt4o_score", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("status", no_wrap=True)
    table.add_column("reason", overflow="fold")

    for vid, r in rows:
        delta = r.get("score_delta")
        delta_str = f"{delta:.0f}" if delta is not None else "-"
        status = r["status"]
        style = STATUS_STYLES.get(status, "white")
        table.add_row(
            vid,
            f"{r['claude_fls']:.0f}" if r["claude_fls"] else "-",
            f"{r['gpt4o_fls']:.0f}" if r["gpt4o_fls"] else "-",
            delta_str,
            f"[{style}]{status}[/{style}]",
            r["reason"],
        )

    console.print(table)

    counts = {"ACCEPTED": 0, "QUARANTINED": 0, "REJECTED": 0}
    for _, r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    total = max(sum(counts.values()), 1)
    console.print(
        f"\n[bold]Summary[/bold]  "
        f"[green]ACCEPTED {counts['ACCEPTED']}[/green]  "
        f"[yellow]QUARANTINED {counts['QUARANTINED']}[/yellow]  "
        f"[red]REJECTED {counts['REJECTED']}[/red]  "
        f"(total {total})"
    )


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

@app.command()
def main(
    scores_dir: Optional[Path] = typer.Option(
        None,
        "--scores-dir",
        help="Validate raw score JSON files from a directory instead of DuckDB.",
    ),
    output_jsonl: Path = typer.Option(
        DEFAULT_RESULTS_PATH,
        "--output-jsonl",
        help="Where to write file-based validation results.",
    ),
    video_id: Optional[str] = typer.Option(
        None, "--video-id", help="Validate a single video."
    ),
    all_: bool = typer.Option(
        False, "--all", help="Validate every un-validated video."
    ),
    revalidate: bool = typer.Option(
        False, "--revalidate", help="Re-run validation on already-validated videos."
    ),
) -> None:
    """Auto-validate dual-teacher scores stored in DuckDB."""
    if scores_dir is not None:
        _validate_score_directory(scores_dir, output_jsonl)
        return

    if not (video_id or all_ or revalidate):
        console.print(
            "[red]Provide --scores-dir, --video-id, --all, or --revalidate.[/red]"
        )
        raise typer.Exit(code=1)

    db_path = _db_path()
    conn = _connect(db_path)
    _ensure_validations_table(conn)
    fls_col, time_col, conf_col = _score_columns(conn)

    if video_id:
        video_ids = [video_id]
    else:
        video_ids = _candidate_video_ids(conn, revalidate=revalidate)

    if not video_ids:
        console.print("[yellow]No videos to validate.[/yellow]")
        return

    rows: list[tuple[str, dict]] = []
    for vid in video_ids:
        teachers = _fetch_latest_teacher_scores(
            conn, vid, fls_col, time_col, conf_col
        )
        duration = _fetch_video_duration(conn, vid)
        result = validate_video(teachers, duration)
        _upsert_validation(conn, vid, result)
        rows.append((vid, result))

    conn.close()
    _render_table(rows)


if __name__ == "__main__":
    app()
