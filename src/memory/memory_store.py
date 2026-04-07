"""DuckDB-backed persistent store for videos, scores, comparisons, and corrections.

All structured data lives here. File-based JSON outputs in memory/ are the
timestamped audit trail; DuckDB is the queryable index.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from src.scoring.schema import (
    Correction,
    CritiqueResult,
    ScoringResult,
    TrainingRun,
    VideoMetadata,
)

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS videos (
    id VARCHAR PRIMARY KEY,
    filename VARCHAR NOT NULL,
    task VARCHAR NOT NULL,
    duration_seconds DOUBLE,
    resolution VARCHAR,
    fps DOUBLE,
    file_hash VARCHAR,
    ingested_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scores (
    id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    source VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    model_version VARCHAR,
    prompt_version VARCHAR,
    scored_at TIMESTAMP,
    completion_time_seconds DOUBLE,
    estimated_penalties DOUBLE,
    estimated_fls_score DOUBLE,
    confidence_score DOUBLE,
    api_cost_usd DOUBLE,
    latency_seconds DOUBLE,
    raw_json JSON,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS critiques (
    id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    teacher_a_score_id VARCHAR,
    teacher_b_score_id VARCHAR,
    critiqued_at TIMESTAMP,
    agreement_score DOUBLE,
    confidence DOUBLE,
    api_cost_usd DOUBLE,
    consensus_json JSON,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS corrections (
    id VARCHAR PRIMARY KEY,
    video_id VARCHAR NOT NULL,
    original_score_id VARCHAR,
    corrected_at TIMESTAMP,
    corrector_role VARCHAR,
    corrected_fields JSON,
    notes TEXT,
    FOREIGN KEY (video_id) REFERENCES videos(id)
);

CREATE TABLE IF NOT EXISTS training_runs (
    id VARCHAR PRIMARY KEY,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    model_base VARCHAR,
    dataset_version VARCHAR,
    n_train_samples INTEGER,
    n_val_samples INTEGER,
    n_test_samples INTEGER,
    config JSON,
    eval_metrics JSON,
    checkpoint_path VARCHAR,
    status VARCHAR
);
"""


class MemoryStore:
    """Persistent store backed by DuckDB."""

    def __init__(self, db_path: str | Path = "data/fls_training.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        self.conn.execute(_SCHEMA_SQL)

    def close(self):
        self.conn.close()

    # --- Videos ---

    def insert_video(self, video: VideoMetadata) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO videos VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [video.id, video.filename, video.task.value, video.duration_seconds,
             video.resolution, video.fps, video.file_hash,
             video.ingested_at.isoformat()],
        )
        logger.info(f"Stored video {video.id}")

    def get_video(self, video_id: str) -> dict | None:
        result = self.conn.execute(
            "SELECT * FROM videos WHERE id = ?", [video_id]
        ).fetchone()
        if result is None:
            return None
        cols = [d[0] for d in self.conn.description]
        return dict(zip(cols, result))

    def list_videos(self) -> list[dict]:
        results = self.conn.execute("SELECT * FROM videos ORDER BY ingested_at DESC").fetchall()
        cols = [d[0] for d in self.conn.description]
        return [dict(zip(cols, r)) for r in results]

    # --- Scores ---

    def insert_score(self, score: ScoringResult) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [score.id, score.video_id, score.source.value, score.model_name,
             score.model_version, score.prompt_version, score.scored_at.isoformat(),
             score.completion_time_seconds, score.estimated_penalties,
             score.estimated_fls_score, score.confidence_score,
             score.api_cost_usd, score.latency_seconds,
             score.model_dump_json()],
        )
        logger.info(f"Stored score {score.id} for video {score.video_id}")

    def get_scores_for_video(self, video_id: str) -> list[dict]:
        results = self.conn.execute(
            "SELECT * FROM scores WHERE video_id = ? ORDER BY scored_at DESC",
            [video_id],
        ).fetchall()
        cols = [d[0] for d in self.conn.description]
        return [dict(zip(cols, r)) for r in results]

    # --- Critiques ---

    def insert_critique(self, critique: CritiqueResult) -> None:
        consensus_json = critique.consensus_score.model_dump_json() if critique.consensus_score else "{}"
        self.conn.execute(
            "INSERT OR REPLACE INTO critiques VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [critique.id, critique.video_id, critique.teacher_a_score_id,
             critique.teacher_b_score_id, critique.critiqued_at.isoformat(),
             critique.agreement_score, critique.confidence,
             critique.api_cost_usd, consensus_json],
        )
        logger.info(f"Stored critique {critique.id} for video {critique.video_id}")

    # --- Corrections ---

    def insert_correction(self, correction: Correction) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO corrections VALUES (?, ?, ?, ?, ?, ?, ?)",
            [correction.id, correction.video_id, correction.original_score_id,
             correction.corrected_at.isoformat(), correction.corrector_role.value,
             json.dumps(correction.corrected_fields), correction.notes],
        )
        logger.info(f"Stored correction {correction.id} for video {correction.video_id}")

    # --- Training Runs ---

    def insert_training_run(self, run: TrainingRun) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO training_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [run.id, run.started_at.isoformat(),
             run.completed_at.isoformat() if run.completed_at else None,
             run.model_base, run.dataset_version,
             run.n_train_samples, run.n_val_samples, run.n_test_samples,
             json.dumps(run.config), json.dumps(run.eval_metrics),
             run.checkpoint_path, run.status],
        )

    # --- Queries for Training Data ---

    def get_training_candidates(self, min_confidence: float = 0.7) -> list[dict]:
        """Get all scores suitable for training: high-confidence consensus or corrected."""
        results = self.conn.execute("""
            SELECT s.*, c.corrected_fields
            FROM scores s
            LEFT JOIN corrections c ON s.video_id = c.video_id
            WHERE s.source IN ('critique_consensus', 'expert_correction')
               OR s.confidence_score >= ?
            ORDER BY s.scored_at DESC
        """, [min_confidence]).fetchall()
        cols = [d[0] for d in self.conn.description]
        return [dict(zip(cols, r)) for r in results]

    def get_total_cost(self) -> float:
        """Total API spend across all scoring."""
        result = self.conn.execute("SELECT SUM(api_cost_usd) FROM scores").fetchone()
        return result[0] or 0.0

    def get_stats(self) -> dict:
        """Summary statistics."""
        return {
            "total_videos": self.conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0],
            "total_scores": self.conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0],
            "total_critiques": self.conn.execute("SELECT COUNT(*) FROM critiques").fetchone()[0],
            "total_corrections": self.conn.execute("SELECT COUNT(*) FROM corrections").fetchone()[0],
            "total_training_runs": self.conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0],
            "total_api_cost": self.get_total_cost(),
        }
