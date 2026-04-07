"""Persistent memory store backed by DuckDB + JSON files on disk.

All state is designed to be committed to GitHub after each session.
The DuckDB file is ephemeral (rebuilt from JSON on each load).
The JSON files in memory/ are the source of truth.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb

from src.scoring.schema import ScoringResult, VideoRecord, CorrectionRecord
from src.feedback.schema import FeedbackReport, TraineeProfile


class MemoryStore:
    """Unified access to all FLS training memory."""

    def __init__(self, base_dir: str | Path = "."):
        self.base = Path(base_dir)
        self.scores_dir = self.base / "memory" / "scores"
        self.feedback_dir = self.base / "memory" / "feedback"
        self.corrections_dir = self.base / "memory" / "corrections"
        self.ledger_path = self.base / "memory" / "learning_ledger.jsonl"
        self.profile_path = self.base / "memory" / "trainee_profile.json"

        # Ensure dirs exist
        for d in [self.scores_dir, self.feedback_dir, self.corrections_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.db = duckdb.connect(":memory:")
        self._init_tables()
        self._load_from_disk()

    def _init_tables(self):
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id VARCHAR PRIMARY KEY,
                filename VARCHAR,
                task VARCHAR,
                duration_seconds DOUBLE,
                resolution VARCHAR,
                fps INTEGER,
                file_hash VARCHAR,
                ingested_at TIMESTAMP,
                recording_note VARCHAR
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id VARCHAR PRIMARY KEY,
                video_id VARCHAR,
                source VARCHAR,
                model_name VARCHAR,
                completion_time DOUBLE,
                fls_score DOUBLE,
                confidence DOUBLE,
                scored_at TIMESTAMP
            )
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id VARCHAR PRIMARY KEY,
                video_id VARCHAR,
                score_id VARCHAR,
                headline VARCHAR,
                fls_score DOUBLE,
                completion_time DOUBLE,
                attempt_number INTEGER,
                generated_at TIMESTAMP
            )
        """)

    def _load_from_disk(self):
        """Rebuild DuckDB tables from JSON files on disk."""
        for f in self.scores_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                self.db.execute(
                    "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?)",
                    [data.get("id"), data.get("video_id"), data.get("source"),
                     data.get("model_name"), data.get("completion_time_seconds"),
                     data.get("estimated_fls_score"), data.get("confidence_score"),
                     data.get("scored_at")]
                )
            except Exception:
                pass

    # === Score operations ===

    def save_score(self, result: ScoringResult) -> Path:
        path = self.scores_dir / f"{result.id}.json"
        path.write_text(result.model_dump_json(indent=2))
        self.db.execute(
            "INSERT OR REPLACE INTO scores VALUES (?,?,?,?,?,?,?,?)",
            [result.id, result.video_id, result.source, result.model_name,
             result.completion_time_seconds, result.estimated_fls_score,
             result.confidence_score, result.scored_at.isoformat()]
        )
        self._append_ledger("score_saved", {"score_id": result.id, "video_id": result.video_id})
        return path

    def get_score(self, score_id: str) -> Optional[ScoringResult]:
        path = self.scores_dir / f"{score_id}.json"
        if path.exists():
            return ScoringResult.model_validate_json(path.read_text())
        return None

    def get_scores_for_video(self, video_id: str) -> list[ScoringResult]:
        results = []
        for f in self.scores_dir.glob("*.json"):
            data = json.loads(f.read_text())
            if data.get("video_id") == video_id:
                results.append(ScoringResult.model_validate(data))
        return results

    def get_all_scores(self) -> list[ScoringResult]:
        results = []
        for f in sorted(self.scores_dir.glob("*.json")):
            try:
                results.append(ScoringResult.model_validate_json(f.read_text()))
            except Exception:
                pass
        return results

    # === Feedback operations ===

    def save_feedback(self, report: FeedbackReport) -> Path:
        path = self.feedback_dir / f"{report.feedback_id}.json"
        path.write_text(report.model_dump_json(indent=2))
        self._append_ledger("feedback_saved", {
            "feedback_id": report.feedback_id,
            "video_id": report.video_id
        })
        return path

    def get_feedback(self, feedback_id: str) -> Optional[FeedbackReport]:
        path = self.feedback_dir / f"{feedback_id}.json"
        if path.exists():
            return FeedbackReport.model_validate_json(path.read_text())
        return None

    def get_all_feedback(self) -> list[FeedbackReport]:
        results = []
        for f in sorted(self.feedback_dir.glob("*.json")):
            try:
                results.append(FeedbackReport.model_validate_json(f.read_text()))
            except Exception:
                pass
        return results

    # === Correction operations ===

    def save_correction(self, correction: CorrectionRecord) -> Path:
        path = self.corrections_dir / f"{correction.correction_id}.json"
        path.write_text(correction.model_dump_json(indent=2))
        self._append_ledger("correction_saved", {
            "correction_id": correction.correction_id,
            "score_id": correction.score_id
        })
        return path

    # === Trainee profile ===

    def get_trainee_profile(self) -> TraineeProfile:
        if self.profile_path.exists():
            return TraineeProfile.model_validate_json(self.profile_path.read_text())
        return TraineeProfile()

    def save_trainee_profile(self, profile: TraineeProfile):
        self.profile_path.write_text(profile.model_dump_json(indent=2))

    def rebuild_trainee_profile(self) -> TraineeProfile:
        """Rebuild profile from all scored data."""
        scores = self.get_all_scores()
        if not scores:
            return TraineeProfile()

        scores_sorted = sorted(scores, key=lambda s: s.scored_at)
        times = [s.completion_time_seconds for s in scores_sorted]
        fls_scores = [s.estimated_fls_score for s in scores_sorted]

        profile = TraineeProfile(
            total_attempts=len(scores_sorted),
            first_attempt_date=scores_sorted[0].scored_at,
            last_attempt_date=scores_sorted[-1].scored_at,
            best_time_seconds=min(times),
            best_fls_score=max(fls_scores),
            baseline_time=times[0] if times else 0,
            current_plateau_time=sum(times[-5:]) / len(times[-5:]) if len(times) >= 5 else times[-1],
        )

        # Phase averages from last 5
        last5 = scores_sorted[-5:]
        placements, throws, cuts = [], [], []
        for s in last5:
            for pt in s.phase_timings:
                if pt.phase.value == "suture_placement":
                    placements.append(pt.duration_seconds)
                elif pt.phase.value in ("first_throw", "second_throw", "third_throw"):
                    throws.append(pt.duration_seconds)
                elif pt.phase.value == "suture_cut":
                    cuts.append(pt.duration_seconds)

        if placements:
            profile.avg_placement_last5 = sum(placements) / len(placements)
        if throws:
            profile.avg_throws_last5 = sum(throws) / len(throws)
        if cuts:
            profile.avg_cutting_last5 = sum(cuts) / len(cuts)

        # Bottleneck
        phase_avgs = {
            "placement": profile.avg_placement_last5,
            "throws": profile.avg_throws_last5,
            "cutting": profile.avg_cutting_last5,
        }
        if any(phase_avgs.values()):
            profile.bottleneck_phase = max(phase_avgs, key=phase_avgs.get)

        self.save_trainee_profile(profile)
        return profile

    # === Ledger ===

    def _append_ledger(self, event_type: str, data: dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event_type": event_type,
            "data": data,
        }
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # === Stats ===

    def get_stats(self) -> dict:
        n_scores = len(list(self.scores_dir.glob("*.json")))
        n_feedback = len(list(self.feedback_dir.glob("*.json")))
        n_corrections = len(list(self.corrections_dir.glob("*.json")))
        return {
            "total_scores": n_scores,
            "total_feedback": n_feedback,
            "total_corrections": n_corrections,
            "ledger_entries": sum(1 for _ in open(self.ledger_path)) if self.ledger_path.exists() else 0,
            "profile_exists": self.profile_path.exists(),
        }
