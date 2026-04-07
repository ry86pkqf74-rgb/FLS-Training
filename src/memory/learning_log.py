"""Learning log: append-only ledger + timestamped file outputs.

Two complementary systems:
1. learning_ledger.jsonl — single append-only log of every event
2. memory/{category}/YYYY-MM-DD/{id}_{context}_{timestamp}.json — individual files
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.scoring.schema import (
    CritiqueResult,
    LearningEvent,
    ScoringResult,
    Correction,
)

logger = logging.getLogger(__name__)


class LearningLog:
    """Manages the learning ledger and file-based output."""

    def __init__(self, memory_path: str | Path = "memory"):
        self.memory_path = Path(memory_path)
        self.ledger_path = self.memory_path / "learning_ledger.jsonl"
        self.memory_path.mkdir(parents=True, exist_ok=True)

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _ts_str(self, dt: datetime | None = None) -> str:
        """YYYYMMDD_HHMMSS format for filenames."""
        dt = dt or self._now()
        return dt.strftime("%Y%m%d_%H%M%S")

    def _date_str(self, dt: datetime | None = None) -> str:
        """YYYY-MM-DD format for directory names."""
        dt = dt or self._now()
        return dt.strftime("%Y-%m-%d")

    # --- Ledger ---

    def append_event(self, event_type: str, data: dict) -> None:
        """Append an event to the learning ledger."""
        event = LearningEvent(
            timestamp=self._now(),
            event_type=event_type,
            data=data,
        )
        with open(self.ledger_path, "a") as f:
            f.write(event.model_dump_json() + "\n")
        logger.debug(f"Ledger event: {event_type}")

    def read_events(
        self,
        event_type: str | None = None,
        after: datetime | None = None,
        before: datetime | None = None,
    ) -> list[dict]:
        """Read events from the ledger with optional filters."""
        if not self.ledger_path.exists():
            return []

        events = []
        with open(self.ledger_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                if event_type and event.get("event_type") != event_type:
                    continue
                ts = datetime.fromisoformat(event["timestamp"])
                if after and ts < after:
                    continue
                if before and ts > before:
                    continue
                events.append(event)
        return events

    # --- File-Based Outputs ---

    def _write_json(self, category: str, filename: str, data: dict) -> Path:
        """Write a JSON file to memory/{category}/YYYY-MM-DD/{filename}."""
        date_dir = self.memory_path / category / self._date_str()
        date_dir.mkdir(parents=True, exist_ok=True)
        path = date_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Wrote {path}")
        return path

    def save_score(self, score: ScoringResult) -> Path:
        """Save a scoring result and log the event."""
        model_slug = score.model_name.replace("/", "-").replace(" ", "_")
        filename = f"{score.video_id}_{model_slug}_{self._ts_str()}.json"
        path = self._write_json("scores", filename, score.model_dump())

        self.append_event("frontier_scored", {
            "video_id": score.video_id,
            "score_id": score.id,
            "model": score.model_name,
            "source": score.source.value,
            "fls_score": score.estimated_fls_score,
            "confidence": score.confidence_score,
            "cost_usd": score.api_cost_usd,
        })
        return path

    def save_critique(self, critique: CritiqueResult) -> Path:
        """Save a critique/comparison result and log the event."""
        filename = f"{critique.video_id}_critique_{self._ts_str()}.json"
        path = self._write_json("comparisons", filename, critique.model_dump())

        self.append_event("comparison_generated", {
            "video_id": critique.video_id,
            "critique_id": critique.id,
            "agreement": critique.agreement_score,
            "n_divergences": len(critique.divergences),
            "consensus_fls_score": (
                critique.consensus_score.estimated_fls_score
                if critique.consensus_score else None
            ),
            "cost_usd": critique.api_cost_usd,
        })
        return path

    def save_correction(self, correction: Correction) -> Path:
        """Save an expert correction and log the event."""
        filename = f"{correction.video_id}_correction_{self._ts_str()}.json"
        path = self._write_json("corrections", filename, correction.model_dump())

        self.append_event("correction_submitted", {
            "video_id": correction.video_id,
            "correction_id": correction.id,
            "corrector": correction.corrector_role.value,
            "fields_corrected": list(correction.corrected_fields.keys()),
        })
        return path

    def log_video_ingested(self, video_id: str, filename: str, task: str) -> None:
        self.append_event("video_ingested", {
            "video_id": video_id,
            "filename": filename,
            "task": task,
        })

    def log_training_started(self, run_id: str, n_samples: int, model_base: str) -> None:
        self.append_event("training_started", {
            "run_id": run_id,
            "n_samples": n_samples,
            "model_base": model_base,
        })

    def log_training_completed(self, run_id: str, metrics: dict) -> None:
        self.append_event("training_completed", {
            "run_id": run_id,
            **metrics,
        })

    def log_model_promoted(self, run_id: str, replaces: str) -> None:
        self.append_event("model_promoted", {
            "run_id": run_id,
            "replaces": replaces,
        })

    # --- Summaries ---

    def summarize(self, since: datetime | None = None) -> dict:
        """Summary of learning activity."""
        events = self.read_events(after=since)
        by_type: dict[str, int] = {}
        total_cost = 0.0
        for e in events:
            by_type[e["event_type"]] = by_type.get(e["event_type"], 0) + 1
            total_cost += e.get("data", {}).get("cost_usd", 0)

        return {
            "total_events": len(events),
            "events_by_type": by_type,
            "total_api_cost_usd": round(total_cost, 4),
            "since": since.isoformat() if since else "all_time",
        }
