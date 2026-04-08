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

# Note: this module is stale relative to the current schema — earlier
# revisions imported CritiqueResult/LearningEvent/Correction types that
# no longer exist in src.scoring.schema. The only callers in the current
# codebase are append_event() (prepare_dataset, drift_detector) and
# read_events() (drift_detector). The save_score / save_critique /
# save_correction helpers were never called from anywhere outside this
# file — the real persistence path goes through src.memory.memory_store.
# The dead helpers have been removed and append_event now builds a plain
# dict so the module imports cleanly.

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
        event = {
            "timestamp": self._now().isoformat(),
            "event_type": event_type,
            "data": data,
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
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

    # save_score / save_critique / save_correction were removed on
    # 2026-04-08 — they referenced schema classes (CritiqueResult,
    # Correction) that no longer exist, and no code path called them
    # (persistence goes through src.memory.memory_store instead).

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
