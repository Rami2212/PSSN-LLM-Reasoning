"""Append-only JSONL logging for reproducible PSSN experiment records."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Persist per-example records with consistent run-level metadata."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        experiment_name: str,
        run_id: str | None = None,
        timestamp: datetime | None = None,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string.")
        self.output_path = Path(output_path)
        if self.output_path.suffix.lower() != ".jsonl":
            raise ValueError("Experiment output path must use the .jsonl extension.")
        self.experiment_name = experiment_name.strip()
        self.run_id = run_id or str(uuid.uuid4())
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string.")
        started_at = timestamp or datetime.now(timezone.utc)
        if started_at.tzinfo is None:
            raise ValueError("timestamp must include timezone information.")
        self.timestamp_utc = started_at.astimezone(timezone.utc).isoformat()
        self.run_metadata = dict(run_metadata or {})

    def _enrich(self, record: Mapping[str, Any]) -> dict[str, Any]:
        enriched = dict(record)
        enriched.update(
            {
                "run_id": self.run_id,
                "experiment_name": self.experiment_name,
                "experiment_timestamp_utc": self.timestamp_utc,
                "run_metadata": dict(self.run_metadata),
            }
        )
        return enriched

    def log(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Append one record and return the exact JSON-compatible payload."""
        if not isinstance(record, Mapping):
            raise TypeError("Experiment record must be a mapping.")
        payload = self._enrich(record)
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized + "\n")
        return payload

    def log_many(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Append multiple records in input order."""
        return [self.log(record) for record in records]

