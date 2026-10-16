"""Local and remote lifecycle status with a timestamped heartbeat."""

from __future__ import annotations

import json
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


HEARTBEAT_STALE_AFTER_SECONDS = 120

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStatus:
    """Write atomic progress snapshots and periodically sync them to Drive."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        total_experiments: int,
        persist_callback: Callable[[Path], None] | None = None,
        heartbeat_seconds: int = 45,
    ) -> None:
        self.path = Path(path)
        self.persist_callback = persist_callback
        self.heartbeat_seconds = heartbeat_seconds
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sync_error: BaseException | None = None
        now = utc_now()
        self.data: dict[str, Any] = {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "starting",
            "is_running": True,
            "phase": "startup",
            "started_at_utc": now,
            "updated_at_utc": now,
            "heartbeat_at_utc": now,
            "heartbeat_stale_after_seconds": HEARTBEAT_STALE_AFTER_SECONDS,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "progress": {
                "total_experiments": total_experiments,
                "completed_experiments": 0,
                "pending_experiments": total_experiments,
                "new_attempts_this_session": 0,
                "correct": 0,
                "incorrect": 0,
                "incomplete": 0,
                "failed": 0,
                "current_experiment": None,
            },
        }

    def _write_locked(self) -> None:
        now = utc_now()
        self.data["updated_at_utc"] = now
        self.data["heartbeat_at_utc"] = now
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.data, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)
        if self.persist_callback:
            self.persist_callback(self.path)

    def start(self, *, phase: str = "startup", metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.data["phase"] = phase
            if metadata:
                self.data["metadata"] = metadata
            self._write_locked()
        self._thread = threading.Thread(target=self._heartbeat_loop, name="pssn-run-heartbeat", daemon=True)
        self._thread.start()

    def update(self, *, status: str | None = None, phase: str | None = None,
               current_experiment: dict[str, Any] | None = None,
               progress: dict[str, Any] | None = None,
               metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            if status is not None:
                self.data["status"] = status
                self.data["is_running"] = status in {"starting", "running"}
            if phase is not None:
                self.data["phase"] = phase
            if current_experiment is not None or phase in {"saving_checkpoint", "completed", "stopped", "paused", "failed"}:
                self.data["progress"]["current_experiment"] = current_experiment
            if progress:
                self.data["progress"].update(progress)
            if metadata:
                self.data.setdefault("metadata", {}).update(metadata)
            self._write_locked()
        if self._sync_error:
            raise RuntimeError("Could not sync run status to Drive") from self._sync_error

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                with self._lock:
                    if self.data["is_running"]:
                        self._write_locked()
            except BaseException as exc:  # surfaced on the next main-thread status update
                self._sync_error = exc

    def close(self, status: str, *, error: str | None = None) -> None:
        if status not in {"completed", "stopped", "paused", "failed"}:
            raise ValueError("final status must be completed, stopped, paused, or failed")
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        with self._lock:
            self.data["status"] = status
            self.data["is_running"] = False
            self.data["phase"] = status
            self.data["finished_at_utc"] = utc_now()
            if error:
                self.data["last_error"] = error
            self._write_locked()
