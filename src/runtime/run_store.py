"""SQLite-backed run ledger for durable local observability."""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .events import RunEvent


_ACTIVE_STATUSES = ("queued", "running", "cancelling")


def _utc_iso(timestamp: float | None = None) -> str:
    return datetime.fromtimestamp(
        timestamp if timestamp is not None else time.time(),
        tz=timezone.utc,
    ).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class RunStore:
    """Persist run summaries and structured events without storing API secrets."""

    def __init__(self, db_path: str | Path = "outputs/runs/runs.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    backend TEXT NOT NULL,
                    adversarial INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    current_state TEXT NOT NULL DEFAULT 'idle',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT,
                    elapsed_seconds REAL NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL DEFAULT 0,
                    coverage REAL NOT NULL DEFAULT 0,
                    supported_count INTEGER NOT NULL DEFAULT 0,
                    claim_count INTEGER NOT NULL DEFAULT 0,
                    source_count INTEGER NOT NULL DEFAULT 0,
                    num_searches INTEGER NOT NULL DEFAULT 0,
                    num_replan INTEGER NOT NULL DEFAULT 0,
                    evidence_gap_rounds INTEGER NOT NULL DEFAULT 0,
                    adversarial_rounds INTEGER NOT NULL DEFAULT 0,
                    revision_attempted INTEGER NOT NULL DEFAULT 0,
                    revision_accepted INTEGER NOT NULL DEFAULT 0,
                    report_path TEXT NOT NULL DEFAULT '',
                    evidence_path TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    kind TEXT NOT NULL,
                    state TEXT NOT NULL,
                    message TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (run_id, sequence),
                    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_runs_created_at
                    ON runs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_run_events_lookup
                    ON run_events(run_id, sequence DESC);
                """
            )

    def create_run(
        self,
        run_id: str,
        *,
        query: str,
        backend: str,
        adversarial: bool,
    ) -> None:
        now = _utc_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, query, backend, adversarial, status,
                    current_state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'queued', 'idle', ?, ?)
                """,
                (run_id, query.strip(), backend.strip(), int(adversarial), now, now),
            )

    def record_event(self, event: RunEvent) -> None:
        payload_json = _json_dumps(event.payload)
        updated_at = _utc_iso(event.timestamp)
        status = None
        if event.kind == "run_started":
            status = "running"
        elif event.kind == "cancellation_requested":
            status = "cancelling"
        elif event.kind == "run_failed":
            status = "failed"
        elif event.kind == "run_completed":
            status = str(event.payload.get("status") or "complete")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO run_events (
                    run_id, sequence, timestamp, kind, state, message, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.run_id,
                    event.sequence,
                    event.timestamp,
                    event.kind,
                    event.state,
                    event.message,
                    payload_json,
                ),
            )
            if status is None:
                connection.execute(
                    "UPDATE runs SET current_state = ?, updated_at = ? WHERE run_id = ?",
                    (event.state, updated_at, event.run_id),
                )
            else:
                connection.execute(
                    """
                    UPDATE runs
                    SET current_state = ?, status = ?, updated_at = ?
                    WHERE run_id = ?
                    """,
                    (event.state, status, updated_at, event.run_id),
                )

    def mark_cancelling(self, run_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'cancelling', updated_at = ?
                WHERE run_id = ? AND status IN ('queued', 'running')
                """,
                (_utc_iso(), run_id),
            )
            return cursor.rowcount > 0

    def complete_run(
        self,
        run_id: str,
        *,
        status: str,
        report_path: str = "",
        evidence_path: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata = dict(metadata or {})
        audit = dict(metadata.get("evidence_audit") or {})
        revision = dict(metadata.get("evidence_revision") or {})
        claim_count = len(audit.get("claims") or [])
        compact_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"evidence_audit"}
        }
        compact_metadata["evidence_summary"] = {
            "coverage": audit.get("coverage", 0.0),
            "supported_count": audit.get("supported_count", 0),
            "refuted_count": audit.get("refuted_count", 0),
            "not_enough_evidence_count": audit.get("not_enough_evidence_count", 0),
            "claim_count": claim_count,
        }
        now = _utc_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs SET
                    status = ?, current_state = 'done', updated_at = ?, completed_at = ?,
                    elapsed_seconds = ?, confidence = ?, coverage = ?,
                    supported_count = ?, claim_count = ?, source_count = ?,
                    num_searches = ?, num_replan = ?, evidence_gap_rounds = ?,
                    adversarial_rounds = ?, revision_attempted = ?,
                    revision_accepted = ?, report_path = ?, evidence_path = ?,
                    metadata_json = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    now,
                    now,
                    float(metadata.get("elapsed_seconds", 0.0) or 0.0),
                    float(metadata.get("confidence", 0.0) or 0.0),
                    float(audit.get("coverage", 0.0) or 0.0),
                    int(audit.get("supported_count", 0) or 0),
                    claim_count,
                    int(metadata.get("source_count", audit.get("source_count", 0)) or 0),
                    int(metadata.get("num_searches", 0) or 0),
                    int(metadata.get("num_replan", 0) or 0),
                    int(metadata.get("evidence_gap_rounds", 0) or 0),
                    int(metadata.get("adversarial_rounds", 0) or 0),
                    int(bool(revision.get("attempted"))),
                    int(bool(revision.get("accepted"))),
                    str(report_path or ""),
                    str(evidence_path or ""),
                    _json_dumps(compact_metadata),
                    run_id,
                ),
            )

    def fail_run(self, run_id: str, error: str) -> None:
        now = _utc_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET status = 'failed', current_state = 'failed', updated_at = ?,
                    completed_at = ?, error = ?
                WHERE run_id = ?
                """,
                (now, now, str(error or "")[:2000], run_id),
            )

    def recover_interrupted(self) -> int:
        """Mark runs left active by a prior process as interrupted."""
        now = _utc_iso()
        placeholders = ", ".join("?" for _ in _ACTIVE_STATUSES)
        with self._connect() as connection:
            cursor = connection.execute(
                f"""
                UPDATE runs
                SET status = 'interrupted', current_state = 'failed',
                    updated_at = ?, completed_at = ?,
                    error = CASE WHEN error = '' THEN 'Process ended before the run reached a terminal state.' ELSE error END
                WHERE status IN ({placeholders})
                """,
                (now, now, *_ACTIVE_STATUSES),
            )
            return cursor.rowcount

    def list_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        safe_limit = max(1, min(200, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self._run_row(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._run_row(row) if row is not None else None

    def get_events(self, run_id: str, limit: int = 200) -> list[dict[str, Any]]:
        safe_limit = max(1, min(1000, int(limit)))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM run_events
                WHERE run_id = ? ORDER BY sequence DESC LIMIT ?
                """,
                (run_id, safe_limit),
            ).fetchall()
        events = []
        for row in reversed(rows):
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json") or "{}")
            events.append(event)
        return events

    @staticmethod
    def _run_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["adversarial"] = bool(item.get("adversarial"))
        item["revision_attempted"] = bool(item.get("revision_attempted"))
        item["revision_accepted"] = bool(item.get("revision_accepted"))
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except json.JSONDecodeError:
            item["metadata"] = {}
            item.pop("metadata_json", None)
        return item


__all__ = ["RunStore"]
