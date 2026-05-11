"""Local SQLite store for agent state: command history + small key/value store."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


# Status values match the OPS command lifecycle exactly.
STATUS_PENDING = "pending"
STATUS_EXECUTING = "executing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_EXPIRED = "expired"
STATUS_CANCELLED = "cancelled"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS commands (
    command_id      INTEGER PRIMARY KEY,        -- OPS-assigned integer ID
    command_type    TEXT NOT NULL,              -- "calibrate", "request_diagnostics", etc.
    payload         TEXT,                       -- JSON, nullable
    received_at     TEXT NOT NULL,              -- ISO 8601 UTC
    acknowledged_at TEXT,
    completed_at    TEXT,
    status          TEXT NOT NULL,              -- pending|executing|completed|failed|expired|cancelled
    result          TEXT,                       -- JSON or null
    error           TEXT
);

CREATE INDEX IF NOT EXISTS idx_commands_status ON commands(status);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentStateStore:
    """Thread-safe SQLite store for command history and small key/value state.

    The kv table holds the canonical config hash and any cached operator
    thresholds from the latest OPS config payload.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        # New connection per use — sqlite3 connections are not thread-safe by default.
        conn = sqlite3.connect(str(self._db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def _init_schema(self) -> None:
        with self._lock, self.conn() as conn:
            conn.executescript(_SCHEMA)

    # ---- commands ----------------------------------------------------------

    def record_received(
        self,
        command_id: int,
        command_type: str,
        payload: Optional[dict[str, Any]],
    ) -> bool:
        """Insert a new command in `pending` status. Returns False if the
        command_id already exists (idempotent: OPS may re-list pending commands
        on every poll until acknowledged)."""
        with self._lock, self.conn() as conn:
            try:
                conn.execute(
                    "INSERT INTO commands "
                    "(command_id, command_type, payload, received_at, status) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        command_id,
                        command_type,
                        json.dumps(payload) if payload is not None else None,
                        _utcnow_iso(),
                        STATUS_PENDING,
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def mark_executing(self, command_id: int) -> None:
        with self._lock, self.conn() as conn:
            conn.execute(
                "UPDATE commands "
                "SET status=?, acknowledged_at=? "
                "WHERE command_id=?",
                (STATUS_EXECUTING, _utcnow_iso(), command_id),
            )

    def mark_completed(
        self, command_id: int, result: Optional[dict[str, Any]] = None
    ) -> None:
        with self._lock, self.conn() as conn:
            conn.execute(
                "UPDATE commands "
                "SET status=?, completed_at=?, result=? "
                "WHERE command_id=?",
                (
                    STATUS_COMPLETED,
                    _utcnow_iso(),
                    json.dumps(result) if result is not None else None,
                    command_id,
                ),
            )

    def mark_failed(self, command_id: int, error: str) -> None:
        with self._lock, self.conn() as conn:
            conn.execute(
                "UPDATE commands "
                "SET status=?, completed_at=?, error=? "
                "WHERE command_id=?",
                (STATUS_FAILED, _utcnow_iso(), error, command_id),
            )

    def mark_expired(self, command_id: int) -> None:
        with self._lock, self.conn() as conn:
            conn.execute(
                "UPDATE commands "
                "SET status=?, completed_at=? "
                "WHERE command_id=?",
                (STATUS_EXPIRED, _utcnow_iso(), command_id),
            )

    def get_executing_commands(self) -> list[dict[str, Any]]:
        """Commands that we acknowledged but haven't completed yet. On agent
        restart, these need to be reconciled with OPS (mark failed locally
        and tell OPS, since we lost the in-flight work)."""
        with self._lock, self.conn() as conn:
            rows = conn.execute(
                "SELECT command_id, command_type, payload "
                "FROM commands WHERE status=?",
                (STATUS_EXECUTING,),
            ).fetchall()
        return [
            {
                "command_id": r["command_id"],
                "command_type": r["command_type"],
                "payload": json.loads(r["payload"]) if r["payload"] else None,
            }
            for r in rows
        ]

    # ---- key/value ---------------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        with self._lock, self.conn() as conn:
            row = conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set(self, key: str, value: str) -> None:
        with self._lock, self.conn() as conn:
            conn.execute(
                "INSERT INTO kv (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_json(self, key: str) -> Optional[Any]:
        raw = self.get(key)
        return json.loads(raw) if raw is not None else None

    def set_json(self, key: str, value: Any) -> None:
        self.set(key, json.dumps(value))

    # Convenience accessors
    def get_config_hash(self) -> Optional[str]:
        return self.get("config_hash")

    def set_config_hash(self, h: str) -> None:
        self.set("config_hash", h)

    def get_ops_config_cache(self) -> Optional[dict[str, Any]]:
        """The most recent config payload OPS sent us (used to read operator
        thresholds for alert evaluation)."""
        return self.get_json("ops_config_cache")

    def set_ops_config_cache(self, cfg: dict[str, Any]) -> None:
        self.set_json("ops_config_cache", cfg)
