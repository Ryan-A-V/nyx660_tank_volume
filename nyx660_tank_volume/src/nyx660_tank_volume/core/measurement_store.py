"""
Local SQLite storage for measurement history.

Stores every measurement result with a 24-hour rolling retention.
Thread-safe for concurrent access from the measurement loop,
API endpoints, and the cloud agent.

No additional dependencies — sqlite3 is in the Python standard library.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from nyx660_tank_volume.core.measurement import MeasurementResult

logger = logging.getLogger(__name__)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT NOT NULL,
    estimated_volume_m3 REAL NOT NULL,
    estimated_volume_liters REAL,
    relative_fill_ratio REAL,
    occupied_surface_area_m2 REAL NOT NULL,
    average_fill_height_m REAL NOT NULL,
    max_fill_height_m REAL NOT NULL,
    valid_pixel_ratio REAL NOT NULL,
    notes TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_measurements_timestamp
ON measurements (timestamp_utc);
"""


class MeasurementStore:
    """
    Thread-safe SQLite store for measurement history.

    All public methods acquire a lock before accessing the database
    so the measurement loop, API handlers, and cloud agent can all
    call into this safely.
    """

    def __init__(self, db_path: str, retention_hours: int = 24) -> None:
        self.db_path = db_path
        self.retention_hours = retention_hours
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(CREATE_TABLE_SQL)
                conn.execute(CREATE_INDEX_SQL)
                conn.commit()
            finally:
                conn.close()
        logger.info("Measurement store initialised at %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, result: MeasurementResult) -> None:
        """Save a measurement result and prune old records."""
        notes_json = json.dumps(result.notes)
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """
                    INSERT INTO measurements (
                        timestamp_utc, estimated_volume_m3,
                        estimated_volume_liters, relative_fill_ratio,
                        occupied_surface_area_m2, average_fill_height_m,
                        max_fill_height_m, valid_pixel_ratio, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.timestamp_utc,
                        result.estimated_volume_m3,
                        result.estimated_volume_liters,
                        result.relative_fill_ratio,
                        result.occupied_surface_area_m2,
                        result.average_fill_height_m,
                        result.max_fill_height_m,
                        result.valid_pixel_ratio,
                        notes_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        self._prune()

    def get_latest(self) -> Optional[dict]:
        """Return the most recent measurement as a dict, or None."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT * FROM measurements ORDER BY id DESC LIMIT 1"
                ).fetchone()
                return self._row_to_dict(row) if row else None
            finally:
                conn.close()

    def get_history(
        self,
        hours: Optional[float] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 10000,
    ) -> list[dict]:
        """
        Return measurement history as a list of dicts.

        Filter by:
          - hours: last N hours from now
          - since/until: ISO timestamp range
          - limit: max number of records (default 10000)
        """
        conditions = []
        params: list = []

        if hours is not None:
            cutoff = (
                datetime.now(timezone.utc) - timedelta(hours=hours)
            ).isoformat()
            conditions.append("timestamp_utc >= ?")
            params.append(cutoff)
        if since is not None:
            conditions.append("timestamp_utc >= ?")
            params.append(since)
        if until is not None:
            conditions.append("timestamp_utc <= ?")
            params.append(until)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT * FROM measurements
            {where}
            ORDER BY timestamp_utc DESC
            LIMIT ?
        """
        params.append(limit)

        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(query, params).fetchall()
                return [self._row_to_dict(r) for r in rows]
            finally:
                conn.close()

    def get_count(self) -> int:
        """Return total number of stored measurements."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM measurements"
                ).fetchone()
                return row["cnt"] if row else 0
            finally:
                conn.close()

    def get_stats(self) -> dict:
        """Return summary statistics for the stored measurements."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) as count,
                        MIN(timestamp_utc) as oldest,
                        MAX(timestamp_utc) as newest,
                        AVG(estimated_volume_m3) as avg_volume_m3,
                        AVG(valid_pixel_ratio) as avg_valid_pixel_ratio
                    FROM measurements
                    """
                ).fetchone()
                if row and row["count"] > 0:
                    return {
                        "count": row["count"],
                        "oldest": row["oldest"],
                        "newest": row["newest"],
                        "avg_volume_m3": round(row["avg_volume_m3"], 4),
                        "avg_valid_pixel_ratio": round(
                            row["avg_valid_pixel_ratio"], 4
                        ),
                    }
                return {"count": 0}
            finally:
                conn.close()

    def _prune(self) -> None:
        """Remove records older than the retention window."""
        cutoff = (
            datetime.now(timezone.utc)
            - timedelta(hours=self.retention_hours)
        ).isoformat()
        with self._lock:
            conn = self._connect()
            try:
                result = conn.execute(
                    "DELETE FROM measurements WHERE timestamp_utc < ?",
                    (cutoff,),
                )
                if result.rowcount > 0:
                    logger.debug("Pruned %d old measurements", result.rowcount)
                conn.commit()
            finally:
                conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        if "notes" in d and isinstance(d["notes"], str):
            try:
                d["notes"] = json.loads(d["notes"])
            except (json.JSONDecodeError, TypeError):
                d["notes"] = []
        # Remove internal db fields from the output
        d.pop("id", None)
        d.pop("created_at", None)
        return d
