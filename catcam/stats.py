from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def day_bounds(dt: datetime) -> tuple[float, float]:
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


class StatsStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts REAL NOT NULL, "
                "clip_name TEXT)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def record_event(self, timestamp: float, clip_name: str | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO events (ts, clip_name) VALUES (?, ?)",
                (timestamp, clip_name),
            )
            return int(cur.lastrowid)

    def count_between(self, start_ts: float, end_ts: float) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM events WHERE ts >= ? AND ts < ?",
                (start_ts, end_ts),
            )
            return int(cur.fetchone()[0])

    def events_between(self, start_ts: float, end_ts: float) -> list[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT ts, clip_name FROM events "
                "WHERE ts >= ? AND ts < ? ORDER BY ts ASC",
                (start_ts, end_ts),
            )
            return [{"ts": row[0], "clip_name": row[1]} for row in cur.fetchall()]
