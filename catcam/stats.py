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
            # labels 表由 FeedbackStore 维护（同一个 db）。这里也 IF NOT EXISTS 一下，
            # 保证统计查询的 LEFT JOIN 永远有表可连（无论两个 store 谁先初始化）。
            conn.execute(
                "CREATE TABLE IF NOT EXISTS labels ("
                "clip_name TEXT PRIMARY KEY, "
                "is_drinking INTEGER NOT NULL, "
                "ts REAL)"
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
        # 「真实喝水」= 没被人工标注成「没喝」的事件。
        # COALESCE(is_drinking,1)：未标注或标注「喝了」算 1，标注「没喝」(0) 被排除。
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM events e "
                "LEFT JOIN labels l ON e.clip_name = l.clip_name "
                "WHERE e.ts >= ? AND e.ts < ? AND COALESCE(l.is_drinking, 1) <> 0",
                (start_ts, end_ts),
            )
            return int(cur.fetchone()[0])

    def daily_counts(self, end_dt: datetime, days: int) -> list[tuple[str, int]]:
        """近 days 天（含 end_dt 当天）每天的喝水次数，缺的天补 0，按时间升序。

        返回 [(MM-DD, count), ...]，用于周/月趋势图。
        """
        end_day = end_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        start_day = end_day - timedelta(days=days - 1)
        out: list[tuple[str, int]] = []
        for i in range(days):
            d0 = start_day + timedelta(days=i)
            d1 = d0 + timedelta(days=1)
            out.append((d0.strftime("%m-%d"), self.count_between(d0.timestamp(), d1.timestamp())))
        return out

    def events_between(self, start_ts: float, end_ts: float) -> list[dict]:
        # 同 count_between：排除被标注「没喝」的事件，让时间点列表与计数一致。
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT e.ts, e.clip_name FROM events e "
                "LEFT JOIN labels l ON e.clip_name = l.clip_name "
                "WHERE e.ts >= ? AND e.ts < ? AND COALESCE(l.is_drinking, 1) <> 0 "
                "ORDER BY e.ts ASC",
                (start_ts, end_ts),
            )
            return [{"ts": row[0], "clip_name": row[1]} for row in cur.fetchall()]
