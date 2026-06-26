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
            # 迁移：记录「测试模型」当时对这段的预测（1/0/NULL）及是哪个版本预测的，
            # 之后和人工标注一比就是模型在真实数据上的命中率。
            ecols = [r[1] for r in conn.execute("PRAGMA table_info(events)")]
            if "predicted" not in ecols:
                conn.execute("ALTER TABLE events ADD COLUMN predicted INTEGER")
            if "predicted_by" not in ecols:
                conn.execute("ALTER TABLE events ADD COLUMN predicted_by TEXT")
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

    def record_event(
        self,
        timestamp: float,
        clip_name: str | None = None,
        predicted: int | None = None,
        predicted_by: str | None = None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO events (ts, clip_name, predicted, predicted_by) VALUES (?, ?, ?, ?)",
                (timestamp, clip_name, predicted, predicted_by),
            )
            return int(cur.lastrowid)

    def model_hitrate(self, version: str) -> dict:
        """某版本「测试模型」在真实数据上的命中率：它预测过、且该段已人工标注的，预测对了多少。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT e.predicted, l.is_drinking FROM events e "
                "JOIN labels l ON e.clip_name = l.clip_name "
                "WHERE e.predicted IS NOT NULL AND e.predicted_by = ?",
                (version,),
            ).fetchall()
        total = len(rows)
        correct = sum(1 for pred, label in rows if int(pred) == int(label))
        return {"total": total, "correct": correct,
                "rate": (correct / total) if total else None}

    def clip_predictions(self) -> dict:
        """{clip_name: 预测bool}，给视频列表显示「模型怎么判」。一段多次取最新。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT clip_name, predicted FROM events "
                "WHERE clip_name IS NOT NULL AND predicted IS NOT NULL ORDER BY ts ASC"
            ).fetchall()
        return {name: bool(pred) for name, pred in rows}

    def count_between(self, start_ts: float, end_ts: float) -> int:
        # 「真实喝水」= 被 AI/人工明确标注为「喝水」(is_drinking=1) 的事件。
        # 口径：只数确认喝水的——未标注 / 标「没喝」/ 无对应 clip 的事件都不计。
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT COUNT(*) FROM events e "
                "JOIN labels l ON e.clip_name = l.clip_name "
                "WHERE e.ts >= ? AND e.ts < ? AND l.is_drinking = 1",
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
        # 同 count_between：只列被确认「喝水」的事件，让时间点列表与计数一致。
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT e.ts, e.clip_name FROM events e "
                "JOIN labels l ON e.clip_name = l.clip_name "
                "WHERE e.ts >= ? AND e.ts < ? AND l.is_drinking = 1 "
                "ORDER BY e.ts ASC",
                (start_ts, end_ts),
            )
            return [{"ts": row[0], "clip_name": row[1]} for row in cur.fetchall()]
