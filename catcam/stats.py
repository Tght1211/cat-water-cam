from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def day_bounds(dt: datetime) -> tuple[float, float]:
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


def bucket_events(events: list[dict]) -> dict:
    """把 events_between 的事件按「小时(0-23)」和「星期(周一=0…周日=6)」分桶。

    纯函数（只看每条的 ts），给趋势页的「时段分布 / 星期分布」两张图用。
    """
    hourly = [0] * 24
    weekday = [0] * 7
    for e in events:
        dt = datetime.fromtimestamp(e["ts"])
        hourly[dt.hour] += 1
        weekday[dt.weekday()] += 1
    return {"hourly": hourly, "weekday": weekday}


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
            if "predicted_confidence" not in ecols:
                conn.execute("ALTER TABLE events ADD COLUMN predicted_confidence REAL")
            if 'candidate_only' not in ecols:
                conn.execute('ALTER TABLE events ADD COLUMN candidate_only INTEGER NOT NULL DEFAULT 0')
            if 'water_review' not in ecols:
                conn.execute('ALTER TABLE events ADD COLUMN water_review INTEGER NOT NULL DEFAULT 0')
            if 'review_candidate' not in ecols:
                conn.execute('ALTER TABLE events ADD COLUMN review_candidate INTEGER')
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
        candidate_only: bool = False,
        review_candidate: bool = True,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO events (ts, clip_name, predicted, predicted_by, candidate_only, review_candidate) VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, clip_name, predicted, predicted_by,int(candidate_only),int(review_candidate)),
            )
            return int(cur.lastrowid)

    def review_candidates(self):
        with self._conn() as conn:
            return {name: bool(value) for name,value in conn.execute(
                'SELECT clip_name,MAX(review_candidate) FROM events GROUP BY clip_name')}

    def set_review_candidate(self, name, required):
        with self._conn() as conn:
            conn.execute('UPDATE events SET review_candidate=? WHERE clip_name=?', (int(required),name))

    def unclassified_clips(self):
        with self._conn() as conn:
            return [row[0] for row in conn.execute(
                'SELECT DISTINCT clip_name FROM events WHERE review_candidate IS NULL ORDER BY ts DESC')]

    def requires_visual_review(self,clip_name):
        with self._conn() as conn:
            row=conn.execute('SELECT MAX(candidate_only) FROM events WHERE clip_name=?',(clip_name,)).fetchone()
        return bool(row and row[0])

    def set_water_review(self,clip_name,required):
        with self._conn() as c:
            c.execute('UPDATE events SET water_review=? WHERE clip_name=?',(int(required),clip_name))

    def requires_water_review(self,clip_name):
        with self._conn() as c:
            row=c.execute('SELECT MAX(water_review) FROM events WHERE clip_name=?',(clip_name,)).fetchone()
        return bool(row and row[0])

    def set_prediction(
        self,
        clip_name: str,
        predicted: int,
        predicted_by: str | None,
        confidence: float | None = None,
    ) -> None:
        """回填某段的「测试模型预测」（影子模式：录完后台判完再写）。更新该 clip 的事件行。"""
        with self._conn() as conn:
            conn.execute(
                "UPDATE events SET predicted = ?, predicted_by = ?, predicted_confidence = ? "
                "WHERE clip_name = ?",
                (int(predicted), predicted_by, confidence, clip_name),
            )

    def model_hitrate(self, version: str, exclude_names=()) -> dict:
        """某版本「测试模型」在真实数据上的命中率：它预测过、且该段已人工标注的，预测对了多少。"""
        with self._conn() as conn:
            cols = {r[1] for r in conn.execute('PRAGMA table_info(labels)')}
            truth_filter = " AND (l.source IS NULL OR l.source IN ('human','ai'))" if 'source' in cols else ''
            target = "COALESCE(l.action_drinking,l.is_drinking)" if "action_drinking" in cols else "l.is_drinking"
            rows = conn.execute(
                f"SELECT e.predicted, {target}, e.clip_name FROM events e "
                "JOIN labels l ON e.clip_name = l.clip_name "
                "WHERE e.predicted IS NOT NULL AND e.predicted_by = ?" + truth_filter,
                (version,),
            ).fetchall()
        excluded=set(exclude_names)
        rows=[(pred,label) for pred,label,name in rows if name not in excluded]
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

    def clip_prediction_details(self) -> dict:
        """返回每段最新的模型判断、喝水概率和模型版本，供人工复核排序。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT clip_name, predicted, predicted_confidence, predicted_by FROM events "
                "WHERE clip_name IS NOT NULL AND predicted IS NOT NULL ORDER BY ts ASC"
            ).fetchall()
        return {
            name: {
                "drinking": bool(pred),
                "confidence": confidence,
                "by": predicted_by,
            }
            for name, pred, confidence, predicted_by in rows
        }

    def clip_metadata(self):
        """一次读取事件，保留最新预测及任一事件要求复核的语义。"""
        result = {}
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT clip_name,predicted,predicted_confidence,predicted_by,"
                "candidate_only,water_review,review_candidate FROM events "
                "WHERE clip_name IS NOT NULL ORDER BY ts ASC,id ASC"
            ).fetchall()
        for name, pred, confidence, version, visual, water, candidate in rows:
            item = result.setdefault(name, {"candidate_only": False, "water_review": False,
                                             "review_candidate": False})
            item['candidate_only'] |= bool(visual)
            item['water_review'] |= bool(water)
            item['review_candidate'] |= bool(candidate)
            if pred is not None:
                item['prediction'] = {"drinking": bool(pred), "confidence": confidence, "by": version}
        return result

    def drinking_records(self):
        """确认喝水记录至少相隔 60 秒，原始事件和人工标注不修改。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT MIN(e.ts),e.clip_name FROM events e JOIN labels l "
                "ON e.clip_name=l.clip_name WHERE l.is_drinking=1 "
                "GROUP BY e.clip_name ORDER BY MIN(e.ts),e.clip_name"
            ).fetchall()
        records = []
        for timestamp, name in rows:
            if not records or timestamp - records[-1]['ts'] >= 60:
                records.append({'ts': timestamp, 'clip_name': name})
        return records

    def count_between(self, start_ts: float, end_ts: float) -> int:
        return len(self.events_between(start_ts, end_ts))

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
        # 先按完整历史去重，再按日期/区间筛选，跨午夜口径仍一致。
        return [e for e in self.drinking_records() if start_ts <= e['ts'] < end_ts]
