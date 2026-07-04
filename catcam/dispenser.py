"""饮水机状态：滤芯周期、蓄水记录、喝水量(ml)自校准、剩余水量反推。

不测体积——靠「一整个蓄水周期的总量 ÷ 期间喝水次数」池化反推每次约几 ml，越用越准。
`DispenserStore` 只管自身状态与校准；「期间喝了几次」由 web 层用 StatsStore 算好传进来（单一职责）。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def estimate_remaining(refill_ml: float, drinks_since: int, ml_per_drink: float) -> float:
    """剩余水量 = 上次加的量 − 之后喝水次数×每次 ml，夹到 0（不给负数）。"""
    return max(0.0, refill_ml - drinks_since * ml_per_drink)


def filter_days_left(last_change_ts: float, cycle_days: float, now: float) -> float:
    """滤芯还剩几天 = 周期 − 距上次更换的天数（可为负=已超期）。"""
    return cycle_days - (now - last_change_ts) / 86400.0


class DispenserStore:
    def __init__(self, db_path, default_ml_per_drink: float = 30.0, default_cycle_days: int = 30):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ml_per_drink = default_ml_per_drink
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS dispenser ("
                "id INTEGER PRIMARY KEY CHECK (id = 1), "
                "last_filter_change_ts REAL NOT NULL DEFAULT 0, "
                "filter_cycle_days INTEGER NOT NULL DEFAULT 30, "
                "last_refill_ts REAL NOT NULL DEFAULT 0, "
                "last_refill_ml REAL NOT NULL DEFAULT 0, "
                "calib_sum_ml REAL NOT NULL DEFAULT 0, "
                "calib_drinks INTEGER NOT NULL DEFAULT 0)"
            )
            # 单行状态：首次插入默认值（周期用配置默认）
            conn.execute(
                "INSERT OR IGNORE INTO dispenser (id, filter_cycle_days) VALUES (1, ?)",
                (default_cycle_days,),
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _row(self, conn) -> dict:
        cur = conn.execute(
            "SELECT last_filter_change_ts, filter_cycle_days, last_refill_ts, "
            "last_refill_ml, calib_sum_ml, calib_drinks FROM dispenser WHERE id = 1"
        )
        c = cur.fetchone()
        return {
            "last_filter_change_ts": c[0], "filter_cycle_days": c[1],
            "last_refill_ts": c[2], "last_refill_ml": c[3],
            "calib_sum_ml": c[4], "calib_drinks": c[5],
        }

    def ml_per_drink(self, row: dict) -> float:
        """池化估计：累计总量 / 累计喝水次数；没校准过就用默认。"""
        if row["calib_drinks"] > 0:
            return row["calib_sum_ml"] / row["calib_drinks"]
        return self.default_ml_per_drink

    def get_state(self) -> dict:
        with self._conn() as conn:
            row = self._row(conn)
        row["ml_per_drink"] = self.ml_per_drink(row)
        return row

    def refill(self, ml: float, now: float, prev_drinks: int) -> None:
        """记一次蓄水。若上一次蓄水存在且期间喝过水，把上周期(总量,次数)并入校准池。"""
        with self._conn() as conn:
            row = self._row(conn)
            sum_ml, drinks = row["calib_sum_ml"], row["calib_drinks"]
            if row["last_refill_ts"] > 0 and row["last_refill_ml"] > 0 and prev_drinks > 0:
                sum_ml += row["last_refill_ml"]
                drinks += prev_drinks
            conn.execute(
                "UPDATE dispenser SET last_refill_ts = ?, last_refill_ml = ?, "
                "calib_sum_ml = ?, calib_drinks = ? WHERE id = 1",
                (now, ml, sum_ml, drinks),
            )

    def mark_filter(self, now: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE dispenser SET last_filter_change_ts = ? WHERE id = 1", (now,))

    def set_cycle(self, days: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE dispenser SET filter_cycle_days = ? WHERE id = 1", (int(days),))
