import sqlite3
from datetime import datetime, timedelta

from catcam.stats import StatsStore, bucket_events


def test_bucket_events_by_hour_and_weekday():
    # 2026-06-22 是周一(weekday=0)；构造几条不同小时/星期的事件
    mon_9 = datetime(2026, 6, 22, 9, 30, 0)      # 周一 9 点
    mon_9b = datetime(2026, 6, 22, 9, 5, 0)      # 周一 9 点（再来一次）
    mon_21 = datetime(2026, 6, 22, 21, 0, 0)     # 周一 21 点
    wed_9 = datetime(2026, 6, 24, 9, 0, 0)       # 周三 9 点
    events = [{"ts": d.timestamp(), "clip_name": "x"} for d in (mon_9, mon_9b, mon_21, wed_9)]
    out = bucket_events(events)
    assert len(out["hourly"]) == 24 and len(out["weekday"]) == 7
    assert out["hourly"][9] == 3                  # 三条落在 9 点
    assert out["hourly"][21] == 1
    assert sum(out["hourly"]) == 4
    assert out["weekday"][0] == 3                 # 周一 3 条
    assert out["weekday"][2] == 1                 # 周三 1 条
    assert sum(out["weekday"]) == 4


def test_bucket_events_empty():
    out = bucket_events([])
    assert out["hourly"] == [0] * 24 and out["weekday"] == [0] * 7


def test_daily_counts_buckets_and_fills_zero(tmp_path):
    db = tmp_path / "db.sqlite"
    stats = StatsStore(db)
    today = datetime(2026, 6, 22, 12, 0, 0)
    stats.record_event(today.timestamp(), "t1.mp4")
    stats.record_event(today.timestamp(), "t2.mp4")
    yday = today - timedelta(days=1)
    stats.record_event(yday.timestamp(), "y1.mp4")
    # 口径：只数确认喝水的——给三段都标 is_drinking=1
    with sqlite3.connect(db) as c:
        for name in ("t1.mp4", "t2.mp4", "y1.mp4"):
            c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES (?, 1, NULL)", (name,))

    week = stats.daily_counts(today, 7)
    assert len(week) == 7
    assert week[-1] == ("06-22", 2)      # 今天 2 次
    assert week[-2] == ("06-21", 1)      # 昨天 1 次
    assert week[0][1] == 0               # 一周前补 0


def test_count_only_includes_confirmed_drinking(tmp_path):
    db = tmp_path / "db.sqlite"
    stats = StatsStore(db)
    now = datetime(2026, 6, 22, 12, 0, 0)
    start, end = now.timestamp() - 10, now.timestamp() + 10
    stats.record_event(now.timestamp(), "clip_a.mp4")
    stats.record_event(now.timestamp(), "clip_b.mp4")
    stats.record_event(now.timestamp(), None)   # 无 clip → 永远不计
    # 口径翻转：未标注一律不计
    assert stats.count_between(start, end) == 0

    # 标 a「喝了」(1) → 计入；标 b「没喝」(0) → 不计
    with sqlite3.connect(db) as c:
        c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES ('clip_a.mp4', 1, NULL)")
        c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES ('clip_b.mp4', 0, NULL)")
    assert stats.count_between(start, end) == 1
    names = [e["clip_name"] for e in stats.events_between(start, end)]
    assert names == ["clip_a.mp4"]


def test_model_hitrate_and_predictions(tmp_path):
    stats = StatsStore(tmp_path / "db.sqlite")
    # 测试模型 v1 对三段的预测
    stats.record_event(1.0, "a.mp4", predicted=1, predicted_by="v1")  # 判真喝水
    stats.record_event(2.0, "b.mp4", predicted=1, predicted_by="v1")  # 判真喝水
    stats.record_event(3.0, "c.mp4", predicted=0, predicted_by="v1")  # 判没喝
    stats.record_event(4.0, "d.mp4", predicted=1, predicted_by="v2")  # 别的版本
    # 人工标注：a 真喝(对)、b 没喝(错)、c 没喝(对)
    with sqlite3.connect(tmp_path / "db.sqlite") as conn:
        for name, val in [("a.mp4", 1), ("b.mp4", 0), ("c.mp4", 0)]:
            conn.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES (?, ?, NULL)", (name, val))
    hr = stats.model_hitrate("v1")
    assert hr["total"] == 3 and hr["correct"] == 2          # a、c 对，b 错
    assert abs(hr["rate"] - 2 / 3) < 1e-9
    assert stats.model_hitrate("v9")["rate"] is None         # 没有该版本预测
    preds = stats.clip_predictions()
    assert preds["a.mp4"] is True and preds["c.mp4"] is False


def test_model_hitrate_uses_only_human_labels(tmp_path):
    from catcam.feedback import FeedbackStore
    stats = StatsStore(tmp_path / "db.sqlite")
    FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    for name in ("human.mp4", "ai.mp4", "local.mp4"):
        stats.record_event(1, name, predicted=0, predicted_by="v1")
    with sqlite3.connect(stats.db_path) as conn:
        for name, source, truth in [("human.mp4", "human", 1), ("ai.mp4", "ai", 0), ("local.mp4", "local", 0)]:
            conn.execute("INSERT INTO labels (clip_name,is_drinking,source) VALUES (?,?,?)", (name, truth, source))
    result = stats.model_hitrate("v1")
    assert result["total"] == 1 and result["fn"] == 1 and result["correct"] == 0
