import sqlite3
from datetime import datetime, timedelta

from catcam.stats import StatsStore


def test_daily_counts_buckets_and_fills_zero(tmp_path):
    stats = StatsStore(tmp_path / "db.sqlite")
    today = datetime(2026, 6, 22, 12, 0, 0)
    stats.record_event(today.timestamp())
    stats.record_event(today.timestamp())
    yday = today - timedelta(days=1)
    stats.record_event(yday.timestamp())

    week = stats.daily_counts(today, 7)
    assert len(week) == 7
    assert week[-1] == ("06-22", 2)      # 今天 2 次
    assert week[-2] == ("06-21", 1)      # 昨天 1 次
    assert week[0][1] == 0               # 一周前补 0


def test_count_excludes_clips_labeled_not_drinking(tmp_path):
    stats = StatsStore(tmp_path / "db.sqlite")
    now = datetime(2026, 6, 22, 12, 0, 0)
    start, end = now.timestamp() - 10, now.timestamp() + 10
    stats.record_event(now.timestamp(), "clip_a.mp4")
    stats.record_event(now.timestamp(), "clip_b.mp4")
    stats.record_event(now.timestamp(), None)   # 没有 clip 的事件也算
    assert stats.count_between(start, end) == 3

    # 把 clip_b 标注成「没喝」→ 应从计数与时间点里排除
    with sqlite3.connect(tmp_path / "db.sqlite") as c:
        c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES ('clip_b.mp4', 0, NULL)")
    assert stats.count_between(start, end) == 2
    names = [e["clip_name"] for e in stats.events_between(start, end)]
    assert "clip_b.mp4" not in names and "clip_a.mp4" in names

    # 标注「喝了」(1) 仍计入
    with sqlite3.connect(tmp_path / "db.sqlite") as c:
        c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES ('clip_a.mp4', 1, NULL)")
    assert stats.count_between(start, end) == 2


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
