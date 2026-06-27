import sqlite3
from datetime import datetime
from catcam.stats import StatsStore, day_bounds


def _label(db_path, clip_name, is_drinking):
    """直接往同一个库的 labels 表写一行（StatsStore 也会 CREATE 这张表）。"""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO labels (clip_name, is_drinking, ts) VALUES (?, ?, 0) "
            "ON CONFLICT(clip_name) DO UPDATE SET is_drinking=excluded.is_drinking",
            (clip_name, 1 if is_drinking else 0),
        )


def test_day_bounds():
    dt = datetime(2026, 6, 19, 14, 30, 0)
    start, end = day_bounds(dt)
    assert datetime.fromtimestamp(start) == datetime(2026, 6, 19, 0, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2026, 6, 20, 0, 0, 0)


def test_only_confirmed_drinking_counts(tmp_path):
    db = tmp_path / "t.db"
    store = StatsStore(db)
    store.record_event(100.0, "a.mp4")   # 标「喝水」→ 计
    store.record_event(150.0, "b.mp4")   # 标「没喝」→ 不计
    store.record_event(180.0, "c.mp4")   # 未标注 → 不计（口径翻转）
    _label(db, "a.mp4", True)
    _label(db, "b.mp4", False)
    assert store.count_between(0.0, 1000.0) == 1
    assert store.count_between(0.0, 120.0) == 1
    assert store.count_between(120.0, 1000.0) == 0


def test_events_between_only_confirmed_sorted(tmp_path):
    db = tmp_path / "t.db"
    store = StatsStore(db)
    store.record_event(300.0, "b.mp4")
    store.record_event(100.0, "a.mp4")
    store.record_event(200.0, "u.mp4")   # 未标注 → 不出现
    _label(db, "a.mp4", True)
    _label(db, "b.mp4", True)
    events = store.events_between(0.0, 1000.0)
    assert [e["ts"] for e in events] == [100.0, 300.0]
    assert events[0]["clip_name"] == "a.mp4"


def test_record_returns_id(tmp_path):
    store = StatsStore(tmp_path / "t.db")
    first = store.record_event(1.0)
    second = store.record_event(2.0)
    assert second == first + 1


def test_set_prediction_updates_event(tmp_path):
    db = tmp_path / "t.db"
    store = StatsStore(db)
    store.record_event(100.0, "clip_p.mp4")          # 初始无预测
    store.set_prediction("clip_p.mp4", 1, "v5")
    assert store.clip_predictions().get("clip_p.mp4") is True
    hr = store.model_hitrate("v5")                    # 还没标注 → total 0
    assert hr["total"] == 0
