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
    store.set_prediction("clip_p.mp4", 1, "v5", 0.82)
    assert store.clip_predictions().get("clip_p.mp4") is True
    assert store.clip_prediction_details()["clip_p.mp4"] == {
        "drinking": True, "confidence": 0.82, "by": "v5"
    }
    hr = store.model_hitrate("v5")                    # 还没标注 → total 0
    assert hr["total"] == 0


def test_drinking_records_minimum_interval_and_boundaries(tmp_path):
    db = tmp_path / 't.db'
    store = StatsStore(db)
    # 59 秒归同次；恰好 60 秒可记下一次；重复事件不能重复计数。
    for ts, name in [(100, 'a'), (115, 'b'), (159, 'c'), (160, 'd'), (219, 'e'), (220, 'f'), (280, 'g')]:
        store.record_event(ts, name)
        _label(db, name, True)
    store.record_event(101, 'a')
    assert [e['ts'] for e in store.drinking_records()] == [100, 280]
    assert store.count_between(0, 300) == 2
    assert store.events_between(110, 160) == []
    assert store.count_between(0, 160) + store.count_between(160, 300) == 2
    # 改判会重新计算，原始片段/标注均保留。
    _label(db, 'a', False)
    assert [e['ts'] for e in store.drinking_records()] == [115, 280]
    assert len(store.clip_metadata()) == 7


def test_drinking_interval_across_midnight(tmp_path):
    db = tmp_path / 't.db'
    store = StatsStore(db)
    midnight = datetime(2026, 10, 13).timestamp()
    for delta, name in [(-20, 'a'), (10, 'b'), (40, 'c'), (100, 'd')]:
        store.record_event(midnight + delta, name)
        _label(db, name, True)
    assert store.count_between(midnight - 86400, midnight) == 1
    assert store.count_between(midnight, midnight + 86400) == 1
    assert store.events_between(midnight, midnight + 86400)[0]['clip_name'] == 'd'


def test_continuous_drinking_longer_than_a_minute_is_one_visit(tmp_path):
    db = tmp_path / 't.db'
    store = StatsStore(db)
    timestamps = [100 + offset for offset in range(0, 181, 15)]
    for timestamp in reversed(timestamps):
        name = f'clip_{timestamp}.mp4'
        store.record_event(timestamp, name)
        _label(db, name, True)
    assert store.count_between(0, 1000) == 1
    assert store.drinking_records() == [{'ts': 100, 'clip_name': 'clip_100.mp4'}]
    assert store.events_between(160, 1000) == []
    assert len(store.clip_metadata()) == len(timestamps)


def test_exact_minute_gap_allows_new_visit_without_clock_minute_buckets(tmp_path):
    db = tmp_path / 't.db'
    store = StatsStore(db)
    for timestamp, name in [(119.9, 'a'), (120.1, 'b'), (180.1, 'c')]:
        store.record_event(timestamp, name)
        _label(db, name, True)
    assert [event['clip_name'] for event in store.drinking_records()] == ['a', 'c']


def test_non_drinking_and_duplicate_events_do_not_extend_visit(tmp_path):
    db = tmp_path / 't.db'
    store = StatsStore(db)
    for timestamp, name, drinking in [(100, 'a', True), (140, 'negative', False),
                                      (150, 'pending', None), (160, 'b', True)]:
        store.record_event(timestamp, name)
        if drinking is not None:
            _label(db, name, drinking)
    store.record_event(155, 'a')
    assert [event['clip_name'] for event in store.drinking_records()] == ['a', 'b']
    _label(db, 'pending', True)
    assert [event['clip_name'] for event in store.drinking_records()] == ['a']
    _label(db, 'pending', False)
    assert [event['clip_name'] for event in store.drinking_records()] == ['a', 'b']


def test_event_bulk_metadata_preserves_latest_prediction_and_review_flags(tmp_path):
    store = StatsStore(tmp_path / 't.db')
    store.record_event(100, 'a', predicted=1, predicted_by='v1', candidate_only=True)
    store.record_event(200, 'a', predicted=0, predicted_by='v2', review_candidate=False)
    store.set_water_review('a', True)
    item = store.clip_metadata()['a']
    assert item['prediction'] == store.clip_prediction_details()['a']
    assert item['candidate_only'] and item['water_review'] and item['review_candidate']


def test_october_13_split_recordings_count_as_two_visits(tmp_path):
    db = tmp_path / 't.db'
    store = StatsStore(db)
    times = [(10, 7, 59, 435000), (10, 8, 14, 529000), (10, 8, 29, 612000),
             (19, 44, 13, 119000), (19, 44, 28, 204000),
             (19, 44, 43, 278000), (19, 44, 58, 372000)]
    for i, (h, m, s, us) in enumerate(times):
        ts = datetime(2026, 10, 13, h, m, s, us).timestamp()
        name = f'clip_{int(ts * 1000)}.mp4'
        store.record_event(ts, name)
        _label(db, name, True)
    start, end = day_bounds(datetime(2026, 10, 13))
    assert store.count_between(start, end) == 2
    assert len(store.clip_metadata()) == 7
