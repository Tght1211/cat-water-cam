from datetime import datetime
from catcam.stats import StatsStore, day_bounds


def test_day_bounds():
    dt = datetime(2026, 6, 19, 14, 30, 0)
    start, end = day_bounds(dt)
    assert datetime.fromtimestamp(start) == datetime(2026, 6, 19, 0, 0, 0)
    assert datetime.fromtimestamp(end) == datetime(2026, 6, 20, 0, 0, 0)


def test_record_and_count(tmp_path):
    store = StatsStore(tmp_path / "t.db")
    store.record_event(100.0, "clip_100000.mp4")
    store.record_event(150.0, "clip_150000.mp4")
    store.record_event(500.0, None)
    assert store.count_between(0.0, 200.0) == 2
    assert store.count_between(0.0, 1000.0) == 3


def test_events_between_sorted_with_fields(tmp_path):
    store = StatsStore(tmp_path / "t.db")
    store.record_event(300.0, "b.mp4")
    store.record_event(100.0, "a.mp4")
    events = store.events_between(0.0, 1000.0)
    assert [e["ts"] for e in events] == [100.0, 300.0]
    assert events[0]["clip_name"] == "a.mp4"


def test_record_returns_id(tmp_path):
    store = StatsStore(tmp_path / "t.db")
    first = store.record_event(1.0)
    second = store.record_event(2.0)
    assert second == first + 1
