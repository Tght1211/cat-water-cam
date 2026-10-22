import sqlite3
from datetime import datetime

import pytest

from catcam import database
from catcam.dispenser import DispenserStore
from catcam.feedback import FeedbackStore
from catcam.health import MonitoringStore, health_summary
from catcam.live_updates import DataRevision
from catcam.stats import StatsStore


def assert_closed(connection):
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")


def test_transaction_commits_and_closes(tmp_path):
    db = tmp_path / "catcam.db"
    with database.connect(db) as connection:
        connection.execute("CREATE TABLE sample (value INTEGER)")
        connection.execute("INSERT INTO sample VALUES (1)")
    assert_closed(connection)
    with database.connect(db) as reader:
        assert reader.execute("SELECT value FROM sample").fetchall() == [(1,)]


def test_transaction_rolls_back_and_closes_on_error(tmp_path):
    db = tmp_path / "catcam.db"
    with database.connect(db) as setup:
        setup.execute("CREATE TABLE sample (value INTEGER)")
    with pytest.raises(ValueError):
        with database.connect(db) as connection:
            connection.execute("INSERT INTO sample VALUES (1)")
            raise ValueError("测试事务失败")
    assert_closed(connection)
    with database.connect(db) as reader:
        assert reader.execute("SELECT * FROM sample").fetchall() == []


def test_repeated_monitoring_and_web_queries_release_all_connections(tmp_path, monkeypatch):
    # 保留所有连接引用，确保释放不依赖垃圾回收的时机。
    connections = []
    original_connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(database.sqlite3, "connect", tracked_connect)
    db = tmp_path / "catcam.db"
    stats = StatsStore(db)
    feedback = FeedbackStore(db, tmp_path / "training")
    dispenser = DispenserStore(db)
    monitoring = MonitoringStore(db)
    revision = DataRevision([db])
    now = datetime(2026, 10, 17, 12)
    for index in range(20):
        monitoring.observe(now.timestamp() + index, {"usable": True})
        stats.record_event(now.timestamp() + index, f"clip_{index}.mp4")
        health_summary(stats, monitoring, now)
        feedback.label_states()
        dispenser.get_state()
        revision.get()
    assert len(connections) > 1000
    for connection in connections:
        assert_closed(connection)
