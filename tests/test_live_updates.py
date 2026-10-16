import sqlite3
from catcam.live_updates import DataRevision


def test_revision_tracks_corrections_and_water_but_ignores_monitor_samples(tmp_path):
    db=tmp_path/'db'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE events (id INTEGER PRIMARY KEY, predicted INTEGER)')
        c.execute('CREATE TABLE monitoring (ts REAL)')
        c.execute('CREATE TABLE water_additions (ml REAL)')
        c.execute('INSERT INTO events VALUES(1,0)')
    revision=DataRevision([db]);first=revision.get()
    assert revision.get()==first
    with sqlite3.connect(db) as c:c.execute('INSERT INTO monitoring VALUES(123)')
    assert revision.get()==first
    with sqlite3.connect(db) as c:c.execute('UPDATE events SET predicted=1 WHERE id=1')
    changed=revision.get();assert changed!=first
    with sqlite3.connect(db) as c:c.execute('INSERT INTO water_additions VALUES(950)')
    assert revision.get()!=changed


def test_revision_detects_wal_changes(tmp_path):
    db=tmp_path/'db'
    with sqlite3.connect(db) as c:
        c.execute('PRAGMA journal_mode=WAL')
        c.execute('CREATE TABLE labels (name TEXT, truth INTEGER)');c.commit()
        revision=DataRevision([db]);before=revision.get()
        c.execute("INSERT INTO labels VALUES('clip',1)");c.commit()
        assert revision.get()!=before
