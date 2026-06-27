import numpy as np
from catcam.recorder import ClipRecorder
from catcam.feedback import FeedbackStore, extract_frames


def _make_clip(tmp_path, ts=1.0, n=6):
    rec = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    return rec.save_clip([frame for _ in range(n)], timestamp=ts)


def test_extract_frames_writes_jpgs(tmp_path):
    clip = _make_clip(tmp_path)
    out = tmp_path / "frames"
    paths = extract_frames(clip, out, max_frames=3)
    assert 1 <= len(paths) <= 3
    for p in paths:
        assert p.exists()
        assert p.suffix == ".jpg"


def test_label_clip_stores_label_and_frames(tmp_path):
    clip = _make_clip(tmp_path)
    store = FeedbackStore(db_path=tmp_path / "fb.db", training_dir=tmp_path / "train")
    store.label_clip(clip, is_drinking=True, max_frames=3)
    assert store.get_label(clip.name) is True
    drinking_dir = tmp_path / "train" / "drinking"
    assert drinking_dir.exists()
    assert len(list(drinking_dir.glob("*.jpg"))) >= 1


def test_label_clip_negative_goes_to_not_drinking(tmp_path):
    clip = _make_clip(tmp_path)
    store = FeedbackStore(db_path=tmp_path / "fb.db", training_dir=tmp_path / "train")
    store.label_clip(clip, is_drinking=False, max_frames=2)
    assert store.get_label(clip.name) is False
    assert (tmp_path / "train" / "not_drinking").exists()


def test_relabel_overwrites(tmp_path):
    clip = _make_clip(tmp_path)
    store = FeedbackStore(db_path=tmp_path / "fb.db", training_dir=tmp_path / "train")
    store.label_clip(clip, is_drinking=True, max_frames=2)
    store.label_clip(clip, is_drinking=False, max_frames=2)
    assert store.get_label(clip.name) is False


def test_get_label_none_when_unlabeled(tmp_path):
    store = FeedbackStore(db_path=tmp_path / "fb.db", training_dir=tmp_path / "train")
    assert store.get_label("nope.mp4") is None


def test_ai_label_stores_source_meta(tmp_path):
    clip = _make_clip(tmp_path)
    store = FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    store.label_clip(clip, True, max_frames=3, source="ai", confidence=0.8, reason="舌头接触水面")
    meta = store.label_meta(clip.name)
    assert meta == {"is_drinking": True, "source": "ai", "confidence": 0.8, "reason": "舌头接触水面"}
    assert store.label_source(clip.name) == "ai"
    assert list((tmp_path / "training" / "drinking").glob("*.jpg"))


def test_human_label_defaults_source_human(tmp_path):
    clip = _make_clip(tmp_path)
    store = FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    store.label_clip(clip, False)  # 不传 source → human
    assert store.label_source(clip.name) == "human"
    m = store.label_meta(clip.name)
    assert m["source"] == "human" and m["confidence"] is None and m["reason"] is None


def test_label_source_none_when_unlabeled(tmp_path):
    store = FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    assert store.label_source("nope.mp4") is None
    assert store.label_meta("nope.mp4") is None


def test_migration_backfills_source_human(tmp_path):
    import sqlite3
    db = tmp_path / "old.db"
    # 模拟老库：labels 表没有 source 列，已有一条人工标注
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE labels (clip_name TEXT PRIMARY KEY, is_drinking INTEGER NOT NULL, ts REAL)")
    con.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES ('old.mp4', 1, 1.0)")
    con.commit(); con.close()
    store = FeedbackStore(db, tmp_path / "training")  # 触发迁移 + 回填
    assert store.label_source("old.mp4") == "human"


def test_record_machine_label_no_frames_and_respects_human(tmp_path):
    store = FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    # 机器标（source=local）：写进 labels 但不抽训练帧
    store.record_machine_label("m.mp4", True, source="local", confidence=0.7, reason="本地判")
    assert store.get_label("m.mp4") is True
    assert store.label_source("m.mp4") == "local"
    assert not (tmp_path / "training" / "drinking").exists()   # 没抽帧
    # 人工标过的不被机器覆盖
    clip = _make_clip(tmp_path, ts=7.0)
    store.label_clip(clip, True)                                # 人工=喝水
    store.record_machine_label(clip.name, False, source="local")  # 机器想改成没喝
    assert store.get_label(clip.name) is True                  # 仍是人工的
    assert store.label_source(clip.name) == "human"
