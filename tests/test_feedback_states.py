import numpy as np
import cv2

from catcam.feedback import FeedbackStore
from catcam.recorder import ClipRecorder


def _clip(rec, ts):
    frames = [np.full((32, 32, 3), 100, np.uint8) for _ in range(4)]
    return rec.save_clip(frames, ts)


def test_label_states_and_mark_trained(tmp_path):
    rec = ClipRecorder(tmp_path / "clips", max_clips=100, fps=10)
    fb = FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    a, b, c = _clip(rec, 1.0), _clip(rec, 2.0), _clip(rec, 3.0)

    fb.label_clip(a, True)
    fb.label_clip(b, True)
    fb.label_clip(c, False)
    s = fb.label_states()
    assert s == {"labeled": 3, "drinking": 2, "not_drinking": 1, "untrained": 3, "trained": 0}

    # 训练后：全部标记为已训练
    fb.mark_trained("v1")
    s = fb.label_states()
    assert s["untrained"] == 0 and s["trained"] == 3

    # 改一个标注 → 它重新变「未训练」
    fb.label_clip(c, True)
    s = fb.label_states()
    assert s["untrained"] == 1 and s["trained"] == 2
    assert s["drinking"] == 3 and s["not_drinking"] == 0


def test_migration_adds_trained_version_column(tmp_path):
    import sqlite3
    db = tmp_path / "db.sqlite"
    # 造一个「老库」：labels 表没有 trained_version 列
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE labels (clip_name TEXT PRIMARY KEY, is_drinking INTEGER NOT NULL, ts REAL)")
        conn.execute("INSERT INTO labels VALUES ('old.mp4', 1, NULL)")
    fb = FeedbackStore(db, tmp_path / "training")   # 初始化应自动补列
    s = fb.label_states()
    assert s["labeled"] == 1 and s["untrained"] == 1   # 老数据默认未训练
