from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import cv2


def extract_frames(clip_path: Path, out_dir: Path, max_frames: int) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip_path))
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames:
        return []
    step = max(1, len(frames) // max_frames)
    chosen = frames[::step][:max_frames]
    stem = Path(clip_path).stem
    written: list[Path] = []
    for i, frame in enumerate(chosen):
        p = out_dir / f"{stem}_{i}.jpg"
        cv2.imwrite(str(p), frame)
        written.append(p)
    return written


class FeedbackStore:
    def __init__(self, db_path: Path, training_dir: Path):
        self.db_path = Path(db_path)
        self.training_dir = Path(training_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS labels ("
                "clip_name TEXT PRIMARY KEY, "
                "is_drinking INTEGER NOT NULL, "
                "ts REAL)"
            )
            # 老库迁移：加一列记录「这条标注是在哪个模型版本里训过的」。
            # NULL = 还没被任何训练用过（已标注未训练）。
            cols = [r[1] for r in conn.execute("PRAGMA table_info(labels)")]
            if "trained_version" not in cols:
                conn.execute("ALTER TABLE labels ADD COLUMN trained_version TEXT")

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def label_clip(self, clip_path: Path, is_drinking: bool, max_frames: int = 5) -> None:
        clip_path = Path(clip_path)
        # 改标注 = 数据变了 → trained_version 清回 NULL，让它重新算「未训练」。
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO labels (clip_name, is_drinking, ts, trained_version) "
                "VALUES (?, ?, ?, NULL) "
                "ON CONFLICT(clip_name) DO UPDATE SET "
                "is_drinking=excluded.is_drinking, ts=excluded.ts, trained_version=NULL",
                (clip_path.name, 1 if is_drinking else 0, time.time()),
            )
        sub = "drinking" if is_drinking else "not_drinking"
        extract_frames(clip_path, self.training_dir / sub, max_frames)

    def get_label(self, clip_name: str) -> bool | None:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT is_drinking FROM labels WHERE clip_name = ?", (clip_name,)
            )
            row = cur.fetchone()
        if row is None:
            return None
        return bool(row[0])

    def label_states(self) -> dict:
        """标注数据的各状态计数，供训练页展示、避免重复训练。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(is_drinking),0), "
                "COALESCE(SUM(CASE WHEN trained_version IS NULL THEN 1 ELSE 0 END),0), "
                "COALESCE(SUM(CASE WHEN trained_version IS NOT NULL THEN 1 ELSE 0 END),0) "
                "FROM labels"
            ).fetchone()
        labeled, drinking, untrained, trained = (int(x) for x in row)
        return {
            "labeled": labeled,
            "drinking": drinking,
            "not_drinking": labeled - drinking,
            "untrained": untrained,   # 已标注、还没进过训练
            "trained": trained,       # 已标注、已被某次训练用过
        }

    def mark_trained(self, version: str) -> None:
        """一次训练完成后，把当前所有标注标记为「已被该版本训练」。"""
        with self._conn() as conn:
            conn.execute("UPDATE labels SET trained_version = ?", (version,))
