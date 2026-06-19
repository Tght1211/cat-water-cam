from __future__ import annotations

import sqlite3
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

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def label_clip(self, clip_path: Path, is_drinking: bool, max_frames: int = 5) -> None:
        clip_path = Path(clip_path)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO labels (clip_name, is_drinking, ts) VALUES (?, ?, NULL) "
                "ON CONFLICT(clip_name) DO UPDATE SET is_drinking=excluded.is_drinking",
                (clip_path.name, 1 if is_drinking else 0),
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
