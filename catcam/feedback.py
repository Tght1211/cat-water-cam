from __future__ import annotations

import sqlite3
import time
import shutil
import math
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
            if "source" not in cols:
                conn.execute("ALTER TABLE labels ADD COLUMN source TEXT")
                # 老库已有的标注都是人工标的，回填 source='human'，让「人工 > AI 不覆盖」对历史段也成立。
                conn.execute("UPDATE labels SET source='human' WHERE source IS NULL")
            if "confidence" not in cols:
                conn.execute("ALTER TABLE labels ADD COLUMN confidence REAL")
            if "reason" not in cols:
                conn.execute("ALTER TABLE labels ADD COLUMN reason TEXT")
            if 'action_drinking' not in cols:
                conn.execute('ALTER TABLE labels ADD COLUMN action_drinking INTEGER')
            if 'outcome' not in cols:
                conn.execute('ALTER TABLE labels ADD COLUMN outcome TEXT')
            if 'drinking_seconds' not in cols:
                conn.execute('ALTER TABLE labels ADD COLUMN drinking_seconds REAL')
            conn.execute('CREATE TABLE IF NOT EXISTS clip_durations (clip_name TEXT PRIMARY KEY, video_seconds REAL NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS clip_notes (clip_name TEXT PRIMARY KEY, reason TEXT NOT NULL)')

    def clip_notes(self):
        with self._conn() as conn:
            return dict(conn.execute('SELECT clip_name,reason FROM clip_notes'))

    def set_note(self, name, reason):
        if not isinstance(reason,str) or len(reason)>500:
            raise ValueError('说明不能超过 500 字')
        with self._conn() as conn:
            conn.execute('INSERT INTO clip_notes VALUES (?,?) ON CONFLICT(clip_name) DO UPDATE SET reason=excluded.reason',
                         (name,reason.strip()))
            conn.execute("UPDATE labels SET reason=? WHERE clip_name=? AND source='human'",(reason.strip(),name))

    def record_video_duration(self, clip_path, seconds=None):
        path=Path(clip_path)
        if seconds is None:
            if not path.exists():
                return
            cap=cv2.VideoCapture(str(path))
            try:
                fps=cap.get(cv2.CAP_PROP_FPS)
                seconds=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps if fps>0 else 0
            finally:
                cap.release()
        if not math.isfinite(seconds) or seconds<=0:
            return
        with self._conn() as conn:
            conn.execute('INSERT INTO clip_durations VALUES(?,?) ON CONFLICT(clip_name) DO UPDATE SET video_seconds=excluded.video_seconds',
                         (path.name,seconds))

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def label_clip(
        self, clip_path: Path, is_drinking: bool, max_frames: int = 5,
        source: str = "human", confidence: float | None = None, reason: str | None = None, outcome: str | None = None,
    ) -> bool:
        clip_path = Path(clip_path)
        if reason is None:
            reason = self.clip_notes().get(clip_path.name)
        if outcome is not None and outcome not in ('drinking','dry_lick','no_action'):
            raise ValueError('请选择实际喝到水、舔水但没喝到或没有喝水动作')
        if outcome is not None and is_drinking != (outcome=='drinking'):
            raise ValueError('结果与喝水状态不一致')
        action = outcome!='no_action' if outcome is not None else bool(is_drinking)
        outcome = outcome or ('drinking' if is_drinking else 'no_action')
        if reason is not None and (not isinstance(reason,str) or len(reason)>500):
            raise ValueError('原因说明过长或格式无效')
        if clip_path.exists():
            self.record_video_duration(clip_path)
            archive=self.training_dir/'videos'/clip_path.name
            archive.parent.mkdir(parents=True,exist_ok=True)
            if not archive.exists():
                temp=archive.with_suffix('.tmp')
                shutil.copyfile(clip_path,temp)
                temp.replace(archive)
        # 改标注 = 数据变了 → trained_version 清回 NULL，让它重新算「未训练」。
        with self._conn() as conn:
            conn.execute('BEGIN IMMEDIATE')
            prior=conn.execute('SELECT source FROM labels WHERE clip_name=?',(clip_path.name,)).fetchone()
            if source != 'human' and prior and prior[0] == 'human':
                return False
            conn.execute(
                "INSERT INTO labels (clip_name,is_drinking,ts,trained_version,source,confidence,reason,action_drinking,outcome) "
                "VALUES (?,?,?,NULL,?,?,?,?,?) ON CONFLICT(clip_name) DO UPDATE SET "
                "trained_version=CASE WHEN COALESCE(labels.action_drinking,labels.is_drinking)=excluded.action_drinking "
                "AND COALESCE(labels.source,'human')=excluded.source THEN labels.trained_version ELSE NULL END, "
                "is_drinking=excluded.is_drinking,ts=excluded.ts,source=excluded.source,confidence=excluded.confidence,reason=excluded.reason, "
                "action_drinking=excluded.action_drinking,outcome=excluded.outcome, "
                "drinking_seconds=CASE WHEN excluded.is_drinking=1 THEN labels.drinking_seconds ELSE NULL END",
                (clip_path.name,int(is_drinking),time.time(),source,confidence,reason,int(action),outcome),
            )
        sub = "drinking" if action else "not_drinking"
        # 改判时移除旧类别的帧，避免同一段同时成为正、负样本。
        opposite = "not_drinking" if action else "drinking"
        for old in (self.training_dir / opposite).glob(clip_path.stem + '_*.jpg'):
            old.unlink()
        extract_frames(clip_path, self.training_dir / sub, max_frames)
        return True

    def record_machine_label(self, clip_name: str, is_drinking: bool, source: str = "local",
                             confidence: float | None = None, reason: str | None = None) -> bool:
        """机器判定写 labels 行（驱动计数/邮件），**不抽训练帧**（判定不是训练真值）。

        不覆盖人工/AI 标注（human/ai 优先）。返回是否写入。
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT source FROM labels WHERE clip_name = ?", (clip_name,)
            ).fetchone()
            if row is not None and row[0] in ("human", "ai"):
                return False
            conn.execute(
                "INSERT INTO labels (clip_name, is_drinking, ts, trained_version, source, confidence, reason) "
                "VALUES (?, ?, ?, NULL, ?, ?, ?) "
                "ON CONFLICT(clip_name) DO UPDATE SET "
                "is_drinking=excluded.is_drinking, ts=excluded.ts, source=excluded.source, "
                "confidence=excluded.confidence, reason=excluded.reason",
                (clip_name, 1 if is_drinking else 0, time.time(), source, confidence, reason),
            )
            conn.execute('UPDATE labels SET action_drinking=?,outcome=? WHERE clip_name=?',
                         (int(is_drinking),'drinking' if is_drinking else 'no_action',clip_name))
        return True

    def get_label(self, clip_name: str) -> bool | None:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT is_drinking FROM labels WHERE clip_name = ?", (clip_name,)
            )
            row = cur.fetchone()
        if row is None:
            return None
        return bool(row[0])

    def label_source(self, clip_name: str) -> str | None:
        """该段标注来源：'human' / 'ai' / None（未标注）。AILabeler 用它实现「人工 > AI」。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT source FROM labels WHERE clip_name = ?", (clip_name,)
            ).fetchone()
        return None if row is None else row[0]

    def label_meta(self, clip_name: str) -> dict | None:
        """该段标注的完整元信息，供视频页显示来源徽标；未标注返回 None。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT l.is_drinking,l.source,l.confidence,l.reason,l.drinking_seconds,d.video_seconds,l.action_drinking,l.outcome "
                "FROM labels l LEFT JOIN clip_durations d ON l.clip_name=d.clip_name WHERE l.clip_name = ?",
                (clip_name,),
            ).fetchone()
        return self._label_meta_row(row)

    @staticmethod
    def _label_meta_row(row):
        if row is None:
            return None
        estimated=round(row[5]*.8,1) if row[0] and row[5] else None
        effective=(row[4] if row[4] is not None else estimated) if row[0] else None
        return {"is_drinking": bool(row[0]), "action_drinking":bool(row[6]) if row[6] is not None else bool(row[0]), "outcome":row[7] or ("drinking" if row[0] else "no_action"), "source": row[1],
                "confidence": row[2], "reason": row[3], 'drinking_seconds': row[4],
                'estimated_drinking_seconds':estimated,'effective_drinking_seconds':effective,
                'duration_source':('human' if row[4] is not None else 'video_80_percent') if effective is not None else None}

    def clip_metadata(self):
        """列表只批量读取数据库，不在请求中打开录像或回写时长。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT l.clip_name,l.is_drinking,l.source,l.confidence,l.reason,"
                "l.drinking_seconds,d.video_seconds,l.action_drinking,l.outcome "
                "FROM labels l LEFT JOIN clip_durations d ON l.clip_name=d.clip_name"
            ).fetchall()
            durations = dict(conn.execute('SELECT clip_name,video_seconds FROM clip_durations'))
            notes = dict(conn.execute('SELECT clip_name,reason FROM clip_notes'))
        return {"meta": {r[0]: self._label_meta_row(r[1:]) for r in rows},
                "durations": durations, "notes": notes}

    def set_duration(self, clip_name, seconds):
        with self._conn() as c:
            c.execute('UPDATE labels SET drinking_seconds=? WHERE clip_name=? AND is_drinking=1 AND source=\'human\'',
                      (seconds,clip_name))

    def label_states(self) -> dict:
        """标注数据的各状态计数，供训练页展示、避免重复训练。"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*), "
                "COALESCE(SUM(COALESCE(action_drinking,is_drinking)),0), "
                "COALESCE(SUM(CASE WHEN trained_version IS NULL THEN 1 ELSE 0 END),0), "
                "COALESCE(SUM(CASE WHEN trained_version IS NOT NULL THEN 1 ELSE 0 END),0) "
                "FROM labels WHERE source IS NULL OR source IN ('human','ai')"
            ).fetchone()
        labeled, drinking, untrained, trained = (int(x) for x in row)
        return {
            "labeled": labeled,
            "drinking": drinking,
            "not_drinking": labeled - drinking,
            "untrained": untrained,   # 已标注、还没进过训练
            "trained": trained,       # 已标注、已被某次训练用过
        }

    def mark_trained(self, version: str, snapshot=None) -> None:
        """一次训练完成后，把当前所有标注标记为「已被该版本训练」。"""
        with self._conn() as conn:
            if snapshot is None:
                conn.execute("UPDATE labels SET trained_version = ? WHERE source IS NULL OR source != 'local'", (version,))
            else:
                conn.executemany(
                    "UPDATE labels SET trained_version = ? WHERE clip_name = ? AND ts IS ? "
                    "AND COALESCE(action_drinking,is_drinking) = ? AND (source IS NULL OR source != 'local')",
                    [(version, name, ts, label) for name, label, ts in snapshot],
                )
