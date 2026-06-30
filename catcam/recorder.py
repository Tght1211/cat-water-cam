from __future__ import annotations

from pathlib import Path

import cv2


def clip_filename(timestamp: float) -> str:
    return f"clip_{int(timestamp * 1000)}.mp4"


def open_writer(path: Path, fps: int, size: tuple[int, int]):
    """优先用 H.264(avc1) —— 浏览器 <video> 才认；这台 OpenCV 装不出就退回 mp4v。

    注意：mp4v(MPEG-4 Part 2) 文件能存能用播放器看，但 Chrome/Safari 的 <video>
    多半解不了，会黑屏 0:00。所以网页要播就必须 avc1。
    """
    for cc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), float(fps), size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"VideoWriter 打不开 {path}（avc1/mp4v 都不可用？）")


def prune_dir(clips_dir: Path, max_clips: int, is_deletable=None) -> None:
    """保留最多 max_clips 段。超量时**从最旧开始、只删「可删」的**段，删够为止。

    is_deletable(clip_name)->bool：哪些段允许被裁掉（如只删「没喝」）。None=任意旧段都可删（旧行为）。
    受保护的段（喝水/未判定）即便最旧也不删——宁可超出上限也保住它们。
    """
    clips = sorted(clips_dir.glob("clip_*.mp4"))   # 旧→新
    if max_clips <= 0:
        for old in clips:
            old.unlink()
        return
    over = len(clips) - max_clips
    if over <= 0:
        return
    deletable = is_deletable or (lambda name: True)
    removed = 0
    for p in clips:               # 最旧优先
        if removed >= over:
            break
        if deletable(p.name):
            p.unlink()
            removed += 1


class ClipRecorder:
    def __init__(self, clips_dir: Path, max_clips: int, fps: int, is_deletable=None):
        self.clips_dir = Path(clips_dir)
        self.max_clips = max_clips
        self.fps = fps
        # 裁剪时哪些段可删（如只删「没喝」）；None=旧行为（删最旧）。
        self.is_deletable = is_deletable
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def save_clip(self, frames: list, timestamp: float) -> Path:
        if not frames:
            raise ValueError("frames 为空，无法录制")
        path = self.clips_dir / clip_filename(timestamp)
        height, width = frames[0].shape[:2]
        writer = open_writer(path, self.fps, (width, height))
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
        prune_dir(self.clips_dir, self.max_clips, self.is_deletable)
        return path

    def list_clips(self) -> list[Path]:
        return sorted(self.clips_dir.glob("clip_*.mp4"), reverse=True)
