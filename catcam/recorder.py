from __future__ import annotations

from pathlib import Path

import cv2


def clip_filename(timestamp: float) -> str:
    return f"clip_{int(timestamp * 1000)}.mp4"


def prune_dir(clips_dir: Path, max_clips: int) -> None:
    clips = sorted(clips_dir.glob("clip_*.mp4"))
    for old in clips[:-max_clips] if max_clips > 0 else clips:
        old.unlink()


class ClipRecorder:
    def __init__(self, clips_dir: Path, max_clips: int, fps: int):
        self.clips_dir = Path(clips_dir)
        self.max_clips = max_clips
        self.fps = fps
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def save_clip(self, frames: list, timestamp: float) -> Path:
        if not frames:
            raise ValueError("frames 为空，无法录制")
        path = self.clips_dir / clip_filename(timestamp)
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, float(self.fps), (width, height))
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(
                f"VideoWriter 打不开 {path}（编码器 mp4v 不可用？）"
            )
        try:
            for f in frames:
                writer.write(f)
        finally:
            writer.release()
        prune_dir(self.clips_dir, self.max_clips)
        return path

    def list_clips(self) -> list[Path]:
        return sorted(self.clips_dir.glob("clip_*.mp4"), reverse=True)
