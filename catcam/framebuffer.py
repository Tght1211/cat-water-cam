from __future__ import annotations

from collections import deque


class FrameBuffer:
    def __init__(self, seconds: float, fps: int):
        self.maxlen = max(1, int(seconds * fps))
        self._buf: deque = deque(maxlen=self.maxlen)

    def add(self, timestamp: float, frame) -> None:
        self._buf.append((timestamp, frame))

    def all_frames(self) -> list:
        return [frame for _, frame in self._buf]
