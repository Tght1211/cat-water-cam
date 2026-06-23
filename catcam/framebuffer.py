from __future__ import annotations

import threading
from collections import deque


class FrameBuffer:
    def __init__(self, seconds: float, fps: int):
        self.maxlen = max(1, int(seconds * fps))
        self._buf: deque = deque(maxlen=self.maxlen)
        # 采集线程 add、检测线程 all_frames（录制时）会并发访问，加锁防 deque 迭代中被改。
        self._lock = threading.Lock()

    def add(self, timestamp: float, frame) -> None:
        with self._lock:
            self._buf.append((timestamp, frame))

    def all_frames(self) -> list:
        with self._lock:
            return [frame for _, frame in self._buf]
