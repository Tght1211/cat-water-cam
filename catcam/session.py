"""整段会话录制：从「猫开始喝水」录到「猫离开」，时长可变。

和旧的「固定 clip_seconds 缓冲 dump」不同，这里是个状态机：
- 空闲：猫在水碗持续 dwell_seconds → 开始录制（先把回放缓冲里的「凑近过程」写进去做 pre-roll）。
- 录制：每帧写入；猫还在就刷新「最后在场时间」。
- 结束：猫离开持续 end_grace_seconds（或到达 max_session_seconds 封顶）→ 收尾存盘，进冷却。

只在采集线程里按帧率喂帧（writer fps 一致，播放速度才对）。检测线程只负责把「猫是否在碗里」
喂进来。写文件全程只有采集线程一个写者，无需为 writer 上锁。
"""
from __future__ import annotations

from dataclasses import dataclass

from catcam.recorder import clip_filename, open_writer, prune_dir


@dataclass
class SessionResult:
    timestamp: float       # 会话开始时间
    clip_name: str         # 存盘文件名
    photo: object          # 触发时的代表帧（猫在喝水），供邮件用


class DrinkSession:
    def __init__(
        self,
        recorder,
        dwell_seconds: float,
        end_grace_seconds: float,
        max_session_seconds: float,
        cooldown_seconds: float,
    ):
        self.recorder = recorder  # ClipRecorder：取 clips_dir / fps / max_clips
        self.dwell_seconds = dwell_seconds
        self.end_grace_seconds = end_grace_seconds
        self.max_session_seconds = max_session_seconds
        self.cooldown_seconds = cooldown_seconds
        self._writer = None
        self._start: float = 0.0
        self._last_present: float = 0.0
        self._cooldown_until: float = 0.0
        self._path = None
        self._photo = None

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def update(self, now, frame, in_roi, in_roi_since, frame_buffer) -> SessionResult | None:
        """采集线程每帧（按 fps 节奏）调用。返回非 None 表示刚收尾了一段会话。"""
        if self._writer is None:
            if now < self._cooldown_until:
                return None
            # 猫已在水碗持续够久 → 开录
            if in_roi and in_roi_since is not None and now - in_roi_since >= self.dwell_seconds:
                self._begin(now, frame, frame_buffer)
            return None

        # 录制中：写当前帧
        self._writer.write(frame)
        if in_roi:
            self._last_present = now
        ended = (
            now - self._last_present >= self.end_grace_seconds
            or now - self._start >= self.max_session_seconds
        )
        if ended:
            return self._finish(now)
        return None

    def _begin(self, now, frame, frame_buffer) -> None:
        height, width = frame.shape[:2]
        self._path = self.recorder.clips_dir / clip_filename(now)
        self._writer = open_writer(self._path, self.recorder.fps, (width, height))
        # pre-roll：把回放缓冲（含凑近过程 + dwell 这几秒）先写进去，避免漏掉开头。
        for f in frame_buffer.all_frames():
            self._writer.write(f)
        self._start = now
        self._last_present = now
        self._photo = frame

    def _finish(self, now) -> SessionResult:
        self._writer.release()
        self._writer = None
        prune_dir(self.recorder.clips_dir, self.recorder.max_clips)
        self._cooldown_until = now + self.cooldown_seconds
        res = SessionResult(timestamp=self._start, clip_name=self._path.name, photo=self._photo)
        self._path = None
        self._photo = None
        return res

    def close(self) -> SessionResult | None:
        """进程退出时收尾未完成的录制（避免半截文件丢失）。"""
        if self._writer is None:
            return None
        return self._finish(self._last_present)
