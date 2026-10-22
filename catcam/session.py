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

from catcam.recorder import clip_filename, open_writer


@dataclass
class SessionResult:
    timestamp: float       # 会话开始时间
    clip_name: str         # 存盘文件名
    photo: object          # 触发时的代表帧（猫在喝水），供邮件用
    audio_start: float = 0.0  # 该段对应的音频窗口起点（回溯了 preroll），供 mux 切片
    audio_end: float = 0.0    # 音频窗口终点（收尾时刻）
    candidate_only: bool = False
    presence_start: float | None = None  # 水碗区域首次持续在场；不是逐帧舔水标注
    presence_end: float | None = None    # 最后一次在场；不含离开宽限
    end_reason: str = "left"


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
        self._writing_path = None
        self._photo = None
        self._audio_start: float = 0.0
        self._candidate_only=False
        self._presence_start = None
        self._last_frame_at = 0.0
        self._await_leave = False

    @property
    def recording(self) -> bool:
        return self._writer is not None

    def update(self, now, frame, in_roi, in_roi_since, frame_buffer, candidate_only=False) -> SessionResult | None:
        """采集线程每帧（按 fps 节奏）调用。返回非 None 表示刚收尾了一段会话。"""
        if self._writer is None:
            # 达到手动设置的时长上限后，必须先离场，不能把同一次喝水切成多次。
            if self._await_leave:
                if not in_roi:
                    self._await_leave = False
                return None
            if now < self._cooldown_until:
                return None
            # 猫已在水碗持续够久 → 开录
            if in_roi and in_roi_since is not None and now - in_roi_since >= self.dwell_seconds:
                self._begin(now, frame, frame_buffer)
                self._presence_start = in_roi_since
                self._candidate_only=candidate_only
            return None

        # 录制中：写当前帧
        self._writer.write(frame)
        self._last_frame_at = now
        self._candidate_only |= candidate_only
        if in_roi:
            self._last_present = now
        left = not in_roi and now - self._last_present >= self.end_grace_seconds
        capped = self.max_session_seconds > 0 and now - self._start >= self.max_session_seconds
        if left or capped:
            self._await_leave = capped and not left and self.cooldown_seconds > 0
            return self._finish(now, "left" if left else "limit")
        return None

    def _begin(self, now, frame, frame_buffer) -> None:
        height, width = frame.shape[:2]
        self._path = self.recorder.clips_dir / clip_filename(now)
        self._writing_path = self._path.with_name('.' + self._path.stem + '.recording.mp4')
        self._writer = open_writer(self._writing_path, self.recorder.fps, (width, height))
        # pre-roll：把回放缓冲（含凑近过程 + dwell 这几秒）先写进去，避免漏掉开头。
        preroll_frames = frame_buffer.all_frames()
        for f in preroll_frames:
            self._writer.write(f)
        self._start = now
        self._last_frame_at = now
        self._last_present = now
        self._photo = frame
        # 音频窗口起点：从开录时刻回溯 preroll 帧对应的时长（帧按 fps 节奏），与视频开头对齐。
        self._audio_start = now - len(preroll_frames) / max(1, self.recorder.fps)

    def _finish(self, now, reason="left") -> SessionResult:
        self._writer.release()
        self._writer = None
        self._writing_path.replace(self._path)
        self._writing_path = None
        self.recorder.prune()
        # 一分钟从本次触发算起；录制中的会话始终持续到离开，不受冷却截断。
        self._cooldown_until = self._start + self.cooldown_seconds
        res = SessionResult(timestamp=self._start, clip_name=self._path.name, photo=self._photo,
                            audio_start=self._audio_start, audio_end=now,candidate_only=self._candidate_only,
                            presence_start=self._presence_start, presence_end=self._last_present,
                            end_reason=reason)
        self._path = None
        self._photo = None
        return res

    def close(self) -> SessionResult | None:
        """进程退出时收尾未完成的录制（避免半截文件丢失）。"""
        if self._writer is None:
            return None
        self._await_leave = True
        return self._finish(self._last_frame_at, "interrupted")
