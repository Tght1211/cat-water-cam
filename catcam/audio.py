"""录制音频：持续采音进有界环形缓冲，会话结束按时间窗切片、mux 进那段无声视频。

视频链路（OpenCV）不碰音频；音频完全走旁路。核心是**旁路采集 + 事后合成**：
- `AudioRing`：一个持续跑的 ffmpeg（avfoundation → s16le PCM）把裸音频喂进带 wall-clock
  时间轴的环形 bytearray；`slice(start,end)` 按时间窗取回 PCM。
- `mux_audio_into`：把切出的 PCM 用 ffmpeg 合进对应 mp4（`-c:v copy`，视频帧原样不动）。

**处处 fail-open**：没麦克风权限 / 无 ffmpeg / 采音起不来 → 当作无音频，视频照录（无声），绝不抛。
⚠️ macOS 下 ffmpeg 打开无授权的麦克风会**卡住**——所以有看门狗：启动若干秒没出数据就判定不可用并杀掉。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

BYTES_PER_SAMPLE = 2  # s16le 单声道


def capture_cmd(fmt: str, device: str, sample_rate: int) -> list[str]:
    """持续采音：从 fmt/device 抽单声道 PCM(s16le) 到 stdout。"""
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-f", fmt, "-i", device,
        "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-",
    ]


def mux_cmd(video: str, pcm_raw: str, out: str, sample_rate: int) -> list[str]:
    """把裸 PCM 合进视频：视频轨直接 copy（帧不变），音频编 aac，按较短的收尾。"""
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(video),
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", str(pcm_raw),
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(out),
    ]


def muted_cmd(video: str) -> list[str]:
    """去音轨版：丢音频、视频 copy，碎片化 mp4 便于流式（管道输出，不可 seek）。"""
    return [
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-i", str(video), "-an", "-c:v", "copy",
        "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1",
    ]


def mux_audio_into(video_path, pcm: bytes, sample_rate: int = 16000) -> bool:
    """把 pcm 合进 video_path（原子替换）。成功 True；任何问题 → 保留原无声视频、返回 False。"""
    video_path = Path(video_path)
    if not pcm:
        return False
    if shutil.which("ffmpeg") is None:
        return False
    out = video_path.with_suffix(".mux.mp4")   # 同目录，保证 os.replace 是同一文件系统
    raw = None
    try:
        fd, raw = tempfile.mkstemp(suffix=".raw")
        with os.fdopen(fd, "wb") as f:
            f.write(pcm)
        r = subprocess.run(mux_cmd(str(video_path), raw, str(out), sample_rate),
                           capture_output=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            os.replace(out, video_path)
            return True
        return False
    except Exception:  # noqa: BLE001 fail-open：合成失败不能影响录制
        return False
    finally:
        if raw and os.path.exists(raw):
            try:
                os.remove(raw)
            except OSError:
                pass
        if out.exists():
            try:
                out.unlink()
            except OSError:
                pass


class AudioRing:
    """持续采音进有界环形缓冲；按 wall-clock 时间窗切片取 PCM。

    时间轴：首块到达时锚定「绝对采样 0」的 wall 时间 `_t0_abs`，之后 wall(sample i) = _t0_abs + i/sr。
    丢弃只动 `_base_sample`（buf 头部丢多少采样），`_t0_abs` 恒定，时间映射始终成立。
    """

    def __init__(self, fmt: str, device: str, sample_rate: int = 16000,
                 max_seconds: float = 120.0, watchdog_seconds: float = 5.0):
        self.fmt = fmt
        self.device = device
        self.sr = sample_rate
        self.max_bytes = int(max_seconds * sample_rate) * BYTES_PER_SAMPLE
        self.watchdog_seconds = watchdog_seconds
        self.available = False
        self._buf = bytearray()
        self._t0_abs: float | None = None   # 绝对采样 0 的 wall 时间
        self._base_sample = 0               # buf[0] 对应的绝对采样序号（已丢弃数）
        self._total_samples = 0             # 累计喂入的采样数（含已丢弃）
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._stop = False

    # ---- 纯逻辑（可单测，不碰 ffmpeg）----
    def _ingest(self, now: float, chunk: bytes) -> None:
        n = len(chunk) // BYTES_PER_SAMPLE
        if n == 0:
            return
        if self._t0_abs is None:
            # 这块刚到达（约结束于 now），其起点约 now - n/sr → 即绝对采样 0 的 wall 时间
            self._t0_abs = now - n / self.sr
        self._buf += chunk
        self._total_samples += n
        excess = len(self._buf) - self.max_bytes
        if excess > 0:
            excess -= excess % BYTES_PER_SAMPLE   # 对齐到整采样
            if excess > 0:
                del self._buf[:excess]
                self._base_sample += excess // BYTES_PER_SAMPLE

    def slice(self, start_ts: float, end_ts: float) -> bytes | None:
        with self._lock:
            if self._t0_abs is None:
                return None
            s0 = round((start_ts - self._t0_abs) * self.sr)
            s1 = round((end_ts - self._t0_abs) * self.sr)
            s0 = max(s0, self._base_sample)
            s1 = min(s1, self._total_samples)
            if s1 <= s0:
                return None
            b0 = (s0 - self._base_sample) * BYTES_PER_SAMPLE
            b1 = (s1 - self._base_sample) * BYTES_PER_SAMPLE
            return bytes(self._buf[b0:b1])

    # ---- 采集（起线程；测试不触及）----
    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                capture_cmd(self.fmt, self.device, self.sr),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
        except Exception as e:  # noqa: BLE001
            print(f"音频采集启动失败（{e}），本次录制将为无声。")
            return
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _reader(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        while not self._stop:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            with self._lock:
                self._ingest(time.time(), chunk)
            self.available = True

    def _watchdog(self) -> None:
        deadline = time.time() + self.watchdog_seconds
        while not self._stop and time.time() < deadline:
            if self.available:
                return
            time.sleep(0.2)
        if not self.available:
            print("音频采集未在规定时间内产出数据（多半是麦克风权限没给），"
                  "本次录制将为无声。去『系统设置→隐私与安全性→麦克风』给终端授权后重启。")
            self.stop()

    def stop(self) -> None:
        self._stop = True
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
