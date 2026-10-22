"""读取本机 WoSee 的摄像头 PCM 音轨，复用现有会话音频缓冲和合成。"""
from __future__ import annotations

import threading
import time

from catcam.audio import AudioRing
from catcam.source import WoSeeCapture


class WoSeeAudioRing(AudioRing):
    def __init__(self, source, max_seconds=120, connector=None):
        # WoSee 已解码成 16 kHz 单声道 s16le，不采本机麦克风。
        super().__init__('wosee', '摄像头音轨', sample_rate=16000, max_seconds=max_seconds)
        self.bridge = WoSeeCapture(source)
        self.connector = connector
        self.thread = None
        self.socket = None
        self.error = None
        self._stopped = threading.Event()

    def start(self):
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _receive_chunk(self, now, chunk):
        if not isinstance(chunk, bytes) or len(chunk) % 2 or len(chunk) > 256000:
            raise ValueError('摄像头音频不是有效的 PCM16')
        if not chunk:
            return
        with self._lock:
            self._ingest(now, chunk)
            self.available = True
            self.error = None

    def _reader(self):
        try:
            if self.connector is None:
                from websockets.sync.client import connect
                self.connector = connect
        except ImportError:
            self.error = '缺少 websockets 依赖，摄像头视频继续录制但暂时无声'
            return
        while not self._stopped.is_set():
            try:
                status = self.bridge._json('/api/status')
                if status.get('serial') != self.bridge.serial:
                    raise ValueError('摄像头设备已切换')
                token = self.bridge._json('/api/config')['token']
                url = self.bridge.base.replace('http://', 'ws://', 1)+'/api/audio/listen'
                with self.connector(url, subprotocols=['wosee-audio', token], proxy=None,
                                    open_timeout=5, close_timeout=2, max_size=256000) as socket:
                    self.socket = socket
                    # 断线后重新锚定时间，不能把两段音频接成没有间隔的声音。
                    with self._lock:
                        self._buf.clear()
                        self._t0_abs = None
                        self._base_sample = self._total_samples = 0
                    last_audio = time.monotonic()
                    checked = last_audio
                    while not self._stopped.is_set():
                        if time.monotonic()-checked >= 2:
                            if self.bridge._json('/api/status').get('serial') != self.bridge.serial:
                                raise ValueError('摄像头设备已切换')
                            checked = time.monotonic()
                        try:
                            packet = socket.recv(timeout=1)
                        except TimeoutError:
                            if time.monotonic()-last_audio > 5:
                                raise TimeoutError('摄像头音频中断')
                            continue
                        if not isinstance(packet, bytes):
                            raise ValueError('摄像头音频服务返回错误')
                        self._receive_chunk(time.time(), packet)
                        last_audio = time.monotonic()
            except Exception:
                # 不回显可能携带本机访问令牌的 WebSocket 异常。
                with self._lock:
                    self.available = False
                    self.error = '摄像头音轨暂不可用，正在重连；视频录制继续'
            finally:
                self.socket = None
            self._stopped.wait(2)

    def slice(self, start_ts, end_ts):
        # 补齐前置缺口，避免晚到的音轨被错误地移到视频开头。
        with self._lock:
            if self._t0_abs is None or end_ts <= start_ts:
                return None
            start = round((start_ts-self._t0_abs)*self.sr)
            end = round((end_ts-self._t0_abs)*self.sr)
            first = max(start, self._base_sample)
            last = min(end, self._total_samples)
            if last <= first:
                return None
            return (b'\0\0'*(first-start)
                    + bytes(self._buf[(first-self._base_sample)*2:(last-self._base_sample)*2])
                    + b'\0\0'*(end-last))

    def status(self):
        status = super().status()
        status.update(source='wosee_camera', process_alive=self.thread is not None and self.thread.is_alive(),
                      error=self.error)
        return status

    def stop(self):
        self._stopped.set()
        self.available = False
        socket = self.socket
        if socket is not None:
            try:
                socket.close()
            except Exception:
                pass
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)
