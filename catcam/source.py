from __future__ import annotations

import json
import re
import time
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

import cv2
import numpy as np


class WoSeeCapture:
    is_live_stream = True

    def __init__(self, source: str, opener=None, clock=time.time, sleep=time.sleep):
        parsed = urlsplit(source)
        serial = parsed.path.strip("/")
        if (parsed.scheme != "wosee" or parsed.hostname not in ("127.0.0.1", "localhost") or
                parsed.username or parsed.password or parsed.query or parsed.fragment or
                not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", serial)):
            raise ValueError("WoSee 视频源须为 wosee://127.0.0.1:18765/设备序列号，仅连接本机服务")
        port = parsed.port or 18765
        if not 1024 <= port <= 65534:
            raise ValueError("WoSee 本机端口无效")
        self.base = f"http://{parsed.hostname}:{port}"
        self.serial = serial
        self.opener = opener or build_opener(ProxyHandler({}))
        self.clock = clock
        self.sleep = sleep
        self.response = None
        self.closed = False
        self.timestamp = None
        self.sequence = None
        self.last_signature = None
        self.retry_at = 0.0
        self.failures = 0

    def isOpened(self):
        return not self.closed

    def set(self, property_id, value):
        return False

    def _json(self, path, payload=None):
        headers = {}
        data = None
        if payload is not None:
            config = self._json("/api/config")
            headers = {"Content-Type": "application/json", "X-Workspace-Token": config["token"]}
            data = json.dumps(payload).encode()
        request = Request(self.base + path, data=data, headers=headers)
        with self.opener.open(request, timeout=15 if payload is not None else 4) as response:
            raw = response.read(65537)
            if len(raw) > 65536:
                raise ValueError("本机视频服务响应超过限制")
            return json.loads(raw)

    def _connect(self):
        try:
            self.response = self.opener.open(self.base + "/api/video.mjpg?serial=" + self.serial, timeout=4)
        except HTTPError as error:
            code = error.code
            error.close()
            if code != 503:
                raise
            result = self._json("/api/resume", {"serial": self.serial, "signaling_mode": "cloud", "allow_saved_login": True})
            if result.get("login_required"):
                raise ValueError("请先在 WoSee 管理页面登录并保存账号")
            if result.get("stopped_by_user"):
                raise ValueError("摄像头已在管理台手动停止，请先恢复连接")
            self.response = self.opener.open(self.base + "/api/video.mjpg?serial=" + self.serial, timeout=4)
        self.last_signature = None

    def _frame(self):
        for attempt in range(8):
            boundary = self.response.readline(256)
            if not boundary:
                raise EOFError("WoSee 视频流已断开")
            if boundary.strip() == b"--wosee":
                break
        else:
            raise ValueError("WoSee 视频帧边界无效")
        headers = {}
        total = 0
        while True:
            line = self.response.readline(2049)
            total += len(line)
            if not line or total > 8192 or len(line) > 2048:
                raise ValueError("WoSee 视频帧头无效")
            if line in (b"\r\n", b"\n"):
                break
            name, value = line.decode("ascii").split(":", 1)
            headers[name.lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        if headers.get("x-camera-serial") != self.serial:
            raise ValueError("视频所属设备与配置不一致")
        if not 0 < length <= 4 * 1024 * 1024 or headers.get("content-type") != "image/jpeg":
            raise ValueError("WoSee 视频帧格式或长度无效")
        encoded = self.response.read(length)
        if len(encoded) != length:
            raise EOFError("WoSee 视频帧尚未完整接收")
        stamp = float(headers["x-frame-timestamp"])
        sequence = int(headers["x-frame-sequence"])
        signature = (sequence, stamp)
        if not self.clock() - 3 <= stamp <= self.clock() + 2 or signature == self.last_signature:
            return False, None
        frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("WoSee 视频 JPEG 无法解码")
        self.timestamp, self.sequence, self.last_signature = stamp, sequence, signature
        self.failures = 0
        return True, frame

    def read(self):
        if self.closed:
            return False, None
        delay = self.retry_at - self.clock()
        if delay > 0:
            self.sleep(min(delay, 0.25))
            return False, None
        try:
            if self.response is None:
                self._connect()
            return self._frame()
        except (OSError, URLError, EOFError, HTTPException, ValueError, KeyError, cv2.error):
            if self.response is not None:
                self.response.close()
            self.response = None
            self.failures += 1
            self.retry_at = self.clock() + min(5.0, 0.5 * 2 ** min(self.failures, 4))
            return False, None

    def release(self):
        self.closed = True
        if self.response is not None:
            self.response.close()
            self.response = None


def open_capture(config):
    source = config.video_source if config.video_source else config.camera_index
    if isinstance(source, str) and source.startswith("wosee:"):
        return WoSeeCapture(source)
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
    return capture
