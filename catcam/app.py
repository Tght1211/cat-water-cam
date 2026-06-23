from __future__ import annotations

import threading
import time
from datetime import datetime

import cv2
import uvicorn

from catcam.config import load_config
from catcam.detector import DrinkingDetector
from catcam.feedback import FeedbackStore
from catcam.framebuffer import FrameBuffer
from catcam.mailer import Emailer
from catcam.netutil import lan_ip
from catcam import nightvision
from catcam.pipeline import Pipeline
from catcam.recorder import ClipRecorder
from catcam.simple import MotionGrayDetector
from catcam.stats import StatsStore
from catcam.trainer import TrainingManager
from catcam.vision import CatDetector
from catcam.web import create_app


class LatestFrame:
    """采集线程写、网页/检测线程读的最新帧（含时间戳与是否夜间）。

    预览（MJPEG/快照）只要帧；检测线程还要 now/night。加锁、返回拷贝避免并发改。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._now = None
        self._frame = None
        self._night = False

    def set(self, now: float, frame, night: bool) -> None:
        with self._lock:
            self._now = now
            self._frame = frame
            self._night = night

    def get(self):
        """供网页预览：只返回帧拷贝。"""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def get_state(self):
        """供检测线程：返回 (now, frame拷贝, night)。"""
        with self._lock:
            if self._frame is None:
                return None
            return self._now, self._frame.copy(), self._night


def _serve_web(app, host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main(config_path: str = "config.json") -> None:
    cfg = load_config(config_path)

    cat_detector = CatDetector.from_path(cfg.yolo_model, cfg.cat_confidence)
    stats = StatsStore(cfg.db_path)
    recorder = ClipRecorder(cfg.clips_dir, cfg.max_clips, cfg.fps)
    feedback = FeedbackStore(cfg.db_path, cfg.training_dir)
    emailer = Emailer(cfg)
    trainer = TrainingManager(
        cfg.training_dir, cfg.models_dir, cfg.cls_base_model, cfg.train_epochs, cfg.train_imgsz
    )
    pipeline = Pipeline(
        cat_detector=cat_detector,
        drinking_detector=DrinkingDetector(cfg.dwell_seconds, cfg.cooldown_seconds),
        frame_buffer=FrameBuffer(cfg.clip_seconds, cfg.fps),
        recorder=recorder,
        stats=stats,
        bowl_roi_ratio=cfg.bowl_roi,
        min_overlap_ratio=cfg.min_overlap_ratio,
        presence_detector=MotionGrayDetector(),
    )

    latest = LatestFrame()
    app = create_app(stats, recorder, feedback, latest.get, cfg.clips_dir, trainer)
    threading.Thread(
        target=_serve_web, args=(app, cfg.web_host, cfg.web_port), daemon=True
    ).start()
    if cfg.web_host == "0.0.0.0":
        print(
            f"网页已启动（局域网）；本机 http://127.0.0.1:{cfg.web_port}"
            f"，同局域网设备 http://{lan_ip()}:{cfg.web_port}"
        )
    else:
        print(
            f"网页已启动（绑定 {cfg.web_host}:{cfg.web_port}）；"
            f"本机访问 http://127.0.0.1:{cfg.web_port}"
        )
    if emailer.enabled:
        print(f"邮件提醒已开启 → {emailer.to}（最小间隔 {int(emailer.min_interval)}s）")

    source = cfg.video_source if cfg.video_source else cfg.camera_index
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"打不开视频源： {source!r}")

    buf_interval = 1.0 / max(1, cfg.fps)  # 回放缓冲按 fps 节奏喂，保证 clip 时长/速度正确

    # 采集线程：全速读相机 → 更新预览（让网页流畅），按 fps 节奏喂回放缓冲。
    # 检测（YOLO，慢）不放这里，避免拖慢出图。
    def _capture() -> None:
        last_buf = 0.0
        while True:
            ok, raw = cap.read()
            if not ok:
                time.sleep(buf_interval)
                continue
            now = time.time()
            night = nightvision.is_dark(raw, cfg.night_brightness_threshold)
            frame = nightvision.enhance_lowlight(raw) if night else raw
            latest.set(now, frame, night)
            if now - last_buf >= buf_interval:
                pipeline.observe(now, frame)
                last_buf = now

    threading.Thread(target=_capture, daemon=True).start()

    # 检测循环：按自己的节奏取最新帧跑识别；预览不受其拖累。
    try:
        while True:
            state = latest.get_state()
            if state is None:
                time.sleep(cfg.detect_interval_seconds)
                continue
            now, frame, night = state
            if night and not cfg.record_at_night:
                time.sleep(cfg.detect_interval_seconds)
                continue
            clip = pipeline.detect(now, frame, night=night)
            if clip:
                print(f"记录一次喝水： {clip}")
                # 邮件发送（含 SMTP 往返）放后台线程，绝不阻塞检测；限流在 emailer 内部判定。
                snapshot = frame.copy()
                threading.Thread(
                    target=emailer.notify_drinking,
                    args=(stats, snapshot, datetime.now()),
                    daemon=True,
                ).start()
            time.sleep(cfg.detect_interval_seconds)
    finally:
        cap.release()
