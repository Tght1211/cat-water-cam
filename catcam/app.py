from __future__ import annotations

import threading
import time

import cv2
import uvicorn

from catcam.config import load_config
from catcam.detector import DrinkingDetector
from catcam.feedback import FeedbackStore
from catcam.framebuffer import FrameBuffer
from catcam.pipeline import Pipeline
from catcam.recorder import ClipRecorder
from catcam.stats import StatsStore
from catcam.vision import CatDetector
from catcam.web import create_app


class LatestFrame:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None

    def set(self, frame) -> None:
        with self._lock:
            self._frame = frame

    def get(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


def _serve_web(app, host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main(config_path: str = "config.json") -> None:
    cfg = load_config(config_path)

    cat_detector = CatDetector.from_path(cfg.yolo_model, cfg.cat_confidence)
    stats = StatsStore(cfg.db_path)
    recorder = ClipRecorder(cfg.clips_dir, cfg.max_clips, cfg.fps)
    feedback = FeedbackStore(cfg.db_path, cfg.training_dir)
    pipeline = Pipeline(
        cat_detector=cat_detector,
        drinking_detector=DrinkingDetector(cfg.dwell_seconds, cfg.cooldown_seconds),
        frame_buffer=FrameBuffer(cfg.clip_seconds, cfg.fps),
        recorder=recorder,
        stats=stats,
        bowl_roi_ratio=cfg.bowl_roi,
        min_overlap_ratio=cfg.min_overlap_ratio,
    )

    latest = LatestFrame()
    app = create_app(stats, recorder, feedback, latest.get, cfg.clips_dir)
    threading.Thread(
        target=_serve_web, args=(app, cfg.web_host, cfg.web_port), daemon=True
    ).start()
    print(f"网页已启动： http://127.0.0.1:{cfg.web_port}")

    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"打不开摄像头 index={cfg.camera_index}")

    interval = 1.0 / max(1, cfg.fps)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(interval)
                continue
            now = time.time()
            latest.set(frame)
            clip = pipeline.process(now, frame)
            if clip:
                print(f"记录一次喝水： {clip}")
            time.sleep(interval)
    finally:
        cap.release()
