from __future__ import annotations

import threading
import time

import uvicorn

from catcam.app import LatestFrame
from catcam.dispenser import DispenserStore
from catcam.feedback import FeedbackStore
from catcam.nightvision import is_dark, assess_quality
from catcam.recorder import ClipRecorder
from catcam.source import open_capture
from catcam.stats import StatsStore
from catcam.web import create_app
from catcam.replay import ReplayArchive


def run_preview(config):
    latest = LatestFrame()
    capture = open_capture(config)
    stopped = threading.Event()

    def collect():
        while not stopped.is_set():
            available, frame = capture.read()
            if not available:
                latest.clear()
                stopped.wait(0.1)
                continue
            stamp = getattr(capture, "timestamp", None) or time.time()
            quality = assess_quality(frame, config.bowl_roi, max(50, config.minimum_monitor_brightness))
            latest.set(stamp, frame, is_dark(frame, config.night_brightness_threshold), quality)

    stats = StatsStore(config.db_path)
    recorder = ClipRecorder(config.clips_dir, config.max_clips, config.fps)
    feedback = FeedbackStore(config.db_path, config.training_dir)
    dispenser = DispenserStore(config.db_path, config.dispenser_default_ml_per_drink, config.filter_cycle_days)
    app = create_app(stats, recorder, feedback, latest.get, config.clips_dir,
                     dispenser=dispenser, dispenser_low_water_pct=config.dispenser_low_water_pct,
                     web_password_hash=config.web_password_hash, web_session_secret=config.web_session_secret,
                     live_status_provider=lambda: {**latest.status(), 'detection_enabled': False},
                     replay_archive=ReplayArchive(recorder, stats, feedback, config.video_source))
    threading.Thread(target=collect, daemon=True).start()
    print(f"真实视频预览已启动：http://127.0.0.1:{config.web_port}；不运行识别、录音、自动录制、外部 AI 或邮件提醒。")
    try:
        uvicorn.run(app, host="127.0.0.1", port=config.web_port, log_level="warning")
    finally:
        stopped.set()
        capture.release()
