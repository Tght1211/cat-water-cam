"""本机演示/试标注入口（无需摄像头）。

塞入几段示例视频到 data/clips，然后只启动本地网页，让你立刻打开
http://127.0.0.1:8000 试用「标注平台」（每段视频的 👍/👎 按钮 + 已标/未标状态）。
真实采集请用 `python -m catcam`（需摄像头），或在 config.json 的 video_source
里填一段录好的视频文件路径。

用法： python -m catcam.demo
"""
from __future__ import annotations

import cv2
import numpy as np
import uvicorn

from catcam.config import load_config
from catcam.feedback import FeedbackStore
from catcam.recorder import ClipRecorder
from catcam.stats import StatsStore
from catcam.web import create_app


def _sample_frame(width: int, height: int, label: str):
    frame = np.full((height, width, 3), 60, dtype=np.uint8)
    cv2.putText(
        frame, label, (16, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2, cv2.LINE_AA,
    )
    return frame


def seed_sample_clips(
    recorder: ClipRecorder, n: int, frames_per_clip: int = 6,
    width: int = 320, height: int = 240,
) -> list:
    """造 n 段示例 mp4（递增时间戳，命名/排序与真实一致），返回写出的路径。"""
    paths = []
    for i in range(n):
        frames = [
            _sample_frame(width, height, f"DEMO {i + 1}")
            for _ in range(frames_per_clip)
        ]
        paths.append(recorder.save_clip(frames, timestamp=float(i + 1)))
    return paths


def main(config_path: str = "config.json") -> None:
    cfg = load_config(config_path)
    stats = StatsStore(cfg.db_path)
    recorder = ClipRecorder(cfg.clips_dir, cfg.max_clips, cfg.fps)
    feedback = FeedbackStore(cfg.db_path, cfg.training_dir)

    if not recorder.list_clips():
        seed_sample_clips(recorder, n=3)
        print("已塞入 3 段示例视频，可在网页上试标注。")

    placeholder = _sample_frame(cfg.frame_width, cfg.frame_height, "DEMO MODE - no camera")
    app = create_app(stats, recorder, feedback, lambda: placeholder, cfg.clips_dir)
    print(
        f"演示网页已启动（绑定 {cfg.web_host}:{cfg.web_port}）；"
        f"本机访问 http://127.0.0.1:{cfg.web_port}"
    )
    uvicorn.run(app, host=cfg.web_host, port=cfg.web_port, log_level="warning")


if __name__ == "__main__":
    main()
