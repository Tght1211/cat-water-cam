"""本地视频裁判：冻结 s3d 主干提 1024 维特征 + 一层 torch logistic 头判「真喝水/没喝」。

第二阶段（离线核心）：提取器/头/判定都做成可注入、可单测；真实 s3d 主干懒加载（torchvision 已装）。
重依赖（torch/torchvision）只在真正用主干时才 import，单测走假提取器，不碰真权重/网络。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FEATURE_DIM = 1024   # s3d avgpool 后的特征维度（实测）
CLIP_FRAMES = 16     # 每段抽多少帧喂主干


def read_clip_frames(clip_path, n: int = CLIP_FRAMES) -> list[np.ndarray]:
    """从 clip 均匀抽 n 帧，返回 RGB(HWC uint8) 列表；帧不足则重复补齐到 n。"""
    cap = cv2.VideoCapture(str(clip_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        return []
    step = max(1, len(frames) // n)
    chosen = frames[::step][:n]
    while len(chosen) < n:               # 不足 n：重复最后一帧补齐
        chosen.append(chosen[-1])
    return chosen
