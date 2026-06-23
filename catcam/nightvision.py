"""夜间/弱光处理：这个摄像头没有红外，晚上画面几乎全黑。

策略（只为识别「行为」，不追求画质）：
1. 判断当前帧是否「暗」（整体亮度低于阈值）。
2. 暗则做弱光增强——LAB 的 L 通道做 CLAHE，把藏在阴影里的那点对比度拉开，
   猫的轮廓就能浮出来（比单纯「反色」更有用：反色不增加信息，CLAHE 才真正
   把暗部细节拉成可见范围）。增强后的帧同时用于网页/录制/识别。
3. 识别上，夜间靠「画面变化」（运动）而非颜色——见 simple.py 的 night 分支。
"""
from __future__ import annotations

import cv2

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def mean_brightness(frame) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def is_dark(frame, threshold: float = 50.0) -> bool:
    return mean_brightness(frame) < threshold


def enhance_lowlight(frame):
    """弱光增强：LAB 的 L 通道 CLAHE，拉出暗部轮廓而不整体过曝。返回 BGR。"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    lab2 = cv2.merge((_clahe.apply(l), a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
