"""无需训练的「简单模型」：画面变化 + 灰蓝色英短猫，先把候选片段都录下来。

思路（用户要的快速起步）：模型还没训练好之前，不强求识别「真喝水」，而是
宁可多录——只要水碗区域里既有「明显画面变化（有东西动进来）」又有「成片的
灰/蓝灰色块（英短蓝猫的低饱和中等明度毛色）」，就当作一次候选喝水记录下来。
之后人在网页上点 👍/👎 标注，再用 trainer.py 训练真正的「真喝水/没喝」分类器。

它和 YOLO 认猫是「或」的关系（见 pipeline）：YOLO 认出猫、或这个启发式命中，
都算「猫在水碗」。这样召回更高，候选更多，标注数据攒得更快。
"""
from __future__ import annotations

import cv2
import numpy as np

from catcam.geometry import Rect


def gray_ratio(roi_bgr) -> float:
    """ROI 里「灰/蓝灰」像素占比：低饱和 + 中等明度（排除白地板、黑色饮水机机身）。"""
    if roi_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    mask = (s < 60) & (v > 45) & (v < 185)
    return float(mask.mean())


def motion_ratio(prev_gray, cur_gray, diff_thresh: int = 22) -> float:
    """两帧灰度图里「明显变化」的像素占比。"""
    if prev_gray is None or prev_gray.shape != cur_gray.shape:
        return 0.0
    diff = cv2.absdiff(prev_gray, cur_gray)
    return float((diff > diff_thresh).mean())


class MotionGrayDetector:
    def __init__(
        self,
        motion_thresh: float = 0.02,
        gray_thresh: float = 0.06,
        night_motion_thresh: float = 0.04,
    ):
        self.motion_thresh = motion_thresh
        self.gray_thresh = gray_thresh
        # 夜间靠运动单独判定，门槛抬高一点压住弱光下的传感器噪点抖动。
        self.night_motion_thresh = night_motion_thresh
        self._prev_gray = None  # 上一帧 ROI 的灰度图，用于算画面变化

    def present(self, frame, bowl_px: Rect, night: bool = False) -> bool:
        """水碗区域里是否「有（灰蓝）东西在动」——把它当作猫凑过来了。

        白天：要求「画面变化」+「灰蓝猫色块」。
        夜间（night=True）：颜色不可靠，只看「画面变化」是否够大。
        """
        x1, y1, x2, y2 = (int(round(v)) for v in bowl_px)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        roi = frame[y1:y2, x1:x2]
        cur_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        m = motion_ratio(self._prev_gray, cur_gray)
        self._prev_gray = cur_gray
        if night:
            return m >= self.night_motion_thresh
        return m >= self.motion_thresh and gray_ratio(roi) >= self.gray_thresh
