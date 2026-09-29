"""夜间/弱光处理：这个摄像头没有红外，晚上画面几乎全黑。

先检查原始水碗区域是否有足够光照；不足时应返回未知，不能把传感器噪声当作动作。
弱光增强只能帮助已有细节显现，无法恢复黑暗中没有采集到的信息。
录制/训练保留原始帧；仅在光照检查通过后对检测帧做增强。
"""
from __future__ import annotations

import cv2
import numpy as np

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


def visibility_status(frame, roi=(0., 0., 1., 1.), minimum=20.0, low_light=50.0):
    """Assess the raw bowl region. Brightening a black image cannot recover evidence."""
    if frame is None:
        return {"status": "unknown", "can_judge": False, "reason": "尚未收到画面"}
    from catcam.geometry import ratio_rect_to_pixels
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = ratio_rect_to_pixels(roi, w, h)
    region = frame[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]
    if not region.size:
        return {"status": "unknown", "can_judge": False, "reason": "饮水区域为空，请调整位置"}
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    dark_fraction = float(np.mean(gray < minimum))
    insufficient = brightness < minimum or dark_fraction >= .85
    status = "insufficient_light" if insufficient else "low_light" if brightness < low_light else "ok"
    return {"status": status, "can_judge": not insufficient,
            "brightness": round(brightness, 1), "dark_fraction": round(dark_fraction, 3),
            "reason": "光线不足，无法判断是否喝水；请给水碗补光" if insufficient else
                      "画面较暗，请复核判断或适当补光" if status == "low_light" else "光照可用"}


def clip_is_visible(path, roi=(0., 0., 1., 1.), minimum=20.0):
    from catcam.videojudge import read_clip_frames
    frames = read_clip_frames(path, n=4)
    if not frames:
        return False
    usable = sum(visibility_status(cv2.cvtColor(f, cv2.COLOR_RGB2BGR), roi, minimum)["can_judge"] for f in frames)
    return usable >= 3
