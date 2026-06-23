import numpy as np

from catcam.nightvision import enhance_lowlight, is_dark, mean_brightness
from catcam.simple import MotionGrayDetector


def test_is_dark_distinguishes_day_night():
    dark = np.full((40, 40, 3), 8, dtype=np.uint8)
    bright = np.full((40, 40, 3), 200, dtype=np.uint8)
    assert is_dark(dark) is True
    assert is_dark(bright) is False


def test_enhance_lifts_dark_frame():
    # 一张几乎全黑、但藏着一点点对比度的帧，增强后整体应更亮、可见
    frame = np.full((40, 40, 3), 6, dtype=np.uint8)
    frame[10:30, 10:30] = 14
    out = enhance_lowlight(frame)
    assert out.shape == frame.shape
    assert mean_brightness(out) > mean_brightness(frame)


def test_night_detector_fires_on_motion_without_color():
    # 夜间：近黑帧里有运动但没有「灰蓝」颜色，也应触发
    d = MotionGrayDetector()
    bowl = (0, 0, 40, 40)
    dark = np.full((40, 40, 3), 5, dtype=np.uint8)
    assert d.present(dark, bowl, night=True) is False   # 基线
    moved = np.full((40, 40, 3), 5, dtype=np.uint8)
    moved[5:35, 5:35] = 60                               # 一片东西动进来
    assert d.present(moved, bowl, night=True) is True
