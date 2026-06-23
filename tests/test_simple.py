import numpy as np

from catcam.simple import MotionGrayDetector, gray_ratio, motion_ratio


def _gray_block(h, w):
    # 低饱和、中等明度 → 算「灰蓝猫色」
    img = np.full((h, w, 3), 120, dtype=np.uint8)
    return img


def test_gray_ratio_high_for_gray_low_for_white():
    gray = _gray_block(40, 40)
    white = np.full((40, 40, 3), 245, dtype=np.uint8)  # 白地板：明度过高，应排除
    assert gray_ratio(gray) > 0.9
    assert gray_ratio(white) < 0.1


def test_motion_ratio_zero_without_change():
    a = np.zeros((30, 30), dtype=np.uint8)
    assert motion_ratio(None, a) == 0.0
    assert motion_ratio(a, a) == 0.0


def test_detector_fires_on_gray_motion_in_roi():
    d = MotionGrayDetector()
    bowl = (0, 0, 40, 40)
    bg = np.full((40, 40, 3), 245, dtype=np.uint8)  # 白底，静止
    assert d.present(bg, bowl) is False              # 建立基线
    cat = _gray_block(40, 40)                         # 灰猫移入 → 有变化 + 灰色
    assert d.present(cat, bowl) is True


def test_detector_quiet_when_static_gray():
    d = MotionGrayDetector()
    bowl = (0, 0, 40, 40)
    cat = _gray_block(40, 40)
    d.present(cat, bowl)                              # 基线
    assert d.present(cat, bowl) is False              # 没动 → 不触发
