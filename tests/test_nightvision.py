import numpy as np

from catcam.nightvision import enhance_lowlight, is_dark, mean_brightness
from catcam.simple import MotionGrayDetector
from catcam.nightvision import visibility_status


def test_visibility_checks_bowl_not_bright_background():
    frame = np.full((100, 100, 3), 200, np.uint8)
    frame[30:70, 30:70] = 5
    state = visibility_status(frame, (.3, .3, .7, .7))
    assert state["status"] == "insufficient_light" and not state["can_judge"]
    assert visibility_status(frame)["can_judge"]


def test_visibility_unknown_dim_and_usable():
    assert not visibility_status(None)["can_judge"]
    assert visibility_status(np.full((40, 40, 3), 35, np.uint8))["status"] == "low_light"
    assert visibility_status(np.full((40, 40, 3), 120, np.uint8))["status"] == "ok"
    assert not visibility_status(np.ones((40, 40, 3), np.uint8), (0, 0, 0, 0))["can_judge"]


def test_clip_visibility_requires_most_sampled_frames_to_be_visible(tmp_path):
    import cv2
    from catcam.nightvision import clip_is_visible
    for name, values, expected in [("dark", [5]*4, False), ("usable", [120]*3+[5], True),
                                    ("mixed", [120]*2+[5]*2, False)]:
        path = tmp_path / f"{name}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (32, 24))
        for value in values:
            writer.write(np.full((24, 32, 3), value, np.uint8))
        writer.release()
        assert clip_is_visible(path) is expected


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
