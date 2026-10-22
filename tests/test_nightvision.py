import numpy as np

from catcam.nightvision import enhance_lowlight, enhance_preview, is_dark, mean_brightness
from catcam.simple import MotionGrayDetector
from catcam.nightvision import LowLightPresenceDetector,PreviewEnhancer
from catcam.nightvision import assess_quality


def test_lit_drinking_area_is_usable_with_dark_background():
    frame = np.full((100, 100, 3), 5, dtype=np.uint8)
    frame[30:70, 30:50] = 40
    frame[30:70, 50:70] = 90
    assert is_dark(frame)
    quality = assess_quality(frame, (.3, .3, .7, .7), 50)
    assert quality['usable']
    assert quality['status'] == 'normal'


def test_dark_drinking_area_stays_blocked_with_bright_background():
    frame = np.full((100, 100, 3), 180, dtype=np.uint8)
    frame[30:70, 30:50] = 5
    frame[30:70, 50:70] = 15
    assert not is_dark(frame)
    assert not assess_quality(frame, (.3, .3, .7, .7), 50)['usable']


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


def test_preview_reveals_existing_shadow_without_modifying_input():
    frame=np.full((80,80,3),6,dtype=np.uint8)
    frame[20:60,20:60]=18
    original=frame.copy();out=enhance_preview(frame)
    assert np.array_equal(frame,original)
    assert mean_brightness(out)>mean_brightness(frame)*2
    assert out[35:45,35:45].mean()>out[:10,:10].mean()


def test_preview_keeps_black_frame_black():
    frame=np.zeros((80,80,3),dtype=np.uint8)
    assert np.array_equal(enhance_preview(frame),frame)


def test_dark_face_is_detected_and_remains_until_it_leaves():
    d=LowLightPresenceDetector();roi=(.3,.3,.7,.7)
    background=np.full((240,320,3),16,dtype=np.uint8)
    for i in range(25):assert not d.observe(background,roi,i*.1)['present']
    face=background.copy();face[35:120,105:210]=5
    for i in range(25,80):assert d.observe(face,roi,i*.1)['present']
    assert not d.observe(background,roi,8)['present']


def test_lit_closeup_head_remains_present_without_motion_for_two_minutes():
    d=LowLightPresenceDetector();roi=(.3,.3,.7,.7)
    background=np.full((240,320,3),180,dtype=np.uint8)
    for i in range(25):assert not d.observe(background,roi,i*.1)['present']
    face=background.copy();face[35:120,105:210]=80
    for i in range(25,1225):assert d.observe(face,roi,i*.1)['present']
    assert not d.observe(background,roi,122.5)['present']


def test_lowlight_noise_does_not_trigger_candidate():
    d=LowLightPresenceDetector();rng=np.random.default_rng(42)
    for i in range(50):
        frame=rng.integers(11,17,(240,320,3),dtype=np.uint8)
        assert not d.observe(frame,(.3,.3,.7,.7),i*.1)['present']


def test_uniform_exposure_change_is_not_a_face():
    d=LowLightPresenceDetector();roi=(.3,.3,.7,.7)
    for i in range(25):d.observe(np.full((240,320,3),20,dtype=np.uint8),roi,i*.1)
    assert not d.observe(np.full((240,320,3),10,dtype=np.uint8),roi,2.5)['present']


def test_full_darkness_cannot_enable_contour_capture():
    d=LowLightPresenceDetector()
    for i in range(40):
        result=d.observe(np.zeros((240,320,3),dtype=np.uint8),(.3,.3,.7,.7),i*.1)
        assert not result['ready'] and not result['present']


def test_preview_temporal_filter_keeps_changed_pixels_current():
    p=PreviewEnhancer();bg=np.full((80,80,3),20,dtype=np.uint8)
    p.enhance(bg);face=bg.copy();face[20:60,20:60]=5;p.enhance(face)
    assert p.previous[35:45,35:45].mean()==5


def test_night_detector_fires_on_motion_without_color():
    # 夜间：近黑帧里有运动但没有「灰蓝」颜色，也应触发
    d = MotionGrayDetector()
    bowl = (0, 0, 40, 40)
    dark = np.full((40, 40, 3), 5, dtype=np.uint8)
    assert d.present(dark, bowl, night=True) is False   # 基线
    moved = np.full((40, 40, 3), 5, dtype=np.uint8)
    moved[5:35, 5:35] = 60                               # 一片东西动进来
    assert d.present(moved, bowl, night=True) is True
