import cv2
import numpy as np

from catcam.framebuffer import FrameBuffer
from catcam.recorder import ClipRecorder
from catcam.session import DrinkSession


def _frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def _make(tmp_path, dwell=2.0, grace=2.0, maxlen=60.0, cooldown=10.0):
    rec = ClipRecorder(tmp_path / "clips", max_clips=10, fps=10)
    return DrinkSession(rec, dwell, grace, maxlen, cooldown), rec


def test_records_from_dwell_until_leave(tmp_path):
    s, rec = _make(tmp_path)
    fb = FrameBuffer(5.0, 10)
    f = _frame()
    t = 0.0
    # 猫刚出现：未到 dwell，不开录
    assert s.update(t, f, in_roi=True, in_roi_since=0.0, frame_buffer=fb) is None
    assert not s.recording
    # 在场满 dwell(2s) → 开录
    s.update(2.1, f, True, 0.0, fb)
    assert s.recording
    # 持续在场一段时间
    for t in [2.5, 3.0, 4.0, 5.0]:
        assert s.update(t, f, True, 0.0, fb) is None
    # 猫离开：还没到 grace，不收尾
    assert s.update(6.0, f, False, None, fb) is None
    # 离开持续超过 grace(2s) → 收尾存盘
    res = s.update(8.5, f, False, None, fb)
    assert res is not None
    assert not s.recording
    assert res.clip_name.endswith(".mp4")
    assert (rec.clips_dir / res.clip_name).exists()
    # 会话时间戳是开始时刻，不是结束
    assert abs(res.timestamp - 2.1) < 1e-6


def test_caps_at_max_session(tmp_path):
    s, _ = _make(tmp_path, dwell=1.0, grace=5.0, maxlen=4.0)
    fb = FrameBuffer(5.0, 10)
    f = _frame()
    s.update(1.1, f, True, 0.0, fb)        # 开录
    assert s.recording
    assert s.update(3.0, f, True, 0.0, fb) is None
    # 即便猫一直在，超过 max_session(4s) 也强制收尾
    res = s.update(5.2, f, True, 0.0, fb)
    assert res is not None and not s.recording


def test_preroll_frames_written(tmp_path):
    s, rec = _make(tmp_path, dwell=1.0, grace=1.0)
    fb = FrameBuffer(5.0, 10)
    for i in range(20):           # 缓冲里塞 20 帧「凑近过程」
        fb.add(float(i), _frame())
    s.update(1.1, _frame(), True, 0.0, fb)   # 开录会把缓冲全写进去
    res = s.update(3.0, _frame(), False, None, fb)
    cap = cv2.VideoCapture(str(rec.clips_dir / res.clip_name))
    n = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    assert n >= 20               # 至少包含 pre-roll 的那些帧
