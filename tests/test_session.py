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


def test_weak_frames_during_session_require_visual_review(tmp_path):
    s,rec=_make(tmp_path);fb=FrameBuffer(5,10);f=_frame()
    s.update(3,f,True,0,fb)
    s.update(4,f,True,0,fb,candidate_only=True)
    result=s.update(7,f,False,None,fb)
    assert result.candidate_only


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
    # 音频窗口：终点是收尾时刻；起点回溯 preroll（缓冲帧数/fps）
    assert res.audio_end == 8.5
    assert res.audio_start <= 2.1
    assert abs(res.audio_start - (2.1 - len(fb.all_frames()) / rec.fps)) < 1e-6


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


def test_unlimited_session_keeps_whole_visit_past_one_minute(tmp_path):
    s, rec = _make(tmp_path, dwell=2, grace=3, maxlen=0, cooldown=60)
    fb = FrameBuffer(5, 10)
    frame = _frame()
    for i in range(21):
        fb.add(i / 10, frame)
    s.update(2, frame, True, 0, fb)
    for i in range(21, 1201):
        assert s.update(i / 10, frame, True, 0, fb) is None
    for i in range(1201, 1230):
        assert s.update(i / 10, frame, False, None, fb) is None
    result = s.update(123, frame, False, None, fb)
    assert result.presence_start == 0
    assert result.presence_end == 120
    assert result.audio_end == 123
    assert result.end_reason == 'left'
    cap = cv2.VideoCapture(str(rec.clips_dir / result.clip_name))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 1230
    cap.release()
    # 已超过开始时的一分钟，离开后的下一次独立到访可正常录制。
    s.update(126, frame, True, 124, fb)
    assert s.recording
    s.close()


def test_short_visit_cannot_retrigger_within_a_minute(tmp_path):
    s, _ = _make(tmp_path, dwell=0, grace=2, maxlen=0, cooldown=60)
    fb, frame = FrameBuffer(5, 10), _frame()
    s.update(100, frame, True, 100, fb)
    assert s.update(103, frame, False, None, fb)
    s.update(159.9, frame, True, 150, fb)
    assert not s.recording
    s.update(160, frame, True, 150, fb)
    assert s.recording
    s.close()


def test_configured_cap_does_not_repeat_same_visit(tmp_path):
    s, _ = _make(tmp_path, dwell=0, grace=2, maxlen=5, cooldown=60)
    fb, frame = FrameBuffer(5, 10), _frame()
    s.update(100, frame, True, 100, fb)
    assert s.update(105, frame, True, 100, fb).end_reason == 'limit'
    s.update(180, frame, True, 100, fb)
    assert not s.recording
    s.update(181, frame, False, None, fb)
    s.update(182, frame, True, 182, fb)
    assert s.recording
    s.close()


def test_interrupted_end_uses_last_written_frame(tmp_path):
    s, _ = _make(tmp_path, dwell=0, grace=3, maxlen=0)
    fb, frame = FrameBuffer(5, 10), _frame()
    s.update(100, frame, True, 100, fb)
    s.update(101, frame, False, None, fb)
    result = s.close()
    assert result.audio_end == 101
    assert result.presence_end == 100
    assert result.end_reason == 'interrupted'
