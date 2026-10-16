import cv2
import numpy as np

from catcam.framebuffer import FrameBuffer
from catcam.motion import MotionDetector
from catcam.recorder import ClipRecorder
from catcam.session import DrinkSession


def test_motion_anywhere_including_dark_background_and_non_cat_colors():
    detector = MotionDetector()
    blank = np.zeros((240, 320, 3), np.uint8)
    assert not detector.observe(0, blank)
    assert not detector.observe(0.1, blank)
    changed = blank.copy()
    changed[5:25, 5:25] = (0, 0, 255)
    assert detector.observe(0.2, changed)


def test_static_sensor_noise_does_not_trigger():
    rng = np.random.default_rng(1)
    detector = MotionDetector()
    for i in range(60):
        frame = rng.integers(98, 103, size=(240, 320, 3), dtype=np.uint8)
        assert not detector.observe(i / 30, frame)


def test_motion_records_at_30fps_without_dwell_or_cooldown(tmp_path):
    recorder = ClipRecorder(tmp_path, 1000, 30)
    session = DrinkSession(recorder, 0, 0.2, 1, 0)
    buffer = FrameBuffer(0.1, 30)
    detector = MotionDetector()
    completed = []
    for i in range(120):
        now = i / 30
        frame = np.zeros((48, 64, 3), np.uint8)
        if 10 <= i < 100:
            frame[10:30, (i % 30):(i % 30) + 15] = 255
        moving = detector.observe(now, frame)
        buffer.add(now, frame)
        result = session.update(now, frame, moving, now if moving else None, buffer)
        if result:
            completed.append(result)
    final = session.close()
    if final:
        completed.append(final)
    assert len(completed) >= 3  # 持续变化跨分段，不受旧的 60 秒冷却限制。
    for result in completed:
        cap = cv2.VideoCapture(str(tmp_path / result.clip_name))
        try:
            assert cap.get(cv2.CAP_PROP_FPS) == 30
            assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 3
        finally:
            cap.release()


def test_retention_is_three_days_even_for_protected_and_archived_video(tmp_path):
    clips = tmp_path / 'clips'
    archive = tmp_path / 'training' / 'videos'
    archive.mkdir(parents=True)
    recorder = ClipRecorder(clips, 1, 30, is_deletable=lambda _: False,
                            retention_days=3, archive_dirs=(archive,))
    now = 1_800_000_000
    timestamps = [now - 3*86400 - 1, now - 3*86400, now - 1]
    for directory in (clips, archive):
        for ts in timestamps:
            (directory / f'clip_{ts*1000}.mp4').write_bytes(b'video')
    recorder.prune(now)
    for directory in (clips, archive):
        assert sorted(p.name for p in directory.glob('*.mp4')) == [
            f'clip_{ts*1000}.mp4' for ts in timestamps[1:]
        ]
