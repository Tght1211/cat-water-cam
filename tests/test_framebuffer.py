from catcam.framebuffer import FrameBuffer


def test_maxlen_from_seconds_and_fps():
    fb = FrameBuffer(seconds=4.0, fps=10)
    assert fb.maxlen == 40


def test_keeps_only_latest_maxlen_frames():
    fb = FrameBuffer(seconds=1.0, fps=3)   # maxlen = 3
    for i in range(5):
        fb.add(timestamp=float(i), frame=f"f{i}")
    assert fb.all_frames() == ["f2", "f3", "f4"]


def test_maxlen_never_zero():
    fb = FrameBuffer(seconds=0.0, fps=0)
    assert fb.maxlen == 1
