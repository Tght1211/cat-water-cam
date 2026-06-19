import numpy as np
from catcam.app import LatestFrame


def test_latest_frame_roundtrip():
    lf = LatestFrame()
    assert lf.get() is None
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    lf.set(frame)
    got = lf.get()
    assert got is not None
    # get() 返回拷贝，改动原帧不影响已取出的
    frame[0, 0, 0] = 99
    assert got[0, 0, 0] == 0
