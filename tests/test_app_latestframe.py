import numpy as np
from catcam.app import LatestFrame, VideoJudgeRuntime
from catcam.videojudge import DrinkingHead


def test_latest_frame_roundtrip():
    lf = LatestFrame()
    assert lf.get() is None
    assert lf.get_state() is None
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    lf.set(123.0, frame, True)
    got = lf.get()
    assert got is not None
    # get() 返回拷贝，改动原帧不影响已取出的
    frame[0, 0, 0] = 99
    assert got[0, 0, 0] == 0
    # get_state 带上时间戳与夜间标志，返回的帧是拷贝（改它不影响下次取出）
    now, sframe, night = lf.get_state()
    assert now == 123.0 and night is True
    sframe[1, 1, 1] = 77
    assert lf.get_state()[1][1, 1, 1] != 77


def test_video_judge_runtime_hot_switches_atomically(tmp_path):
    path = tmp_path / "head.npz"
    DrinkingHead(
        weight=np.ones(4), bias=0, mean=np.zeros(4), std=np.ones(4)
    ).save(path)
    runtime = VideoJudgeRuntime()
    assert runtime.snapshot() == (None, "shadow")
    runtime.activate({"id": "v2", "path": str(path)}, "gate")
    judge, mode = runtime.snapshot()
    assert judge.version == "v2" and mode == "gate"
    runtime.clear()
    assert runtime.snapshot() == (None, "shadow")
