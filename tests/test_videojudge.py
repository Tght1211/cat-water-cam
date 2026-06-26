import numpy as np
import cv2
from catcam.videojudge import read_clip_frames


def _make_clip(path, n=20, color=(120, 130, 140)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    for _ in range(n):
        w.write(np.full((24, 32, 3), color, np.uint8))  # BGR
    w.release()


def test_read_clip_frames_samples_n_and_is_rgb(tmp_path):
    clip = tmp_path / "a.mp4"; _make_clip(clip, n=20, color=(200, 100, 50))  # BGR
    frames = read_clip_frames(clip, n=16)
    assert len(frames) == 16
    f = frames[0]
    assert f.shape == (24, 32, 3) and f.dtype == np.uint8
    # 写入 BGR(200,100,50)；若做了 BGR→RGB，读回应是 R≈50 小、B≈200 大（mp4v 有损，给容差）。
    r, g, b = (int(x) for x in f[0, 0])
    assert r < 128 < b      # 没换通道的话会是 R=200、B=50，断言失败
    assert abs(r - 50) < 25 and abs(b - 200) < 25


def test_read_clip_frames_short_clip_pads(tmp_path):
    clip = tmp_path / "b.mp4"; _make_clip(clip, n=5)
    frames = read_clip_frames(clip, n=16)
    assert len(frames) == 16  # 帧不够时重复补齐到 n
