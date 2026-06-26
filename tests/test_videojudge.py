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


from catcam.videojudge import DrinkingHead, FEATURE_DIM


def _synth(n=40, dim=8, seed=0):
    """造线性可分的两类特征：第 0 维正=喝水、负=没喝。"""
    rng = np.random.default_rng(seed)
    Xpos = rng.normal(0, 0.3, (n, dim)); Xpos[:, 0] += 2.0
    Xneg = rng.normal(0, 0.3, (n, dim)); Xneg[:, 0] -= 2.0
    X = np.vstack([Xpos, Xneg]).astype(np.float32)
    y = np.array([1] * n + [0] * n)
    return X, y


def test_head_learns_separable_features():
    X, y = _synth()
    head = DrinkingHead.fit(X, y, dim=8, epochs=300, seed=0)
    drink, conf = head.predict(X[0])      # 第 0 维 +2 → 喝水
    assert drink is True and conf > 0.8
    drink2, _ = head.predict(X[-1])       # 第 0 维 -2 → 没喝
    assert drink2 is False


def test_head_save_load_roundtrip(tmp_path):
    X, y = _synth()
    head = DrinkingHead.fit(X, y, dim=8, epochs=200, seed=1)
    p = tmp_path / "head.npz"; head.save(p)
    again = DrinkingHead.load(p)
    assert again.dim == 8
    # 同一输入，存取前后预测一致
    assert again.predict(X[0])[0] == head.predict(X[0])[0]
    assert abs(again.predict(X[0])[1] - head.predict(X[0])[1]) < 1e-5


def test_head_handles_imbalance():
    # 少数喝水(5) + 多数没喝(50)，靠 pos_weight 不至于全判没喝
    Xp, _ = _synth(n=5, dim=8); Xn, _ = _synth(n=50, dim=8, seed=9)
    X = np.vstack([Xp[:5], Xn[50:]]); y = np.concatenate([[1] * 5, [0] * 50])
    head = DrinkingHead.fit(X, y, dim=8, epochs=400, seed=0)
    pos = np.zeros(8, np.float32); pos[0] = 2.0
    assert head.predict(pos)[0] is True


from catcam.videojudge import LocalVideoClipJudge, S3DFeatureExtractor
from catcam.judge import Verdict


class _FakeExtractor:
    """假提取器：不碰真模型，按第 0 帧像素给个确定特征。"""
    def __init__(self, dim=8): self.dim = dim
    def extract(self, frames):
        v = np.zeros(self.dim, np.float32)
        v[0] = 2.0 if frames[0][0, 0, 0] > 128 else -2.0
        return v


def test_local_judge_returns_verdict(tmp_path):
    clip = tmp_path / "a.mp4"; _make_clip(clip, n=16, color=(0, 0, 255))  # BGR→R 大→喝水
    X, y = _synth(dim=8)
    head = DrinkingHead.fit(X, y, dim=8, epochs=300, seed=0)
    judge = LocalVideoClipJudge(_FakeExtractor(8), head, version="v3")
    v = judge.judge(clip)
    assert isinstance(v, Verdict) and v.by == "v3"
    assert v.drinking is True and 0.0 <= v.confidence <= 1.0


def test_local_judge_fail_open_on_empty(tmp_path):
    # 抽帧为空（坏 clip）→ 返回 None，不崩
    bad = tmp_path / "empty.mp4"; bad.write_bytes(b"not a video")
    head = DrinkingHead.fit(*_synth(dim=8), dim=8, epochs=50)
    judge = LocalVideoClipJudge(_FakeExtractor(8), head, version="v3")
    assert judge.judge(bad) is None


def test_s3d_extractor_real_smoke():
    # 真 s3d：权重已缓存；一段 16 帧假画面 → 1024 维特征。验证真实主干路径可用。
    ext = S3DFeatureExtractor()
    frames = [np.full((24, 32, 3), 100 + i, np.uint8) for i in range(16)]
    feat = ext.extract(frames)
    assert feat.shape == (FEATURE_DIM,) and feat.dtype == np.float32
