import numpy as np
import cv2
from catcam.video_trainer import feature_cache_path, gather_dataset, train_video_head
from catcam.videojudge import DrinkingHead
from catcam.feedback import FeedbackStore
from catcam.models import ModelRegistry


class _FakeExtractor:
    def __init__(self, dim=8): self.dim = dim
    def extract(self, frames):
        v = np.zeros(self.dim, np.float32); v[0] = float(frames[0][0, 0, 0]); return v


def _clip(path, val):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    for _ in range(16):
        w.write(np.full((24, 32, 3), val, np.uint8))
    w.release()


def test_feature_cache_path(tmp_path):
    p = feature_cache_path(tmp_path, "clip_123.mp4")
    assert p == tmp_path / "features" / "clip_123.npy"


def test_gather_dataset_joins_features_and_labels(tmp_path):
    clips = tmp_path / "clips"; clips.mkdir()
    training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    # 两段喝水(亮)、两段没喝(暗)
    for name, val, drink in [("a.mp4", 200, True), ("b.mp4", 210, True),
                             ("c.mp4", 10, False), ("d.mp4", 20, False)]:
        _clip(clips / name, val)
        store.label_clip(clips / name, drink)
    X, y, names = gather_dataset(clips, training, store, _FakeExtractor(8), dim=8)
    assert X.shape == (4, 8) and set(y) == {0, 1} and len(names) == 4
    # 第二次调用应命中缓存（不再 extract）：特征文件已存在
    assert (training / "features" / "a.npy").exists()


def test_train_video_head_registers_version(tmp_path):
    clips = tmp_path / "clips"; clips.mkdir()
    training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    for i in range(6):
        _clip(clips / f"p{i}.mp4", 200); store.label_clip(clips / f"p{i}.mp4", True)
    for i in range(6):
        _clip(clips / f"n{i}.mp4", 10); store.label_clip(clips / f"n{i}.mp4", False)
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    res = train_video_head(clips, training, store, registry,
                           models_dir=tmp_path / "models", extractor=_FakeExtractor(8),
                           dim=8, epochs=200, created_ts=111.0)
    assert res["version"] == "v1"
    assert res["top1"] >= 0.8
    entry = registry.get("v1")
    assert entry["base"] == "s3d+head"
    head = DrinkingHead.load(entry["path"])
    assert head.predict(np.array([2.0] + [0.0] * 7, np.float32))[0] in (True, False)
