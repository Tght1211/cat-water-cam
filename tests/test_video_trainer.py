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


def test_gather_excludes_source_local(tmp_path):
    clips = tmp_path / "clips"; clips.mkdir()
    training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    _clip(clips / "ai.mp4", 200); store.label_clip(clips / "ai.mp4", True)         # source=human
    _clip(clips / "loc.mp4", 200); store.record_machine_label("loc.mp4", True, source="local")
    X, y, names = gather_dataset(clips, training, store, _FakeExtractor(8), dim=8)
    assert "loc.mp4" not in names and "ai.mp4" in names     # 本地判定不进训练集


def test_video_training_manager_runs_and_reports(tmp_path):
    from catcam.video_trainer import VideoTrainingManager
    from catcam.models import ModelRegistry
    clips = tmp_path / "clips"; clips.mkdir()
    training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    for i in range(6):
        _clip(clips / f"p{i}.mp4", 200); store.label_clip(clips / f"p{i}.mp4", True)
    for i in range(6):
        _clip(clips / f"n{i}.mp4", 10); store.label_clip(clips / f"n{i}.mp4", False)
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    mgr = VideoTrainingManager(clips, training, store, registry, tmp_path / "models",
                               extractor=_FakeExtractor(8), dim=8, epochs=200)
    mgr._run()                                   # 同步跑一次（避开线程时序）
    s = mgr.status()
    assert s["state"] == "done"
    assert s["result"]["version"] == "v1"
    assert "召回" in s["detail"]
    assert s["models"][0]["base"] == "s3d+head"


def test_video_training_manager_error_on_too_few(tmp_path):
    from catcam.video_trainer import VideoTrainingManager
    from catcam.models import ModelRegistry
    clips = tmp_path / "clips"; clips.mkdir()
    store = FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    mgr = VideoTrainingManager(clips, tmp_path / "training", store, registry, tmp_path / "models",
                               extractor=_FakeExtractor(8), dim=8)
    mgr._run()
    s = mgr.status()
    assert s["state"] == "error" and "不够" in s["detail"]


def test_extract_and_cache_self_heals_corrupt(tmp_path):
    from catcam.video_trainer import extract_and_cache, feature_cache_path
    clips = tmp_path / "clips"; clips.mkdir()
    training = tmp_path / "training"
    _clip(clips / "a.mp4", 200)
    # 放一个损坏缓存（只有半截 npy 头、无数据），模拟上次 np.save 写一半崩了
    cache = feature_cache_path(training, "a.mp4"); cache.parent.mkdir(parents=True)
    cache.write_bytes(b"\x93NUMPY\x01\x00short")
    feat = extract_and_cache(clips / "a.mp4", training, _FakeExtractor(8), dim=8)
    assert feat is not None and feat.shape == (8,)        # 没崩，重抽成功
    # 覆盖成了合法缓存，再读一次正常
    feat2 = extract_and_cache(clips / "a.mp4", training, _FakeExtractor(8), dim=8)
    assert feat2.shape == (8,)
