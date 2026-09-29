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
    assert entry["evaluation"]["policy"] == "human-grouped-holdout-v1"
    assert not entry["evaluation"]["release"]["eligible"]
    head = DrinkingHead.load(entry["path"])
    assert head.predict(np.array([2.0] + [0.0] * 7, np.float32))[0] in (True, False)


def test_gather_excludes_source_local(tmp_path):
    clips = tmp_path / "clips"; clips.mkdir()
    training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    _clip(clips / "ai.mp4", 200); store.label_clip(clips / "ai.mp4", True)         # source=human
    _clip(clips / "loc.mp4", 200); store.record_machine_label("loc.mp4", True, source="local")
    _clip(clips / "external.mp4", 200); store.label_clip(clips / "external.mp4", True, source="ai")
    X, y, names = gather_dataset(clips, training, store, _FakeExtractor(8), dim=8)
    assert "loc.mp4" not in names and "ai.mp4" in names     # 本地判定不进训练集
    assert "external.mp4" not in names


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


def test_gather_dataset_reports_progress(tmp_path):
    clips = tmp_path / "clips"; clips.mkdir()
    training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    for name, drink in [("a.mp4", True), ("b.mp4", True), ("c.mp4", False)]:
        _clip(clips / name, 200 if drink else 10); store.label_clip(clips / name, drink)
    seen = []
    gather_dataset(clips, training, store, _FakeExtractor(8), dim=8,
                   progress_cb=lambda info: seen.append(info))
    extracting = [i for i in seen if i["phase"] == "extracting"]
    assert extracting, "应至少报一次 extracting"
    assert extracting[-1]["done"] == extracting[-1]["total"] == 3   # 三段处理完
    assert [i["done"] for i in extracting] == [1, 2, 3]             # 递增


def test_video_training_manager_exposes_progress_fields(tmp_path):
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
                               extractor=_FakeExtractor(8), dim=8, epochs=100)
    # 运行中（未跑前）状态不应崩；progress 字段在 running 时出现
    seen_phases = []
    orig = mgr._on_progress
    def _spy(info):
        seen_phases.append(info.get("phase")); orig(info)
    mgr._on_progress = _spy
    mgr._run()
    assert "preparing" in seen_phases and "extracting" in seen_phases and "training" in seen_phases
    s = mgr.status()
    assert s["state"] == "done"


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


def test_gather_uses_cache_after_clip_pruned(tmp_path):
    # 喝水正样本难攒的一个原因：clip 被 max_clips 裁掉。但特征缓存(~4KB)应保住这条正样本。
    clips = tmp_path / "clips"; clips.mkdir()
    training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    _clip(clips / "drink.mp4", 200); store.label_clip(clips / "drink.mp4", True)
    # 先缓存它的特征（模拟之前训练过一次）
    from catcam.video_trainer import extract_and_cache, feature_cache_path
    extract_and_cache(clips / "drink.mp4", training, _FakeExtractor(8), dim=8)
    assert feature_cache_path(training, "drink.mp4").exists()
    # 现在 clip 文件被裁掉了
    (clips / "drink.mp4").unlink()
    X, y, names = gather_dataset(clips, training, store, _FakeExtractor(8), dim=8)
    assert "drink.mp4" in names and 1 in y     # 靠缓存仍计入这条正样本


class _ValExtractor:
    def __init__(self, v): self.v = v
    def extract(self, frames):
        import numpy as np
        return np.full(8, self.v, np.float32)


def test_extract_and_cache_force_reextracts(tmp_path):
    from catcam.video_trainer import extract_and_cache
    clips = tmp_path / "clips"; clips.mkdir(); training = tmp_path / "training"
    _clip(clips / "a.mp4", 200)
    assert extract_and_cache(clips / "a.mp4", training, _ValExtractor(1.0), dim=8)[0] == 1.0
    # 不 force：读旧缓存(1.0)，即便换了提取器
    assert extract_and_cache(clips / "a.mp4", training, _ValExtractor(9.0), dim=8)[0] == 1.0
    # force：重抽得新值(9.0)并覆盖缓存
    assert extract_and_cache(clips / "a.mp4", training, _ValExtractor(9.0), dim=8, force=True)[0] == 9.0
    assert extract_and_cache(clips / "a.mp4", training, _ValExtractor(0.0), dim=8)[0] == 9.0


def test_force_falls_back_to_cache_when_clip_pruned(tmp_path):
    from catcam.video_trainer import extract_and_cache
    clips = tmp_path / "clips"; clips.mkdir(); training = tmp_path / "training"
    _clip(clips / "a.mp4", 200)
    extract_and_cache(clips / "a.mp4", training, _FakeExtractor(8), dim=8)
    (clips / "a.mp4").unlink()                      # mp4 被裁掉
    f = extract_and_cache(clips / "a.mp4", training, _FakeExtractor(8), dim=8, force=True)
    assert f is not None and f.shape == (8,)        # force 也退回缓存（唯一副本）


def test_train_video_head_rebuild_ok(tmp_path):
    from catcam.models import ModelRegistry
    clips = tmp_path / "clips"; clips.mkdir(); training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    for i in range(6):
        _clip(clips / f"p{i}.mp4", 200); store.label_clip(clips / f"p{i}.mp4", True)
    for i in range(6):
        _clip(clips / f"n{i}.mp4", 10); store.label_clip(clips / f"n{i}.mp4", False)
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    res = train_video_head(clips, training, store, registry, tmp_path / "models",
                           extractor=_FakeExtractor(8), dim=8, epochs=100,
                           created_ts=1.0, rebuild=True)
    assert res["version"] == "v1"


def test_retraining_same_data_reuses_version_and_rebuild_is_unique(tmp_path, monkeypatch):
    clips = tmp_path / "clips"; clips.mkdir()
    training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    for i in range(8):
        name = f"sample{i}.mp4"
        _clip(clips / name, 200 if i % 2 else 10)
        store.label_clip(clips / name, bool(i % 2))
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    args = (clips, training, store, registry, tmp_path / "models")
    kwargs = dict(extractor=_FakeExtractor(8), dim=8, epochs=20, created_ts=1)
    first = train_video_head(*args, **kwargs)
    again = train_video_head(*args, **kwargs)
    assert again["reused"] and again["version"] == first["version"]
    assert len(registry.list()) == 1
    registry.set_active("v1")
    rebuilt = train_video_head(*args, **kwargs, rebuild=True)
    assert rebuilt["comparison"]["status"] == "compared"
    assert rebuilt["version"] == "v2"
    assert registry.get("v1")["path"] != registry.get("v2")["path"]
    assert registry.active_id() == "v1"
    monkeypatch.setattr(DrinkingHead, "fit", lambda *a, **k: DrinkingHead(np.zeros(8), -100, np.zeros(8), np.ones(8)))
    regressed = train_video_head(*args, **kwargs, rebuild=True)
    assert regressed["comparison"]["status"] == "compared"
    assert any("退步" in reason for reason in regressed["release"]["reasons"])
    assert registry.active_id() == "v1"


def test_wrong_shape_or_nan_cache_is_recomputed(tmp_path):
    from catcam.video_trainer import extract_and_cache, _save_npy
    clip = tmp_path / "clip.mp4"
    _clip(clip, 200)
    cache = feature_cache_path(tmp_path, clip.name)
    for invalid in (np.zeros(2), np.full(8, np.nan)):
        _save_npy(cache, invalid)
        feat = extract_and_cache(clip, tmp_path, _FakeExtractor(8), dim=8)
        assert feat.shape == (8,) and np.isfinite(feat).all()


def test_correction_removes_old_class_frames(tmp_path):
    clip = tmp_path / "sample.mp4"
    _clip(clip, 200)
    training = tmp_path / "training"
    store = FeedbackStore(tmp_path / "db.sqlite", training)
    store.label_clip(clip, True)
    assert list((training / "drinking").glob("sample_*.jpg"))
    store.label_clip(clip, False)
    assert not list((training / "drinking").glob("sample_*.jpg"))
    assert list((training / "not_drinking").glob("sample_*.jpg"))
