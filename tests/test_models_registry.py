import numpy as np

from catcam.classifier import ActiveModel
from catcam.models import ModelRegistry, install_bundled_video_model


def test_registry_add_list_activate(tmp_path):
    reg = ModelRegistry(tmp_path / "registry.json")
    assert reg.active_id() is None and reg.list() == []
    a = reg.add(path=tmp_path / "a.pt", top1=0.8, image_counts={"drinking": 5, "not_drinking": 9},
                label_counts={"labeled": 3}, base="b.pt", epochs=10, imgsz=96, created_ts=1.0)
    b = reg.add(path=tmp_path / "b.pt", top1=0.9, image_counts={}, label_counts={},
                base="b.pt", epochs=10, imgsz=96, created_ts=2.0)
    assert a["id"] == "v1" and b["id"] == "v2"
    assert [m["id"] for m in reg.list()] == ["v2", "v1"]   # 新的在前
    reg.set_active("v2")
    assert reg.active_id() == "v2"
    assert reg.active_path().endswith("b.pt")
    reg.set_active(None)                                   # 停用
    assert reg.active_id() is None and reg.active_path() is None


def test_registry_persists(tmp_path):
    p = tmp_path / "registry.json"
    reg = ModelRegistry(p)
    reg.add(path="x.pt", top1=0.5, image_counts={}, label_counts={}, base="b", epochs=1, imgsz=64, created_ts=1.0)
    reg.set_active("v1")
    reg2 = ModelRegistry(p)                                # 重新加载
    assert reg2.active_id() == "v1" and len(reg2.list()) == 1


def test_registry_persists_active_mode(tmp_path):
    p = tmp_path / "registry.json"
    reg = ModelRegistry(p)
    reg.add(path="x.pt", top1=0.5, image_counts={}, label_counts={}, base="b", epochs=1, imgsz=64, created_ts=1.0)
    reg.set_active("v1", "gate")
    assert ModelRegistry(p).active_mode() == "gate"


def test_install_bundled_video_model_only_for_empty_registry(tmp_path):
    source = tmp_path / "bundled.npz"
    source.write_bytes(b"model")
    models_dir = tmp_path / "models"
    reg = ModelRegistry(models_dir / "registry.json")
    entry = install_bundled_video_model(reg, models_dir, source=source)
    assert entry is not None and entry["base"] == "s3d+head"
    assert reg.active_id() == entry["id"] and reg.active_mode() == "shadow"
    assert (models_dir / "pretrained_s3d_head_v11.npz").read_bytes() == b"model"
    assert install_bundled_video_model(reg, models_dir, source=source) is None


def test_install_bundled_video_model_does_not_touch_existing_registry(tmp_path):
    source = tmp_path / "bundled.npz"; source.write_bytes(b"model")
    reg = ModelRegistry(tmp_path / "registry.json")
    reg.add(path="mine.npz", top1=1.0, image_counts={}, label_counts={},
            base="s3d+head", epochs=1, imgsz=224, created_ts=1.0)
    assert install_bundled_video_model(reg, tmp_path / "models", source=source) is None
    assert [m["path"] for m in reg.list()] == ["mine.npz"]


class _AlwaysNo:
    def is_drinking(self, frame):
        return False


def test_active_model_modes():
    am = ActiveModel()
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    # 没模型：放行，预测 None
    assert am.gate(f) is True and am.predict(f) is None
    # 测试(shadow)模式：恒放行（兜底不受影响），但仍给出预测
    am.set(_AlwaysNo(), "v1")               # 默认 shadow
    assert am.mode == "shadow"
    assert am.gate(f) is True               # 关键：不拦截
    assert am.predict(f) is False           # 但记录它判「没喝」
    # 过滤(gate)模式：判「没喝」→ 拦
    am.set(_AlwaysNo(), "v1", "gate")
    assert am.mode == "gate" and am.gate(f) is False
    am.clear()
    assert am.gate(f) is True and am.predict(f) is None
