import numpy as np

from catcam.classifier import ActiveModel
from catcam.models import ModelRegistry


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


class _AlwaysNo:
    def is_drinking(self, frame):
        return False


def test_active_model_gate():
    am = ActiveModel()
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    assert am.confirm(f) is True            # 没启用 → 放行
    am.set(_AlwaysNo(), "v1")
    assert am.confirm(f) is False           # 启用且判「没喝」→ 拦
    assert am.active_id == "v1"
    am.clear()
    assert am.confirm(f) is True            # 停用 → 放行
