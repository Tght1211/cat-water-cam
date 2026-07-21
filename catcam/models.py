"""模型版本登记表：每次训练产出一个带版本号 + 准确率 + 样本数的模型，可选哪个生效。

存成 data/models/registry.json（人也能看）。`active` 指向当前生效的版本 id（None=不启用，
只用简单模型）。生效的模型会在录制前确认「真喝水」，见 classifier.py / pipeline.py。
"""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path


BUNDLED_VIDEO_MODEL = "pretrained_s3d_head_v11.npz"
BUNDLED_VIDEO_TOP1 = 0.8914728682170543
BUNDLED_VIDEO_COUNTS = {"drinking": 57, "not_drinking": 462}


def install_bundled_video_model(
    registry: "ModelRegistry", models_dir: Path, source: Path | None = None
) -> dict | None:
    """为空 registry 安装随包发布的视频分类头；已有任何版本时保持用户数据不变。"""
    if registry.list():
        return None
    source = source or Path(__file__).with_name("assets") / BUNDLED_VIDEO_MODEL
    if not source.exists():
        return None
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / BUNDLED_VIDEO_MODEL
    shutil.copyfile(source, destination)
    entry = registry.add(
        path=destination,
        top1=BUNDLED_VIDEO_TOP1,
        image_counts=dict(BUNDLED_VIDEO_COUNTS),
        label_counts=dict(BUNDLED_VIDEO_COUNTS),
        base="s3d+head",
        epochs=300,
        imgsz=224,
        created_ts=1784482239.4945889,
    )
    registry.set_active(entry["id"], "shadow")
    return entry


class ModelRegistry:
    def __init__(self, registry_path: Path):
        self.path = Path(registry_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                d.setdefault("active", None)
                d.setdefault("active_mode", "shadow")
                d.setdefault("next_seq", 1)
                d.setdefault("models", [])
                return d
            except (json.JSONDecodeError, OSError):
                pass
        return {"active": None, "active_mode": "shadow", "next_seq": 1, "models": []}

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add(self, *, path, top1, image_counts, label_counts, base, epochs, imgsz, created_ts) -> dict:
        with self._lock:
            seq = self._data["next_seq"]
            self._data["next_seq"] = seq + 1
            entry = {
                "id": f"v{seq}",
                "version": seq,
                "created_ts": created_ts,
                "top1": top1,
                "image_counts": image_counts,   # 实际训练用的抽帧张数 {drinking,not_drinking}
                "label_counts": label_counts,    # 当时的标注段数快照
                "path": str(path),
                "base": base,
                "epochs": epochs,
                "imgsz": imgsz,
            }
            self._data["models"].insert(0, entry)  # 最新的排最前
            self._save()
            return entry

    def list(self) -> list[dict]:
        with self._lock:
            return [dict(m) for m in self._data["models"]]

    def active_id(self):
        with self._lock:
            return self._data.get("active")

    def active_mode(self):
        with self._lock:
            return self._data.get("active_mode", "shadow")

    def get(self, model_id: str):
        with self._lock:
            for m in self._data["models"]:
                if m["id"] == model_id:
                    return dict(m)
            return None

    def set_active(self, model_id, mode: str = "shadow") -> None:
        with self._lock:
            if model_id is not None and not any(m["id"] == model_id for m in self._data["models"]):
                raise KeyError(model_id)
            self._data["active"] = model_id
            self._data["active_mode"] = mode if mode in ("shadow", "gate") else "shadow"
            self._save()

    def active_path(self):
        with self._lock:
            mid = self._data.get("active")
            for m in self._data["models"]:
                if m["id"] == mid:
                    return m["path"]
            return None
