"""模型版本登记表：每次训练产出一个带版本号 + 准确率 + 样本数的模型，可选哪个生效。

存成 data/models/registry.json（人也能看）。`active` 指向当前生效的版本 id（None=不启用，
只用简单模型）。生效的模型会在录制前确认「真喝水」，见 classifier.py / pipeline.py。
"""
from __future__ import annotations

import json
import threading
from pathlib import Path


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
                d.setdefault("next_seq", 1)
                d.setdefault("models", [])
                return d
            except (json.JSONDecodeError, OSError):
                pass
        return {"active": None, "next_seq": 1, "models": []}

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

    def get(self, model_id: str):
        with self._lock:
            for m in self._data["models"]:
                if m["id"] == model_id:
                    return dict(m)
            return None

    def set_active(self, model_id) -> None:
        with self._lock:
            if model_id is not None and not any(m["id"] == model_id for m in self._data["models"]):
                raise KeyError(model_id)
            self._data["active"] = model_id
            self._save()

    def active_path(self):
        with self._lock:
            mid = self._data.get("active")
            for m in self._data["models"]:
                if m["id"] == mid:
                    return m["path"]
            return None
