"""「真喝水 / 没喝」分类器：把训练好的 YOLOv8-cls 模型用于录制前确认。

`ActiveModel` 是个可热插拔的持有者：网页切换生效版本时换里面的分类器，采集/检测线程
通过 `confirm(frame)` 询问。没启用任何模型时一律放行（= 旧行为，只靠简单模型多录）。
确认出错也放行（fail-open），坏模型不至于把录制全掐死。
"""
from __future__ import annotations

import threading

DRINKING_CLASS = "drinking"


class DrinkingClassifier:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_path(cls, path: str) -> "DrinkingClassifier":
        from ultralytics import YOLO

        return cls(YOLO(path))

    def is_drinking(self, frame) -> bool:
        results = self.model(frame, verbose=False)
        res = results[0]
        top1 = int(res.probs.top1)
        return self.model.names[top1] == DRINKING_CLASS


class ActiveModel:
    """当前生效分类器的线程安全持有者，可在运行时热替换。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._clf = None
        self._id = None

    def set(self, classifier, model_id: str) -> None:
        with self._lock:
            self._clf = classifier
            self._id = model_id

    def clear(self) -> None:
        with self._lock:
            self._clf = None
            self._id = None

    @property
    def active_id(self):
        with self._lock:
            return self._id

    def confirm(self, frame) -> bool:
        """没启用模型 → True（放行）；启用了 → 模型判「真喝水」才 True；出错 → True（放行）。"""
        with self._lock:
            clf = self._clf
        if clf is None:
            return True
        try:
            return clf.is_drinking(frame)
        except Exception:  # noqa: BLE001 — 坏模型不许掐死录制
            return True
