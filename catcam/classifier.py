"""「真喝水 / 没喝」分类器 + 两种运行模式。

设计要点（兜底 + 测试模型）：简单模型/YOLO 是**兜底**，永远负责多录候选、保证召回；
训练出来的分类器是**测试模型**，默认 `shadow`（测试）模式——只预测、**绝不拦截录制**，
预测结果记到事件上供和人工标注对比评估。只有你信任它了，才切到 `gate`（过滤）模式让它
过滤「好奇凑近没喝」。这样一个还没训好的模型不会把录制全掐死。

`ActiveModel` 可热插拔：网页切版本/切模式时换里面的分类器与模式。
- `gate(frame)`：录制是否放行。shadow 恒放行（兜底不受影响）；gate 模式才用模型判，出错放行。
- `predict(frame)`：模型怎么判（True/False/None），只用于评估，不影响录制。
"""
from __future__ import annotations

import threading

DRINKING_CLASS = "drinking"
SHADOW = "shadow"   # 测试模式：只预测、不拦截
GATE = "gate"       # 过滤模式：信任模型，用它过滤误触


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
    """当前生效分类器的线程安全持有者，可在运行时热替换模型与模式。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._clf = None
        self._id = None
        self._mode = SHADOW

    def set(self, classifier, model_id: str, mode: str = SHADOW) -> None:
        with self._lock:
            self._clf = classifier
            self._id = model_id
            self._mode = mode if mode in (SHADOW, GATE) else SHADOW

    def clear(self) -> None:
        with self._lock:
            self._clf = None
            self._id = None
            self._mode = SHADOW

    @property
    def active_id(self):
        with self._lock:
            return self._id

    @property
    def mode(self):
        with self._lock:
            return self._mode

    def predict(self, frame):
        """模型怎么判：True=真喝水 / False=没喝 / None=没模型或出错。只评估用，不影响录制。"""
        with self._lock:
            clf = self._clf
        if clf is None:
            return None
        try:
            return clf.is_drinking(frame)
        except Exception:  # noqa: BLE001
            return None

    def gate(self, frame) -> bool:
        """录制是否放行。没模型/测试(shadow)模式 → 恒 True（兜底不受影响）；
        过滤(gate)模式 → 模型判「真喝水」才放行，出错放行（坏模型不许掐死录制）。"""
        with self._lock:
            clf = self._clf
            mode = self._mode
        if clf is None or mode != GATE:
            return True
        try:
            return clf.is_drinking(frame)
        except Exception:  # noqa: BLE001
            return True
