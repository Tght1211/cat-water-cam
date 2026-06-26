"""本地视频裁判：冻结 s3d 主干提 1024 维特征 + 一层 torch logistic 头判「真喝水/没喝」。

第二阶段（离线核心）：提取器/头/判定都做成可注入、可单测；真实 s3d 主干懒加载（torchvision 已装）。
重依赖（torch/torchvision）只在真正用主干时才 import，单测走假提取器，不碰真权重/网络。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

FEATURE_DIM = 1024   # s3d avgpool 后的特征维度（实测）
CLIP_FRAMES = 16     # 每段抽多少帧喂主干


def read_clip_frames(clip_path, n: int = CLIP_FRAMES) -> list[np.ndarray]:
    """从 clip 均匀抽 n 帧，返回 RGB(HWC uint8) 列表；帧不足则重复补齐到 n。"""
    cap = cv2.VideoCapture(str(clip_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    if not frames:
        return []
    step = max(1, len(frames) // n)
    chosen = frames[::step][:n]
    while len(chosen) < n:               # 不足 n：重复最后一帧补齐
        chosen.append(chosen[-1])
    return chosen


class DrinkingHead:
    """一层 logistic 回归头：标准化特征 → Linear(dim,1) → sigmoid。

    fit 用 BCEWithLogitsLoss(pos_weight=neg/pos) 处理类别不平衡；固定随机种子可复现。
    存盘内容：权重/偏置 + 训练集特征均值方差（预测时同样标准化）。
    """

    def __init__(self, weight, bias, mean, std):
        self._w = np.asarray(weight, np.float32).reshape(-1)   # (dim,)
        self._b = float(bias)
        self._mean = np.asarray(mean, np.float32).reshape(-1)
        self._std = np.asarray(std, np.float32).reshape(-1)

    @property
    def dim(self) -> int:
        return int(self._w.shape[0])

    @classmethod
    def fit(cls, X, y, dim: int = FEATURE_DIM, epochs: int = 300, lr: float = 0.05,
            seed: int = 0) -> "DrinkingHead":
        import torch
        torch.manual_seed(seed)
        X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
        mean = X.mean(0); std = X.std(0) + 1e-6
        Xn = (X - mean) / std
        n_pos = max(1.0, float((y == 1).sum())); n_neg = max(1.0, float((y == 0).sum()))
        Xt = torch.from_numpy(Xn); yt = torch.from_numpy(y).reshape(-1, 1)
        lin = torch.nn.Linear(dim, 1)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]))
        opt = torch.optim.Adam(lin.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(lin(Xt), yt).backward()
            opt.step()
        w = lin.weight.detach().numpy().reshape(-1)
        b = float(lin.bias.detach().numpy().reshape(-1)[0])
        return cls(w, b, mean, std)

    def predict_proba(self, feat) -> float:
        feat = np.asarray(feat, np.float32).reshape(-1)
        z = float(np.dot((feat - self._mean) / self._std, self._w) + self._b)
        return 1.0 / (1.0 + np.exp(-z))

    def predict(self, feat) -> tuple[bool, float]:
        p = self.predict_proba(feat)
        return bool(p >= 0.5), float(p)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weight=self._w, bias=np.float32(self._b),
                 mean=self._mean, std=self._std)
        appended = Path(str(path) + ".npz")        # np.savez 对无 .npz 的路径会自动补
        if appended.exists() and appended != path:
            appended.rename(path)

    @classmethod
    def load(cls, path) -> "DrinkingHead":
        with np.load(str(path)) as d:
            return cls(d["weight"], float(d["bias"]), d["mean"], d["std"])
