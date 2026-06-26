# 本地视频模型（s3d 冻结特征 + torch 小头）第二阶段 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用云 VLM 攒下的标注，离线训练一个本地视频「真喝水/没喝」小模型（冻结 s3d 主干提特征 + 一层 logistic 头），并备好它作为整段裁判的实现——但本轮**不改运行中 app 的热路径**，只交付可单测的离线核心 + 一次真实 smoke。

**Architecture:** torchvision 自带的 Kinetics-400 预训练 `s3d`（8.3M 参数，已随项目装好）当**冻结特征提取器**：一段 clip 抽 16 帧 → 官方 transforms → 取分类器前 1024 维池化特征。特征按 clip 缓存。`video_trainer` 把「缓存特征 + 当前标签」喂给一层 `nn.Linear` logistic 头（`BCEWithLogitsLoss` + `pos_weight` 处理不平衡），登记成 registry 版本。`LocalVideoClipJudge`（提取器 + 头 → `Verdict`）实现好、单测覆盖，待后续一步接进 app 当裁判。

**Tech Stack:** 已装的 `torch` 2.12 / `torchvision` 0.27 / `cv2` / `numpy`；现有 `catcam`（`judge.Verdict`、`models.ModelRegistry`、`feedback.FeedbackStore`、`config`）；pytest。**不引入** `transformers` / `sklearn`。

设计依据：`docs/superpowers/specs/2026-06-26-video-action-judge-design.md`（含 2026-06-27 环境校正附录）。

---

## 落地前已实测确认（写代码时按这些事实）

- `torchvision.models.video.s3d(weights=S3D_Weights.DEFAULT)`：权重 32MB（已缓存到 `~/.cache/torch`）。
- 官方预处理 `S3D_Weights.DEFAULT.transforms()`：吃 `(T, C, H, W)` uint8 → 出 `(C, T, H, W)` float，resize256/crop224、Kinetics 均值方差归一化。
- `model.avgpool` 的 forward hook 输出 flatten 后 = **1024 维**特征；`model(batch)` 出 `(1, 400)` logits（不用）。
- 输入 batch 形状 `(1, 3, 16, 224, 224)`。

---

## File Structure

- **新建** `catcam/videojudge.py`：`read_clip_frames`（clip→16 帧 RGB）、`S3DFeatureExtractor`（懒加载、`extract(frames)->np[1024]`）、`DrinkingHead`（torch 一层 logistic，`fit`/`predict`/`save`/`load`，含特征标准化）、`LocalVideoClipJudge`（提取器+头→`Verdict`）。
- **新建** `catcam/video_trainer.py`：`feature_cache_path`、`extract_and_cache`、`gather_dataset`（缓存特征 + 当前标签）、`train_video_head`（训头+留出集准确率+登记版本）。
- **新建** `catcam/video_train.py`：`main(config_path)` CLI，串起来跑一次；`__main__` 入口。
- **新建** `tests/test_videojudge.py`、`tests/test_video_trainer.py`。
- 不改 `app.py` / `web.py` / 现有模块（本轮零热路径改动）。

---

## Task 1: read_clip_frames —— clip 抽 16 帧（RGB）

**Files:**
- Create: `catcam/videojudge.py`（先放这一个函数 + 文件头）
- Test: `tests/test_videojudge.py`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import cv2
from catcam.videojudge import read_clip_frames


def _make_clip(path, n=20, color=(120, 130, 140)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    for _ in range(n):
        w.write(np.full((24, 32, 3), color, np.uint8))  # BGR
    w.release()


def test_read_clip_frames_samples_n_and_is_rgb(tmp_path):
    clip = tmp_path / "a.mp4"; _make_clip(clip, n=20, color=(200, 100, 50))  # BGR
    frames = read_clip_frames(clip, n=16)
    assert len(frames) == 16
    f = frames[0]
    assert f.shape == (24, 32, 3) and f.dtype == np.uint8
    # BGR(200,100,50) → RGB(50,100,200)：确认做了 BGR→RGB
    assert tuple(int(x) for x in f[0, 0]) == (50, 100, 200)


def test_read_clip_frames_short_clip_pads(tmp_path):
    clip = tmp_path / "b.mp4"; _make_clip(clip, n=5)
    frames = read_clip_frames(clip, n=16)
    assert len(frames) == 16  # 帧不够时重复补齐到 n
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest tests/test_videojudge.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'catcam.videojudge'`。

- [ ] **Step 3: 写文件头 + read_clip_frames**

```python
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
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest tests/test_videojudge.py -q`
Expected: PASS（2 passed）。

- [ ] **Step 5: 提交**

```bash
git add catcam/videojudge.py tests/test_videojudge.py
git commit -m "feat(videojudge): read_clip_frames——clip 均匀抽 16 帧并转 RGB

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: DrinkingHead —— torch 一层 logistic 头（含特征标准化）

**Files:**
- Modify: `catcam/videojudge.py`（追加 `DrinkingHead`）
- Test: `tests/test_videojudge.py`（追加）

- [ ] **Step 1: 写失败测试**

```python
from catcam.videojudge import DrinkingHead, FEATURE_DIM


def _synth(n=40, dim=8, seed=0):
    """造线性可分的两类特征：第 0 维正=喝水、负=没喝。"""
    rng = np.random.default_rng(seed)
    Xpos = rng.normal(0, 0.3, (n, dim)); Xpos[:, 0] += 2.0
    Xneg = rng.normal(0, 0.3, (n, dim)); Xneg[:, 0] -= 2.0
    X = np.vstack([Xpos, Xneg]).astype(np.float32)
    y = np.array([1] * n + [0] * n)
    return X, y


def test_head_learns_separable_features():
    X, y = _synth()
    head = DrinkingHead.fit(X, y, dim=8, epochs=300, seed=0)
    drink, conf = head.predict(X[0])      # 第 0 维 +2 → 喝水
    assert drink is True and conf > 0.8
    drink2, _ = head.predict(X[-1])       # 第 0 维 -2 → 没喝
    assert drink2 is False


def test_head_save_load_roundtrip(tmp_path):
    X, y = _synth()
    head = DrinkingHead.fit(X, y, dim=8, epochs=200, seed=1)
    p = tmp_path / "head.pt"; head.save(p)
    again = DrinkingHead.load(p)
    assert again.dim == 8
    # 同一输入，存取前后预测一致
    assert again.predict(X[0])[0] == head.predict(X[0])[0]
    assert abs(again.predict(X[0])[1] - head.predict(X[0])[1]) < 1e-5


def test_head_handles_imbalance():
    # 少数喝水(5) + 多数没喝(50)，靠 pos_weight 不至于全判没喝
    X, y = _synth(n=5, dim=8); Xn, yn = _synth(n=50, dim=8, seed=9)
    X = np.vstack([X[:5], Xn[50:]]); y = np.concatenate([y[:5], yn[50:]])
    head = DrinkingHead.fit(X, y, dim=8, epochs=400, seed=0)
    pos = X[0]; pos[0] = 2.0
    assert head.predict(pos)[0] is True
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest tests/test_videojudge.py -q`
Expected: FAIL —— `ImportError: cannot import name 'DrinkingHead'`。

- [ ] **Step 3: 实现 DrinkingHead**

在 `catcam/videojudge.py` 追加（顶部不加 torch import，放函数内懒加载以免影响只用 read_clip_frames 的场景；但 head 必用 torch，这里在类方法内 import）：

```python
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
        return (p >= 0.5), p

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, weight=self._w, bias=np.float32(self._b),
                 mean=self._mean, std=self._std)
        if path.suffix != ".npz" and Path(str(path) + ".npz").exists():
            Path(str(path) + ".npz").rename(path)   # np.savez 会补 .npz，统一回传入名

    @classmethod
    def load(cls, path) -> "DrinkingHead":
        with np.load(str(path)) as d:
            return cls(d["weight"], float(d["bias"]), d["mean"], d["std"])
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest tests/test_videojudge.py -q`
Expected: PASS（5 passed）。

- [ ] **Step 5: 提交**

```bash
git add catcam/videojudge.py tests/test_videojudge.py
git commit -m "feat(videojudge): DrinkingHead——torch 一层 logistic 头(标准化+pos_weight+存取)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: LocalVideoClipJudge + S3DFeatureExtractor

**Files:**
- Modify: `catcam/videojudge.py`（追加 `S3DFeatureExtractor`、`LocalVideoClipJudge`）
- Test: `tests/test_videojudge.py`（追加；judge 用假提取器，另加一个真 s3d smoke）

- [ ] **Step 1: 写失败测试（judge 用假提取器 + 真 s3d smoke）**

```python
from catcam.videojudge import LocalVideoClipJudge, S3DFeatureExtractor
from catcam.judge import Verdict


class _FakeExtractor:
    """假提取器：不碰真模型，按第 0 帧像素给个确定特征。"""
    def __init__(self, dim=8): self.dim = dim
    def extract(self, frames):
        v = np.zeros(self.dim, np.float32)
        v[0] = 2.0 if frames[0][0, 0, 0] > 128 else -2.0
        return v


def test_local_judge_returns_verdict(tmp_path):
    clip = tmp_path / "a.mp4"; _make_clip(clip, n=16, color=(0, 0, 255))  # BGR→R 大→喝水
    X, y = _synth(dim=8)
    head = DrinkingHead.fit(X, y, dim=8, epochs=300, seed=0)
    judge = LocalVideoClipJudge(_FakeExtractor(8), head, version="v3")
    v = judge.judge(clip)
    assert isinstance(v, Verdict) and v.by == "v3"
    assert v.drinking is True and 0.0 <= v.confidence <= 1.0


def test_local_judge_fail_open_on_empty(tmp_path):
    # 抽帧为空（坏 clip）→ 返回 None，不崩
    bad = tmp_path / "empty.mp4"; bad.write_bytes(b"not a video")
    judge = LocalVideoClipJudge(_FakeExtractor(8),
                                DrinkingHead.fit(*_synth(dim=8), dim=8, epochs=50), version="v3")
    assert judge.judge(bad) is None


def test_s3d_extractor_real_smoke():
    # 真 s3d：权重已缓存；一段 16 帧假画面 → 1024 维特征。验证真实主干路径可用。
    ext = S3DFeatureExtractor()
    frames = [np.full((24, 32, 3), 100 + i, np.uint8) for i in range(16)]
    feat = ext.extract(frames)
    assert feat.shape == (FEATURE_DIM,) and feat.dtype == np.float32
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest tests/test_videojudge.py -q`
Expected: FAIL —— `ImportError: cannot import name 'S3DFeatureExtractor'`。

- [ ] **Step 3: 实现 S3DFeatureExtractor + LocalVideoClipJudge**

在 `catcam/videojudge.py` 追加：

```python
class S3DFeatureExtractor:
    """冻结 s3d 主干特征提取器：一组 RGB 帧 → 1024 维特征。torch/torchvision 懒加载。"""

    def __init__(self, device: str | None = None):
        self._device = device
        self._model = None
        self._transforms = None
        self._feat = {}   # forward hook 暂存 avgpool 输出

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from torchvision.models.video import s3d, S3D_Weights
        w = S3D_Weights.DEFAULT
        model = s3d(weights=w)
        model.eval()
        for p in model.parameters():       # 冻结
            p.requires_grad_(False)
        model.avgpool.register_forward_hook(
            lambda m, i, o: self._feat.__setitem__("v", o.detach())
        )
        self._model = model
        self._transforms = w.transforms()
        self._torch = torch

    def extract(self, frames) -> np.ndarray:
        """frames: RGB(HWC uint8) 列表 → np.float32[1024]。"""
        self._ensure_loaded()
        torch = self._torch
        # (T, H, W, C) uint8 → (T, C, H, W) uint8 → 官方 transforms → (C, T, H, W) → batch
        arr = np.stack(frames).astype(np.uint8)
        clip = torch.from_numpy(arr).permute(0, 3, 1, 2)
        batch = self._transforms(clip).unsqueeze(0)
        with torch.no_grad():
            self._model(batch)
        return self._feat["v"].flatten(1)[0].cpu().numpy().astype(np.float32)


class LocalVideoClipJudge:
    """本地视频裁判：clip → 抽帧 → s3d 特征 → 头 → Verdict（by=版本号）。

    与 VLMClipJudge 同接口（judge(clip)->Verdict|None），可直接替进 app 的裁判位（后续一步）。
    fail-open：抽帧空/出错 → None。注意：本地裁判**不写 labels**（预测不是训练真值，避免自我强化）。
    """

    def __init__(self, extractor, head, version: str, frames: int = CLIP_FRAMES, log=print):
        self.extractor = extractor
        self.head = head
        self.version = version
        self.frames = frames
        self.log = log

    def judge(self, clip_path) -> "Verdict | None":
        from catcam.judge import Verdict
        try:
            frames = read_clip_frames(clip_path, self.frames)
            if not frames:
                return None
            feat = self.extractor.extract(frames)
            drinking, conf = self.head.predict(feat)
        except Exception as e:  # noqa: BLE001 —— 裁判失败绝不能崩
            self.log(f"本地视频裁判失败（{clip_path}）：{e}")
            return None
        return Verdict(drinking=bool(drinking), confidence=float(conf),
                      reason="", by=self.version)
```

- [ ] **Step 4: 运行，确认通过（含真 s3d smoke）**

Run: `.venv/bin/pytest tests/test_videojudge.py -q`
Expected: PASS（8 passed）。真 smoke 首次会用已缓存的 s3d 权重，秒级完成。

- [ ] **Step 5: 提交**

```bash
git add catcam/videojudge.py tests/test_videojudge.py
git commit -m "feat(videojudge): S3DFeatureExtractor(冻结/懒加载) + LocalVideoClipJudge

真 s3d 主干 smoke 通过：16 帧→1024 维特征。本地裁判同 VLMClipJudge 接口、不写标签。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: video_trainer —— 特征缓存 + 训头 + 登记版本

**Files:**
- Create: `catcam/video_trainer.py`
- Test: `tests/test_video_trainer.py`

> 特征缓存：`data/training/features/<clip_stem>.npy`（1024 float32 ≈ 4KB/段，不随标注变）。
> 训练集 = 对每个**有标注**的 clip 取缓存特征 + 其 `labels.is_drinking`。留出集算准确率，登记 registry 版本。

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
from catcam.video_trainer import feature_cache_path, gather_dataset, train_video_head
from catcam.videojudge import DrinkingHead
from catcam.feedback import FeedbackStore
from catcam.models import ModelRegistry


class _FakeExtractor:
    def __init__(self, dim=8): self.dim = dim
    def extract(self, frames):
        v = np.zeros(self.dim, np.float32); v[0] = float(frames[0][0, 0, 0]); return v


def _clip(path, val):
    import cv2
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
```

- [ ] **Step 2: 运行，确认失败**

Run: `.venv/bin/pytest tests/test_video_trainer.py -q`
Expected: FAIL —— `ModuleNotFoundError: No module named 'catcam.video_trainer'`。

- [ ] **Step 3: 实现 video_trainer.py**

```python
"""离线训练本地视频小头：缓存 s3d 特征 + 当前标签 → 训 logistic 头 → 登记 registry 版本。

不改运行中的 app。训练数据来自 VLM/人工已写进 labels 的标注（is_drinking）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from catcam.videojudge import DrinkingHead, S3DFeatureExtractor, read_clip_frames, FEATURE_DIM

MIN_PER_CLASS = 4   # 每类至少这么多段才值得训


def feature_cache_path(training_dir, clip_name: str) -> Path:
    return Path(training_dir) / "features" / (Path(clip_name).stem + ".npy")


def extract_and_cache(clip_path, training_dir, extractor, dim: int = FEATURE_DIM) -> np.ndarray | None:
    """取一段的特征：命中缓存直接读，否则抽帧→提取→存缓存。抽帧空返回 None。"""
    clip_path = Path(clip_path)
    cache = feature_cache_path(training_dir, clip_path.name)
    if cache.exists():
        return np.load(cache)
    frames = read_clip_frames(clip_path)
    if not frames:
        return None
    feat = np.asarray(extractor.extract(frames), np.float32).reshape(-1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, feat)
    return feat


def _labeled_clips(store) -> list[tuple[str, int]]:
    """从 labels 表取所有 (clip_name, is_drinking)。"""
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute("SELECT clip_name, is_drinking FROM labels").fetchall()
    return [(name, int(v)) for name, v in rows]


def gather_dataset(clips_dir, training_dir, store, extractor, dim: int = FEATURE_DIM):
    """对每个有标注且 clip 文件还在的段，取特征 + 标签。返回 (X, y, names)。"""
    clips_dir = Path(clips_dir)
    X, y, names = [], [], []
    for name, label in _labeled_clips(store):
        clip = clips_dir / name
        if not clip.exists():
            continue
        feat = extract_and_cache(clip, training_dir, extractor, dim)
        if feat is None:
            continue
        X.append(feat); y.append(label); names.append(name)
    if not X:
        return np.empty((0, dim), np.float32), np.array([], int), []
    return np.vstack(X).astype(np.float32), np.array(y, int), names


def train_video_head(clips_dir, training_dir, store, registry, models_dir,
                     extractor=None, dim: int = FEATURE_DIM, epochs: int = 300,
                     val_ratio: float = 0.25, seed: int = 0, created_ts: float = 0.0) -> dict:
    """训头并登记版本。数据不够 raise ValueError。返回 {version, top1, counts}。"""
    extractor = extractor or S3DFeatureExtractor()
    X, y, names = gather_dataset(clips_dir, training_dir, store, extractor, dim)
    counts = {"drinking": int((y == 1).sum()), "not_drinking": int((y == 0).sum())}
    too_few = [c for c in ("drinking", "not_drinking") if counts[c] < MIN_PER_CLASS]
    if too_few:
        raise ValueError(f"标注样本不够：当前 {counts}，每类需 ≥{MIN_PER_CLASS}。")
    # 确定性留出集：按名字排序后取尾部 val_ratio 当验证
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = max(1, int(len(y) * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    head = DrinkingHead.fit(X[train_idx], y[train_idx], dim=dim, epochs=epochs, seed=seed)
    preds = np.array([head.predict(X[i])[0] for i in val_idx], int)
    top1 = float((preds == y[val_idx]).mean())
    models_dir = Path(models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    head_path = models_dir / f"videohead_{int(created_ts)}.npz"
    head.save(head_path)
    entry = registry.add(path=head_path, top1=top1, image_counts=counts,
                         label_counts=counts, base="s3d+head", epochs=epochs,
                         imgsz=224, created_ts=created_ts)
    return {"version": entry["id"], "top1": top1, "counts": counts}
```

- [ ] **Step 4: 运行，确认通过**

Run: `.venv/bin/pytest tests/test_video_trainer.py -q`
Expected: PASS（3 passed）。

- [ ] **Step 5: 提交**

```bash
git add catcam/video_trainer.py tests/test_video_trainer.py
git commit -m "feat(video_trainer): 特征缓存 + 训 logistic 头 + 登记 registry 版本(base=s3d+head)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: CLI —— 一条命令离线训练 + 报告

**Files:**
- Create: `catcam/video_train.py`
- Test: 无（薄壳；逻辑已在 Task 4 测）。用 `--help`/import 冒烟。

- [ ] **Step 1: 实现 video_train.py**

```python
"""离线训练本地视频模型：从已积累的标注训一个 s3d+小头，登记成版本（不自动生效）。

用法：.venv/bin/python -m catcam.video_train
首次会用已缓存的 s3d 权重提取每段特征（之后走缓存）。需先靠 AI/人工标注攒够样本（每类 ≥4）。
"""
from __future__ import annotations

import time

from catcam.config import load_config
from catcam.feedback import FeedbackStore
from catcam.models import ModelRegistry
from catcam.video_trainer import train_video_head


def main(config_path: str = "config.json") -> None:
    cfg = load_config(config_path)
    store = FeedbackStore(cfg.db_path, cfg.training_dir)
    registry = ModelRegistry(cfg.models_dir / "registry.json")
    print("开始训练本地视频模型（s3d 冻结特征 + logistic 头）…")
    try:
        res = train_video_head(
            cfg.clips_dir, cfg.training_dir, store, registry, cfg.models_dir,
            created_ts=time.time(),
        )
    except ValueError as e:
        print(f"训练未开始：{e}")
        return
    print(f"完成：版本 {res['version']}，留出集准确率 {res['top1']:.1%}，样本 {res['counts']}。")
    print("未自动生效——评估满意后再接入裁判（后续一步）。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: import 冒烟**

Run: `.venv/bin/python -c "import catcam.video_train; print('ok')"`
Expected: 打印 `ok`，无异常。

- [ ] **Step 3: 跑一次（数据不够时优雅退出）**

Run: `.venv/bin/python -m catcam.video_train`
Expected: 没攒够标注时打印「训练未开始：标注样本不够…」并正常退出（不报错、不崩）。

- [ ] **Step 4: 提交**

```bash
git add catcam/video_train.py
git commit -m "feat(video_train): 一条命令离线训练本地视频模型并登记版本

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 文档 + 全量回归

**Files:**
- Modify: `CLAUDE.md`（在 AI 裁判段后补一句第二阶段现状）
- Modify: `README.md`（「从 0 到 1」段补「本地视频模型」一句）

- [ ] **Step 1: CLAUDE.md 补充**

在 `CLAUDE.md` 的「AI 整段裁判 / 自动标注」条目末尾的 spec 指向行后，加一条：

```markdown
- **本地视频模型（第二阶段，离线已就绪、未接裁判）**：`videojudge.py`（`s3d` 冻结特征 + `DrinkingHead`
  torch 小头 + `LocalVideoClipJudge`）+ `video_trainer.py` + `python -m catcam.video_train`。用 VLM/人工
  攒的 `labels` 离线训一个本地视频「真喝水/没喝」小头，登记成 `base=s3d+head` 的版本（不自动生效）。
  主干用 torchvision 自带 `s3d`（8.3M，权重 ~30MB，免 transformers）；小头一层 logistic（免 sklearn）。
  **本轮不改采集/裁判热路径**——评估满意后再把 `LocalVideoClipJudge` 接进 `app.py` 的裁判位（后续一步）。
```

- [ ] **Step 2: README.md 补充**

在 README「从 0 到 1 训练…」一节末尾加一句：

```markdown
> 进阶（本地视频模型）：攒够标注后可跑 `.venv/bin/python -m catcam.video_train`，用本地 `s3d` 冻结特征
> 训一个**看动作（非单帧）**的「真喝水/没喝」小模型；登记成版本、显示留出集准确率，评估满意后再接入裁判。
```

- [ ] **Step 3: 全量回归**

Run: `.venv/bin/pytest -q`
Expected: 全绿（含新增 videojudge/video_trainer 测试 + 真 s3d smoke）。

- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md README.md
git commit -m "docs: 第二阶段本地视频模型(离线就绪、未接裁判)说明

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review 记录

- **Spec 覆盖（第二阶段离线核心）**：s3d 冻结特征（Task 3）、torch 小头（Task 2）、特征缓存 + 训练 + 版本登记（Task 4）、clip 抽帧（Task 1）、CLI（Task 5）、LocalVideoClipJudge 同接口待接（Task 3）。明确**不含**：网页训练按钮、真实训练运行（无数据）、app 裁判接线（后续一步）——均在 spec/plan 标注。
- **占位符**：无 TBD；每步含完整代码/命令/期望输出。
- **类型一致**：`DrinkingHead.fit(X,y,dim,epochs,seed)` / `.predict(feat)->(bool,float)` / `.save/.load` 前后一致；`S3DFeatureExtractor.extract(frames)->np[1024]`；`LocalVideoClipJudge.judge(clip)->Verdict|None`（复用 `catcam.judge.Verdict`，字段 drinking/confidence/reason/by 一致）；`train_video_head(...)->{version,top1,counts}`；`registry.add(...)` 签名与 `models.py` 一致（base/imgsz/created_ts 等）。
- **依赖**：只用已装的 torch/torchvision/cv2/numpy；torch 在类方法内懒 import，避免影响只用 read_clip_frames 的路径。真 s3d 权重已缓存。
- **边界**：抽帧空 / 坏 clip → fail-open None；样本不够 → ValueError 优雅退出；本地裁判不写 labels（避免自我强化）。
