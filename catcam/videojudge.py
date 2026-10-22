"""本地视频裁判：冻结 s3d 主干提 1024 维特征 + 一层 torch logistic 头判「真喝水/没喝」。

第二阶段（离线核心）：提取器/头/判定都做成可注入、可单测；真实 s3d 主干懒加载（torchvision 已装）。
重依赖（torch/torchvision）只在真正用主干时才 import，单测走假提取器，不碰真权重/网络。
"""
from __future__ import annotations

import io
import os
import json
import threading
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

    def __init__(self, weight, bias, mean, std, threshold=0.5, metadata=None):
        self._w = np.asarray(weight, np.float32).reshape(-1)   # (dim,)
        self._b = float(bias)
        self._mean = np.asarray(mean, np.float32).reshape(-1)
        self._std = np.asarray(std, np.float32).reshape(-1)
        self.threshold = float(threshold)
        self.metadata = metadata or {"backend": "s3d-legacy"}

    @property
    def dim(self) -> int:
        return int(self._w.shape[0])

    @classmethod
    def fit(cls, X, y, dim: int = FEATURE_DIM, epochs: int = 300, lr: float = 0.05,
            seed: int = 0) -> "DrinkingHead":
        import torch
        torch.manual_seed(seed)
        X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
        mean = X.mean(0); std = np.maximum(X.std(0), 0.05)
        Xn = (X - mean) / std
        n_pos = max(1.0, float((y == 1).sum())); n_neg = max(1.0, float((y == 0).sum()))
        Xt = torch.from_numpy(Xn); yt = torch.from_numpy(y).reshape(-1, 1)
        lin = torch.nn.Linear(dim, 1)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]))
        opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=0.1)
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
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))))

    def predict(self, feat) -> tuple[bool, float]:
        p = self.predict_proba(feat)
        return bool(p >= self.threshold), float(p)

    def save(self, path) -> None:
        # 经 BytesIO + 原子替换：绕开 numpy 的 FILE*(tofile) 路径（守护进程/3.14 线程里会失败、
        # 把文件写一半留成损坏头），且写入是原子的、文件名就是传入的 path（不靠 .npz 自动补后改名）。
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        np.savez(buf, weight=self._w, bias=np.float32(self._b),
                 mean=self._mean, std=self._std, threshold=self.threshold,
                 metadata=json.dumps(self.metadata))
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path) -> "DrinkingHead":
        with np.load(io.BytesIO(Path(path).read_bytes())) as d:
            return cls(d["weight"], float(d["bias"]), d["mean"], d["std"],
                       float(d['threshold']) if 'threshold' in d else 0.5,
                       json.loads(str(d['metadata'])) if 'metadata' in d else None)


class S3DFeatureExtractor:
    """冻结 s3d 主干特征提取器：一组 RGB 帧 → 1024 维特征。torch/torchvision 懒加载。"""

    def __init__(self, device: str | None = None, weights_path=None):
        self._device = device
        self.weights_path = weights_path
        self._model = None
        self._transforms = None
        self._torch = None
        self._feat = {}   # forward hook 暂存 avgpool 输出
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from torchvision.models.video import s3d, S3D_Weights
        w = S3D_Weights.DEFAULT
        model = s3d(weights=w)
        if self.weights_path:
            if str(self.weights_path).endswith('.safetensors'):
                from safetensors.torch import load_file
                state = load_file(str(self.weights_path))
            else:
                state = torch.load(self.weights_path,map_location='cpu',weights_only=True)
            model.load_state_dict(state,strict=False)
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
        with self._lock:
            return self._extract(frames)

    def _extract(self, frames) -> np.ndarray:
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

    def judge(self, clip_path):
        from catcam.judge import Verdict
        try:
            if hasattr(self.extractor, 'extract_clip'):
                feat = self.extractor.extract_clip(clip_path)
                if feat is None:
                    return None
            else:
                frames = read_clip_frames(clip_path, self.frames)
                if not frames:
                    return None
                feat = self.extractor.extract(frames)
            drinking, conf = self.head.predict(feat)
        except Exception as e:  # noqa: BLE001 —— 裁判失败绝不能崩
            self.log(f"本地视频裁判失败（{clip_path}）：{e}")
            return None
        return Verdict(drinking=bool(drinking), confidence=float(conf),
                       reason="", by=self.version,
                       uncertain=abs(conf-self.head.threshold) < .15)


def read_temporal_windows(path, n=16, seconds=2.0, windows=3):
    """均匀选择最多三个连续两秒窗口，保持动作的时间尺度；仅解码选中的帧。"""
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total <= 0 or fps <= 0:
            return []
        span = min(total, max(n, int(seconds * fps)))
        starts = np.unique(np.linspace(0, max(0, total-span), windows).astype(int))
        result = []
        for start in starts:
            frames = []
            for idx in np.linspace(start, start+span-1, n).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(frames) == n:
                result.append(frames)
        return result
    finally:
        cap.release()


class TemporalFeatureExtractor:
    def __init__(self, backend='s3d-temporal-v2', roi=None, weights_path=None):
        self.backend = backend
        self.roi = roi
        self.weights_path = str(weights_path) if weights_path else None
        self.cache_key = backend
        if roi is not None:
            import hashlib
            self.cache_key += '-' + hashlib.sha256(json.dumps(roi).encode()).hexdigest()[:12]
        if weights_path:
            import hashlib
            self.cache_key += '-' + hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()[:12]
        self.backbone = S3DFeatureExtractor(weights_path=weights_path) if backend.startswith('s3d-') else VideoMAEFeatureExtractor()
        self.dim = 2048 if backend.startswith('s3d-') else 1536

    def extract_clip(self, path):
        windows = read_temporal_windows(path)
        if not windows:
            return None
        if self.roi is not None:
            # 给饮水区域留出猫头的上下文，降低其他背景对个性化训练的干扰。
            x1,y1,x2,y2 = self.roi
            cropped = []
            for frames in windows:
                h,w = frames[0].shape[:2]
                left,right = int(max(0,x1-.1)*w), int(min(1,x2+.1)*w)
                top,bottom = int(max(0,y1-.1)*h), int(min(1,y2+.1)*h)
                cropped.append([f[top:bottom,left:right] for f in frames])
            windows = cropped
        features = np.vstack([self.backbone.extract(frames) for frames in windows])
        return np.concatenate([features.mean(0), features.max(0)]).astype(np.float32)


class VideoMAEFeatureExtractor:
    """可选 Hugging Face 主干；首次加载下载，后续使用本机缓存。"""
    model_id = 'MCG-NJU/videomae-base-finetuned-kinetics'

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def extract(self, frames):
        with self._lock:
            import torch
            from transformers import VideoMAEImageProcessor, VideoMAEModel
            if self._model is None:
                self._processor = VideoMAEImageProcessor.from_pretrained(self.model_id)
                self._model = VideoMAEModel.from_pretrained(self.model_id).eval()
            inputs = self._processor(list(frames), return_tensors='pt')
            with torch.inference_mode():
                return self._model(**inputs).last_hidden_state.mean(1)[0].cpu().numpy().astype(np.float32)


def extractor_for_backend(backend, roi=None, weights_path=None):
    if backend == 's3d-legacy':
        return S3DFeatureExtractor()
    if backend in ('s3d-temporal-v2', 'videomae-temporal-v1'):
        return TemporalFeatureExtractor(backend, roi)
    if backend == 's3d-finetuned-v1' and weights_path:
        return TemporalFeatureExtractor(backend,roi,weights_path)
    raise ValueError(f'不支持的视频特征版本：{backend}')
