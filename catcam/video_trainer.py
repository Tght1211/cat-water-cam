"""离线训练本地视频小头：缓存 s3d 特征 + 当前标签 → 训 logistic 头 → 登记 registry 版本。

不改运行中的 app。训练数据来自 VLM/人工已写进 labels 的标注（is_drinking）。
"""
from __future__ import annotations

import io
import os
import threading
import time
from pathlib import Path

import numpy as np

from catcam.videojudge import DrinkingHead, S3DFeatureExtractor, read_clip_frames, FEATURE_DIM

MIN_PER_CLASS = 4   # 每类至少这么多段才值得训


def feature_cache_path(training_dir, clip_name: str) -> Path:
    return Path(training_dir) / "features" / (Path(clip_name).stem + ".npy")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """先写临时文件再原子替换——任何中途失败都不会留下半截缓存文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _save_npy(path: Path, arr: np.ndarray) -> None:
    """经 BytesIO 存 .npy：纯 Python 写，绕开 numpy 的 FILE*(tofile) 路径
    （守护进程/3.14 线程里 `_fdopen` 会失败、把文件写一半留成损坏头）。原子落盘。"""
    buf = io.BytesIO()
    np.save(buf, arr)
    _atomic_write_bytes(Path(path), buf.getvalue())


def _load_npy(path: Path) -> np.ndarray:
    """经 BytesIO 读 .npy：纯 Python 读，绕开 FILE*(fromfile)。"""
    return np.load(io.BytesIO(Path(path).read_bytes()))


def extract_and_cache(clip_path, training_dir, extractor, dim: int = FEATURE_DIM):
    """取一段的特征：命中缓存直接读，否则抽帧→提取→存缓存。抽帧空返回 None。

    缓存损坏/截断（上次写一半崩了）当作未命中：重抽并覆盖（自愈）。
    """
    clip_path = Path(clip_path)
    cache = feature_cache_path(training_dir, clip_path.name)
    if cache.exists():
        try:
            return _load_npy(cache)
        except Exception:  # noqa: BLE001 —— 损坏缓存不致命：重抽覆盖
            pass
    frames = read_clip_frames(clip_path)
    if not frames:
        return None
    feat = np.asarray(extractor.extract(frames), np.float32).reshape(-1)
    _save_npy(cache, feat)
    return feat


def _labeled_clips(store) -> list[tuple[str, int]]:
    """从 labels 表取所有 (clip_name, is_drinking)；排除 source='local'（机器自身判定，不当训练真值）。"""
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute(
            "SELECT clip_name, is_drinking FROM labels WHERE source IS NULL OR source != 'local'"
        ).fetchall()
    return [(name, int(v)) for name, v in rows]


def gather_dataset(clips_dir, training_dir, store, extractor, dim: int = FEATURE_DIM):
    """对每个有标注且 clip 文件还在的段，取特征 + 标签。返回 (X, y, names)。"""
    clips_dir = Path(clips_dir)
    X, y, names = [], [], []
    for name, label in _labeled_clips(store):
        # 不预判 clip 是否存在：extract_and_cache 先看特征缓存——clip 即便被 max_clips 裁掉，
        # 只要特征缓存(~4KB)还在就保住这条样本（喝水正样本稀少，绝不能因裁剪丢）。
        # 既无缓存、clip 也没了 → extract_and_cache 返回 None，跳过。
        feat = extract_and_cache(clips_dir / name, training_dir, extractor, dim)
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
    # 确定性留出集：固定种子打乱后取头部 val_ratio 当验证
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = max(1, int(len(y) * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    head = DrinkingHead.fit(X[train_idx], y[train_idx], dim=dim, epochs=epochs, seed=seed)
    yval = y[val_idx]
    preds = np.array([head.predict(X[i])[0] for i in val_idx], int)
    top1 = float((preds == yval).mean())
    # 类别不平衡下 top1 会骗人（全猜「没喝」也能很高）：另报喝水类的召回/精确，和「全猜没喝」基线。
    n_pos = int((yval == 1).sum()); n_neg = int((yval == 0).sum())
    tp = int(((preds == 1) & (yval == 1)).sum()); pp = int((preds == 1).sum())
    drinking_recall = (tp / n_pos) if n_pos else None        # 抓到了几成真喝水
    drinking_precision = (tp / pp) if pp else None
    naive_baseline = (max(n_pos, n_neg) / len(yval)) if len(yval) else None  # 全猜多数类的准确率
    models_dir = Path(models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    head_path = models_dir / f"videohead_{int(created_ts)}.npz"
    head.save(head_path)
    entry = registry.add(path=head_path, top1=top1, image_counts=counts,
                         label_counts=counts, base="s3d+head", epochs=epochs,
                         imgsz=224, created_ts=created_ts)
    return {"version": entry["id"], "top1": top1, "counts": counts,
            "val_counts": {"drinking": n_pos, "not_drinking": n_neg},
            "drinking_recall": drinking_recall, "drinking_precision": drinking_precision,
            "naive_baseline": naive_baseline}


def _pct(x) -> str:
    return f"{x:.0%}" if isinstance(x, float) else "—"


class VideoTrainingManager:
    """网页一键训练本地视频模型：后台线程跑 train_video_head，随时查状态。

    与单帧 TrainingManager 并存。extractor 可注入（测试塞假提取器）；None=用真 s3d。
    训完登记成 base=s3d+head 的版本（不自动生效）。
    """

    def __init__(self, clips_dir, training_dir, feedback, registry, models_dir,
                 extractor=None, dim: int = FEATURE_DIM, epochs: int = 300):
        self.clips_dir = clips_dir
        self.training_dir = training_dir
        self.feedback = feedback
        self.registry = registry
        self.models_dir = models_dir
        self._extractor = extractor
        self._dim = dim
        self._epochs = epochs
        self._lock = threading.Lock()
        self._state = "idle"   # idle | running | done | error
        self._detail = ""
        self._result: dict | None = None

    def status(self) -> dict:
        with self._lock:
            base = {"state": self._state, "detail": self._detail, "result": self._result}
        if self.registry is not None:
            base["models"] = self.registry.list()
            base["active"] = self.registry.active_id()
        return base

    def start(self) -> bool:
        with self._lock:
            if self._state == "running":
                return False
            self._state = "running"
            self._detail = "训练中…（首次要为每段抽 s3d 特征，可能要几分钟）"
            self._result = None
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        try:
            res = train_video_head(
                self.clips_dir, self.training_dir, self.feedback, self.registry, self.models_dir,
                extractor=self._extractor, dim=self._dim, epochs=self._epochs, created_ts=time.time(),
            )
            detail = (f"完成 {res['version']} · 喝水召回 {_pct(res['drinking_recall'])} "
                      f"精确 {_pct(res['drinking_precision'])}（top1 {_pct(res['top1'])}，"
                      f"全猜没喝基线 {_pct(res['naive_baseline'])}；样本 👍{res['counts']['drinking']}"
                      f"/👎{res['counts']['not_drinking']}）。未自动生效。")
            with self._lock:
                self._state = "done"; self._result = res; self._detail = detail
        except ValueError as e:   # 样本不够等可预期问题
            with self._lock:
                self._state = "error"; self._detail = str(e)
        except Exception as e:    # noqa: BLE001
            import traceback
            print("视频训练失败，完整堆栈：\n" + traceback.format_exc(), flush=True)
            with self._lock:
                self._state = "error"; self._detail = f"训练失败：{e}"
