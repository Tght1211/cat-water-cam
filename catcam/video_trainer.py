"""离线训练本地视频小头：缓存 s3d 特征 + 当前标签 → 训 logistic 头 → 登记 registry 版本。

不改运行中的 app。只使用人工标注；验证集固定留出，AI 标签等待人工确认。
"""
from __future__ import annotations

import io
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from catcam.videojudge import DrinkingHead, S3DFeatureExtractor, read_clip_frames, FEATURE_DIM
from catcam.evaluation import holdout_split, classification_metrics, release_assessment, sample_group

MIN_PER_CLASS = 4   # 每类至少这么多段才值得训


def feature_cache_path(training_dir, clip_name: str) -> Path:
    return Path(training_dir) / "features" / (Path(clip_name).stem + ".npy")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """先写临时文件再原子替换——任何中途失败都不会留下半截缓存文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _save_npy(path: Path, arr: np.ndarray) -> None:
    """经 BytesIO 存 .npy：纯 Python 写，绕开 numpy 的 FILE*(tofile) 路径
    （守护进程/3.14 线程里 `_fdopen` 会失败、把文件写一半留成损坏头）。原子落盘。"""
    buf = io.BytesIO()
    np.save(buf, arr)
    _atomic_write_bytes(Path(path), buf.getvalue())


def _load_npy(path: Path) -> np.ndarray:
    """经 BytesIO 读 .npy：纯 Python 读，绕开 FILE*(fromfile)。"""
    return np.load(io.BytesIO(Path(path).read_bytes()))


def _valid_feature(feat, dim):
    feat = np.asarray(feat, np.float32)
    if feat.shape != (dim,) or not np.isfinite(feat).all():
        raise ValueError("特征维度不匹配或包含非有限数值，请重建特征")
    return feat


def extract_and_cache(clip_path, training_dir, extractor, dim: int = FEATURE_DIM, force: bool = False,
                      diagnostics=None):
    """取一段的特征：命中缓存直接读，否则抽帧→提取→存缓存。抽帧空返回 None。

    缓存损坏/截断（上次写一半崩了）当作未命中：重抽并覆盖（自愈）。
    force=True（从头重建）：忽略缓存、强制重抽并覆盖——但 mp4 已被裁掉时仍退回缓存（缓存是唯一副本）。
    """
    clip_path = Path(clip_path)
    def count(key):
        if diagnostics is not None:
            diagnostics[key] = diagnostics.get(key, 0) + 1
    cache = feature_cache_path(training_dir, clip_path.name)
    if not force and cache.exists():
        try:
            feat = _valid_feature(_load_npy(cache), dim)
            count("cache_hits")
            return feat
        except Exception:  # noqa: BLE001 —— 损坏缓存不致命：重抽覆盖
            pass
    frames = read_clip_frames(clip_path)
    if not frames:
        # mp4 不在了（多半被 max_clips 裁掉）：force 也只能退回缓存
        if cache.exists():
            try:
                feat = _valid_feature(_load_npy(cache), dim)
                count("cache_hits")
                return feat
            except Exception:  # noqa: BLE001
                return None
        return None
    feat = _valid_feature(np.asarray(extractor.extract(frames), np.float32).reshape(-1), dim)
    _save_npy(cache, feat)
    count("extracted")
    return feat


def _labeled_clips(store) -> list[tuple[str, int]]:
    """只使用人工确认标签；AI 和本地预测留给复核，不作为训练真值。"""
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        rows = conn.execute(
            "SELECT clip_name, is_drinking FROM labels WHERE source IS NULL OR source = 'human' ORDER BY clip_name"
        ).fetchall()
    return [(name, int(v)) for name, v in rows]


def gather_dataset(clips_dir, training_dir, store, extractor, dim: int = FEATURE_DIM,
                   force: bool = False, progress_cb=None, diagnostics=None):
    """对每个有标注且 clip 文件还在的段，取特征 + 标签。返回 (X, y, names)。

    force=True：从头重建——对还有 mp4 的段强制重抽特征（被裁掉的退回缓存）。
    progress_cb（可选）：每处理完一段回调 `{"phase":"extracting","done":k,"total":N}`
    （done 含被跳过的段，最终必达 total）。回调抛异常会被吞掉，绝不拖累抽取。
    """
    clips_dir = Path(clips_dir)
    X, y, names = [], [], []
    labeled = _labeled_clips(store)
    total = len(labeled)
    for i, (name, label) in enumerate(labeled):
        # 不预判 clip 是否存在：extract_and_cache 先看特征缓存——clip 即便被 max_clips 裁掉，
        # 只要特征缓存(~4KB)还在就保住这条样本（喝水正样本稀少，绝不能因裁剪丢）。
        # 既无缓存、clip 也没了 → extract_and_cache 返回 None，跳过。
        feat = extract_and_cache(clips_dir / name, training_dir, extractor, dim, force=force,
                                 diagnostics=diagnostics)
        if feat is not None:
            X.append(feat); y.append(label); names.append(name)
        elif diagnostics is not None:
            diagnostics["skipped"] = diagnostics.get("skipped", 0) + 1
        if progress_cb is not None:
            try:
                progress_cb({"phase": "extracting", "done": i + 1, "total": total,
                             "performance": dict(diagnostics or {})})
            except Exception:  # noqa: BLE001 进度回调绝不能拖累抽取
                pass
    if not X:
        return np.empty((0, dim), np.float32), np.array([], int), []
    return np.vstack(X).astype(np.float32), np.array(y, int), names


def train_video_head(clips_dir, training_dir, store, registry, models_dir,
                     extractor=None, dim: int = FEATURE_DIM, epochs: int = 300,
                     val_ratio: float = 0.25, seed: int = 0, created_ts: float = 0.0,
                     rebuild: bool = False, progress_cb=None) -> dict:
    """训头并登记版本。数据不够 raise ValueError。返回 {version, top1, counts}。

    只在人工训练分区上拟合，固定验证分区不参与拟合；rebuild 额外强制重抽特征。
    progress_cb（可选）：依次报 `preparing`（加载/下载 s3d）→ `extracting`（逐段，由 gather_dataset 发）
    → `training`（拟合小头）。回调抛异常会被吞掉。
    """
    def _emit(info: dict) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(info)
        except Exception:  # noqa: BLE001 进度回调绝不能拖累训练
            pass

    started = time.perf_counter()
    labeled = _labeled_clips(store)
    initial = {"drinking": sum(v == 1 for _, v in labeled), "not_drinking": sum(v == 0 for _, v in labeled)}
    if min(initial.values()) < MIN_PER_CLASS:
        raise ValueError(f"人工标注样本不够：当前 {initial}，每类需 ≥{MIN_PER_CLASS}；尚未开始特征提取。")
    if any(len({sample_group(name) for name, label in labeled if label == c}) < 2 for c in (0, 1)):
        raise ValueError("每类需包含至少两个时段的人工标注，才能独立验证；尚未开始特征提取。")
    performance = {"cache_hits": 0, "extracted": 0, "skipped": 0}
    _emit({"phase": "preparing", "total": len(labeled)})
    extractor = extractor or S3DFeatureExtractor()
    X, y, names = gather_dataset(clips_dir, training_dir, store, extractor, dim,
                                 force=rebuild, progress_cb=progress_cb, diagnostics=performance)
    performance["feature_seconds"] = round(time.perf_counter() - started, 3)
    performance["device"] = getattr(extractor, "device", "custom")
    counts = {"drinking": int((y == 1).sum()), "not_drinking": int((y == 0).sum())}
    too_few = [c for c in ("drinking", "not_drinking") if counts[c] < MIN_PER_CLASS]
    if too_few:
        raise ValueError(f"标注样本不够：当前 {counts}，每类需 ≥{MIN_PER_CLASS}。")
    _emit({"phase": "training", "performance": dict(performance)})
    train_idx, val_idx = holdout_split(names, y, Path(training_dir) / "holdout.json", val_ratio, seed)
    manifest = {"train": [names[i] for i in train_idx], "validation": [names[i] for i in val_idx]}
    fingerprint = hashlib.sha256(json.dumps(list(zip(names, y.tolist()))).encode() + X.tobytes()).hexdigest()
    parameters = {"epochs": epochs, "seed": seed, "weight_decay": .01}
    for existing in registry.list():
        evidence = existing.get("evaluation") or {}
        if (not rebuild and evidence.get("dataset_fingerprint") == fingerprint
                and evidence.get("parameters") == parameters and evidence.get("manifest") == manifest
                and Path(existing["path"]).exists()):
            performance.update(fit_seconds=0.0, evaluation_seconds=0.0,
                               total_seconds=round(time.perf_counter() - started, 3))
            return {"version": existing["id"], "counts": counts, "reused": True,
                    "performance": performance, "validation_examples": evidence.get("validation_examples", []),
                    **{k: evidence[k] for k in ("top1", "drinking_recall", "drinking_precision",
                        "balanced_accuracy", "f1", "confusion", "val_counts", "naive_baseline", "release", "comparison")}}
    fit_started = time.perf_counter()
    head = DrinkingHead.fit(X[train_idx], y[train_idx], dim=dim, epochs=epochs, seed=seed)
    performance["fit_seconds"] = round(time.perf_counter() - fit_started, 3)
    evaluation_started = time.perf_counter()
    _emit({"phase": "evaluating", "performance": dict(performance)})
    metrics = classification_metrics(head, X[val_idx], y[val_idx])
    assessment = release_assessment(metrics)
    comparison = {"status": "no_active_model"}
    old = None
    incumbent = registry.get(registry.active_id())
    if incumbent:
        previous = (incumbent.get("evaluation") or {}).get("manifest")
        if previous is None:
            comparison = {"status": "unknown_training_history", "version": incumbent["id"]}
        elif {sample_group(n) for n in previous["train"]} & {sample_group(n) for n in manifest["validation"]}:
            comparison = {"status": "overlapping_training_data", "version": incumbent["id"]}
        else:
            try:
                old = DrinkingHead.load(incumbent["path"])
                old_metrics = classification_metrics(old, X[val_idx], y[val_idx])
                comparison = {"status": "compared", "version": incumbent["id"], "metrics": old_metrics}
                comparison["delta"] = {k: metrics[k] - old_metrics[k] for k in
                                       ("drinking_recall", "drinking_precision", "balanced_accuracy")}
                comparison["misses_reduced"] = old_metrics["confusion"]["fn"] - metrics["confusion"]["fn"]
                comparison["false_alarms_reduced"] = old_metrics["confusion"]["fp"] - metrics["confusion"]["fp"]
                if any(metrics[k] + .02 < old_metrics[k] for k in ("drinking_recall", "drinking_precision", "balanced_accuracy")):
                    assessment["reasons"].append("与当前模型相比，召回、精确率或平衡准确率退步超过 2 个百分点")
            except (OSError, ValueError, KeyError):
                old = None
                comparison = {"status": "model_unavailable", "version": incumbent["id"]}
        if comparison["status"] != "compared":
            # Unknown legacy history is reported, never presented as evidence of improvement.
            comparison["note"] = "无法无泄漏对比旧版；只能依据候选的独立验证结果判断"
    assessment["eligible"] = not assessment["reasons"]
    examples = []
    for i in val_idx:
        prediction, probability = head.predict(X[i])
        correct = bool(prediction) == bool(y[i])
        was_correct = bool(old.predict(X[i])[0]) == bool(y[i]) if old is not None else None
        if not correct or was_correct is False:
            examples.append({"clip": names[i], "label": bool(y[i]), "prediction": bool(prediction),
                             "probability": float(probability),
                             "outcome": "fixed" if correct else "regressed" if was_correct else "missed" if y[i] else "false_alarm"})
    examples.sort(key=lambda e: (e["outcome"] == "fixed", e["clip"]))
    performance["evaluation_seconds"] = round(time.perf_counter() - evaluation_started, 3)
    performance["total_seconds"] = round(time.perf_counter() - started, 3)
    evaluation = {**metrics, "manifest": manifest, "dataset_fingerprint": fingerprint,
                  "policy": "human-grouped-holdout-v1", "parameters": parameters,
                  "comparison": comparison, "release": assessment,
                  "performance": performance, "validation_examples": examples[:12]}
    models_dir = Path(models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    head_path = models_dir / f"videohead_{int(created_ts)}_{uuid.uuid4().hex[:8]}.npz"
    head.save(head_path)
    entry = registry.add(path=head_path, top1=metrics["top1"], image_counts=counts,
                         label_counts=counts, base="s3d+head", epochs=epochs,
                         imgsz=224, created_ts=created_ts, evaluation=evaluation)
    return {"version": entry["id"], "counts": counts, **metrics,
            "release": assessment, "comparison": comparison,
            "performance": performance, "validation_examples": examples[:12]}


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
        self._extractor = extractor or S3DFeatureExtractor()
        self._dim = dim
        self._epochs = epochs
        self._lock = threading.Lock()
        self._state = "idle"   # idle | running | done | error
        self._detail = ""
        self._result: dict | None = None
        self._rebuild = False  # 本次是否从头重建特征缓存
        self._phase = ""       # preparing（加载 s3d）| extracting（逐段抽特征）| training（拟合小头）
        self._done = 0
        self._total = 0
        self._started = None
        self._performance = {}

    def _on_progress(self, info: dict) -> None:
        phase = info.get("phase")
        with self._lock:
            if phase:
                self._phase = phase
            self._total = info.get("total", self._total)
            if "performance" in info:
                self._performance = info["performance"]
            if phase == "extracting":
                self._done = info.get("done", self._done)
                self._total = info.get("total", self._total)

    def status(self) -> dict:
        with self._lock:
            base = {"state": self._state, "detail": self._detail, "result": self._result}
            if self._state == "running":
                # Reserve the last portion for fitting/evaluation; do not report
                # completion while a candidate still needs validation.
                if self._phase == "extracting" and self._total:
                    progress = .9 * self._done / self._total
                elif self._phase in ("training", "evaluating"):
                    progress = .95
                else:
                    progress = 0.0
                base["phase"] = self._phase
                base["done"] = self._done
                base["total"] = self._total
                base["progress"] = progress
                base["elapsed_seconds"] = round(time.perf_counter() - self._started, 1) if self._started else 0
                base["performance"] = dict(self._performance)
        if self.registry is not None:
            base["models"] = self.registry.list()
            base["active"] = self.registry.active_id()
        return base

    def start(self, rebuild: bool = False) -> bool:
        with self._lock:
            if self._state == "running":
                return False
            self._state = "running"
            self._rebuild = rebuild
            self._detail = ("从头重建特征 + 训练中…（要为每段重抽 s3d 特征，更慢）" if rebuild
                            else "训练中…（首次要为每段抽 s3d 特征，可能要几分钟）")
            self._result = None
            self._phase = "preparing"
            self._done = 0
            self._total = 0
            self._started = time.perf_counter()
            self._performance = {}
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        try:
            res = train_video_head(
                self.clips_dir, self.training_dir, self.feedback, self.registry, self.models_dir,
                extractor=self._extractor, dim=self._dim, epochs=self._epochs, created_ts=time.time(),
                rebuild=self._rebuild, progress_cb=self._on_progress,
            )
            detail = (f"完成 {res['version']} · 喝水召回 {_pct(res['drinking_recall'])} "
                      f"精确 {_pct(res['drinking_precision'])}（top1 {_pct(res['top1'])}，"
                      f"多数类基线 {_pct(res['naive_baseline'])}；样本 👍{res['counts']['drinking']}"
                      f"/👎{res['counts']['not_drinking']}）。未自动生效。")
            if res.get("reused"):
                detail = "数据和参数未变化，复用已有版本。" + detail
            if not res["release"]["eligible"]:
                detail += "仅限影子模式：" + "；".join(res["release"]["reasons"])
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
