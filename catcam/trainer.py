"""网页一键训练：用 👍/👎 标注帧训练「真喝水 / 没喝」分类器（YOLOv8-cls）。

数据来源：feedback.py 在标注时把每段视频抽帧写进 data/training/{drinking,not_drinking}/。
训练越多次（标注越多）模型越准，这就是「自我迭代」。本轮训练产物只保存+报告
准确率，暂不自动接入采集链路（避免半成品模型误杀候选）。
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional

CLASSES = ("drinking", "not_drinking")
MIN_PER_CLASS = 4  # 每类至少这么多张才值得训练


def training_progress(epochs_done: int, total_epochs: int, elapsed: float) -> dict:
    """按「已完成 epoch / 总 epoch」算进度，按已耗时线性外推剩余秒数（纯函数，可单测）。

    elapsed 应为「训练真正开始」起的耗时（不含基座权重下载/数据准备），否则 ETA 会偏大。
    epochs_done==0 或没耗时 → 还没法估，eta 返回 None。
    """
    if total_epochs <= 0:
        return {"progress": 0.0, "eta_seconds": None}
    progress = min(1.0, max(0.0, epochs_done / total_epochs))
    eta: Optional[float] = None
    if 0 < epochs_done < total_epochs and elapsed > 0:
        per_epoch = elapsed / epochs_done
        eta = per_epoch * (total_epochs - epochs_done)
    return {"progress": progress, "eta_seconds": eta}


def count_images(training_dir: Path) -> dict[str, int]:
    training_dir = Path(training_dir)
    return {c: len(list((training_dir / c).glob("*.jpg"))) for c in CLASSES}


def balance_target(train_counts: dict[str, int]) -> int:
    """类别平衡目标张数 = 各类训练张数的最小值（多数类降采样到它）。"""
    return min(train_counts.values()) if train_counts else 0


def prepare_dataset(training_dir: Path, dataset_dir: Path, val_ratio: float = 0.25, balance: bool = True) -> dict[str, int]:
    """把 data/training/{cls}/ 整理成 YOLO 分类要求的 train/ val/ 两份目录。

    balance=True：对 train 划分做类别平衡——多数类降采样到少数类张数，避免失衡训练
    把模型带歪（自动标注会持续堆负样本，不平衡会越来越严重）。val 保持真实分布。
    """
    training_dir = Path(training_dir)
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    counts: dict[str, int] = {}
    val_imgs: dict[str, list] = {}
    train_imgs: dict[str, list] = {}
    for c in CLASSES:
        imgs = sorted((training_dir / c).glob("*.jpg"))
        counts[c] = len(imgs)
        n_val = max(1, int(len(imgs) * val_ratio)) if len(imgs) > 1 else 0
        val_imgs[c] = imgs[:n_val]
        train_imgs[c] = imgs[n_val:]
        for split in ("train", "val"):
            (dataset_dir / split / c).mkdir(parents=True, exist_ok=True)
    if balance:
        target = balance_target({c: len(train_imgs[c]) for c in CLASSES})
        if target > 0:  # 某类 train 为 0 时 target=0，别把另一类也清空成空训练集
            for c in CLASSES:
                train_imgs[c] = train_imgs[c][:target]  # 降采样到最小类（确定性取前 target 张）
    for c in CLASSES:
        for p in val_imgs[c]:
            shutil.copy(p, dataset_dir / "val" / c / p.name)
        for p in train_imgs[c]:
            shutil.copy(p, dataset_dir / "train" / c / p.name)
    return counts


def train_classifier(
    training_dir: Path, work_dir: Path, base_model: str, epochs: int, imgsz: int,
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> dict:
    """整理数据 → 训练 → 返回 {counts, top1, model}。数据不够则 raise ValueError。

    progress_cb（可选）在训练过程中被回调，喂进度字典：
    `{"phase": "train_start", "total_epochs": N}` / `{"phase": "epoch", "epochs_done": k, "total_epochs": N}`。
    回调里抛异常会被吞掉，绝不影响训练本身。
    """
    training_dir = Path(training_dir)
    work_dir = Path(work_dir)
    counts = count_images(training_dir)
    too_few = [c for c in CLASSES if counts[c] < MIN_PER_CLASS]
    if too_few:
        need = "、".join(f"{c}≥{MIN_PER_CLASS}" for c in too_few)
        raise ValueError(
            f"标注样本不够：当前 {counts}，需要 {need}。"
            f"去网页上多点几段视频的 👍/👎 再训练。"
        )

    # 必须用绝对路径：传相对 project 时 ultralytics 会把它塞进 runs/classify/<project>/，
    # 害得我们按 work_dir/cls/... 找不到 best.pt。绝对路径则原样用作输出目录。
    save_root = Path(work_dir).resolve()
    dataset_dir = save_root / "dataset"
    prepare_dataset(training_dir, dataset_dir)

    from ultralytics import YOLO

    model = YOLO(base_model)

    def _emit(info: dict) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(info)
        except Exception:  # noqa: BLE001 进度回调绝不能拖累训练
            pass

    def _on_train_start(trainer):  # 基座权重已就绪、第一个 epoch 即将开始
        _emit({"phase": "train_start", "total_epochs": getattr(trainer, "epochs", epochs)})

    def _on_fit_epoch_end(trainer):  # 每个 epoch（train+val）跑完
        done = getattr(trainer, "epoch", 0) + 1  # trainer.epoch 是 0 起的当前 epoch 序号
        _emit({"phase": "epoch", "epochs_done": done,
               "total_epochs": getattr(trainer, "epochs", epochs)})

    model.add_callback("on_train_start", _on_train_start)
    model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
    results = model.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=imgsz,
        project=str(save_root),
        name="cls",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    top1 = None
    rd = getattr(results, "results_dict", None)
    if isinstance(rd, dict):
        top1 = rd.get("metrics/accuracy_top1")
    best = save_root / "cls" / "weights" / "best.pt"
    return {"counts": counts, "top1": top1, "model": str(best) if best.exists() else None}


class TrainingManager:
    """给网页用的训练状态机：一键开始、后台线程跑、随时查状态。

    训完产出一个**带版本号的模型**登记进 registry（不自动生效），并把当时所有标注标记为
    「已被该版本训练」，这样训练页能区分「已标注未训练 / 已标注已训练」、避免重复无效训练。
    """

    def __init__(
        self,
        training_dir: Path,
        work_dir: Path,
        base_model: str,
        epochs: int,
        imgsz: int,
        feedback=None,
        registry=None,
    ):
        self.training_dir = Path(training_dir)
        self.work_dir = Path(work_dir)
        self.base_model = base_model
        self.epochs = epochs
        self.imgsz = imgsz
        self.feedback = feedback
        self.registry = registry
        self._lock = threading.Lock()
        self._state = "idle"  # idle | running | done | error
        self._detail = ""
        self._result: dict | None = None
        self._phase = ""        # preparing（准备/下载权重）| training（跑 epoch）
        self._epoch = 0
        self._total_epochs = epochs
        self._train_started_ts: float | None = None  # 训练真正开始（用于 ETA，排除下载耗时）

    def _on_progress(self, info: dict) -> None:
        phase = info.get("phase")
        with self._lock:
            if phase == "train_start":
                self._phase = "training"
                self._train_started_ts = time.time()
                self._total_epochs = info.get("total_epochs") or self.epochs
            elif phase == "epoch":
                self._phase = "training"
                self._epoch = info.get("epochs_done", self._epoch)
                self._total_epochs = info.get("total_epochs") or self._total_epochs

    def status(self) -> dict:
        with self._lock:
            base = {"state": self._state, "detail": self._detail, "result": self._result,
                    "image_counts": count_images(self.training_dir)}
            if self._state == "running":
                elapsed = (time.time() - self._train_started_ts) if self._train_started_ts else 0.0
                prog = training_progress(self._epoch, self._total_epochs, elapsed)
                base["phase"] = self._phase
                base["epoch"] = self._epoch
                base["total_epochs"] = self._total_epochs
                base["progress"] = prog["progress"]
                base["eta_seconds"] = prog["eta_seconds"]
        if self.feedback is not None:
            base["label_states"] = self.feedback.label_states()
        if self.registry is not None:
            base["models"] = self.registry.list()
            base["active"] = self.registry.active_id()
        return base

    def start(self) -> bool:
        with self._lock:
            if self._state == "running":
                return False
            self._state = "running"
            self._detail = "训练中…（首次会下载分类基座权重，CPU 上需几分钟）"
            self._result = None
            self._phase = "preparing"
            self._epoch = 0
            self._total_epochs = self.epochs
            self._train_started_ts = None
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        try:
            label_snapshot = self.feedback.label_states() if self.feedback is not None else None
            res = train_classifier(
                self.training_dir, self.work_dir, self.base_model, self.epochs, self.imgsz,
                progress_cb=self._on_progress,
            )
            now = time.time()
            version = None
            if self.registry is not None and res.get("model"):
                # 把本次 best.pt 另存成带时间戳的版本文件，登记进 registry。
                src = Path(res["model"])
                versioned = self.work_dir / f"model_{int(now)}.pt"
                if src.exists():
                    shutil.copy(src, versioned)
                entry = self.registry.add(
                    path=versioned, top1=res.get("top1"), image_counts=res.get("counts"),
                    label_counts=label_snapshot, base=self.base_model,
                    epochs=self.epochs, imgsz=self.imgsz, created_ts=now,
                )
                version = entry["id"]
                if self.feedback is not None:
                    self.feedback.mark_trained(version)
            acc = res.get("top1")
            acc_txt = f"{acc:.1%}" if isinstance(acc, (int, float)) else "—"
            vtxt = f"{version} · " if version else ""
            with self._lock:
                self._state = "done"
                self._detail = f"训练完成（{vtxt}验证集准确率 {acc_txt}）。未自动生效，可在下方启用。"
                self._result = {**res, "version": version, "created_ts": now}
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._state = "error"
                self._detail = str(e)
                self._result = None
