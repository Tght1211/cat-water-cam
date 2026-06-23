"""网页一键训练：用 👍/👎 标注帧训练「真喝水 / 没喝」分类器（YOLOv8-cls）。

数据来源：feedback.py 在标注时把每段视频抽帧写进 data/training/{drinking,not_drinking}/。
训练越多次（标注越多）模型越准，这就是「自我迭代」。本轮训练产物只保存+报告
准确率，暂不自动接入采集链路（避免半成品模型误杀候选）。
"""
from __future__ import annotations

import shutil
import threading
from pathlib import Path

CLASSES = ("drinking", "not_drinking")
MIN_PER_CLASS = 4  # 每类至少这么多张才值得训练


def count_images(training_dir: Path) -> dict[str, int]:
    training_dir = Path(training_dir)
    return {c: len(list((training_dir / c).glob("*.jpg"))) for c in CLASSES}


def prepare_dataset(training_dir: Path, dataset_dir: Path, val_ratio: float = 0.25) -> dict[str, int]:
    """把 data/training/{cls}/ 整理成 YOLO 分类要求的 train/ val/ 两份目录。"""
    training_dir = Path(training_dir)
    dataset_dir = Path(dataset_dir)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    counts: dict[str, int] = {}
    for c in CLASSES:
        imgs = sorted((training_dir / c).glob("*.jpg"))
        counts[c] = len(imgs)
        n_val = max(1, int(len(imgs) * val_ratio)) if len(imgs) > 1 else 0
        for split in ("train", "val"):
            (dataset_dir / split / c).mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(imgs):
            split = "val" if i < n_val else "train"
            shutil.copy(p, dataset_dir / split / c / p.name)
    return counts


def train_classifier(
    training_dir: Path, work_dir: Path, base_model: str, epochs: int, imgsz: int
) -> dict:
    """整理数据 → 训练 → 返回 {counts, top1, model}。数据不够则 raise ValueError。"""
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

    dataset_dir = work_dir / "dataset"
    prepare_dataset(training_dir, dataset_dir)

    from ultralytics import YOLO

    model = YOLO(base_model)
    results = model.train(
        data=str(dataset_dir),
        epochs=epochs,
        imgsz=imgsz,
        project=str(work_dir),
        name="cls",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    top1 = None
    rd = getattr(results, "results_dict", None)
    if isinstance(rd, dict):
        top1 = rd.get("metrics/accuracy_top1")
    best = work_dir / "cls" / "weights" / "best.pt"
    return {"counts": counts, "top1": top1, "model": str(best) if best.exists() else None}


class TrainingManager:
    """给网页用的训练状态机：一键开始、后台线程跑、随时查状态。"""

    def __init__(self, training_dir: Path, work_dir: Path, base_model: str, epochs: int, imgsz: int):
        self.training_dir = Path(training_dir)
        self.work_dir = Path(work_dir)
        self.base_model = base_model
        self.epochs = epochs
        self.imgsz = imgsz
        self._lock = threading.Lock()
        self._state = "idle"  # idle | running | done | error
        self._detail = ""
        self._result: dict | None = None

    def status(self) -> dict:
        with self._lock:
            return {"state": self._state, "detail": self._detail, "result": self._result,
                    "counts": count_images(self.training_dir)}

    def start(self) -> bool:
        with self._lock:
            if self._state == "running":
                return False
            self._state = "running"
            self._detail = "训练中…（首次会下载分类基座权重，CPU 上需几分钟）"
            self._result = None
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        try:
            res = train_classifier(
                self.training_dir, self.work_dir, self.base_model, self.epochs, self.imgsz
            )
            acc = res.get("top1")
            acc_txt = f"{acc:.1%}" if isinstance(acc, (int, float)) else "—"
            with self._lock:
                self._state = "done"
                self._detail = f"训练完成，验证集准确率 {acc_txt}"
                self._result = res
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._state = "error"
                self._detail = str(e)
                self._result = None
