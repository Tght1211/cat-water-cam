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
    def _pct(x):
        return f"{x:.1%}" if isinstance(x, float) else "—"
    print(f"完成：版本 {res['version']}，样本 {res['counts']}。")
    print(f"  留出集 top1={_pct(res['top1'])}（全猜「没喝」基线={_pct(res['naive_baseline'])}）"
          f" · 留出集分布 {res['val_counts']}")
    print(f"  ⚠️ 真正看这两个：喝水召回={_pct(res['drinking_recall'])}"
          f" 喝水精确={_pct(res['drinking_precision'])}——"
          f"召回低 = 漏判喝水。喝水样本太少时这俩才是真信号，top1 会被多数类带高。")
    print("未自动生效——评估满意后再接入裁判（后续一步）。")


if __name__ == "__main__":
    main()
