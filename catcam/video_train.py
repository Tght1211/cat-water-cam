"""离线训练本地视频模型：从已积累的标注训一个 s3d+小头，登记成版本（不自动生效）。

用法：.venv/bin/python -m catcam.video_train
复用在线识别与训练的特征缓存。需先积累人工标注（每类 ≥4）。
"""
from __future__ import annotations

import time

from catcam.config import load_config
from catcam.feedback import FeedbackStore
from catcam.models import ModelRegistry
from catcam.video_trainer import train_video_head
from catcam.videojudge import S3DFeatureExtractor


def main(config_path: str = "config.json") -> None:
    cfg = load_config(config_path)
    store = FeedbackStore(cfg.db_path, cfg.training_dir)
    registry = ModelRegistry(cfg.models_dir / "registry.json")
    print("开始训练本地视频模型（s3d 冻结特征 + logistic 头）…")
    try:
        res = train_video_head(
            cfg.clips_dir, cfg.training_dir, store, registry, cfg.models_dir,
            created_ts=time.time(),
            extractor=S3DFeatureExtractor(cfg.video_device),
        )
    except ValueError as e:
        print(f"训练未开始：{e}")
        return
    def _pct(x):
        return f"{x:.1%}" if isinstance(x, float) else "—"
    print(f"完成：版本 {res['version']}，样本 {res['counts']}。")
    print(f"  留出集 top1={_pct(res['top1'])}（多数类基线={_pct(res['naive_baseline'])}）"
          f" · 留出集分布 {res['val_counts']}")
    print(f"  ⚠️ 真正看这两个：喝水召回={_pct(res['drinking_recall'])}"
          f" 喝水精确={_pct(res['drinking_precision'])}——"
          f"召回低 = 漏判喝水。喝水样本太少时这俩才是真信号，top1 会被多数类带高。")
    c = res["confusion"]
    print(f"  漏掉喝水 {c['fn']} 段，误报喝水 {c['fp']} 段。")
    p = res["performance"]
    print(f"  复用 {p['cache_hits']} 段 / 新处理 {p['extracted']} 段 / 跳过 {p['skipped']} 段；"
          f"视频处理 {p['feature_seconds']}s，拟合 {p['fit_seconds']}s，评估 {p['evaluation_seconds']}s。")
    if res.get("reused"):
        print("数据和参数未变化，复用已有版本。")
    print("未自动生效——请在网页模型版本中启用影子观察；历史预测不会自动更新。")


if __name__ == "__main__":
    main()
