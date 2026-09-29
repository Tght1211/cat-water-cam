"""Persistent, session-grouped human holdout and conservative release evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sample_group(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("clip_") and stem[5:].isdigit():
        # Recorder names contain epoch milliseconds. Nearby recordings share a split.
        return f"half-hour:{int(stem[5:]) // 1_800_000}"
    return name


def holdout_split(names, y, path, ratio=0.25, seed=0):
    """Never move an existing group between training and validation, even after relabeling."""
    if not 0 < ratio < 1:
        raise ValueError("val_ratio 必须在 0 和 1 之间")
    path = Path(path)
    groups = [sample_group(n) for n in names]
    unique = sorted(set(groups))
    if path.exists():
        assignments = json.loads(path.read_text())["groups"]
        for group in unique:
            if group not in assignments:
                fraction = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 2**32
                assignments[group] = "val" if fraction < ratio else "train"
    else:
        # Choose a reproducible grouped, approximately stratified initial split.
        rng = np.random.default_rng(seed)
        best = None
        nval = max(1, round(len(unique) * ratio))
        for _ in range(256):
            selected = set(rng.permutation(unique)[:nval])
            mask = np.array([g in selected for g in groups])
            if any(np.sum(y[mask] == c) < 1 or np.sum(y[~mask] == c) < 2 for c in (0, 1)):
                continue
            score = sum(abs(float(np.mean(mask[y == c])) - ratio) for c in (0, 1))
            if best is None or score < best[0]:
                best = (score, selected)
        if best is None:
            raise ValueError("无法分出独立验证集：请跨时段标注更多喝水和没喝视频，每类训练至少 2 段、验证至少 1 段。")
        assignments = {g: "val" if g in best[1] else "train" for g in unique}
    mask = np.array([assignments[g] == "val" for g in groups])
    if any(np.sum(y[mask] == c) < 1 or np.sum(y[~mask] == c) < 2 for c in (0, 1)):
        raise ValueError("固定验证集或训练集缺少某类样本，请补充跨时段人工标注；不会把旧训练样本移入验证集。")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"schema": 1, "groups": assignments}, ensure_ascii=False))
    tmp.replace(path)
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def classification_metrics(head, X, y):
    preds = np.array([head.predict(x)[0] for x in X], dtype=int)
    tp = int(np.sum((preds == 1) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {"top1": (tp + tn) / len(y), "drinking_recall": recall,
            "drinking_precision": precision, "balanced_accuracy": (recall + specificity) / 2,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "val_counts": {"drinking": tp + fn, "not_drinking": tn + fp},
            "naive_baseline": max(tp + fn, tn + fp) / len(y)}


def release_assessment(metrics):
    reasons = []
    if min(metrics["val_counts"].values()) < 10:
        reasons.append("人工验证集每类至少需要 10 段")
    for key, threshold, label in (("drinking_recall", .8, "喝水召回"),
                                  ("drinking_precision", .8, "喝水精确率"),
                                  ("balanced_accuracy", .8, "平衡准确率")):
        if metrics[key] < threshold:
            reasons.append(f"{label}低于 {threshold:.0%}")
    return {"eligible": not reasons, "reasons": reasons}
