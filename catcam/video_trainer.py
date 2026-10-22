"""离线训练本地视频小头：缓存 s3d 特征 + 当前标签 → 训 logistic 头 → 登记 registry 版本。

不改运行中的 app。训练数据来自 VLM/人工已写进 labels 的标注（is_drinking）。
"""
from __future__ import annotations

import io
import json
import hashlib
from catcam.database import connect
from datetime import datetime
import os
import threading
import time
from pathlib import Path

import numpy as np

from catcam.videojudge import DrinkingHead, S3DFeatureExtractor, read_clip_frames, FEATURE_DIM, extractor_for_backend

MIN_PER_CLASS = 4   # 每类至少这么多段才值得训


def training_clip_path(clips_dir, training_dir, name):
    original=Path(clips_dir)/name
    return original if original.exists() else Path(training_dir)/'videos'/name


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


def extract_and_cache(clip_path, training_dir, extractor, dim: int = FEATURE_DIM, force: bool = False):
    """取一段的特征：命中缓存直接读，否则抽帧→提取→存缓存。抽帧空返回 None。

    缓存损坏/截断（上次写一半崩了）当作未命中：重抽并覆盖（自愈）。
    force=True（从头重建）：忽略缓存、强制重抽并覆盖——但 mp4 已被裁掉时仍退回缓存（缓存是唯一副本）。
    """
    clip_path = Path(clip_path)
    backend = getattr(extractor, 'cache_key', getattr(extractor, 'backend', 's3d-legacy'))
    cache = feature_cache_path(training_dir, clip_path.name)
    if backend != 's3d-legacy':
        cache = Path(training_dir) / 'features' / backend / cache.name
    signature = cache.with_suffix('.json')
    if clip_path.exists() and signature.exists():
        stat = clip_path.stat()
        try:
            info = json.loads(signature.read_text())
        except (ValueError, OSError):
            info = {}
        force = force or info != {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    def valid_cache():
        feat = np.asarray(_load_npy(cache), np.float32).reshape(-1)
        if len(feat) != dim or not np.isfinite(feat).all():
            raise ValueError('特征维度不匹配或包含非法值')
        return feat
    if not force and cache.exists():
        try:
            return valid_cache()
        except Exception:  # noqa: BLE001 —— 损坏缓存不致命：重抽覆盖
            pass
    feat = None
    if clip_path.exists():
        if hasattr(extractor, 'extract_clip'):
            feat = extractor.extract_clip(clip_path)
        else:
            frames = read_clip_frames(clip_path)
            if frames:
                feat = extractor.extract(frames)
    if feat is None:
        # mp4 不在了（多半被 max_clips 裁掉）：force 也只能退回缓存
        if cache.exists():
            try:
                return valid_cache()
            except Exception:  # noqa: BLE001
                return None
        return None
    feat = np.asarray(feat, np.float32).reshape(-1)
    if len(feat) != dim or not np.isfinite(feat).all():
        raise ValueError(f'{clip_path.name} 特征无效，请重建缓存')
    _save_npy(cache, feat)
    stat = clip_path.stat()
    _atomic_write_bytes(signature, json.dumps({'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}).encode())
    return feat


def _labeled_clips(store) -> list[tuple[str, int]]:
    """从 labels 表取所有 (clip_name, is_drinking)；排除 source='local'（机器自身判定，不当训练真值）。"""
    with connect(store.db_path) as conn:
        rows = conn.execute(
            "SELECT clip_name, COALESCE(action_drinking,is_drinking) FROM labels WHERE source IS NULL OR source IN ('human', 'ai') ORDER BY clip_name"
        ).fetchall()
    return [(name, int(v)) for name, v in rows]


def gather_dataset(clips_dir, training_dir, store, extractor, dim: int = FEATURE_DIM,
                   force: bool = False, progress_cb=None, require_video=False):
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
        path = training_clip_path(clips_dir, training_dir, name)
        feat = (extract_and_cache(path, training_dir, extractor, dim, force=force)
                if not require_video or path.is_file() else None)
        if feat is not None:
            X.append(feat); y.append(label); names.append(name)
        if progress_cb is not None:
            try:
                progress_cb({"phase": "extracting", "done": i + 1, "total": total})
            except Exception:  # noqa: BLE001 进度回调绝不能拖累抽取
                pass
    if not X:
        return np.empty((0, dim), np.float32), np.array([], int), []
    return np.vstack(X).astype(np.float32), np.array(y, int), names


def train_video_head(clips_dir, training_dir, store, registry, models_dir,
                     extractor=None, dim: int = FEATURE_DIM, epochs: int = 300,
                     val_ratio: float = 0.25, seed: int = 0, created_ts: float = 0.0,
                     rebuild: bool = False, progress_cb=None, backend='s3d-temporal-v2', roi=None) -> dict:
    """训练分类头并登记包含独立评估结果的版本。

    全部标注参与数据划分；只用训练集拟合，验证集调阈值，测试集独立评估。
    rebuild 仅额外强制重抽特征。val_ratio 同时作为验证和测试比例，默认约50/25/25。
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

    _emit({"phase": "preparing"})           # 构造提取器：首次会下载 ~30MB s3d 权重，最慢的「无反馈」段
    finetune = backend == 's3d-finetuned-v1' and extractor is None
    extractor = extractor or extractor_for_backend('s3d-temporal-v2' if finetune else backend, roi)
    dim = getattr(extractor, 'dim', dim)
    backend = getattr(extractor, 'backend', 's3d-legacy')
    with connect(store.db_path) as conn:
        snapshot = conn.execute("SELECT clip_name, COALESCE(action_drinking,is_drinking), ts FROM labels WHERE source IS NULL OR source IN ('human','ai')").fetchall()
    X, y, names = gather_dataset(clips_dir, training_dir, store, extractor, dim,
                                 force=rebuild, progress_cb=progress_cb, require_video=finetune)
    counts = {"drinking": int((y == 1).sum()), "not_drinking": int((y == 0).sum())}
    too_few = [c for c in ("drinking", "not_drinking") if counts[c] < MIN_PER_CLASS]
    if too_few:
        raise ValueError(f"标注样本不够：当前 {counts}，每类需 ≥{MIN_PER_CLASS}。")
    _emit({"phase": "training"})             # 特征齐了，拟合 logistic 小头（很快）
    train_idx, val_idx, test_idx, split_method = persistent_split(y, names, registry, seed, val_ratio)
    if finetune:
        from catcam.backbone_train import fine_tune_s3d
        paths={name:training_clip_path(clips_dir,training_dir,name) for name in names}
        if not all(path.exists() for path in paths.values()):
            raise ValueError('主干微调需要原视频；缺失视频时请使用分类头训练或恢复录像')
        parent = registry.get(registry.active_id()) if registry.active_id() else None
        initial_weights = None
        if parent and parent.get('base') == 's3d+head':
            parent_head = DrinkingHead.load(parent['path'])
            if parent_head.metadata.get('backend', '').startswith('s3d-'):
                initial_weights = parent_head.metadata.get('weights_path')
        weights_path = fine_tune_s3d(clips_dir,names,y,train_idx,val_idx,models_dir,roi=roi,
                                    progress_cb=_emit,paths=paths,initial_weights=initial_weights)
        extractor = extractor_for_backend('s3d-finetuned-v1',roi,weights_path)
        backend = extractor.backend
        features=[]
        for i,name in enumerate(names):
            feat=extract_and_cache(paths[name],training_dir,extractor,dim=dim)
            if feat is None:
                raise ValueError(f'微调后无法读取{name}，本次训练未发布')
            features.append(feat)
            _emit({'phase':'extracting','done':i+1,'total':len(names)})
        X=np.vstack(features)
    head = DrinkingHead.fit(X[train_idx], y[train_idx], dim=dim, epochs=epochs, seed=seed)
    # 调阈值只用验证集；最终指标来自独立测试集，避免调参后自测虚高。
    probabilities = np.array([head.predict_proba(X[i]) for i in val_idx])
    candidates = np.linspace(0.1, 0.9, 17)
    scores = [classification_metrics(y[val_idx], probabilities >= t)['balanced_accuracy'] for t in candidates]
    best = max(range(len(candidates)), key=lambda i: (scores[i], -abs(candidates[i]-0.5)))
    head.threshold = float(candidates[best])
    head.metadata = {'backend': backend, 'roi': getattr(extractor, 'roi', None),
                     'weights_path':getattr(extractor,'weights_path',None),
                     'sampling': 'continuous-2s-3windows' if backend != 's3d-legacy' else 'legacy-16frames'}
    test_probs = np.array([head.predict_proba(X[i]) for i in test_idx])
    metrics = classification_metrics(y[test_idx], test_probs >= head.threshold)
    warnings = []
    hashes = {}
    for name in names:
        path = training_clip_path(clips_dir,training_dir,name)
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes.setdefault(digest, []).append(name)
    train_names = {names[i] for i in train_idx}
    held_names = {names[i] for i in np.concatenate([val_idx, test_idx])}
    if any(set(group) & train_names and set(group) & held_names for group in hashes.values()):
        warnings.append('发现训练集与留出集包含相同视频副本，当前指标可能虚高，请去除重复副本后重训。')
    if 'day_holdout' not in split_method:
        warnings.append('录像日期不足以按天隔离：相近会话可能造成验证偏乐观，请补充不同日期的数据。')
    if min(metrics['val_counts'].values()) < 10:
        warnings.append('独立测试集每类不足10段，当前评估仅供参考。')
    eligible = (not warnings and metrics['drinking_recall'] >= .8
                and (metrics['drinking_precision'] or 0) >= .8)
    used_names={names[i] for i in np.concatenate([train_idx,val_idx,test_idx])}
    used_snapshot = [r for r in snapshot if r[0] in used_names]
    fingerprint = hashlib.sha256(json.dumps(used_snapshot, sort_keys=True).encode()).hexdigest()
    report = {**metrics, 'counts': counts, 'backend': backend, 'threshold': head.threshold,
              'training_mode': 'partial_backbone_finetune' if finetune else 'frozen_backbone_head_only',
              'trainable_backbone': 'features[-3:]' if finetune else None,
              'parent_version': (parent or {}).get('id') if finetune else None,
              'split_method': split_method, 'warnings': warnings, 'auto_label_eligible': eligible,
              'dataset_fingerprint': fingerprint,
              'split': {'train': [names[i] for i in train_idx], 'validation': [names[i] for i in val_idx],
                        'test': [names[i] for i in test_idx],
                        'retired_holdout':[n for n in names if n not in used_names]},
              'skipped': len(snapshot)-len(names)}
    report['comparison'] = compare_active(registry, X, y, names, test_idx, head, extractor, clips_dir, training_dir)
    models_dir = Path(models_dir); models_dir.mkdir(parents=True, exist_ok=True)
    head_path = models_dir / f"videohead_{time.time_ns()}.npz"
    head.save(head_path)
    entry = registry.add(path=head_path, top1=metrics['top1'], image_counts=counts,
                         label_counts=counts, base="s3d+head", epochs=epochs,
                         imgsz=224, created_ts=created_ts, report=report)
    store.mark_trained(entry['id'], used_snapshot)
    return {'version': entry['id'], **report}


def persistent_split(y, names, registry, seed, ratio):
    """后续训练沿用最早一版的留出视频，避免每轮把考试题拿去训练。"""
    previous = [m for m in reversed(registry.list()) if m.get('report',{} ) and m['report'].get('split')]
    if previous:
        report = previous[-1]['report']
        holdout = set(report['split']['test'])
        seen = {n for m in previous for part in m['report']['split'].values() for n in part}
        reserved = {n for m in previous for part in ('test','retired_holdout') for n in m['report']['split'].get(part,[])}
        if 'day_holdout' not in report['split_method']:
            candidate_train,candidate_val,candidate_test,method = split_dataset(y,names,seed,ratio)
            test_names={names[i] for i in candidate_test}
            # 初期只有一天数据时先用分层留出；新独立日期足够后升级评估。
            if method=='day_holdout' and not test_names&seen and all((y[candidate_test]==c).sum()>=10 for c in (0,1)):
                train=np.array([i for i in candidate_train if names[i] not in reserved],int)
                val=np.array([i for i in candidate_val if names[i] not in reserved],int)
                if all(set(y[g])=={0,1} for g in (train,val)):
                    return train,val,candidate_test,'day_holdout'
        # 后续只用从未参与过任何训练/验证的新录像扩充小测试集。
        # 保留集已达每类10段后固定；避免最初每类1段的小留出集永远限制模型评估。
        index = {n:i for i,n in enumerate(names)}
        for label in (() if 'day_holdout' in report['split_method'] else (0,1)):
            current = sum(y[index[n]] == label for n in holdout if n in index)
            fresh = [n for n in names if n not in seen and y[index[n]] == label]
            holdout.update(fresh[::5][:max(0,10-current)])
        # 按日期划分时，整天都是保留集，新采集的同一天视频仍不得用于训练。
        if 'day_holdout' in report['split_method']:
            def day(n):
                return datetime.fromtimestamp(int(Path(n).stem.removeprefix('clip_'))/1000).date()
            days = {day(n) for n in holdout}
            holdout |= {n for n in names if day(n) in days}
            validation_days = {day(n) for n in report['split']['validation']}
            test = np.array([i for i,n in enumerate(names) if n in holdout],int)
            val = np.array([i for i,n in enumerate(names) if day(n) in validation_days and n not in holdout|reserved],int)
            train = np.array([i for i,n in enumerate(names) if day(n) not in validation_days|days and n not in reserved],int)
            if all(set(y[g]) == {0,1} for g in (train,val,test)):
                return train,val,test,'fixed_day_holdout'
            raise ValueError('按日期保留的训练、验证或测试集类别不足，请补充对应日期的喝水和没喝样本；不会退回随机划分。')
        test = np.array([i for i,n in enumerate(names) if n in holdout], int)
        rest = np.array([i for i,n in enumerate(names) if n not in holdout|reserved], int)
        if set(y[test]) == {0,1} and all((y[rest] == c).sum() >= 3 for c in (0,1)):
            rng = np.random.default_rng(seed)
            train, val = [], []
            for c in (0,1):
                indices = rng.permutation(rest[y[rest] == c])
                count = max(1, int(len(indices)*ratio/(1-ratio)))
                val.extend(indices[:count]); train.extend(indices[count:])
            return np.array(train), np.array(val), test, 'fixed_'+report['split_method'].removeprefix('fixed_')
        raise ValueError('原独立测试样本缺失或纠错后类别不足，请恢复测试录像并补充新的喝水/没喝样本；保留集不会被拿去训练。')
    return split_dataset(y, names, seed, ratio)


def compare_active(registry, X, y, names, indices, head, extractor, clips_dir, training_dir):
    entry = registry.get(registry.active_id()) if registry.active_id() else None
    if not entry or entry.get('base') != 's3d+head':
        return None
    previous_train = set(((entry.get('report') or {}).get('split') or {}).get('train', []))
    indices = np.array([i for i in indices if names[i] not in previous_train], int)
    if not len(indices) or set(y[indices]) != {0,1}:
        return {'version': entry['id'], 'message': '没有足够的共同独立测试数据，不能公平比较。'}
    try:
        old = DrinkingHead.load(entry['path'])
        same = old.metadata == head.metadata
        old_extractor = extractor if same else extractor_for_backend(old.metadata['backend'],old.metadata.get('roi'),old.metadata.get('weights_path'))
        old_features = [X[i] if same else extract_and_cache(training_clip_path(clips_dir,training_dir,names[i]),training_dir,old_extractor,old.dim) for i in indices]
        valid = [(i,f) for i,f in zip(indices,old_features) if f is not None]
        if not valid or set(y[[i for i,_ in valid]]) != {0,1}:
            return {'version':entry['id'], 'message':'旧模型的特征不可用，暂不能比较。'}
        actual = np.array([i for i,_ in valid])
        before = classification_metrics(y[actual],[old.predict(f)[0] for _,f in valid])
        after = classification_metrics(y[actual],[head.predict(X[i])[0] for i in actual])
        return {'version':entry['id'], 'test_names':[names[i] for i in actual],
                'before':before,'after':after,'delta_balanced_accuracy':after['balanced_accuracy']-before['balanced_accuracy'],
                'message':'新旧模型在同一批当前标签的留出视频上比较；结果可能改善，也可能退步。'}
    except Exception as e:
        return {'version':entry['id'],'message':f'旧模型比较未完成：{e}'}


def classification_metrics(y, predictions):
    y, predictions = np.asarray(y), np.asarray(predictions, int)
    tp = int(((y == 1) & (predictions == 1)).sum())
    tn = int(((y == 0) & (predictions == 0)).sum())
    fp = int(((y == 0) & (predictions == 1)).sum())
    fn = int(((y == 1) & (predictions == 0)).sum())
    recall = tp / (tp+fn) if tp+fn else None
    specificity = tn / (tn+fp) if tn+fp else None
    return {'top1': float((y == predictions).mean()),
            'drinking_recall': recall, 'drinking_precision': tp/(tp+fp) if tp+fp else None,
            'balanced_accuracy': ((recall or 0)+(specificity or 0))/2,
            'naive_baseline': float(max((y == 1).sum(), (y == 0).sum()) / len(y)),
            'val_counts': {'drinking': tp+fn, 'not_drinking': tn+fp},
            'confusion': {'true_drinking': tp, 'missed_drinking': fn, 'false_drinking': fp, 'true_not_drinking': tn}}


def split_dataset(y, names, seed=0, holdout_ratio=.25):
    """优先按拍摄日期隔离三份数据；否则按类别分层，并明确标记评估局限。"""
    if not 0 < holdout_ratio < .5:
        raise ValueError('验证与测试比例必须在0到0.5之间')
    rng = np.random.default_rng(seed)
    try:
        days = np.array([datetime.fromtimestamp(int(Path(n).stem.removeprefix('clip_'))/1000).strftime('%Y-%m-%d') for n in names])
        unique = sorted(set(days))
        # 连续日期分块，避免同一会话邻近片段泄露到不同集合。
        if len(unique) >= 3:
            a = max(1, int(len(unique)*(1-2*holdout_ratio))); b = max(a+1, int(len(unique)*(1-holdout_ratio)))
            groups = [np.where(np.isin(days, ds))[0] for ds in (unique[:a], unique[a:b], unique[b:])]
            if all(set(y[g]) == {0, 1} for g in groups):
                return *groups, 'day_holdout'
    except (ValueError, OverflowError, OSError):
        pass
    parts = [[], [], []]
    for label in (0, 1):
        idx = rng.permutation(np.where(y == label)[0])
        n = max(1, int(len(idx)*holdout_ratio))
        parts[0].extend(idx[2*n:]); parts[1].extend(idx[:n]); parts[2].extend(idx[n:2*n])
    return *(np.array(p, int) for p in parts), 'stratified_clip_holdout'


def dataset_diagnostics(clips_dir, training_dir, store):
    rows = _labeled_clips(store)
    usable = [(n, y) for n, y in rows if training_clip_path(clips_dir,training_dir,n).exists() or feature_cache_path(training_dir,n).exists()]
    videos = [(n,y) for n,y in rows if training_clip_path(clips_dir,training_dir,n).exists()]
    with connect(store.db_path) as conn:
        pending={r[0] for r in conn.execute('SELECT clip_name FROM labels WHERE trained_version IS NULL')}
    video_counts={'drinking':sum(y==1 for _,y in videos),'not_drinking':sum(y==0 for _,y in videos)}
    new_counts={'drinking':sum(y==1 and n in pending for n,y in videos),'not_drinking':sum(y==0 and n in pending for n,y in videos)}
    counts = {'drinking': sum(y == 1 for _, y in usable), 'not_drinking': sum(y == 0 for _, y in usable)}
    return {'labels': len(rows), 'available_clips': sum(training_clip_path(clips_dir,training_dir,n).exists() for n,_ in rows),
            'video_counts':video_counts,'new_video_counts':new_counts,'minimum_per_class':MIN_PER_CLASS,
            'legacy_usable': len(usable), 'missing_video': sum(not training_clip_path(clips_dir,training_dir,n).exists() for n,_ in rows),
            'counts': counts, 'ready': min(counts.values()) >= MIN_PER_CLASS,
            'advice': '请标注不同日期、白天与弱光的真实喝水和凑近没喝片段；缺失视频的旧缓存只能供旧特征版本使用。'}


def _pct(x) -> str:
    return f"{x:.0%}" if isinstance(x, float) else "—"


class VideoTrainingManager:
    """网页一键训练本地视频模型：后台线程跑 train_video_head，随时查状态。

    与单帧 TrainingManager 并存。extractor 可注入（测试塞假提取器）；None=用真 s3d。
    训完登记成 base=s3d+head 的版本（不自动生效）。
    """

    def __init__(self, clips_dir, training_dir, feedback, registry, models_dir,
                 extractor=None, dim: int = FEATURE_DIM, epochs: int = 300, roi=None, on_complete=None):
        self.clips_dir = clips_dir
        self.training_dir = training_dir
        self.feedback = feedback
        self.registry = registry
        self.models_dir = models_dir
        self._extractor = extractor
        self._dim = dim
        self._epochs = epochs
        self.roi = roi
        self.on_complete = on_complete
        self._auto_activate = False
        self._lock = threading.Lock()
        self._state = "idle"   # idle | running | done | error
        self._detail = ""
        self._result: dict | None = None
        self._backend = 's3d-temporal-v2'
        self._rebuild = False  # 本次是否从头重建特征缓存
        self._phase = ""       # preparing（加载 s3d）| extracting（逐段抽特征）| training（拟合小头）
        self._done = 0
        self._total = 0
        self._fine_progress = {}

    def _on_progress(self, info: dict) -> None:
        phase = info.get("phase")
        with self._lock:
            if phase:
                self._phase = phase
            if phase == 'finetuning':
                self._fine_progress = {k:info.get(k,0) for k in ('epoch','epochs','loss','val_loss','sample','samples')}
            if phase == "extracting":
                self._done = info.get("done", self._done)
                self._total = info.get("total", self._total)

    def status(self) -> dict:
        with self._lock:
            base = {"state": self._state, "detail": self._detail, "result": self._result}
            if self._state == "running":
                # 进度主要反映「逐段抽特征」（最耗时）：extracting=done/total；
                # preparing（下载 s3d）还没法估=0；training（小头）很快=1。
                if self._phase == "extracting" and self._total:
                    progress = self._done / self._total
                elif self._phase == "training":
                    progress = 1.0
                else:
                    progress = 0.0
                base["phase"] = self._phase
                base["done"] = self._done
                base["total"] = self._total
                base["progress"] = progress
                if self._phase == 'finetuning':
                    base.update(self._fine_progress)
                    fraction=self._fine_progress.get('sample',0)/max(1,self._fine_progress.get('samples',1))
                    base['progress'] = (self._fine_progress.get('epoch',0)+fraction)/self._fine_progress.get('epochs',1)
        if self.registry is not None:
            base["models"] = self.registry.list()
            base["active"] = self.registry.active_id()
        base['dataset'] = dataset_diagnostics(self.clips_dir, self.training_dir, self.feedback)
        return base

    def start(self, rebuild: bool = False, backend='s3d-temporal-v2', auto_activate=False) -> bool:
        if backend not in ('s3d-temporal-v2', 'videomae-temporal-v1', 's3d-legacy','s3d-finetuned-v1'):
            raise ValueError('不支持的训练主干')
        with self._lock:
            if self._state == "running":
                return False
            self._state = "running"
            self._rebuild = rebuild
            self._backend = backend
            self._auto_activate = auto_activate
            self._detail = ("从头重建特征 + 训练中…（要为每段重抽 s3d 特征，更慢）" if rebuild
                            else "训练中…（首次要为每段抽 s3d 特征，可能要几分钟）")
            self._result = None
            self._phase = "preparing"
            self._done = 0
            self._total = 0
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        try:
            res = train_video_head(
                self.clips_dir, self.training_dir, self.feedback, self.registry, self.models_dir,
                extractor=self._extractor, dim=self._dim, epochs=self._epochs, created_ts=time.time(),
                rebuild=self._rebuild, progress_cb=self._on_progress, backend=self._backend, roi=self.roi,
            )
            detail = (f"完成 {res['version']} · 喝水召回 {_pct(res['drinking_recall'])} "
                      f"精确 {_pct(res['drinking_precision'])}（top1 {_pct(res['top1'])}，"
                      f"全猜没喝基线 {_pct(res['naive_baseline'])}；样本 👍{res['counts']['drinking']}"
                      f"/👎{res['counts']['not_drinking']}）。未自动生效。")
            if self._auto_activate and self.on_complete:
                comparison=res.get('comparison') or {}
                regressed=comparison.get('delta_balanced_accuracy',0)<0
                if not res.get('auto_label_eligible'):
                    res['activation_note']='独立评估未达要求，模型已保存供测试，保留原模型。'
                elif regressed:
                    res['activation_note']='共同留出集表现退步，保留原模型；新版本可在模型版本中测试。'
                else:
                    try:
                        res['activation_note']=self.on_complete(self.registry.get(res['version']))
                    except Exception as e:
                        res['activation_note']=f'模型已保存，但启用未完成：{e}'
                detail=detail.replace('未自动生效。',res['activation_note'])
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
