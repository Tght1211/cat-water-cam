"""可分享的 S3D 模型包：权重、预处理与汇总评估；不包含录像、机位或本机路径。"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import time
import zipfile
from pathlib import Path

from catcam.videojudge import DrinkingHead

MAX_BUNDLE_BYTES = 128 * 1024 * 1024
BACKENDS = {'s3d-legacy': 1024, 's3d-temporal-v2': 2048, 's3d-finetuned-v1': 2048}
METRICS = ('top1', 'drinking_recall', 'drinking_precision', 'balanced_accuracy',
           'naive_baseline', 'confusion', 'counts', 'threshold')


def _validate_head(head, backend):
    import numpy as np
    if backend not in BACKENDS or head.dim != BACKENDS[backend]:
        raise ValueError('模型主干或特征维度不兼容')
    if (not 0 < head.threshold < 1 or not np.isfinite(head._b)
            or any(not np.isfinite(x).all() for x in (head._w, head._mean, head._std))
            or len(head._mean) != head.dim or len(head._std) != head.dim
            or (head._std <= 0).any()):
        raise ValueError('模型参数无效')


def export_bundle(entry):
    head = DrinkingHead.load(entry['path'])
    backend = head.metadata.get('backend', 's3d-legacy')
    _validate_head(head, backend)
    weights = head.metadata.get('weights_path')
    if backend == 's3d-finetuned-v1' and (not weights or not Path(weights).is_file()):
        raise ValueError('缺少微调主干权重，不能只导出分类头')
    # 分类头适配沿用官方 S3D 基座；已有缓存时一并打包，便于离线预载。
    from torchvision.models.video import S3D_Weights
    from torch.hub import get_dir
    base_url = S3D_Weights.DEFAULT.url
    if not weights:
        candidate = Path(get_dir())/'checkpoints'/base_url.rsplit('/', 1)[-1]
        weights = candidate if candidate.exists() else None
    files = {}
    if weights:
        import torch
        from safetensors.torch import save, load_file
        state = load_file(str(weights)) if str(weights).endswith('.safetensors') else torch.load(weights, map_location='cpu', weights_only=True)
        files['backbone.safetensors'] = save({k:v.contiguous() for k,v in state.items()})
    metadata = {'backend': backend, 'roi': None,
                'weights_path': 'backbone.safetensors' if weights else None,
                'sampling': head.metadata.get('sampling', 'legacy-16frames' if backend=='s3d-legacy' else 'continuous-2s-3windows')}
    head.metadata = metadata
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'head.npz'
        head.save(path)
        files['head.npz'] = path.read_bytes()
    report = entry.get('report') or {}
    metrics = {key: report[key] for key in METRICS if key in report}
    manifest = {'format': 'catcam-model', 'format_version': 1, 'task': 'cat_drinking_action',
                'backend': backend, 'feature_dim': head.dim,
                'base_model': {'name': 'torchvision/s3d', 'weights': 'KINETICS400_V1', 'url': base_url},
                'preprocessing': {'color': 'RGB', 'frames_per_window':16, 'window_seconds':2,
                                  'windows':3, 'pooling':'mean+max' if backend!='s3d-legacy' else 'avgpool',
                                  'roi':'configured_by_each_household'},
                'metrics':metrics, 'source_version':entry['id'],
                'files':{name:hashlib.sha256(value).hexdigest() for name,value in files.items()}}
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        for name, value in files.items():archive.writestr(name, value)
    if output.tell() > MAX_BUNDLE_BYTES:
        raise ValueError('模型包超过 128 MB')
    return output.getvalue()


def import_bundle(data, registry, models_dir, roi=None):
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError('模型包超过 128 MB')
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            if (len(names)!=len(set(names)) or not {'manifest.json','head.npz'}.issubset(names)
                    or set(names)-{'manifest.json','head.npz','backbone.safetensors'}
                    or sum(i.file_size for i in archive.infolist()) > MAX_BUNDLE_BYTES
                    or archive.getinfo('manifest.json').file_size > 65536
                    or archive.getinfo('head.npz').file_size > 1024*1024):
                raise ValueError('模型包结构或大小无效')
            manifest = json.loads(archive.read('manifest.json'))
            if (manifest.get('format')!='catcam-model' or manifest.get('format_version')!=1
                    or manifest.get('task')!='cat_drinking_action' or manifest.get('backend') not in BACKENDS):
                raise ValueError('模型包格式或任务不兼容')
            expected = set(names)-{'manifest.json'}
            if set(manifest.get('files', {})) != expected:
                raise ValueError('模型文件清单不一致')
            files = {name:archive.read(name) for name in expected}
        for name, value in files.items():
            if hashlib.sha256(value).hexdigest()!=manifest['files'][name]:raise ValueError('模型文件校验失败')
        # NPZ 自身也是 ZIP；限制内部解压大小，避免嵌套压缩包耗尽内存。
        with zipfile.ZipFile(io.BytesIO(files['head.npz'])) as archive:
            if sum(i.file_size for i in archive.infolist()) > 1024*1024:
                raise ValueError('分类头参数过大')
        directory = Path(models_dir)/('import_'+hashlib.sha256(data).hexdigest()[:20])
        directory.mkdir(parents=True, exist_ok=True)
        head_path = directory/'head.npz'
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)/'head.npz';temp.write_bytes(files['head.npz'])
            head = DrinkingHead.load(temp)
        backend = manifest['backend'];_validate_head(head, backend)
        if backend=='s3d-finetuned-v1' and 'backbone.safetensors' not in files:
            raise ValueError('微调模型缺少主干权重')
        # 只使用校验后的本地文件路径；丢弃分享者的绝对路径和机位设置。
        weights = directory/'backbone.safetensors'
        if 'backbone.safetensors' in files:weights.write_bytes(files['backbone.safetensors'])
        head.metadata = {'backend': backend, 'roi': roi,
                         'sampling': 'legacy-16frames' if backend=='s3d-legacy' else 'continuous-2s-3windows',
                         'weights_path':str(weights.resolve()) if weights.exists() else None}
        head.save(head_path)
        metrics = {k:v for k,v in manifest.get('metrics', {}).items() if k in METRICS}
        report = {**metrics, 'backend':backend, 'imported':True,
                  'auto_label_eligible':False, 'warnings':['导入模型尚未在本家庭独立评估，请先测试和少量微调。']}
        return registry.add(path=head_path.resolve(), top1=metrics.get('top1',0), image_counts={},
                            label_counts=metrics.get('counts', {}), base='s3d+head', epochs=0,
                            imgsz=224, created_ts=time.time(), report=report)
    except (KeyError, TypeError, zipfile.BadZipFile, json.JSONDecodeError, OSError) as error:
        raise ValueError('模型包损坏或字段缺失') from error
