import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from catcam.model_bundle import export_bundle, import_bundle
from catcam.models import ModelRegistry
from catcam.videojudge import DrinkingHead


def bundle_fixture(tmp_path):
    weights=tmp_path/'private-house-backbone.safetensors'
    save_file({'example':torch.ones(2)},str(weights))
    head=DrinkingHead(np.ones(2048),.25,np.zeros(2048),np.ones(2048),metadata={
        'backend':'s3d-finetuned-v1','weights_path':str(weights),'roi':[.1,.2,.3,.4]})
    head.save(tmp_path/'head.npz')
    entry={'id':'v2','path':str(tmp_path/'head.npz'),'report':{
        'top1':.9,'counts':{'drinking':12,'not_drinking':30},
        'split':{'train':['private-video.mp4']},'roi':[.1,.2,.3,.4],
        'warnings':['/private/home/recording.mp4']}}
    return export_bundle(entry),head


def test_share_roundtrip_keeps_predictions_and_removes_private_scene(tmp_path):
    data,original=bundle_fixture(tmp_path)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert set(archive.namelist())=={'manifest.json','head.npz','backbone.safetensors'}
        manifest=json.loads(archive.read('manifest.json'))
        assert 'private' not in json.dumps(manifest)
        assert 'split' not in manifest['metrics']
    registry=ModelRegistry(tmp_path/'receiver'/'registry.json')
    entry=import_bundle(data,registry,tmp_path/'receiver',roi=[.2,.3,.8,.9])
    restored=DrinkingHead.load(entry['path'])
    assert restored.metadata['roi']==[.2,.3,.8,.9]
    assert Path(restored.metadata['weights_path']).parent==Path(entry['path']).parent
    assert restored.predict_proba(np.ones(2048)*.01)==original.predict_proba(np.ones(2048)*.01)
    assert registry.active_id() is None
    assert entry['report']['auto_label_eligible'] is False


def test_import_rejects_tampering_and_extra_paths(tmp_path):
    data,_=bundle_fixture(tmp_path)
    registry=ModelRegistry(tmp_path/'received'/'registry.json')
    for extra in ('head.npz','../escape'):
        result=io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(data)) as source,zipfile.ZipFile(result,'w') as dest:
            for name in source.namelist():dest.writestr(name,b'changed' if extra==name else source.read(name))
            if extra.startswith('..'):dest.writestr(extra,b'bad')
        with pytest.raises(ValueError):import_bundle(result.getvalue(),registry,tmp_path/'received')
    assert registry.list()==[]
    assert not (tmp_path/'escape').exists()
