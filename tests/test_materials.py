import io
import json
import zipfile
import numpy as np
import pytest
from fastapi.testclient import TestClient
from catcam.feedback import FeedbackStore
from catcam.recorder import ClipRecorder
from catcam.stats import StatsStore
from catcam.web import create_app
from catcam.materials import MaterialWorkspace,skill_archive


def setup(tmp_path):
    db=tmp_path/'db';feedback=FeedbackStore(db,tmp_path/'training');stats=StatsStore(db)
    recorder=ClipRecorder(tmp_path/'clips',100,5)
    for timestamp in (1,2,3):recorder.save_clip([np.zeros((48,64,3),np.uint8)]*10,timestamp)
    feedback.label_clip(recorder.clips_dir/'clip_1000.mp4',True)
    feedback.label_clip(recorder.clips_dir/'clip_2000.mp4',False)
    app=create_app(stats,recorder,feedback,lambda:None,recorder.clips_dir)
    return TestClient(app),feedback,recorder


def test_skill_accepts_only_confirmed_drinking_without_sdk(tmp_path):
    client,feedback,recorder=setup(tmp_path)
    result=client.get('/api/materials').json()
    assert [row['clip'] for row in result['clips']]==['clip_1000.mp4']
    assert client.get('/api/materials/clip_2000.mp4').status_code==400
    assert client.get('/api/materials/clip_3000.mp4').status_code==400
    assert client.get('/api/materials/clip_1000.mp4').status_code==200
    assert client.get('/api/materials?limit=0').status_code==422


def test_skill_proposals_are_idempotent_and_leave_labels_untouched(tmp_path):
    client,feedback,_=setup(tmp_path)
    metadata=client.get('/api/materials/clip_1000.mp4').json()
    before=feedback.clip_metadata()['meta']['clip_1000.mp4']
    body={'clip':metadata['clip'],'fingerprint':metadata['fingerprint'],'coverage':[[0,1]],
          'segments':[[.2,.9]],'unknown':[[.4,.5]],'reason':'示例','model':'外部测试Agent'}
    first=client.post('/api/materials/proposals',json=body)
    assert first.status_code==200
    result=first.json()
    assert result['result']['segments']==[[.2,.4],[.5,.9]]
    assert result['result']['unknown']==[[.4,.5],[1,2]]
    assert client.post('/api/materials/proposals',json=body).json()['id']==result['id']
    assert len(client.get('/api/materials/proposals').json()['jobs'])==1
    assert client.get('/api/agent/jobs/'+result['id']).json()['result']['source']=='external_agent_proposal'
    assert feedback.clip_metadata()['meta']['clip_1000.mp4']==before


def test_revoked_or_changed_video_cannot_submit(tmp_path):
    client,feedback,recorder=setup(tmp_path)
    meta=client.get('/api/materials/clip_1000.mp4').json()
    body={**meta,'coverage':[[0,1]],'segments':[[.1,.5]],'reason':'示例'}
    path=recorder.clips_dir/'clip_1000.mp4'
    path.touch()
    assert client.post('/api/materials/proposals',json=body).status_code==400
    body['fingerprint']=client.get('/api/materials/clip_1000.mp4').json()['fingerprint']
    feedback.label_clip(path,False)
    assert client.post('/api/materials/proposals',json=body).status_code==400
    assert not client.get('/api/materials/proposals').json()['jobs']


def test_skill_package_is_self_contained(tmp_path):
    archive=zipfile.ZipFile(io.BytesIO(skill_archive()))
    assert 'cat-water-cam/SKILL.md' in archive.namelist()
    assert 'cat-water-cam/scripts/catcam_materials.py' in archive.namelist()
    assert 'cat-water-cam/references/proposals.md' in archive.namelist()
    assert not any(name.endswith(('.mp4','.db')) or 'config.json' in name for name in archive.namelist())
