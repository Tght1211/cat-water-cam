import json
import time
import numpy as np
import pytest
from catcam.codex_video import CodexVideoAgent,validate_proposal
from catcam.config import Config
from catcam.feedback import FeedbackStore
from catcam.recorder import ClipRecorder


def build(tmp_path,runner):
    feedback=FeedbackStore(tmp_path/'db',tmp_path/'training')
    recorder=ClipRecorder(tmp_path/'clips',100,5)
    recorder.save_clip([np.zeros((48,64,3),np.uint8)]*10,1)
    cfg=Config(codex_base_url='http://127.0.0.1:1234/v1',codex_api_key='test-secret',codex_model='mock-vision')
    return CodexVideoAgent(cfg,tmp_path/'config.json',feedback,recorder.clips_dir,runner),feedback


def test_agent_outputs_only_proposal_and_excludes_unknown(tmp_path):
    captured=[]
    def runner(request):
        captured.append(request)
        return {'segments':[[.5,1.5]],'unknown':[[1,1.2]],'reason':'模拟动作区间'}
    agent,feedback=build(tmp_path,runner)
    ident=agent.start(['clip_1000.mp4'])['jobs'][0]
    for _ in range(200):
        job=agent.status(ident)['jobs'][0]
        if job['status'] not in ('queued','running'):break
        time.sleep(.01)
    assert job['status']=='ready'
    assert job['result']['segments']==[[.5,1],[1.2,1.5]]
    assert captured[0]['frames'][0][0]==0
    assert len(captured[0]['frames'])==4
    assert feedback.label_meta('clip_1000.mp4') is None
    assert 'test-secret' not in json.dumps(agent.status())


def test_agent_config_is_private_and_requires_explicit_valid_settings(tmp_path):
    agent,_=build(tmp_path,lambda request:{})
    with pytest.raises(ValueError):agent.configure({'base_url':'https://user:key@example.com/v1'})
    agent.configure({'base_url':'https://service.example/v1','model':'vision-model','api_key':'new-secret','daily_budget':20})
    assert agent.settings()['has_key'] and 'new-secret' not in json.dumps(agent.settings())
    assert json.loads(agent.config_path.read_text())['codex_api_key']=='new-secret'
    assert agent.config_path.stat().st_mode&0o777==0o600
    with pytest.raises(ValueError):agent.start(['../private.mp4'])


@pytest.mark.parametrize('result',[
    {'segments':[[-1,2]],'unknown':[],'reason':'无效'},
    {'segments':[[0,31]],'unknown':[],'reason':'无效'},
    {'segments':[[0,float('nan')]],'unknown':[],'reason':'无效'},
    {'segments':[],'unknown':[],'reason':1},
])
def test_agent_rejects_out_of_window_or_invalid_output(result):
    with pytest.raises(ValueError):validate_proposal(result,0,30)


def test_agent_api_requires_sending_intent_and_never_exposes_key(tmp_path):
    from fastapi.testclient import TestClient
    from catcam.web import create_app
    from catcam.stats import StatsStore
    agent,feedback=build(tmp_path,lambda request:{'segments':[],'unknown':[],'reason':'无明确动作'})
    recorder=ClipRecorder(agent.clips_dir,100,5)
    app=create_app(StatsStore(feedback.db_path),recorder,feedback,lambda:None,agent.clips_dir,codex_agent=agent)
    client=TestClient(app)
    response=client.get('/api/agent/settings')
    assert response.status_code==200 and response.json()['has_key']
    assert 'test-secret' not in response.text
    assert client.post('/api/agent/jobs',json={'clips':['clip_1000.mp4']}).status_code==400
    assert client.get('/api/agent/jobs/nonexistent').status_code==404
