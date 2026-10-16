from types import SimpleNamespace
import json
import httpx
import numpy as np
import pytest
from catcam.ai_budget import AIBudget
from catcam.ai_service import resolve_settings, test_service as run_test


def config():
    return SimpleNamespace(ai_base_url='https://example.test/v1',ai_model='vision',
                           ai_api_key='test-secret',ai_daily_budget=1000,ai_min_interval_seconds=0)


def test_custom_budget_above_fifty_persists(tmp_path):
    settings=resolve_settings({'ai_daily_budget':1000},config())
    budget=AIBudget(tmp_path/'db',settings['limit'],0)
    for i in range(51):assert budget.reserve('vision',now=100+i)
    assert AIBudget(tmp_path/'db',1000,0).limit==1000


@pytest.mark.parametrize('limit',[0,-1,1.5,True,'12.5'])
def test_invalid_budget_rejected(limit):
    with pytest.raises(ValueError):resolve_settings({'ai_daily_budget':limit},config())


def test_connection_uses_unsaved_address_and_fetches_models(tmp_path):
    def handle(request):
        assert str(request.url)=='https://proxy.test/v1/models'
        assert request.headers['Authorization']=='Bearer new-key'
        return httpx.Response(200,json={'data':[{'id':'vision'}]})
    cfg=config()
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result=run_test({'ai_base_url':'https://proxy.test/v1','ai_api_key':'new-key'},cfg,None,tmp_path/'db',client)
    assert result['ok'] and result['model_available']
    assert cfg.ai_api_key=='test-secret' and cfg.ai_base_url=='https://example.test/v1'
    assert not (tmp_path/'db').exists()  # 模型列表不消耗推理预算。


@pytest.mark.parametrize('http_status',[200,502,429])
def test_vision_sends_image_and_charges_every_attempt(tmp_path,http_status):
    def handle(request):
        body=json.loads(request.content)
        assert body['messages'][0]['content'][1]['image_url']['url'].startswith('data:image/jpeg;base64,')
        return httpx.Response(http_status,json={'choices':[{'message':{'content':'看见一个水碗'}}]})
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result=run_test({'kind':'vision'},config(),np.full((50,60,3),100,np.uint8),tmp_path/'db',client)
    assert result['ok']==(http_status==200)
    assert AIBudget(tmp_path/'db',1000,0).status()['used']==1
    assert 'test-secret' not in json.dumps(result)


def test_malformed_vision_reply_charges_failed_attempt(tmp_path):
    with httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={'choices':[]}))) as client:
        result=run_test({'kind':'vision'},config(),np.zeros((30,30,3),np.uint8),tmp_path/'db',client)
    assert not result['ok']
    assert AIBudget(tmp_path/'db',1000,0).status()['requests']=={'failed':1}


def test_missing_camera_never_spends_budget(tmp_path):
    with pytest.raises(ValueError):run_test({'kind':'vision'},config(),None,tmp_path/'db')
    assert not (tmp_path/'db').exists()
