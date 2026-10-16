import numpy as np
import pytest
from fastapi.testclient import TestClient
from catcam.access import hash_password,verify_password,COOKIE,SESSION_SECONDS
from catcam.feedback import FeedbackStore
from catcam.recorder import ClipRecorder
from catcam.stats import StatsStore
from catcam.web import create_app


@pytest.fixture
def client(tmp_path):
    db=tmp_path/'db';stats=StatsStore(db);feedback=FeedbackStore(db,tmp_path/'training')
    recorder=ClipRecorder(tmp_path/'clips',10,5)
    app=create_app(stats,recorder,feedback,lambda:np.zeros((24,32,3),np.uint8),recorder.clips_dir,
                   web_password_hash=hash_password('test-password'),web_session_secret='test-session-secret')
    return TestClient(app)


@pytest.mark.parametrize('path',['/api/clips','/api/setup','/snapshot.jpg','/stream.mjpg','/clips/example.mp4','/docs','/static/app.js'])
def test_no_unauthenticated_access_to_data_or_management(client,path):
    assert client.get(path).status_code==401


def test_password_login_and_protected_preview(client):
    assert client.get('/',follow_redirects=False).headers['location']=='/login'
    assert '访问密码' in client.get('/login').text
    assert client.post('/api/auth/login',json={'password':'wrong'}).status_code==401
    login=client.post('/api/auth/login',json={'password':'test-password'})
    assert login.status_code==200
    assert 'httponly' in login.headers['set-cookie'].lower()
    assert 'samesite=strict' in login.headers['set-cookie'].lower()
    assert client.get('/').status_code==200
    assert client.get('/snapshot.jpg').status_code==200


def test_logout_rejects_reuse_of_the_old_cookie(client):
    client.post('/api/auth/login',json={'password':'test-password'})
    token=client.cookies[COOKIE]
    assert client.post('/api/auth/logout').status_code==200
    assert client.get('/api/clips').status_code==401
    client.cookies.set(COOKIE,token)
    assert client.get('/api/clips').status_code==401


def test_expired_or_modified_cookie_is_rejected(client,monkeypatch):
    import catcam.access as access
    client.post('/api/auth/login',json={'password':'test-password'})
    token=client.cookies[COOKIE];issued=int(token.split('.')[0])
    client.cookies.clear();client.cookies.set(COOKIE,token[:-1]+('0' if token[-1]!='0' else '1'))
    assert client.get('/api/clips').status_code==401
    client.cookies.clear();client.cookies.set(COOKIE,token)
    monkeypatch.setattr(access.time,'time',lambda:issued+SESSION_SECONDS+1)
    assert client.get('/api/clips').status_code==401


def test_cross_site_mutation_is_rejected(client):
    client.post('/api/auth/login',json={'password':'test-password'})
    assert client.post('/api/auth/logout',headers={'Origin':'https://other.example'}).status_code==403
    assert client.get('/api/clips').status_code==200


def test_wrong_password_attempts_are_limited(client):
    for _ in range(10):assert client.post('/api/auth/login',json={'password':'wrong'}).status_code==401
    assert client.post('/api/auth/login',json={'password':'wrong'}).status_code==429


def test_password_hash_is_salted_and_never_contains_plaintext():
    password='example-private-password'
    encoded=hash_password(password)
    assert password not in encoded and encoded!=hash_password(password)
    assert verify_password(password,encoded)
    assert not verify_password('wrong',encoded)


def password_client(save):
    from fastapi import FastAPI
    from catcam.access import install_access
    app=FastAPI()
    install_access(app,hash_password('test-password'),'session-secret',save)
    @app.get('/private')
    def private():return {'ok':True}
    return TestClient(app)


def test_change_password_saves_hash_revokes_old_sessions_and_survives_restart():
    saved=[];client=password_client(saved.append)
    client.post('/api/auth/login',json={'password':'test-password'})
    old=client.cookies[COOKIE]
    result=client.post('/api/auth/password',json={'current_password':'test-password','new_password':'new-password','confirm_password':'new-password'})
    assert result.status_code==200
    assert len(saved)==1 and verify_password('new-password',saved[0])
    assert 'new-password' not in saved[0]
    assert client.get('/private').status_code==200
    client.cookies.clear();client.cookies.set(COOKIE,old)
    assert client.get('/private').status_code==401
    assert client.post('/api/auth/login',json={'password':'test-password'}).status_code==401
    assert client.post('/api/auth/login',json={'password':'new-password'}).status_code==200
    from fastapi import FastAPI
    from catcam.access import install_access
    app=FastAPI();install_access(app,saved[0],'session-secret',saved.append)
    assert TestClient(app).post('/api/auth/login',json={'password':'new-password'}).status_code==200


@pytest.mark.parametrize('current,new,confirm,status',[('wrong','new-password','new-password',400),('test-password','short','short',400),('test-password','new-password','different',400)])
def test_invalid_password_change_does_not_save(current,new,confirm,status):
    saved=[];client=password_client(saved.append)
    client.post('/api/auth/login',json={'password':'test-password'})
    assert client.post('/api/auth/password',json={'current_password':current,'new_password':new,'confirm_password':confirm}).status_code==status
    assert not saved
    assert client.post('/api/auth/login',json={'password':'test-password'}).status_code==200


def test_password_save_failure_preserves_current_password_and_session():
    def fail(encoded):raise OSError('private error')
    client=password_client(fail);client.post('/api/auth/login',json={'password':'test-password'})
    response=client.post('/api/auth/password',json={'current_password':'test-password','new_password':'new-password','confirm_password':'new-password'})
    assert response.status_code==500 and 'private error' not in response.text
    assert client.get('/private').status_code==200
    assert client.post('/api/auth/login',json={'password':'test-password'}).status_code==200


def test_password_change_requires_login():
    saved=[];client=password_client(saved.append)
    assert client.post('/api/auth/password',json={'current_password':'test-password','new_password':'new-password','confirm_password':'new-password'}).status_code==401
    assert not saved
