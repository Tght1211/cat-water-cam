import cv2
import numpy as np
from fastapi.testclient import TestClient

from catcam.feedback import FeedbackStore
from catcam.recorder import ClipRecorder
from catcam.review_routing import ReviewRouter
from catcam.stats import StatsStore
from catcam.web import create_app


def build(tmp_path):
    stats=StatsStore(tmp_path/'db')
    feedback=FeedbackStore(tmp_path/'db',tmp_path/'training')
    recorder=ClipRecorder(tmp_path/'clips',20,30)
    return stats,feedback,recorder


def test_empty_recording_stays_in_replay_even_when_drinking_head_says_yes(tmp_path):
    stats,feedback,recorder=build(tmp_path)
    path=recorder.save_clip([np.zeros((48,64,3),np.uint8)]*10,1)
    stats.record_event(1,path.name,predicted=1,review_candidate=False)
    class NoCat:
        def detect_cats(self,frame):return []
    assert not ReviewRouter(NoCat(),stats,feedback).route(path)
    client=TestClient(create_app(stats,recorder,feedback,lambda:None,recorder.clips_dir))
    summary=client.get('/api/clips/summary').json()
    assert summary['clips']==[path.name]
    assert summary['pending_names']==[]
    assert summary['review_names']==[]
    assert client.post(f'/api/clips/{path.name}/review').status_code==200
    assert client.get('/api/clips/summary').json()['pending_names']==[path.name]


def test_cat_candidate_and_human_feedback_are_preserved(tmp_path):
    stats,feedback,recorder=build(tmp_path)
    path=recorder.save_clip([np.zeros((48,64,3),np.uint8)]*10,1)
    stats.record_event(1,path.name,review_candidate=False)
    class Cat:
        def detect_cats(self,frame):return [(0,0,5,5)]
    assert ReviewRouter(Cat(),stats,feedback).route(path)
    assert stats.review_candidates()[path.name]
    feedback.label_clip(path,False)
    class NoCat:
        def detect_cats(self,frame):raise AssertionError('Human feedback must be preserved')
    assert ReviewRouter(NoCat(),stats,feedback).route(path)


def test_optional_note_does_not_classify_and_survives_quick_judgment(tmp_path):
    stats,feedback,recorder=build(tmp_path)
    path=recorder.save_clip([np.zeros((48,64,3),np.uint8)]*10,1)
    stats.record_event(1,path.name)
    client=TestClient(create_app(stats,recorder,feedback,lambda:None,recorder.clips_dir))
    assert client.post(f'/api/clips/{path.name}/note',json={'reason':'只是路过'}).status_code==200
    assert feedback.get_label(path.name) is None
    response=client.post('/api/feedback',json={'clip':path.name,'is_drinking':False})
    assert response.status_code==200
    assert response.json()['meta']['is_drinking'] is False
    assert response.json()['meta']['reason']=='只是路过'
    assert client.post(f'/api/clips/{path.name}/note',json={'reason':''}).status_code==200
    assert feedback.get_label(path.name) is False


def test_cover_is_actual_middle_frame_not_first(tmp_path):
    stats,feedback,recorder=build(tmp_path)
    frames=[np.full((48,64,3),color,np.uint8) for color in [(0,0,255)]*4+[(0,255,0)]*4+[(255,0,0)]*4]
    path=recorder.save_clip(frames,1)
    client=TestClient(create_app(stats,recorder,feedback,lambda:None,recorder.clips_dir))
    response=client.get(f'/clips/{path.name}/thumb.jpg')
    frame=cv2.imdecode(np.frombuffer(response.content,np.uint8),cv2.IMREAD_COLOR)
    assert frame[:,:,1].mean()>200
    assert frame[:,:,0].mean()<40 and frame[:,:,2].mean()<40
