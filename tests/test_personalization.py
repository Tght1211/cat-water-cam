import sqlite3
import numpy as np
import pytest
from catcam.ai_budget import AIBudget
from catcam.ai_labeler import parse_label
from catcam.dispenser import DispenserStore
from catcam.stats import StatsStore
from catcam.feedback import FeedbackStore
from catcam.video_trainer import persistent_split
from catcam.models import ModelRegistry


def test_budget_persists_and_failures_still_cost(tmp_path):
    budget=AIBudget(tmp_path/'db',2,0)
    one=budget.reserve('model',now=100);budget.finish(one,'failed')
    assert AIBudget(tmp_path/'db',2,0).reserve('model',now=101)
    assert budget.reserve('model',now=102) is None
    assert budget.reserve('model',now=100+86400)


def test_budget_enforces_cooldown(tmp_path):
    budget=AIBudget(tmp_path/'db',40,60)
    assert budget.reserve('a',now=100)
    assert budget.reserve('b',now=130) is None
    assert budget.reserve('b',now=160)


def test_estimated_duration_is_distinct_from_manual_and_machine_truth(tmp_path):
    f=FeedbackStore(tmp_path/'db',tmp_path/'training')
    f.record_video_duration(tmp_path/'a.mp4',20)
    f.record_machine_label('a.mp4',True,confidence=1)
    m=f.label_meta('a.mp4')
    assert m['drinking_seconds'] is None and m['effective_drinking_seconds']==16
    assert m['duration_source']=='video_80_percent'
    assert f.label_states()['labeled']==0
    f.label_clip(tmp_path/'a.mp4',True)
    f.set_duration('a.mp4',10)
    assert f.label_meta('a.mp4')['effective_drinking_seconds']==10
    assert f.label_meta('a.mp4')['duration_source']=='human'
    f.label_clip(tmp_path/'a.mp4',False)
    assert f.label_meta('a.mp4')['effective_drinking_seconds'] is None


def test_cycle_can_calibrate_using_estimated_duration(tmp_path):
    db=tmp_path/'db';s=StatsStore(db);f=FeedbackStore(db,tmp_path/'training');d=DispenserStore(db)
    d.refill(100,100);s.record_event(200,'a.mp4')
    f.label_clip(tmp_path/'a.mp4',True);f.record_video_duration(tmp_path/'a.mp4',25)
    d.finish_cycle(300)
    assert d.calibration(s)['ml_per_second']==5
    assert d.calibration(s)['duration_calibration_estimated'] is True


def test_uncertain_visual_label_does_not_become_negative():
    assert parse_label('{"drinking":null,"confidence":0.5}')['drinking'] is None
    for raw in ('{"drinking":"maybe"}','{"drinking":true,"confidence":2}'):
        with pytest.raises(ValueError):parse_label(raw)


def test_cycle_estimate_updates_after_human_correction(tmp_path):
    db=tmp_path/'db';s=StatsStore(db);f=FeedbackStore(db,tmp_path/'training');d=DispenserStore(db)
    d.refill(2000,100)
    for i in range(10):
        name=f'clip_{i}.mp4';s.record_event(200+i*60,name);f.label_clip(tmp_path/name,True)
    d.finish_cycle(1000,0,200)
    assert d.calibration(s)['ml_per_drink']==180
    f.label_clip(tmp_path/'clip_1.mp4',False)
    assert d.calibration(s)['ml_per_drink']==200
    with pytest.raises(ValueError):d.finish_cycle(2000)


def test_cycle_waits_for_pending_labels_and_uses_duration(tmp_path):
    db=tmp_path/'db';s=StatsStore(db);f=FeedbackStore(db,tmp_path/'training');d=DispenserStore(db)
    d.refill(100,100)
    s.record_event(200,'a.mp4');s.record_event(300,'b.mp4')
    f.label_clip(tmp_path/'a.mp4',True);f.set_duration('a.mp4',10)
    d.finish_cycle(1000)
    assert not d.calibration(s)['calibrated']
    f.label_clip(tmp_path/'b.mp4',True);f.set_duration('b.mp4',15)
    c=d.calibration(s)
    assert c['ml_per_drink']==50 and c['ml_per_second']==4


def test_fixed_test_samples_never_enter_future_training(tmp_path):
    registry=ModelRegistry(tmp_path/'registry.json')
    names=[f'a{i}.mp4' for i in range(12)];y=np.array([0,1]*6)
    train,val,test,method=persistent_split(y,names,registry,0,.25)
    registry.add(path='unused',top1=.8,image_counts={},label_counts={},base='s3d+head',epochs=1,imgsz=224,created_ts=1,
                 report={'split_method':method,'split':{'train':[names[i] for i in train], 'validation':[names[i] for i in val],'test':[names[i] for i in test]}})
    expanded=names+[f'new{i}.mp4' for i in range(10)];labels=np.concatenate([y,np.array([0,1]*5)])
    train2,val2,test2,method=persistent_split(labels,expanded,registry,0,.25)
    assert set(test).issubset(set(test2))
    assert len(test2)>len(test)
    assert not set(test)&set(train2)
    assert method.startswith('fixed_')


@pytest.mark.parametrize('missing_validation_class', [False, True])
def test_day_holdout_stays_isolated_as_new_days_arrive(tmp_path, missing_validation_class):
    from datetime import datetime, timedelta
    registry=ModelRegistry(tmp_path/'registry.json')
    names=[f'clip_{int((datetime(2026,9,1)+timedelta(days=d,seconds=c)).timestamp()*1000)}.mp4'
           for d in range(4) for c in (0,1)]
    registry.add(path='unused',top1=.8,image_counts={},label_counts={},base='s3d+head',epochs=1,imgsz=224,created_ts=1,
                 report={'split_method':'day_holdout','split':{'train':names[:2],'validation':names[2:4],'test':names[4:6]}})
    y=np.array([0,1]*4)
    if missing_validation_class:
        y[3]=0
        with pytest.raises(ValueError,match='按日期'):
            persistent_split(y,names,registry,0,.25)
    else:
        train,val,test,method=persistent_split(y,names,registry,0,.25)
        assert method=='fixed_day_holdout'
        assert set(train)=={0,1,6,7}
        assert set(val)=={2,3} and set(test)=={4,5}


def test_async_ai_cannot_overwrite_human_correction(tmp_path):
    f=FeedbackStore(tmp_path/'db',tmp_path/'training')
    f.label_clip(tmp_path/'a.mp4',False,source='human')
    assert f.label_clip(tmp_path/'a.mp4',True,source='ai',confidence=.99) is False
    assert f.get_label('a.mp4') is False


def test_repeating_same_label_preserves_training_state(tmp_path):
    f=FeedbackStore(tmp_path/'db',tmp_path/'training')
    f.label_clip(tmp_path/'a.mp4',True)
    f.mark_trained('v1')
    f.label_clip(tmp_path/'a.mp4',True)
    assert f.label_states()['trained']==1


def test_rejected_candidate_never_calls_cloud(tmp_path):
    import cv2
    from catcam.ai_labeler import AILabeler
    f=FeedbackStore(tmp_path/'db',tmp_path/'training')
    path=tmp_path/'a.mp4';w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
    for _ in range(16):w.write(np.zeros((32,32,3),np.uint8))
    w.release()
    class Client:
        def __init__(self):self.chat=self;self.completions=self;self.calls=0
        def create(self,**kwargs):self.calls+=1;raise AssertionError('cloud must not be called')
    client=Client();labeler=AILabeler(f,tmp_path/'training','test',client=client,candidate_check=lambda _:False)
    assert labeler.label(path) is None
    assert client.calls==0


def test_archived_video_survives_display_clip_pruning(tmp_path):
    import cv2
    from catcam.video_trainer import training_clip_path
    f=FeedbackStore(tmp_path/'db',tmp_path/'training');path=tmp_path/'a.mp4'
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
    for _ in range(16):w.write(np.zeros((32,32,3),np.uint8))
    w.release();f.label_clip(path,False);path.unlink()
    assert training_clip_path(tmp_path,tmp_path/'training','a.mp4').exists()


def test_registry_sees_external_training_without_losing_versions(tmp_path):
    a=ModelRegistry(tmp_path/'registry.json');b=ModelRegistry(tmp_path/'registry.json')
    arguments=dict(path='weights',top1=.8,image_counts={},label_counts={},base='s3d+head',epochs=1,imgsz=224,created_ts=1)
    assert a.add(**arguments)['id']=='v1'
    assert b.add(**arguments)['id']=='v2'
    assert len(a.list())==2
    a.set_active('v2')
    assert b.active_id()=='v2'


def test_model_recheck_preserves_human_truth_and_hitrate_excludes_training(tmp_path):
    from catcam.model_review import ModelReview
    from catcam.judge import Verdict
    from pathlib import Path
    s=StatsStore(tmp_path/'db');f=FeedbackStore(tmp_path/'db',tmp_path/'training')
    for name in ('training.mp4','new.mp4'):
        s.record_event(100,name);f.label_clip(tmp_path/name,False)
    class Judge:
        version='v2'
        def judge(self,path):return Verdict(True,.95,'','v2')
    judge=Judge()
    class Runtime:
        def snapshot(self):return judge,'gate'
    class Recorder:
        def list_clips(self):return [Path('training.mp4'),Path('new.mp4')]
    review=ModelReview(Runtime(),s,f,Recorder())
    review._run(judge,'gate')
    assert review.status()['done']==2
    assert f.get_label('new.mp4') is False
    assert s.model_hitrate('v2',exclude_names=['training.mp4'])['total']==1


def test_weak_contour_recording_cannot_be_auto_confirmed_on_recheck(tmp_path):
    from catcam.model_review import ModelReview
    from pathlib import Path
    s=StatsStore(tmp_path/'db');f=FeedbackStore(tmp_path/'db',tmp_path/'training')
    s.record_event(100,'weak.mp4',candidate_only=True)
    class Judge:
        version='v2'
        def judge(self,path):raise AssertionError('weak clip must stay for visual review')
    judge=Judge()
    class Runtime:
        def snapshot(self):return judge,'gate'
    class Recorder:
        def list_clips(self):return [Path('weak.mp4')]
    review=ModelReview(Runtime(),s,f,Recorder(),lambda:True);review._run(judge,'gate')
    assert review.status()['state']=='done'
    assert f.get_label('weak.mp4') is None


@pytest.mark.parametrize('mode', ['gate', 'shadow'])
def test_recheck_water_alert_does_not_block_drinking_confirmation(tmp_path, mode):
    from catcam.model_review import ModelReview
    from catcam.judge import Verdict
    from pathlib import Path
    s = StatsStore(tmp_path/'db')
    f = FeedbackStore(tmp_path/'db', tmp_path/'training')
    s.record_event(100, 'drink.mp4')
    s.set_water_review('drink.mp4', True)
    class Judge:
        version = 'v2'
        def judge(self, path):
            return Verdict(True, 1.0, '持续舔水', self.version)
    judge = Judge()
    class Runtime:
        def snapshot(self):
            return judge, mode
    class Recorder:
        def list_clips(self):
            return [Path('drink.mp4')]
    review = ModelReview(Runtime(), s, f, Recorder(), lambda: True)
    review._run(judge, mode)
    assert review.status()['state'] == 'done'
    assert f.get_label('drink.mp4') is True
    assert s.count_between(0, 200) == 1


def test_duration_rate_uses_last_cycle_and_all_continuation_clips(tmp_path):
    db=tmp_path/'db';s=StatsStore(db);f=FeedbackStore(db,tmp_path/'training');d=DispenserStore(db)
    d.refill(100,100)
    s.record_event(200,'first.mp4');f.label_clip(tmp_path/'first.mp4',True);f.set_duration('first.mp4',10)
    d.finish_cycle(300)
    assert d.calibration(s)['ml_per_second']==10
    d.refill(120,400)
    for ts,name,seconds in [(500,'a.mp4',10),(520,'b.mp4',20)]:
        s.record_event(ts,name);s.record_event(ts+1,name)
        f.label_clip(tmp_path/name,True);f.set_duration(name,seconds)
    d.finish_cycle(600)
    report=d.calibration(s)
    assert report['duration_calibration_cycle']['drinks']==1
    assert report['duration_calibration_cycle']['drinking_seconds']==30
    assert report['ml_per_second']==4  # 120/30，不能池化旧周期，也不能丢弃合并访问的后续片段。


def test_duration_rate_never_divides_by_incomplete_seconds(tmp_path):
    db=tmp_path/'db';s=StatsStore(db);f=FeedbackStore(db,tmp_path/'training');d=DispenserStore(db)
    d.refill(100,100)
    for ts,name in [(200,'a.mp4'),(300,'b.mp4')]:
        s.record_event(ts,name);f.label_clip(tmp_path/name,True)
    f.set_duration('a.mp4',10)
    d.finish_cycle(400)
    report=d.calibration(s)
    assert report['ml_per_second'] is None
    assert report['duration_calibration_cycle']['missing_durations']==1
