import numpy as np
from datetime import datetime, timedelta
from catcam.video_trainer import split_dataset, classification_metrics, extract_and_cache
from catcam.videojudge import DrinkingHead, read_temporal_windows, TemporalFeatureExtractor
from catcam.feedback import FeedbackStore


def test_split_has_both_classes_and_no_overlap():
    y = np.array([0]*4+[1]*4)
    parts = split_dataset(y,[f'a{i}.mp4' for i in range(8)])[:3]
    assert all(set(y[p]) == {0,1} for p in parts)
    assert sum(len(p) for p in parts)==8
    assert len(set(np.concatenate(parts)))==8


def test_date_split_prevents_neighbor_session_leakage():
    names,y=[],[]
    for day in range(5):
        for label in (0,1):
            ts=(datetime(2026,9,1)+timedelta(days=day,seconds=label)).timestamp()
            names.append(f'clip_{int(ts*1000)}.mp4');y.append(label)
    train,val,test,method=split_dataset(np.array(y),names)
    assert method=='day_holdout'
    assert max(train)<min(val)<min(test)


def test_majority_guess_exposes_missed_drinking():
    m=classification_metrics(np.array([1,0,0,0,0]), np.zeros(5))
    assert m['top1']==.8
    assert m['drinking_recall']==0
    assert m['confusion']['missed_drinking']==1
    assert m['balanced_accuracy']==.5


def test_metadata_and_threshold_survive_model_reload(tmp_path):
    head=DrinkingHead([0,0],0,[0,0],[1,1],.7,{'backend':'s3d-temporal-v2'})
    head.save(tmp_path/'head.npz')
    other=DrinkingHead.load(tmp_path/'head.npz')
    assert other.threshold==.7
    assert other.metadata==head.metadata
    assert other.predict([0,0])==(False,.5)


def test_correction_removes_old_class_frames(tmp_path):
    s=FeedbackStore(tmp_path/'db', tmp_path/'training')
    old=tmp_path/'training/not_drinking';old.mkdir(parents=True)
    (old/'clip_1_0.jpg').write_bytes(b'old')
    s.label_clip(tmp_path/'clip_1.mp4',True)
    assert not (old/'clip_1_0.jpg').exists()


def test_mark_trained_does_not_hide_concurrent_correction(tmp_path):
    s=FeedbackStore(tmp_path/'db',tmp_path/'training')
    s.label_clip(tmp_path/'a.mp4',True)
    with s._conn() as c: snapshot=c.execute('SELECT clip_name,is_drinking,ts FROM labels').fetchall()
    s.label_clip(tmp_path/'a.mp4',False)
    s.mark_trained('v2',snapshot)
    assert s.label_states()['untrained']==1


def test_cache_rejects_wrong_dimension(tmp_path):
    from catcam.video_trainer import feature_cache_path,_save_npy
    _save_npy(feature_cache_path(tmp_path,'gone.mp4'),np.zeros(8))
    assert extract_and_cache(tmp_path/'gone.mp4',tmp_path,object(),dim=16) is None


def test_temporal_window_is_bounded_and_real_backbone_works(tmp_path):
    import cv2
    path=tmp_path/'clip.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
    rng=np.random.default_rng(10)
    for _ in range(60): writer.write(rng.integers(0,255,(32,32,3),dtype=np.uint8))
    writer.release()
    windows=read_temporal_windows(path)
    assert len(windows)==3 and all(len(w)==16 for w in windows)
    ext=TemporalFeatureExtractor()
    feat=ext.extract_clip(path)
    assert feat.shape==(2048,) and np.isfinite(feat).all()
