import cv2
import numpy as np
import pytest

from catcam.flow import FlowMonitor


def image(seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(60, 210, (120, 160, 3), dtype=np.uint8)


def test_stopped_dwell_and_stale_alert(tmp_path):
    m = FlowMonitor(tmp_path, dwell=3)
    frame = image()
    m.calibrate('stopped', frame)
    base = m.last_seen
    # New reference is manually confirmed; recurring matches require stable observation.
    m.alert = False
    for t in range(1, 6):
        m.observe(base+t, frame, True)
    assert m.status(base+5)['state'] == 'suspected_stopped'
    assert m.status(base+15)['state'] == 'unknown'
    assert m.status(base+15)['alert']
    restarted = FlowMonitor(tmp_path)
    assert restarted.alert
    assert restarted.fault_since == m.fault_since


def test_occlusion_darkness_and_changed_scene_cannot_clear_alert(tmp_path):
    m = FlowMonitor(tmp_path, dwell=2)
    m.calibrate('stopped', image())
    base = m.last_seen
    for t in range(1, 7):
        m.observe(base+t, image(5), True)
    assert m.status(base+6)['state'] == 'unknown'
    assert m.alert
    m.observe(base+7, np.zeros_like(image()), True)
    m.observe(base+8, image(), True, True)
    assert m.status(base+8)['state'] == 'unknown'
    assert m.pending is None
    with pytest.raises(ValueError):
        m.calibrate('flowing', image(), occluded=True)


def test_personal_references_and_roi_reset(tmp_path):
    m = FlowMonitor(tmp_path, dwell=2)
    stopped, flowing = image(), np.flip(image(), axis=1).copy()
    m.calibrate('stopped', stopped)
    m.calibrate('flowing', flowing)
    base = m.last_seen
    for t in range(1, 5):
        m.observe(base+t, stopped, True)
    assert m.alert
    for t in range(5, 9):
        m.observe(base+t, flowing, True)
    assert m.status(base+8)['state'] == 'flowing'
    assert not m.alert
    m.set_roi([.1,.2,.4,.6])
    reloaded = FlowMonitor(tmp_path)
    assert reloaded.roi == (.1,.2,.4,.6)
    assert not reloaded.refs
    for roi in (None, [.1,.2,.1,.6], [float('nan'),.2,.4,.6]):
        with pytest.raises(ValueError):
            m.set_roi(roi)


def test_identical_references_do_not_prove_recovery(tmp_path):
    m = FlowMonitor(tmp_path, dwell=2)
    frame = image()
    m.calibrate('flowing', frame)
    m.calibrate('stopped', frame)
    base = m.last_seen
    for t in range(1, 8):
        m.observe(base+t, frame, True)
    assert m.status(base+7)['state'] == 'unknown'
    assert m.alert


def test_camera_gap_resets_dwell(tmp_path):
    m = FlowMonitor(tmp_path, dwell=3)
    frame = image()
    m.calibrate('stopped', frame)
    m.alert = False
    base = m.last_seen
    m.observe(base+1, frame, True)
    m.observe(base+2, frame, True)
    m.observe(base+30, frame, True)
    m.observe(base+31, frame, True)
    assert not m.alert


def test_flow_api_personal_capture_and_invalid_input(tmp_path):
    from fastapi.testclient import TestClient
    from catcam.web import create_app
    from catcam.stats import StatsStore
    from catcam.recorder import ClipRecorder
    from catcam.feedback import FeedbackStore
    from catcam.dispenser import DispenserStore
    monitor = FlowMonitor(tmp_path/'flow')
    recorder = ClipRecorder(clips_dir=tmp_path/'clips', max_clips=10, fps=5)
    db = tmp_path/'store.db'
    app = create_app(StatsStore(db), recorder, FeedbackStore(db, tmp_path/'train'),
                     lambda: image(), recorder.clips_dir, dispenser=DispenserStore(db, 30, 30),
                     flow_monitor=monitor, flow_calibrate=lambda kind: monitor.calibrate(kind, image()))
    client = TestClient(app)
    assert client.get('/api/dispenser').json()['flow']['state'] == 'unconfigured'
    assert client.post('/api/dispenser/flow/reference', json={'kind':'wrong'}).status_code == 400
    result = client.post('/api/dispenser/flow/reference', json={'kind':'stopped'})
    assert result.status_code == 200
    assert result.json()['has_stopped_reference']
    assert (tmp_path/'flow'/'stopped.jpg').exists()
    assert client.get('/api/dispenser').json()['flow']['alert']
    assert client.post('/api/dispenser/flow/roi', json={'roi':None}).status_code == 400


def test_auto_recovery_history_dismiss_and_restart(tmp_path):
    m=FlowMonitor(tmp_path,dwell=2)
    stopped,flowing=image(),image(22)
    m.calibrate('flowing',flowing)
    m.calibrate('stopped',stopped)
    base=m.last_seen
    for t in range(1,5):m.observe(base+t,flowing,True)
    status=m.status(base+4)
    assert not status['alert']
    assert status['state']=='flowing'
    assert status['recovery']['end']==base+3
    identity=status['recovery']['id']
    restarted=FlowMonitor(tmp_path)
    assert restarted.status()['recovery']['id']==identity
    restarted.dismiss(identity)
    assert FlowMonitor(tmp_path).status()['recovery'] is None
    assert len(restarted.status()['incidents'])==1


def test_legacy_recovery_history_is_migrated(tmp_path):
    import json
    history=[{'ts':100,'state':'suspected_stopped','alert':True},
             {'ts':110,'state':'unknown','alert':True},
             {'ts':130,'state':'flowing','alert':False}]
    (tmp_path/'transitions.jsonl').write_text('\n'.join(json.dumps(row) for row in history))
    status=FlowMonitor(tmp_path).status(140)
    assert status['recovery']['start']==100
    assert status['recovery']['end']==130
    assert not status['alert']


def test_stopped_interval_context_is_limited_to_affected_video(tmp_path):
    m=FlowMonitor(tmp_path)
    m._set_alert(True,100)
    m._set_alert(False,120)
    assert m.was_stopped(105,115)
    assert m.was_stopped(95,110)
    assert not m.was_stopped(120,130)
