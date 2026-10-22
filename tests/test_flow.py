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
    m.calibrate('flowing', image(22))
    base = m.last_seen
    # New reference is manually confirmed; recurring matches require stable observation.
    m.alert = False
    for t in range(1, 7):
        m.observe(base+t, frame, True)
    assert m.status(base+6)['state'] == 'suspected_stopped'
    assert m.status(base+15)['state'] == 'unknown'
    assert m.status(base+15)['alert']
    restarted = FlowMonitor(tmp_path)
    assert restarted.alert
    assert restarted.fault_since == m.fault_since


def test_occlusion_darkness_and_changed_scene_cannot_clear_alert(tmp_path):
    m = FlowMonitor(tmp_path, dwell=2)
    m.calibrate('flowing', image(22))
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


def test_missing_calibration_is_not_hidden_by_occlusion_or_offline(tmp_path):
    m = FlowMonitor(tmp_path)
    m.observe(100, image(), False, True)
    for now in (100, 110):
        status = m.status(now)
        assert status['state'] == 'unconfigured'
        assert not status['calibration_complete']
        assert status['missing_references'] == ['停止出水', '正常出水']
    m.calibrate('stopped', image())
    assert m.status()['missing_references'] == ['正常出水']


def test_stopped_evidence_is_three_real_frames_over_two_seconds_and_survives_restart(tmp_path):
    m = FlowMonitor(tmp_path, roi=(.2,.2,.8,.8), dwell=2)
    frame = image()
    frame[:20,:20] = 0
    m.calibrate('stopped', frame)
    start = m.fault_since
    base = m.last_seen
    for i in range(1, 8):
        current = frame.copy()
        current[:20,:20] = i*25  # 出水区域外的时间标记，验证不是复制同一张图
        m.observe(base+i*.5, current, True)
    evidence = m.evidence(start)
    assert len(evidence) == 3
    times = [ts for ts, _ in evidence]
    assert times[1]-times[0] == times[2]-times[1] == 1
    means = [cv2.imdecode(np.frombuffer(jpg,np.uint8),cv2.IMREAD_COLOR)[2:15,2:15].mean()
             for _, jpg in evidence]
    assert means[0] < means[1] < means[2]
    m.observe(base+4, frame, True)
    assert m.evidence(start) == evidence  # 后续画面不得改写故障发生时的证据
    assert FlowMonitor(tmp_path).evidence(start) == evidence


def test_evidence_waits_for_continuous_unobstructed_frames(tmp_path):
    m = FlowMonitor(tmp_path, dwell=0)
    frame = image()
    m.calibrate('stopped', frame)
    start, base = m.fault_since, m.last_seen
    m.observe(base+.5, frame, True)
    assert m.evidence(start) == []
    m.observe(base+1, frame, True, occluded=True)
    m.observe(base+1.5, frame, True)
    m.observe(base+2, frame, True)
    assert m.evidence(start) == []
    m.observe(base+10, frame, True)
    m.observe(base+10.5, frame, True)
    assert m.evidence(start) == []


def test_one_reference_never_automatically_confirms_flow(tmp_path):
    m = FlowMonitor(tmp_path, dwell=2)
    m.calibrate('stopped', image())
    m.alert = False
    base = m.last_seen
    for i in range(1, 20):
        m.observe(base+i*.5, image(), True)
    assert not m.alert
    assert m.status(base+10)['state'] == 'unconfigured'


def test_sparse_frames_cannot_satisfy_confirmation(tmp_path):
    m = FlowMonitor(tmp_path, dwell=4)
    m.calibrate('stopped', image())
    m.calibrate('flowing', image(22))
    base = m.last_seen
    for t in (1, 3, 5, 7):
        m.observe(base+t, image(), True)
    assert not m.alert  # 时间已够，但有效帧不足
    for t in (7.5, 8, 8.5):
        m.observe(base+t, image(), True)
    assert m.alert


def test_occlusion_breaks_multiframe_confirmation(tmp_path):
    m = FlowMonitor(tmp_path, dwell=2)
    m.calibrate('stopped', image())
    m.calibrate('flowing', image(22))
    base = m.last_seen
    for t in (1, 1.5, 2):
        m.observe(base+t, image(), True)
    m.observe(base+2.5, image(), True, occluded=True)
    m.observe(base+3, image(), True)
    m.observe(base+3.5, image(), True)
    assert not m.alert
    for t in (4, 4.5, 5, 5.5):
        m.observe(base+t, image(), True)
    assert m.alert


def test_outlet_occlusion_only_uses_fresh_boxes_in_selected_region():
    from catcam.flow import OutletOcclusion
    obstruction = OutletOcclusion()
    roi = (.4, .6, .7, .9)
    assert not obstruction.get(roi, 100)
    obstruction.set(100, [(0, 0, 25, 25)], 100, 100)
    assert not obstruction.get(roi, 100.5)  # 猫在画面别处不等于遮挡出水口
    obstruction.set(101, [(40, 60, 70, 90)], 100, 100)
    assert obstruction.get(roi, 101.5)
    assert not obstruction.get(roi, 104.1)  # 不把旧猫框无限期留作遮挡
    assert not obstruction.get((0, 0, .2, .2), 101.5)  # 重新框选立即按新区域计算
    obstruction.set(102, [], 100, 100)
    assert not obstruction.get(roi, 102)


def test_live_reference_reports_specific_blocking_reason(tmp_path):
    m = FlowMonitor(tmp_path)
    with pytest.raises(ValueError, match='尚未提供'):
        m.calibrate_live('flowing', None, now=100)
    with pytest.raises(ValueError, match='超过 2 秒'):
        m.calibrate_live('flowing', (97, image(), False), now=100)
    with pytest.raises(ValueError, match='画质不足.*亮度'):
        m.calibrate_live('flowing', (100, np.zeros_like(image()), False), now=100)
    with pytest.raises(ValueError, match='猫咪遮挡'):
        m.calibrate_live('flowing', (100, image(), False), occluded=True, now=100)
    assert not m.refs
    result = m.calibrate_live('flowing', (100, image(), False), now=100)
    assert result['has_flowing_reference']
    assert not result['alert']


def test_flow_observation_explains_local_quality_and_occlusion(tmp_path):
    m = FlowMonitor(tmp_path)
    m.observe(100, image(), True, occluded=False)
    observation = m.status(100)['observation']
    assert observation['usable'] and not observation['occluded']
    m.observe(101, image(), True, occluded=True)
    assert m.status(101)['observation']['occluded']


def test_video_reference_requires_continuous_real_frames_and_persists(tmp_path):
    m = FlowMonitor(tmp_path, roi=(.1,.1,.9,.9))
    frame = image()
    with pytest.raises(ValueError, match='6 秒'):
        m.calibrate_video('flowing', (100, frame, False), now=100)
    for i in range(61):
        m.observe(100+i*.1, frame, True)
    result = m.calibrate_video('flowing', (106, frame, False), now=106)
    assert result['has_flowing_video']
    assert result['method'] == 'local_video_sequence'
    assert (tmp_path/'flowing.mp4').exists()
    cap = cv2.VideoCapture(str(tmp_path/'flowing.mp4'))
    try:
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 50
        assert 9 < cap.get(cv2.CAP_PROP_FPS) < 11
    finally:
        cap.release()
    restored = FlowMonitor(tmp_path)
    assert restored.status()['has_flowing_video']
    assert np.allclose(restored.temporal_refs['flowing'], m.temporal_refs['flowing'])
    restored.set_roi((.2,.2,.8,.8))
    assert not (tmp_path/'flowing.mp4').exists()
    assert not restored.temporal_refs


def test_video_reference_cannot_bridge_occlusion_or_stream_gap(tmp_path):
    m = FlowMonitor(tmp_path)
    frame = image()
    for i in range(61):
        m.observe(100+i*.1, frame, True)
    m.observe(106.1, frame, True, occluded=True)
    m.observe(106.2, frame, True)
    with pytest.raises(ValueError, match='6 秒'):
        m.calibrate_video('stopped', (106.2, frame, False), now=106.2)
    for i in range(61):
        m.observe(110+i*.1, frame, True)
    m.observe(120, frame, True)
    with pytest.raises(ValueError, match='6 秒'):
        m.calibrate_video('stopped', (120, frame, False), now=120)


def test_video_motion_distinguishes_same_average_appearance(tmp_path):
    m = FlowMonitor(tmp_path, dwell=1)
    frame = image()
    for i in range(61):
        m.observe(100+i*.1, frame, True)
    m.calibrate_video('stopped', (106, frame, False), now=106)
    def flowing(i):
        current = frame.astype(np.int16)
        current[:,:80] += 35 if i%2 else -35
        return np.clip(current, 0, 255).astype(np.uint8)
    # 交替明暗模拟持续水流：中位外观接近静态参考，时间变化不同。
    for i in range(61):
        m.observe(110+i*.1, flowing(i), True)
    m.calibrate_video('flowing', (116, flowing(60), False), now=116)
    assert float(m.temporal_refs['flowing'].mean()) > .02
    for i in range(50):
        m.observe(120+i*.5, flowing(i), True)
    assert m.status(144.5)['state'] == 'flowing'
    assert not m.alert
    for i in range(50):
        m.observe(150+i*.5, frame, True)
    assert m.status(174.5)['state'] == 'suspected_stopped'
    assert m.alert


def test_import_confirmed_video_learns_separate_flow_model(tmp_path):
    frame = image()
    original = FlowMonitor(tmp_path/'original')
    for i in range(61):
        original.observe(100+i*.1, frame, True)
    original.calibrate_video('stopped', (106, frame, False), now=106)
    target = FlowMonitor(tmp_path/'imported')
    result = target.learn_video('stopped', tmp_path/'original'/'stopped.mp4')
    assert result['has_stopped_video']
    assert (tmp_path/'imported'/'stopped.npz').exists()
    assert not result['has_flowing_video']
