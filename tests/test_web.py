import sqlite3
import pytest
import time
import numpy as np
from fastapi.testclient import TestClient

from catcam.stats import StatsStore
from catcam.recorder import ClipRecorder
from catcam.feedback import FeedbackStore
from catcam.dispenser import DispenserStore
from catcam.web import create_app


def _build(tmp_path, frame_provider=lambda: None):
    stats = StatsStore(tmp_path / "s.db")
    recorder = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5)
    feedback = FeedbackStore(db_path=tmp_path / "f.db", training_dir=tmp_path / "train")
    dispenser = DispenserStore(tmp_path / "d.db", default_ml_per_drink=30.0, default_cycle_days=30)
    app = create_app(stats, recorder, feedback, frame_provider, recorder.clips_dir,
                     dispenser=dispenser, dispenser_low_water_pct=0.2)
    return app, stats, recorder, feedback


def test_index_serves_html(tmp_path):
    app, *_ = _build(tmp_path)
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert r.headers["cache-control"] == "no-store"


def test_preview_reports_connected_video_without_marking_monitoring_offline(tmp_path):
    app, stats, recorder, feedback = _build(tmp_path)
    camera = {'connected': True, 'last_frame_age': 0.1, 'width': 1280, 'height': 720,
              'quality': {'usable': True}, 'detection_enabled': False}
    app = create_app(stats, recorder, feedback, lambda: None, recorder.clips_dir,
                     live_status_provider=lambda: dict(camera))
    client = TestClient(app)
    assert client.get('/api/live/status').json() == camera
    health = client.get('/api/health').json()
    assert health['camera']['connected'] is True
    assert health['status'] == 'preview_only'
    assert health['monitoring']['status'] == 'preview'
    assert health['coverage_hours'] == 0
    assert not any('检查摄像头' in message for message in health['actions'])
    camera['connected'] = False
    assert client.get('/api/live/status').json()['connected'] is False
    assert client.get('/api/health').json()['monitoring']['status'] == 'offline'


def test_audio_status_exposes_runtime_health(tmp_path):
    app, stats, recorder, feedback = _build(tmp_path)
    assert TestClient(app).get("/api/audio/status").json() == {
        "enabled": False, "available": False
    }
    app2 = create_app(
        stats, recorder, feedback, lambda: None, recorder.clips_dir,
        audio_status_provider=lambda: {"available": True, "buffered_seconds": 12.5},
    )
    assert TestClient(app2).get("/api/audio/status").json() == {
        "enabled": True, "available": True, "buffered_seconds": 12.5
    }


def test_clips_list_includes_label_status(tmp_path):
    app, _, recorder, feedback = _build(tmp_path)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.save_clip([frame], timestamp=1.0)
    recorder.save_clip([frame], timestamp=2.0)
    feedback.label_clip(recorder.clips_dir / "clip_2000.mp4", True)
    client = TestClient(app)
    body = client.get("/api/clips").json()
    assert body["clips"] == ["clip_2000.mp4", "clip_1000.mp4"]
    assert body["labels"]["clip_2000.mp4"] is True
    assert body["labels"]["clip_1000.mp4"] is None


def test_clip_playback_is_inline_and_explicit_download_is_attachment(tmp_path):
    app, _, recorder, _ = _build(tmp_path)
    recorder.save_clip([np.zeros((48, 64, 3), np.uint8)], timestamp=1.0)
    r = TestClient(app).get("/clips/clip_1000.mp4")
    assert r.status_code == 200
    assert "content-disposition" not in r.headers
    dl = TestClient(app).get("/clips/clip_1000.mp4?download=1")
    assert "attachment" in dl.headers.get("content-disposition", "")


def test_clip_muted_download_falls_back_when_no_ffmpeg(tmp_path, monkeypatch):
    # 无 ffmpeg 时 audio=0 应回退原文件，不 500（fail-open）
    import catcam.web as web
    monkeypatch.setattr(web.shutil, "which", lambda _: None)
    app, _, recorder, _ = _build(tmp_path)
    recorder.save_clip([np.zeros((48, 64, 3), np.uint8)], timestamp=2.0)
    r = TestClient(app).get("/clips/clip_2000.mp4?audio=0")
    assert r.status_code == 200
    assert r.content  # 原文件字节


def test_dispenser_initial_state(tmp_path):
    app, *_ = _build(tmp_path)
    d = TestClient(app).get("/api/dispenser").json()
    assert d["has_refill"] is False and d["has_filter"] is False
    assert d["filter_cycle_days"] == 30
    assert d["ml_per_drink"] == 30.0 and d["calibrated"] is False


def test_dispenser_water_estimate_and_measurement_correction(tmp_path):
    import sqlite3
    app, stats, *_ = _build(tmp_path)
    client = TestClient(app)
    d = client.post("/api/dispenser/refill", json={"ml": 2000}).json()
    assert d["has_refill"] and d["last_refill_ml"] == 2000
    assert d["remaining_ml"] == 2000            # 还没喝
    # 造 3 段「确认喝水」的事件：时刻在「蓄水之后、下次读取之前」这个当下
    ev = time.time()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        stats.record_event(ev, name)
    with sqlite3.connect(stats.db_path) as c:
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES (?, 1, NULL)", (name,))
    d2 = client.get("/api/dispenser").json()
    assert d2["drinks_since_refill"] == 1  # 同一分钟的三段只记一次
    assert d2['remaining_ml'] == 1970  # 用户接受按次数估算
    assert d2['water_source'] == 'estimated'
    measured = client.post('/api/dispenser/observe', json={'ml': 1500}).json()
    assert measured['remaining_ml'] == 1500
    assert d2["today_ml"] == 30


def test_dispenser_filter_countdown_and_config(tmp_path):
    app, *_ = _build(tmp_path)
    client = TestClient(app)
    client.post("/api/dispenser/filter")
    d = client.get("/api/dispenser").json()
    assert d["has_filter"] and d["filter_days_left"] is not None
    assert 29 <= d["filter_days_left"] <= 30 and d["need_filter"] is False
    d2 = client.post("/api/dispenser/config", json={"filter_cycle_days": 45}).json()
    assert d2["filter_cycle_days"] == 45


def test_dispenser_refill_rejects_nonpositive(tmp_path):
    app, *_ = _build(tmp_path)
    r = TestClient(app).post("/api/dispenser/refill", json={"ml": 0})
    assert r.status_code == 400


def test_dispenser_refill_rejects_nonfinite_values(tmp_path):
    app, *_ = _build(tmp_path)
    client = TestClient(app)
    assert client.post("/api/dispenser/refill", json={"ml": "nan"}).status_code == 400
    assert client.post("/api/dispenser/refill", json={"ml": "inf"}).status_code == 400


def test_clips_list_reports_max_clips(tmp_path):
    app, _, recorder, _ = _build(tmp_path)
    body = TestClient(app).get("/api/clips").json()
    assert body["max_clips"] == recorder.max_clips == 10


def test_stats_trend_shape_and_buckets(tmp_path):
    app, stats, _, feedback = _build(tmp_path)
    now = time.time()
    # 两段确认喝水（同一小时），一段没喝（不计）
    stats.record_event(now, "clip_a.mp4")
    stats.record_event(now, "clip_b.mp4")
    stats.record_event(now, "clip_c.mp4")
    with sqlite3.connect(stats.db_path) as c:
        c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES ('clip_a.mp4', 1, NULL)")
        c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES ('clip_b.mp4', 1, NULL)")
        c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES ('clip_c.mp4', 0, NULL)")
    body = TestClient(app).get("/api/stats/trend?days=7").json()
    assert len(body["days"]) == 7
    assert len(body["hourly"]) == 24 and len(body["weekday"]) == 7
    assert body["total"] == 1                      # 同一时刻只记一次
    assert sum(body["hourly"]) == 1 and sum(body["weekday"]) == 1
    assert body["active_days"] == 1
    assert "prev_total" in body


def test_today_stats_counts_recent_event(tmp_path):
    app, stats, *_ = _build(tmp_path)
    stats.record_event(time.time(), "clip_x.mp4")
    # 计数口径：只数被确认「喝水」的段 → 标 clip_x 为 is_drinking=1
    with sqlite3.connect(stats.db_path) as c:
        c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES ('clip_x.mp4', 1, NULL)")
    client = TestClient(app)
    r = client.get("/api/stats/today")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    assert len(body["times"]) == body["count"]


def test_clips_list_and_download(tmp_path):
    app, _, recorder, _ = _build(tmp_path)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.save_clip([frame, frame], timestamp=1.0)
    client = TestClient(app)
    listing = client.get("/api/clips").json()["clips"]
    assert listing == ["clip_1000.mp4"]
    dl = client.get("/clips/clip_1000.mp4")
    assert dl.status_code == 200
    assert len(dl.content) > 0


def test_snapshot_503_without_frame(tmp_path):
    app, *_ = _build(tmp_path, frame_provider=lambda: None)
    client = TestClient(app)
    assert client.get("/snapshot.jpg").status_code == 503


def test_snapshot_returns_jpeg_with_frame(tmp_path):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    app, *_ = _build(tmp_path, frame_provider=lambda: frame)
    client = TestClient(app)
    r = client.get("/snapshot.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert len(r.content) > 0


def test_clip_thumbnail_seeks_to_middle_frame(tmp_path, monkeypatch):
    import catcam.web as web

    app, _, recorder, _ = _build(tmp_path)
    path = recorder.clips_dir / "clip_1000.mp4"
    path.write_bytes(b"placeholder")
    sought = []

    class FakeCapture:
        def __init__(self, _path): pass
        def get(self, prop): return 11 if prop == web.cv2.CAP_PROP_FRAME_COUNT else 0
        def set(self, prop, value): sought.append((prop, value)); return True
        def read(self): return True, np.full((24, 32, 3), 127, np.uint8)
        def release(self): pass

    monkeypatch.setattr(web.cv2, "VideoCapture", FakeCapture)
    r = TestClient(app).get("/clips/clip_1000.mp4/thumb.jpg")
    assert r.status_code == 200
    assert sought == [(web.cv2.CAP_PROP_POS_FRAMES, 5)]


def test_post_feedback_persists_label(tmp_path):
    app, _, recorder, feedback = _build(tmp_path)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.save_clip([frame, frame], timestamp=2.0)
    client = TestClient(app)
    r = client.post("/api/feedback", json={"clip": "clip_2000.mp4", "is_drinking": True})
    assert r.status_code == 200
    assert feedback.get_label("clip_2000.mp4") is True


def test_download_rejects_path_traversal(tmp_path):
    app, *_ = _build(tmp_path)
    client = TestClient(app)
    assert client.get("/clips/..%2F..%2Fsecret.txt").status_code in (400, 404)


def test_clips_list_is_newest_first(tmp_path):
    app, _, recorder, _ = _build(tmp_path)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.save_clip([frame], timestamp=1.0)
    recorder.save_clip([frame], timestamp=3.0)
    recorder.save_clip([frame], timestamp=2.0)
    client = TestClient(app)
    clips = client.get("/api/clips").json()["clips"]
    assert clips == ["clip_3000.mp4", "clip_2000.mp4", "clip_1000.mp4"]


def test_clips_exposes_model_and_label_review_fields(tmp_path):
    app, _, recorder, feedback = _build(tmp_path)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.save_clip([frame], timestamp=1.0)
    feedback.label_clip(recorder.clips_dir / "clip_1000.mp4", True,
                        source="ai", confidence=0.8, reason="舔水")
    client = TestClient(app)
    body = client.get("/api/clips").json()
    assert body["labels"]["clip_1000.mp4"] is True
    assert body["meta"]["clip_1000.mp4"]["source"] == "ai"
    assert body["predictions"] == {}
    assert body["prediction_details"] == {}


def test_clips_exposes_model_prediction_confidence(tmp_path):
    app, stats, recorder, _ = _build(tmp_path)
    frame = np.zeros((48, 64, 3), np.uint8)
    recorder.save_clip([frame], timestamp=3.0)
    stats.record_event(3.0, "clip_3000.mp4")
    stats.set_prediction("clip_3000.mp4", 0, "v3", 0.18)
    body = TestClient(app).get("/api/clips").json()
    assert body["prediction_details"]["clip_3000.mp4"] == {
        "drinking": False, "confidence": 0.18, "by": "v3", "threshold": 0.5
    }


def _build_with_registry(tmp_path, video_model_switch=None, video_model_clear=None):
    from catcam.models import ModelRegistry
    from catcam.classifier import ActiveModel
    stats = StatsStore(tmp_path / "s.db")
    recorder = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5)
    feedback = FeedbackStore(db_path=tmp_path / "f.db", training_dir=tmp_path / "train")
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    active_model = ActiveModel()
    app = create_app(stats, recorder, feedback, lambda: None, recorder.clips_dir,
                     registry=registry, active_model=active_model,
                     video_model_switch=video_model_switch,
                     video_model_clear=video_model_clear)
    return app, stats, recorder, feedback, registry, active_model


def test_activate_s3d_head_version_does_not_500(tmp_path):
    switched = []
    app, stats, recorder, feedback, registry, active_model = _build_with_registry(
        tmp_path, video_model_switch=lambda entry, mode: switched.append((entry["id"], mode)))
    head_path = tmp_path / "videohead_1.npz"; head_path.write_bytes(b"x")
    registry.add(path=head_path, top1=0.9, image_counts={"drinking": 5, "not_drinking": 5},
                 label_counts=None, base="s3d+head", epochs=300, imgsz=224, created_ts=1.0)
    client = TestClient(app)
    r = client.post("/api/model/activate", json={"id": "v1", "mode": "shadow"})
    assert r.status_code == 200
    assert registry.active_id() == "v1" and registry.active_mode() == "shadow"
    # 视频版本不塞进单帧 active_model，而是立即热切换本地视频裁判。
    assert active_model.active_id is None
    assert switched == [("v1", "shadow")]
    assert "立即生效" in (r.json().get("note") or "")


def test_activate_video_model_failure_keeps_registry_unchanged(tmp_path):
    def fail(_entry, _mode):
        raise ValueError("坏模型")

    app, _, _, _, registry, _ = _build_with_registry(tmp_path, video_model_switch=fail)
    head_path = tmp_path / "videohead_1.npz"; head_path.write_bytes(b"x")
    registry.add(path=head_path, top1=0.9, image_counts={}, label_counts={},
                 base="s3d+head", epochs=1, imgsz=224, created_ts=1.0)
    r = TestClient(app).post("/api/model/activate", json={"id": "v1", "mode": "gate"})
    assert r.status_code == 500
    assert registry.active_id() is None


def test_user_can_choose_automatic_model_despite_small_evaluation_set(tmp_path):
    switched = []
    app, _, _, _, registry, _ = _build_with_registry(
        tmp_path, video_model_switch=lambda entry, mode: switched.append(mode))
    registry.add(path=tmp_path/'head.npz', top1=.92, image_counts={}, label_counts={},
                 base='s3d+head', epochs=1, imgsz=224, created_ts=1,
                 report={'auto_label_eligible': False, 'warnings': ['独立测试样本较少']})
    client = TestClient(app)
    assert client.post('/api/model/activate', json={'id': 'v1', 'mode': 'gate'}).status_code == 200
    assert registry.active_mode() == 'gate'
    assert switched == ['gate']
    assert client.post('/api/model/activate', json={'id': 'v1', 'mode': 'invalid'}).status_code == 400
    assert registry.active_mode() == 'gate'


class _FakeVideoTrainer:
    def __init__(self): self.started = 0; self._state = "idle"; self.last_rebuild = None
    def start(self, rebuild=False, backend='s3d-temporal-v2', auto_activate=False): self.started += 1; self.last_rebuild = rebuild; self._state = "running"; return True
    def status(self): return {"state": self._state, "detail": "x", "result": None,
                              "models": [], "active": None}


def test_train_video_endpoints(tmp_path):
    from catcam.models import ModelRegistry
    from catcam.classifier import ActiveModel
    stats = StatsStore(tmp_path / "s.db")
    recorder = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5)
    feedback = FeedbackStore(db_path=tmp_path / "f.db", training_dir=tmp_path / "train")
    vt = _FakeVideoTrainer()
    app = create_app(stats, recorder, feedback, lambda: None, recorder.clips_dir, video_trainer=vt)
    client = TestClient(app)
    assert client.post("/api/train_video").json()["started"] is True   # 无 body → rebuild=False
    assert vt.started == 1 and vt.last_rebuild is False
    assert client.get("/api/train_video/status").json()["state"] == "running"


def test_train_video_rebuild_flag(tmp_path):
    stats = StatsStore(tmp_path / "s.db")
    recorder = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5)
    feedback = FeedbackStore(db_path=tmp_path / "f.db", training_dir=tmp_path / "train")
    vt = _FakeVideoTrainer()
    app = create_app(stats, recorder, feedback, lambda: None, recorder.clips_dir, video_trainer=vt)
    client = TestClient(app)
    assert client.post("/api/train_video", json={"rebuild": True}).json()["started"] is True
    assert vt.last_rebuild is True


def test_train_video_disabled_when_not_wired(tmp_path):
    app, *_ = _build(tmp_path)   # 没传 video_trainer
    client = TestClient(app)
    assert client.post("/api/train_video").json()["started"] is False
    assert client.get("/api/train_video/status").json()["state"] == "disabled"


def test_additive_water_api_retains_drinking_events(tmp_path,monkeypatch):
    app,stats,recorder,feedback=_build(tmp_path)
    client=TestClient(app)
    monkeypatch.setattr('catcam.web.time.time',lambda:1000)
    first=client.post('/api/dispenser/add',json={'added_ml':950,'initial_remaining_ml':0}).json()
    assert first['remaining_ml']==950
    stats.record_event(1050,'drink.mp4')
    feedback.record_machine_label('drink.mp4',True)
    with stats._conn() as c:
        c.execute('INSERT INTO labels(clip_name,is_drinking,ts) VALUES(?,?,?)',('drink.mp4',1,1050))
    monkeypatch.setattr('catcam.web.time.time',lambda:1100)
    second=client.post('/api/dispenser/add',json={'added_ml':1050}).json()
    assert second['last_refill_ts']==1000
    assert second['last_refill_ml']==2000
    assert second['remaining_ml']==1970
    assert second['drinks_since_refill']==1
    assert second['reminder']['low_ml'] is None
    assert not second['need_water']
    assert client.post('/api/dispenser/cycle',json={}).status_code==400
    result=client.post('/api/dispenser/cycle',json={'remaining_ml':200,'stopped_confirmed':True})
    assert result.status_code==200
    assert result.json()['reminder']['stop_residual_ml']==200


def test_dry_lick_feedback_api_preserves_action_truth(tmp_path):
    from catcam.video_trainer import _labeled_clips
    app,stats,recorder,feedback=_build(tmp_path)
    frame=np.zeros((48,64,3),dtype=np.uint8)
    recorder.save_clip([frame],timestamp=2)
    stats.record_event(2,'clip_2000.mp4',predicted=1,predicted_by='v1')
    client=TestClient(app)
    response=client.post('/api/feedback',json={'clip':'clip_2000.mp4','is_drinking':False,'outcome':'dry_lick','reason':'换水时没喝到'})
    assert response.status_code==200
    assert feedback.label_meta('clip_2000.mp4')['reason']=='换水时没喝到'
    assert _labeled_clips(feedback)==[('clip_2000.mp4',1)]
    assert stats.count_between(0,5)==0
    assert client.post('/api/feedback',json={'clip':'clip_2000.mp4','is_drinking':True,'outcome':'dry_lick'}).status_code==400


def test_large_clip_list_never_opens_videos_or_queries_individually(tmp_path, monkeypatch):
    import catcam.web as web
    app, stats, recorder, feedback = _build(tmp_path)
    recorder.max_clips = 2000
    for i in range(1000):
        (recorder.clips_dir / f'clip_{1000000 + i * 1000}.mp4').touch()
    name = 'clip_1000000.mp4'
    feedback.record_video_duration(recorder.clips_dir / name, 12.25)
    with feedback._conn() as conn:
        conn.execute("INSERT INTO labels (clip_name,is_drinking,source) VALUES (?,1,'human')", (name,))
    expected = feedback.label_meta(name)
    def forbidden(*args, **kwargs):
        raise AssertionError('列表不应打开录像、写入数据或逐段查询')
    monkeypatch.setattr(web.cv2, 'VideoCapture', forbidden)
    monkeypatch.setattr(feedback, 'record_video_duration', forbidden)
    monkeypatch.setattr(feedback, 'get_label', forbidden)
    monkeypatch.setattr(feedback, 'label_meta', forbidden)
    monkeypatch.setattr(stats, 'requires_visual_review', forbidden)
    monkeypatch.setattr(stats, 'requires_water_review', forbidden)
    statements = []
    for store in (stats, feedback):
        connect = store._conn
        def traced(connect=connect):
            conn = connect()
            conn.set_trace_callback(statements.append)
            return conn
        monkeypatch.setattr(store, '_conn', traced)
    data = TestClient(app).get('/api/clips').json()
    assert len(data['clips']) == 1000
    assert data['meta'][name] == expected
    assert data['durations'][name] == 12.2
    assert data['durations']['clip_1001000.mp4'] is None
    assert len(statements) <= 5


def test_thumbnail_cache_invalidates_on_video_change(tmp_path, monkeypatch):
    import catcam.web as web
    app, _, recorder, _ = _build(tmp_path)
    path = recorder.clips_dir / 'clip_1000.mp4'
    path.write_bytes(b'placeholder')
    opened = []
    class Capture:
        def __init__(self, path): opened.append(path)
        def get(self, prop): return 30
        def set(self, prop, value): return True
        def read(self): return True, np.zeros((48, 64, 3), np.uint8)
        def release(self): pass
    monkeypatch.setattr(web.cv2, 'VideoCapture', Capture)
    client = TestClient(app)
    for _ in range(2):
        assert client.get('/clips/clip_1000.mp4/thumb.jpg').status_code == 200
    assert len(opened) == 1
    path.write_bytes(b'replaced recording')
    assert client.get('/clips/clip_1000.mp4/thumb.jpg').status_code == 200
    assert len(opened) == 2


def test_clip_api_and_stats_use_same_minute_interval(tmp_path):
    stats = StatsStore(tmp_path / 'shared.db')
    feedback = FeedbackStore(tmp_path / 'shared.db', tmp_path / 'train')
    recorder = ClipRecorder(tmp_path / 'clips', max_clips=10, fps=30)
    for ts in [100, 115, 130, 160, 220]:
        name = f'clip_{ts * 1000}.mp4'
        (recorder.clips_dir / name).touch()
        stats.record_event(ts, name)
        with feedback._conn() as conn:
            conn.execute("INSERT INTO labels(clip_name,is_drinking,source) VALUES (?,1,'human')", (name,))
    app = create_app(stats, recorder, feedback, lambda: None, recorder.clips_dir)
    data = TestClient(app).get('/api/clips').json()
    assert data['drinking_records'] == ['clip_220000.mp4', 'clip_100000.mp4']
    assert len(data['clips']) == 5
    assert stats.count_between(0, 1000) == 2



def _paged_fixture(tmp_path, count=35):
    stats = StatsStore(tmp_path / 'shared.db')
    feedback = FeedbackStore(tmp_path / 'shared.db', tmp_path / 'training')
    recorder = ClipRecorder(tmp_path / 'clips', max_clips=2000, fps=30)
    names = []
    for i in range(count):
        name = f'clip_{1700000000000 + i * 1000}.mp4'
        (recorder.clips_dir / name).touch()
        stats.record_event(1700000000 + i, name, predicted=i % 2, predicted_by='v1')
        names.append(name)
    app = create_app(stats, recorder, feedback, lambda: None, recorder.clips_dir)
    return TestClient(app), stats, feedback, recorder, list(reversed(names))


def test_review_pagination_bounds_payload_and_never_opens_video(tmp_path, monkeypatch):
    import catcam.web as web
    client, stats, feedback, recorder, names = _paged_fixture(tmp_path, 1000)
    def forbidden(*a, **k):
        raise AssertionError('分页不得打开视频')
    monkeypatch.setattr(web.cv2, 'VideoCapture', forbidden)
    data = client.get('/api/clips/page').json()
    assert data['clips'] == names[:12]
    assert data['total'] == 1000
    assert data['counts']['yes'] == data['counts']['no'] == 500
    assert data['next_cursor'] == names[11]
    assert data['snapshot'] == names[0]
    assert 'drinking_records' not in data
    for field in ['labels', 'meta', 'durations', 'review_candidates', 'prediction_details']:
        assert set(data[field]) == set(names[:12])
    assert len(client.get('/api/clips/page').content) < 15000
    assert client.get('/api/clips/page?limit=49').status_code == 422
    assert client.get('/api/clips/page?limit=0').status_code == 422
    assert client.get('/api/clips/page?filter=invalid').status_code == 422
    assert client.get('/api/clips/page?cursor=../x').status_code == 422


def test_review_cursor_handles_new_clips_and_labels_without_skipping(tmp_path):
    client, stats, feedback, recorder, names = _paged_fixture(tmp_path)
    first = client.get('/api/clips/page').json()
    # 第一页标完，新录像加入。旧游标应从第13条继续，不受 OFFSET 位移影响。
    with feedback._conn() as conn:
        conn.executemany("INSERT INTO labels (clip_name,is_drinking,source) VALUES (?,1,'human')",
                         [(name,) for name in first['clips']])
    new = 'clip_1800000000000.mp4'
    (recorder.clips_dir / new).touch()
    stats.record_event(1800000000, new)
    seen = first['clips'][:]
    cursor = first['next_cursor']
    while cursor:
        data = client.get('/api/clips/page', params={'cursor': cursor, 'snapshot': first['snapshot']}).json()
        seen.extend(data['clips'])
        cursor = data['next_cursor']
    assert seen == names
    summary = client.get('/api/clips/summary', params={'since': first['snapshot']}).json()
    assert summary == {'pending_count': 24, 'new_count': 1}
    assert client.get('/api/clips/page').json()['clips'][0] == new


def test_review_page_filters_counts_and_human_review_semantics(tmp_path):
    client, stats, feedback, recorder, names = _paged_fixture(tmp_path, 6)
    # newest predicted yes is human-confirmed false -> model mistake
    with feedback._conn() as conn:
        conn.execute("INSERT INTO labels (clip_name,is_drinking,action_drinking,source) VALUES (?,0,0,'human')", (names[0],))
    stats.set_review_candidate(names[-1], False)
    all_data = client.get('/api/clips/page?filter=all').json()
    assert all_data['counts'] == {'all': 5, 'review': 4, 'yes': 3, 'no': 2, 'mistakes': 1}
    assert client.get('/api/clips/page?filter=mistakes').json()['clips'] == [names[0]]
    assert len(client.get('/api/clips/page?filter=yes').json()['clips']) == 3
    assert len(client.get('/api/clips/page?filter=no').json()['clips']) == 2
    # 人工标注片段仍可在全部中回看，即使路由原本归普通回放。
    stats.set_review_candidate(names[0], False)
    assert names[0] in client.get('/api/clips/page?filter=all').json()['clips']
    assert client.get('/api/clips/page', params={'cursor': names[-1]}).json()['clips'] == []


def test_review_page_skips_drinking_history_aggregation(tmp_path, monkeypatch):
    client, stats, feedback, recorder, names = _paged_fixture(tmp_path, 1000)
    def forbidden():
        raise AssertionError('待复核分页不应扫描喝水历史或合并喝水次数')
    monkeypatch.setattr(stats, 'drinking_records', forbidden)
    result = client.get('/api/clips/page')
    assert result.status_code == 200
    assert len(result.json()['clips']) == 12


def test_thumbnail_conditional_request_skips_decoding(tmp_path, monkeypatch):
    import catcam.web as web
    app, _, recorder, _ = _build(tmp_path)
    path = recorder.clips_dir/'clip_1000.mp4'
    path.write_bytes(b'placeholder')
    class Capture:
        def __init__(self, path): pass
        def get(self, prop): return 30
        def set(self, prop, value): return True
        def read(self): return True, np.zeros((48, 64, 3), np.uint8)
        def release(self): pass
    monkeypatch.setattr(web.cv2, 'VideoCapture', Capture)
    client = TestClient(app)
    result = client.get('/clips/clip_1000.mp4/thumb.jpg')
    assert result.status_code == 200
    def forbidden(*a):
        raise AssertionError('封面未变化时不应重新解码')
    monkeypatch.setattr(web.cv2, 'VideoCapture', forbidden)
    cached = client.get('/clips/clip_1000.mp4/thumb.jpg', headers={'If-None-Match':result.headers['etag']})
    assert cached.status_code == 304
    assert cached.content == b''


def test_thumbnails_for_different_clips_decode_concurrently(tmp_path, monkeypatch):
    import catcam.web as web
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    app, _, recorder, _ = _build(tmp_path)
    first = 'clip_1000.mp4'
    second = next(f'clip_{i}.mp4' for i in range(2000, 2100) if hash(f'clip_{i}.mp4')%32 != hash(first)%32)
    for name in (first, second):
        (recorder.clips_dir/name).write_bytes(b'placeholder')
    barrier = Barrier(2)
    class Capture:
        def __init__(self, path): pass
        def get(self, prop): return 30
        def set(self, prop, value): return True
        def read(self):
            barrier.wait(timeout=3)  # 若仍共用全局解码锁，这里会超时。
            return True, np.zeros((48, 64, 3), np.uint8)
        def release(self): pass
    monkeypatch.setattr(web.cv2, 'VideoCapture', Capture)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda name: client.get(f'/clips/{name}/thumb.jpg'), (first, second)))
    assert all(r.status_code == 200 for r in responses)


def test_high_confidence_machine_labels_remain_reviewable_like_sidebar(tmp_path):
    client, stats, feedback, recorder, names = _paged_fixture(tmp_path, 5)
    feedback.record_machine_label(names[0], True, confidence=1.0)
    feedback.record_machine_label(names[1], False, confidence=0.0)
    feedback.record_machine_label(names[2], True, source='ai', confidence=.99)
    with feedback._conn() as conn:
        conn.execute("INSERT INTO labels(clip_name,is_drinking,source) VALUES (?,1,'human')", (names[3],))
    for i, name in enumerate(names[:3]):
        stats.set_prediction(name, int(i != 1), 'v1', 0.0 if i == 1 else 1.0)
    page = client.get('/api/clips/page?filter=review').json()
    summary = client.get('/api/clips/summary').json()
    assert page['counts']['review'] == len(summary['pending_names']) == 4
    assert page['clips'] == summary['pending_names']
    assert names[3] not in page['clips']
    assert set(names[:3]).issubset(page['clips'])
    compact = client.get('/api/clips/summary', params={'since': page['snapshot']}).json()
    assert compact['pending_count'] == page['counts']['review']


def test_trend_calendar_periods_and_daily_totals(tmp_path):
    from datetime import datetime
    app, stats, _, _ = _build(tmp_path)
    now = datetime.now()
    stats.record_event(now.timestamp(), 'clip_calendar.mp4')
    with sqlite3.connect(stats.db_path) as connection:
        connection.execute("INSERT INTO labels (clip_name,is_drinking) VALUES ('clip_calendar.mp4',1)")
    client = TestClient(app)
    for period, length in [('week', now.weekday()+1), ('month', now.day)]:
        response = client.get('/api/stats/trend', params={'period':period}).json()
        assert len(response['days']) == length
        assert response['days'][-1]['date'] == now.strftime('%m-%d')
        assert sum(day['count'] for day in response['days']) == response['total'] == 1


def test_drinking_intervals_save_reload_and_reject(tmp_path):
    app, stats, recorder, feedback = _build(tmp_path)
    frame=np.zeros((48,64,3),np.uint8)
    recorder.save_clip([frame]*10,timestamp=1.0)
    name='clip_1000.mp4'
    feedback.label_clip(recorder.clips_dir/name,True)
    stats.record_event(1.0,name)
    client=TestClient(app)
    response=client.post(f'/api/clips/{name}/drinking-intervals',json={'segments':[[.1,.6],[.5,1],[1.5,1.8]]})
    assert response.status_code==200
    assert response.json()['seconds']==pytest.approx(1.2)
    meta=client.get('/api/clips').json()['meta'][name]
    assert meta['drinking_intervals']==[[.1,1],[1.5,1.8]]
    assert client.post(f'/api/clips/{name}/drinking-intervals',json={'segments':[[0,99]]}).status_code==400
    assert client.post('/api/feedback',json={'clip':name,'is_drinking':False,'outcome':'no_action'}).status_code==200
    assert stats.count_between(0,10)==0
    assert 'drinking_intervals' not in client.get('/api/clips').json()['meta'][name]
    assert client.post(f'/api/clips/{name}/drinking-intervals',json={'segments':[[0,1]]}).status_code==400


def test_timeline_strip_has_real_frames_and_uses_cache(tmp_path,monkeypatch):
    import cv2
    app, _, recorder, _ = _build(tmp_path)
    recorder.save_clip([np.full((48,64,3),i*20,np.uint8) for i in range(10)],timestamp=1)
    client=TestClient(app)
    response=client.get('/clips/clip_1000.mp4/timeline.jpg')
    assert response.status_code==200
    image=cv2.imdecode(np.frombuffer(response.content,np.uint8),cv2.IMREAD_COLOR)
    assert image.shape[:2]==(54,768)
    assert image[:,:96].mean()<image[:,-96:].mean()
    monkeypatch.setattr(cv2,'VideoCapture',lambda *_:pytest.fail('时间轴缩略帧应使用缓存'))
    assert client.get('/clips/clip_1000.mp4/timeline.jpg').content==response.content
    assert client.get('/clips/clip_1000.mp4/timeline.jpg',headers={'If-None-Match':response.headers['etag']}).status_code==304


def test_zoomed_timeline_samples_only_requested_window(tmp_path):
    import cv2
    app, _, recorder, _ = _build(tmp_path)
    recorder.save_clip([np.full((48,64,3),i*10,np.uint8) for i in range(20)],timestamp=1)
    client=TestClient(app)
    def pixels(part):
        response=client.get('/clips/clip_1000.mp4/timeline.jpg',params={'parts':4,'part':part})
        assert response.status_code==200
        return cv2.imdecode(np.frombuffer(response.content,np.uint8),cv2.IMREAD_COLOR)
    first,last=pixels(0),pixels(3)
    assert first[:,-96:].mean()<last[:,:96].mean()
    assert client.get('/clips/clip_1000.mp4/timeline.jpg?parts=17').status_code==400
    assert client.get('/clips/clip_1000.mp4/timeline.jpg?parts=4&part=4').status_code==400
