import sqlite3
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


def test_clip_download_default_has_attachment(tmp_path):
    app, _, recorder, _ = _build(tmp_path)
    recorder.save_clip([np.zeros((48, 64, 3), np.uint8)], timestamp=1.0)
    r = TestClient(app).get("/clips/clip_1000.mp4")
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")


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


def test_dispenser_refill_then_remaining_drops_with_drinks(tmp_path):
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
    assert d2["drinks_since_refill"] == 3
    assert d2["remaining_ml"] == 2000 - 3 * 30   # 每次 30ml → 剩 1910
    assert d2["today_ml"] >= 3 * 30


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
    assert body["total"] == 2                      # 只数确认喝水
    assert sum(body["hourly"]) == 2 and sum(body["weekday"]) == 2
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


def test_clips_includes_label_meta(tmp_path):
    app, _, recorder, feedback = _build(tmp_path)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    recorder.save_clip([frame], timestamp=1.0)
    feedback.label_clip(recorder.clips_dir / "clip_1000.mp4", True,
                        source="ai", confidence=0.8, reason="舔水")
    client = TestClient(app)
    body = client.get("/api/clips").json()
    assert "meta" in body
    m = body["meta"]["clip_1000.mp4"]
    assert m["source"] == "ai" and m["reason"] == "舔水" and m["is_drinking"] is True


def _build_with_registry(tmp_path):
    from catcam.models import ModelRegistry
    from catcam.classifier import ActiveModel
    stats = StatsStore(tmp_path / "s.db")
    recorder = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5)
    feedback = FeedbackStore(db_path=tmp_path / "f.db", training_dir=tmp_path / "train")
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    active_model = ActiveModel()
    app = create_app(stats, recorder, feedback, lambda: None, recorder.clips_dir,
                     registry=registry, active_model=active_model)
    return app, stats, recorder, feedback, registry, active_model


def test_activate_s3d_head_version_does_not_500(tmp_path):
    app, stats, recorder, feedback, registry, active_model = _build_with_registry(tmp_path)
    head_path = tmp_path / "videohead_1.npz"; head_path.write_bytes(b"x")
    registry.add(path=head_path, top1=0.9, image_counts={"drinking": 5, "not_drinking": 5},
                 label_counts=None, base="s3d+head", epochs=300, imgsz=224, created_ts=1.0)
    client = TestClient(app)
    r = client.post("/api/model/activate", json={"id": "v1", "mode": "shadow"})
    assert r.status_code == 200
    assert registry.active_id() == "v1" and registry.active_mode() == "shadow"
    # 视频版本不该被塞进单帧 active_model（否则会加载成一个 bogus YOLO）；应清空、留给视频裁判。
    assert active_model.active_id is None
    assert "重启" in (r.json().get("note") or "")


class _FakeVideoTrainer:
    def __init__(self): self.started = 0; self._state = "idle"; self.last_rebuild = None
    def start(self, rebuild=False): self.started += 1; self.last_rebuild = rebuild; self._state = "running"; return True
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
