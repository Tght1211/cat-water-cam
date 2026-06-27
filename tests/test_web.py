import sqlite3
import time
import numpy as np
from fastapi.testclient import TestClient

from catcam.stats import StatsStore
from catcam.recorder import ClipRecorder
from catcam.feedback import FeedbackStore
from catcam.web import create_app


def _build(tmp_path, frame_provider=lambda: None):
    stats = StatsStore(tmp_path / "s.db")
    recorder = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5)
    feedback = FeedbackStore(db_path=tmp_path / "f.db", training_dir=tmp_path / "train")
    app = create_app(stats, recorder, feedback, frame_provider, recorder.clips_dir)
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
