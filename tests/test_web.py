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
