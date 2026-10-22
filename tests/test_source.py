import io
import json
from urllib.error import HTTPError

import cv2
import numpy as np
import pytest

from catcam.config import Config
from catcam.source import WoSeeCapture, open_capture


def packet(sequence=1, timestamp=1000):
    frame = np.full((32, 48, 3), (30, 100, 180), np.uint8)
    encoded = cv2.imencode(".jpg", frame)[1].tobytes()
    return (f"--wosee\r\nContent-Type: image/jpeg\r\nContent-Length: {len(encoded)}\r\n"
            f"X-Frame-Sequence: {sequence}\r\nX-Frame-Timestamp: {timestamp}\r\nX-Camera-Serial: SYNTHETIC\r\n\r\n").encode() + encoded + b"\r\n"


class Opener:
    def __init__(self, streams):
        self.streams = iter(streams)
        self.requests = []

    def open(self, request, timeout):
        url = request if isinstance(request, str) else request.full_url
        self.requests.append(request)
        if url.endswith("/api/config"):
            return io.BytesIO(json.dumps({"token": "synthetic-workspace-token"}).encode())
        if url.endswith("/api/resume"):
            assert request.headers["X-workspace-token"] == "synthetic-workspace-token"
            return io.BytesIO(json.dumps({"reused": False, "camera": "192.168.1.2"}).encode())
        result = next(self.streams)
        if isinstance(result, Exception):
            raise result
        return io.BytesIO(result)


def test_wosee_reads_real_jpeg_format_without_resizing():
    capture = WoSeeCapture("wosee://127.0.0.1:18765/SYNTHETIC", Opener([packet()]), clock=lambda: 1000)
    available, frame = capture.read()
    assert available
    assert frame.shape == (32, 48, 3)
    assert capture.sequence == 1
    assert capture.timestamp == 1000
    capture.release()
    assert not capture.isOpened()
    assert capture.read() == (False, None)


def test_wosee_suppresses_duplicate_and_stale_frames():
    capture = WoSeeCapture("wosee://localhost/SYNTHETIC", Opener([packet() + packet() + packet(2, 990) + packet(3)]), clock=lambda: 1000)
    assert capture.read()[0]
    assert capture.read() == (False, None)
    assert capture.read() == (False, None)
    assert capture.read()[0]
    assert capture.sequence == 3
    capture.release()


def test_wosee_requests_resume_without_any_camera_password():
    error = HTTPError("http://127.0.0.1:18765/api/video.mjpg", 503, "no camera frame", {}, None)
    opener = Opener([error, packet()])
    capture = WoSeeCapture("wosee://127.0.0.1:18765/SYNTHETIC", opener, clock=lambda: 1000)
    assert capture.read()[0]
    resume = [request for request in opener.requests if not isinstance(request, str) and request.full_url.endswith("/api/resume")][0]
    assert json.loads(resume.data) == {"serial": "SYNTHETIC", "signaling_mode": "cloud", "allow_saved_login": True}
    assert "password" not in resume.data.decode()
    capture.release()


def test_wosee_reconnect_accepts_sequence_reset_after_service_restart():
    now = [1000.0]
    opener = Opener([packet(100, 1000), packet(1, 1001)])
    capture = WoSeeCapture("wosee://localhost/SYNTHETIC", opener, clock=lambda: now[0], sleep=lambda seconds: None)
    assert capture.read()[0]
    assert capture.read() == (False, None)
    now[0] = 1002
    assert capture.read()[0]
    assert capture.sequence == 1
    capture.release()


@pytest.mark.parametrize("source", ["wosee://192.168.1.2/SYNTHETIC", "wosee://user:secret@localhost/SYNTHETIC",
                                     "wosee://localhost/../other", "wosee://localhost/SYNTHETIC?key=bad"])
def test_wosee_admin_access_is_loopback_only(source):
    with pytest.raises(ValueError):
        WoSeeCapture(source)


def test_wosee_rejects_oversized_frame_before_reading_body():
    raw = b"--wosee\r\nContent-Type: image/jpeg\r\nContent-Length: 9000000\r\n\r\n"
    capture = WoSeeCapture("wosee://localhost/SYNTHETIC", Opener([raw]), clock=lambda: 1000)
    assert capture.read() == (False, None)
    assert capture.response is None


def test_wosee_rejects_other_camera_video():
    raw = packet().replace(b"X-Camera-Serial: SYNTHETIC", b"X-Camera-Serial: OTHER")
    opener = Opener([raw])
    capture = WoSeeCapture("wosee://localhost/SYNTHETIC", opener, clock=lambda: 1000)
    assert capture.read() == (False, None)
    assert capture.response is None
    assert "?serial=SYNTHETIC" in opener.requests[0]


def test_usb_and_file_sources_keep_opencv_capture(monkeypatch):
    calls = []
    class Capture:
        def set(self, property_id, value):
            calls.append((property_id, value))
    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: calls.append(source) or capture)
    assert open_capture(Config(video_source="synthetic.mp4", frame_width=800, frame_height=600)) is capture
    assert calls[0] == "synthetic.mp4"
    assert (cv2.CAP_PROP_FRAME_WIDTH, 800) in calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 600) in calls
