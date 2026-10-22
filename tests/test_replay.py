import io
import json
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

from catcam.feedback import FeedbackStore
from catcam.recorder import ClipRecorder
from catcam.replay import ReplayArchive
from catcam.stats import StatsStore
from catcam.web import create_app


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


class Opener:
    def __init__(self):
        self.serial = 'SYNTHETIC'
        self.requests = []
        self.directory = '/synthetic/recordings'
        self.media = None
        self.failure = None

    def open(self, request, timeout):
        url = request if isinstance(request, str) else request.full_url
        self.requests.append(request)
        path = urlsplit(url).path
        if path.endswith('/catalog'):
            data = {'serial': self.serial, 'directory': self.directory, 'archive_id': 'a' * 24,
                    'segments': [{'name': 'b' * 32 + '/000001.mp4', 'start': 100, 'end': 130},
                                 {'name': 'b' * 32 + '/000002.mp4', 'start': 130, 'end': 160}]}
            return Response(json.dumps(data).encode())
        if path == '/api/recording':
            return Response(json.dumps({'serial': self.serial, 'directory': self.directory,
                                        'config': {'retention_seconds': 600}, 'active': True, 'enabled': True}).encode())
        if self.failure:
            raise HTTPError(url, self.failure, 'synthetic', {}, None)
        self.media = Response(b'0123456789', 206, {'Content-Length': '10', 'Content-Range': 'bytes 10-19/100',
                                                 'Accept-Ranges': 'bytes'})
        return self.media


def build(tmp_path, video_source='wosee://127.0.0.1:18765/SYNTHETIC'):
    stats = StatsStore(tmp_path / 'db.sqlite')
    feedback = FeedbackStore(stats.db_path, tmp_path / 'training')
    recorder = ClipRecorder(tmp_path / 'clips', 100, 10, retention_days=3)
    opener = Opener()
    archive = ReplayArchive(recorder, stats, feedback, video_source, opener)
    app = create_app(stats, recorder, feedback, lambda: None, recorder.clips_dir, replay_archive=archive)
    return TestClient(app), archive, opener


def test_wosee_catalog_uses_rolling_archive_and_time_aligned_records(tmp_path):
    client, archive, opener = build(tmp_path)
    for timestamp, name in [(110, 'drink'), (125, 'continuation'), (140, 'review')]:
        archive.stats.record_event(timestamp, name)
    with archive.feedback._conn() as connection:
        for name in ('drink', 'continuation'):
            connection.execute("INSERT INTO labels(clip_name,is_drinking,source) VALUES (?,1,'human')", (name,))
    (archive.recorder.clips_dir / 'clip_1000.mp4').touch()
    data = client.get('/api/replay').json()
    assert data['source'] == 'wosee'
    assert data['start'] == 100 and data['end'] == 160
    assert data['active'] and data['retention_seconds'] == 600
    assert len(data['segments']) == 2
    assert 'clip_1000.mp4' not in str(data['segments'])
    assert [event['kind'] for event in data['events']] == ['drinking', 'continuation', 'review']
    assert [event['ts'] for event in data['events']] == [110, 125, 140]
    assert data['segments'][0]['url'].startswith('/api/replay/media/wosee/')
    assert all('127.0.0.1:18765' in request for request in opener.requests)


def test_remote_media_proxies_ranges_and_closes_upstream(tmp_path):
    client, archive, opener = build(tmp_path)
    url = client.get('/api/replay').json()['segments'][0]['url']
    response = client.get(url, headers={'Range': 'bytes=10-19'})
    assert response.status_code == 206 and response.content == b'0123456789'
    assert response.headers['content-type'] == 'video/mp4'
    assert response.headers['content-range'] == 'bytes 10-19/100'
    assert 'attachment' not in response.headers.get('content-disposition', '')
    request = opener.requests[-1]
    assert request.headers['Range'] == 'bytes=10-19'
    assert 'serial=SYNTHETIC' in request.full_url and 'archive=' + 'a' * 24 in request.full_url
    assert opener.media.closed


def test_wrong_camera_or_directory_change_does_not_fall_back_to_old_clips(tmp_path):
    client, archive, opener = build(tmp_path)
    opener.serial = 'OTHER'
    assert client.get('/api/replay').status_code == 503
    assert client.get('/api/replay/media/local/clip_1000.mp4').status_code == 404
    opener.serial = 'SYNTHETIC'
    for code in (404, 409, 416):
        opener.failure = code
        response = client.get('/api/replay/media/wosee/' + 'b' * 32 + '/000001.mp4?archive=' + 'a' * 24)
        assert response.status_code == code


def test_media_rejects_untrusted_names_and_multiranges(tmp_path):
    client, archive, opener = build(tmp_path)
    for name in ('x.mp4', 'abcd/000001.mp4', 'b' * 32 + '/../../config.json'):
        with pytest.raises(ValueError):
            archive.open_remote(name, 'a' * 24)
    url = '/api/replay/media/wosee/' + 'b' * 32 + '/000001.mp4?archive=' + 'a' * 24
    assert client.get(url, headers={'Range': 'bytes=1-2,3-4'}).status_code == 400
    assert client.get(url.replace('a' * 24, 'invalid')).status_code == 400
    assert not opener.requests


def test_local_archive_keeps_wall_clock_gaps_and_uses_stored_duration(tmp_path):
    client, archive, opener = build(tmp_path, '')
    for name, duration in [('clip_100000.mp4', 15), ('clip_130000.mp4', 12)]:
        path = archive.recorder.clips_dir / name
        path.write_bytes(b'synthetic-video')
        archive.feedback.record_video_duration(path, duration)
    data = client.get('/api/replay').json()
    assert data['source'] == 'local' and data['retention_seconds'] == 259200
    assert [(item['start'], item['end']) for item in data['segments']] == [(100, 115), (130, 142)]
    assert not opener.requests
    response = client.get(data['segments'][0]['url'], headers={'Range': 'bytes=0-3'})
    assert response.status_code == 206 and response.content == b'synt'
    (archive.recorder.clips_dir / 'clip_100000.mp4').unlink()
    assert client.get(data['segments'][0]['url']).status_code == 404


def test_replay_endpoints_keep_existing_login_protection(tmp_path):
    from catcam.access import hash_password
    client, archive, opener = build(tmp_path)
    app = create_app(archive.stats, archive.recorder, archive.feedback, lambda: None, archive.recorder.clips_dir,
                     replay_archive=archive, web_password_hash=hash_password('synthetic-password'), web_session_secret='synthetic-secret')
    protected = TestClient(app)
    assert protected.get('/api/replay').status_code == 401
    assert protected.get('/api/replay/media/wosee/' + 'b' * 32 + '/000001.mp4?archive=' + 'a' * 24).status_code == 401
    assert not opener.requests
