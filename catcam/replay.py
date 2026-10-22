from __future__ import annotations

import hashlib
import json
import math
import re
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from catcam.source import WoSeeCapture


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        return None


class ReplayArchive:
    def __init__(self, recorder, stats, feedback, video_source="", opener=None):
        self.recorder, self.stats, self.feedback = recorder, stats, feedback
        self.bridge = WoSeeCapture(video_source) if video_source.startswith("wosee:") else None
        self.opener = opener or build_opener(ProxyHandler({}), NoRedirect())

    def read_json(self, path):
        with self.opener.open(self.bridge.base + path, timeout=5) as response:
            body = response.read(16 * 1024 * 1024 + 1)
        if len(body) > 16 * 1024 * 1024:
            raise ValueError("录像目录超过读取限制")
        result = json.loads(body)
        if not isinstance(result, dict):
            raise ValueError("录像服务返回了无效目录")
        return result

    def remote_catalog(self):
        result = self.read_json('/api/recording/catalog?' + urlencode({'serial': self.bridge.serial}))
        if result.get('serial') != self.bridge.serial:
            raise ValueError("录像服务当前选择了另一台摄像头，请在 WoSee 管理台切回本项目设备")
        return result

    def events(self, start, end):
        if start is None:
            return []
        with self.stats._conn() as connection:
            rows = connection.execute(
                'SELECT MIN(ts),clip_name FROM events WHERE clip_name IS NOT NULL GROUP BY clip_name '
                'HAVING MIN(ts)>=? AND MIN(ts)<=? ORDER BY MIN(ts),clip_name', (start, end)).fetchall()
        stored = self.feedback.clip_metadata()
        metadata = self.stats.clip_metadata()
        representatives = {event['clip_name'] for event in self.stats.drinking_records()}
        events = []
        for timestamp, name in rows:
            label = stored['meta'].get(name) or {}
            entry = metadata.get(name, {})
            drinking = label.get('is_drinking') is True
            kind = 'drinking' if name in representatives else 'continuation' if drinking else 'review' if not label and entry.get('review_candidate', True) else 'activity'
            events.append({'ts': timestamp, 'clip': name, 'kind': kind,
                           'label': {'drinking': '确认喝水', 'continuation': '同次喝水 · 后续片段',
                                     'review': '待复核', 'activity': '画面活动'}[kind],
                           'note': label.get('reason') or stored['notes'].get(name) or '',
                           'seconds': label.get('effective_drinking_seconds')})
        return events

    def catalog(self):
        segments = []
        if self.bridge:
            data = self.remote_catalog()
            status = self.read_json('/api/recording')
            if status.get('serial') != self.bridge.serial or status.get('directory') != data.get('directory'):
                raise ValueError("录像目录已切换，请刷新回放")
            archive = data.get('archive_id') or hashlib.sha256(str(data.get('directory', '')).encode()).hexdigest()[:24]
            for item in data.get('segments', []):
                name = item.get('name', '')
                start, end = item.get('start'), item.get('end')
                if (not isinstance(name, str) or not re.fullmatch(r'[0-9a-f]{32}/[0-9]{6}\.mp4', name) or
                        not all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end)) or end <= start):
                    continue
                segments.append({'id': name, 'start': start, 'end': end, 'width': item.get('width'),
                                 'height': item.get('height'), 'url': '/api/replay/media/wosee/' + name + '?' + urlencode({'archive': archive})})
            source, label = 'wosee', 'WoSee · 监控滚动录像'
            retention = status.get('config', {}).get('retention_seconds')
            active, enabled = status.get('active', False), status.get('enabled', False)
        else:
            stored = self.feedback.clip_metadata()
            for path in self.recorder.list_clips():
                match = re.fullmatch(r'clip_([0-9]+)\.mp4', path.name)
                duration = stored['durations'].get(path.name)
                if not match or not duration or duration <= 0:
                    continue
                start = int(match.group(1)) / 1000
                segments.append({'id': path.name, 'start': start, 'end': start + duration,
                                 'url': '/api/replay/media/local/' + path.name})
            source, label = 'local', '本机监控录像 · 活动时段'
            retention = self.recorder.retention_days * 86400 if self.recorder.retention_days else None
            active, enabled = False, False
        segments.sort(key=lambda item: (item['start'], item['id']))
        start = segments[0]['start'] if segments else None
        end = max(item['end'] for item in segments) if segments else None
        return {'source': source, 'source_label': label, 'segments': segments,
                'start': start, 'end': end, 'events': self.events(start, end),
                'retention_seconds': retention, 'active': active, 'enabled': enabled,
                'segment_seconds': sum(item['end'] - item['start'] for item in segments)}

    def open_remote(self, name, archive, range_header=None):
        if not self.bridge or not re.fullmatch(r'[0-9a-f]{32}/[0-9]{6}\.mp4', name):
            raise ValueError("录像标识无效")
        if not re.fullmatch(r'[0-9a-f]{24}', archive or ''):
            raise ValueError("录像目录标识无效，请刷新回放")
        query = urlencode({'serial': self.bridge.serial, 'archive': archive})
        headers = {}
        if range_header:
            if not re.fullmatch(r'bytes=(?:[0-9]+-[0-9]*|-[0-9]+)', range_header):
                raise ValueError("只支持单段视频范围请求")
            headers['Range'] = range_header
        request = Request(self.bridge.base + '/api/recording/files/' + name + '?' + query, headers=headers)
        return self.opener.open(request, timeout=5)

    def local_path(self, name):
        if self.bridge or not re.fullmatch(r'clip_[0-9]+\.mp4', name):
            return None
        path = self.recorder.clips_dir / name
        if path.is_symlink() or not path.is_file() or self.recorder.clips_dir.is_symlink():
            return None
        return path
