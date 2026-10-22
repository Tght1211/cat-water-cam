"""使用用户确认的原始参考画面，保守判断出水口停止与恢复。"""
from pathlib import Path
from datetime import datetime
from collections import deque
import json
import math
import threading
import time

import cv2
import numpy as np

from catcam.nightvision import assess_quality


class FlowMonitor:
    def __init__(self, directory, roi=(0, 0, 1, 1), dwell=20):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.dwell = dwell
        self.roi = tuple(roi)
        metadata = self.directory / 'settings.json'
        if metadata.exists():
            self.roi = tuple(json.loads(metadata.read_text())['roi'])
        self.refs = {}
        for kind in ('stopped', 'flowing'):
            frame = cv2.imread(str(self.directory / f'{kind}.jpg')) if (self.directory / f'{kind}.jpg').exists() else None
            if frame is not None:
                self.refs[kind] = self._feature(frame)
        self.previous = None
        self._motion_window = deque(maxlen=5)
        self.last_seen = 0
        self.pending = None
        self.pending_samples = 0
        self.since = 0
        self.state = 'learning' if self.refs else 'unconfigured'
        self.alert = False
        self.fault_since = 0
        self.changed_at = 0
        self.distance = None
        self._evidence_frames = deque(maxlen=8)  # 每半秒一帧，仅保留最近几秒
        history = self.directory / 'transitions.jsonl'
        if history.exists():
            lines = history.read_text().splitlines()
            if lines:
                try:
                    last = json.loads(lines[-1])
                    self.alert = bool(last.get('alert'))
                    self.fault_since = last.get('fault_since',last['ts'] if self.alert else 0)
                except (ValueError, TypeError):
                    pass

        self.incidents_path = self.directory / 'incidents.json'
        self.incidents = json.loads(self.incidents_path.read_text()) if self.incidents_path.exists() else []
        if not self.incidents_path.exists() and history.exists():
            active = None
            for line in history.read_text().splitlines():
                try: item = json.loads(line)
                except ValueError: continue
                if item.get('alert') and active is None:
                    start = item.get('fault_since') or item['ts']
                    active = {'id':str(start),'start':start,'end':None,'dismissed':False}
                    self.incidents.append(active)
                elif not item.get('alert') and item.get('state')=='flowing' and active is not None:
                    active['end'] = item['ts']
                    active = None
            if self.incidents:
                self._save_incidents()
        if self.alert and not any(row['end'] is None for row in self.incidents):
            started = self.fault_since or time.time()
            self.incidents.append({'id':str(started),'start':started,'end':None,'dismissed':False})
            self._save_incidents()

    def _save_incidents(self):
        temp = self.incidents_path.with_suffix('.tmp')
        temp.write_text(json.dumps(self.incidents))
        temp.replace(self.incidents_path)

    def _set_alert(self, value, now):
        if value and not self.alert:
            self.fault_since = now
            self.incidents.append({'id':str(now),'start':now,'end':None,'dismissed':False})
            self._save_incidents()
        elif not value and self.alert:
            for row in self.incidents:
                if row['end'] is None:
                    row['end'] = now
            self.fault_since = 0
            self._save_incidents()
        self.alert = value
        if value:
            self._save_evidence(now)

    def _remember_frame(self, now, frame):
        self._evidence_frames.append((now, frame.copy()))
        while self._evidence_frames and now-self._evidence_frames[0][0] > 3:
            self._evidence_frames.popleft()

    def _save_evidence(self, now):
        """每次停水固定保存约两秒内的三帧；重试邮件仍使用同一组现场证据。"""
        incident = next((row for row in reversed(self.incidents) if row['end'] is None), None)
        frames = list(self._evidence_frames)
        if incident is None or incident.get('evidence_times') or len(frames) < 3:
            return
        if now-frames[-1][0] > 1 or any(b[0]-a[0] > 1 for a,b in zip(frames,frames[1:])):
            return
        end = frames[-1][0]
        selected = [min(frames, key=lambda item: abs(item[0]-target)) for target in (end-2, end-1, end)]
        if any(abs(item[0]-target) > .35 for item,target in zip(selected,(end-2,end-1,end))):
            return  # 不能复制一帧冒充连续画面，也不能跨断流拼图。
        directory = self.directory / f"evidence_{int(incident['start']*1000)}"
        try:
            directory.mkdir(exist_ok=True)
            for i, (_, frame) in enumerate(selected):
                ok, encoded = cv2.imencode('.jpg', frame)
                if not ok:
                    return
                temporary = directory / f'{i}.tmp'
                temporary.write_bytes(encoded.tobytes())
                temporary.replace(directory / f'{i}.jpg')
            incident['evidence_times'] = [ts for ts, _ in selected]
            self._save_incidents()
        except OSError:
            incident.pop('evidence_times', None)
            # 图片保存失败不能阻塞采集；邮件会等待下一次完整证据。

    def evidence(self, incident_start):
        """返回指定停水事件的三张原始现场图与采集时间，按时间顺序排列。"""
        with self.lock:
            incident = next((row for row in self.incidents if row['start'] == incident_start), None)
            if incident is None or len(incident.get('evidence_times', [])) != 3:
                return []
            directory = self.directory / f"evidence_{int(incident['start']*1000)}"
            try:
                return [(ts, (directory / f'{i}.jpg').read_bytes())
                        for i, ts in enumerate(incident['evidence_times'])]
            except OSError:
                return []

    def was_stopped(self,start,end):
        with self.lock:
            return any(row["start"]<end and (row["end"] is None or row["end"]>start) for row in self.incidents)

    def dismiss(self, incident_id):
        with self.lock:
            for row in self.incidents:
                if row['id']==incident_id and row['end'] is not None:
                    for earlier in self.incidents:
                        if earlier['end'] is not None and earlier['end']<=row['end']:
                            earlier['dismissed'] = True
                    self._save_incidents()
                    return self.status()
            raise ValueError('没有可关闭的恢复提醒')

    def _feature(self, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.roi
        crop = frame[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
        gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (64, 48)).astype(np.float32)/255
        return gray - gray.mean()

    @staticmethod
    def _distance(feature, reference):
        # Compare small translations to tolerate slight camera/bowl movement.
        border = 3
        core = feature[border:-border,border:-border]
        best = 1.0
        for dy in range(-border,border+1):
            for dx in range(-border,border+1):
                sample = reference[border+dy:reference.shape[0]-border+dy,border+dx:reference.shape[1]-border+dx]
                sample = sample-sample.mean()
                target = core-core.mean()
                best = min(best,float(np.mean(np.abs(target-sample))))
        return best

    def _state(self, state, now):
        if self.state != state:
            self.state = state
            self.changed_at = now
            with (self.directory / 'transitions.jsonl').open('a') as log:
                log.write(json.dumps({'ts': now, 'state': state, 'alert': self.alert, 'fault_since':self.fault_since})+'\n')

    def calibrate(self, kind, frame, usable=True, occluded=False):
        if kind not in ('stopped', 'flowing'):
            raise ValueError('请选择停止出水或正在出水')
        if frame is None or not usable or occluded:
            raise ValueError('请在光线充足、猫咪未遮挡出水口时确认')
        with self.lock:
            if not cv2.imwrite(str(self.directory / f'{kind}.jpg'), frame):
                raise ValueError('参考画面保存失败')
            self.refs[kind] = self._feature(frame)
            temp = self.directory / 'settings.tmp'
            temp.write_text(json.dumps({'roi': self.roi}))
            temp.replace(self.directory / 'settings.json')
            self.pending = None
            self.previous = None
            self._motion_window.clear()
            self.last_seen = time.time()
            self._remember_frame(self.last_seen, frame)
            self._set_alert(kind == 'stopped', self.last_seen)
            self._state('suspected_stopped' if self.alert else 'flowing', self.last_seen)
            return self.status()

    def set_roi(self, roi):
        try:
            roi = tuple(float(v) for v in roi)
        except (TypeError, ValueError):
            raise ValueError('区域坐标无效')
        if len(roi) != 4 or not all(0 <= v <= 1 for v in roi) or roi[2]-roi[0] < .03 or roi[3]-roi[1] < .03:
            raise ValueError('区域过小或超出画面')
        with self.lock:
            self.roi = roi
            temp = self.directory / 'settings.tmp'
            temp.write_text(json.dumps({'roi': roi}))
            temp.replace(self.directory / 'settings.json')
            for kind in ('stopped', 'flowing'):
                (self.directory / f'{kind}.jpg').unlink(missing_ok=True)
            self.refs.clear()
            self.previous = None
            self._motion_window.clear()
            self.pending = None
            self._evidence_frames.clear()
            self._state('unconfigured', time.time())
            # A new crop cannot establish that a previously observed fault is fixed.
            return self.status()

    def observe(self, now, frame, usable, occluded=False):
        with self.lock:
            if now-self.last_seen < .5:
                return
            gap = now-self.last_seen
            self.last_seen = now
            if gap > 3:
                self.pending = None
                self.previous = None
                self._motion_window.clear()
                self._evidence_frames.clear()
            valid = usable and not occluded and assess_quality(frame, self.roi, 50)['usable']
            if valid:
                self._remember_frame(now, frame)
                if self.alert:
                    self._save_evidence(now)
            else:
                self._evidence_frames.clear()
            # 未校准是明确的配置状态，不能被运动或遮挡掩盖成含糊的「未知」。
            if len(self.refs) < 2:
                self.pending = None
                self.pending_samples = 0
                self._state('unconfigured', now)
                return
            if not valid:
                self.pending = None
                self.previous = None
                self._motion_window.clear()
                self._state('unknown', now)
                return
            feature = self._feature(frame)
            motion = float(np.mean(np.abs(feature-self.previous))) if self.previous is not None else None
            self.previous = feature
            self._motion_window.append(feature)
            # 比较整段窗口与其中位画面，避免只看相邻帧漏掉缓慢/周期性水流。
            sequence = np.stack(self._motion_window)
            temporal_change = float(np.max(np.mean(np.abs(sequence-np.median(sequence, axis=0)), axis=(1, 2))))
            distances = {k: self._distance(feature,v) for k, v in self.refs.items()}
            self.distance = distances.get('stopped')
            target = None
            stopped = distances.get('stopped', 1)
            flowing = distances.get('flowing', 1)
            # Overlapping references cannot distinguish a stopped pump from a running one.
            separation = float(np.mean(np.abs(self.refs['stopped']-self.refs['flowing']))) if len(self.refs)==2 else 1
            margin = max(.003, separation*.2)
            if stopped < .035 and flowing-stopped > margin and motion is not None and motion < .012 and temporal_change < .012:
                target = 'suspected_stopped'
            elif flowing < .035 and stopped-flowing > margin:
                target = 'flowing'
            if target is None:
                self.pending = None
                self._state('unknown', now)
            elif self.pending != target:
                self.pending, self.since = target, now
                self.pending_samples = 1
                self._state('learning', now)
            else:
                self.pending_samples += 1
                # 持续时间与有效帧数双门槛，稀疏采样不能冒充连续确认。
                if now-self.since >= self.dwell and self.pending_samples >= max(3, math.ceil(self.dwell*1.5)):
                    self._set_alert(target == 'suspected_stopped', now)
                    self._state(target, now)

    def status(self, now=None):
        with self.lock:
            now = time.time() if now is None else now
            state = ('unconfigured' if len(self.refs) < 2 else
                     'unknown' if self.last_seen and now-self.last_seen > 5 else self.state)
            messages = {'suspected_stopped': '画面与停止出水参考相符，请检查蓄水、供电和水泵。',
                        'flowing': '画面与您确认的正常出水参考相符。',
                        'unknown': '暂无法判断：可能遮挡、光线不足、画面变化或摄像头离线。',
                        'learning': f'正在连续比对出水口，需至少 {self.dwell:g} 秒、{max(3, math.ceil(self.dwell*1.5))} 帧一致证据；出水状态不影响喝水记录。',
                        'unconfigured': '尚未校准，不能自动判断停水。请框选出水口，分别保存停止和正常出水画面。'}
            missing = [label for kind, label in (('stopped', '停止出水'), ('flowing', '正常出水'))
                       if kind not in self.refs]
            today = datetime.fromtimestamp(now).replace(hour=0,minute=0,second=0,microsecond=0).timestamp()
            downtime = sum(max(0,min(now,row['end'] or now)-max(today,row['start'])) for row in self.incidents)
            recovery = next((row.copy() for row in reversed(self.incidents) if row['end'] is not None and not row['dismissed']),None)
            return {'incidents':list(reversed(self.incidents[-30:])), 'recovery':recovery, 'today_downtime_seconds':round(downtime), 'state': state, 'alert': self.alert, 'message': messages[state],
                    'calibration_complete': not missing, 'missing_references': missing,
                    'roi': self.roi, 'has_stopped_reference': 'stopped' in self.refs,
                    'has_flowing_reference': 'flowing' in self.refs,
                    'last_observed_at': self.last_seen, 'changed_at': self.changed_at, 'fault_since':self.fault_since,
                    'confirmation_seconds': self.dwell, 'confirmation_frames': max(3, math.ceil(self.dwell*1.5)),
                    'pending_samples': self.pending_samples if self.pending else 0,
                    'method': 'local_reference_sequence', 'stopped_distance': self.distance}
