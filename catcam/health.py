"""监控完整度和饮水习惯变化。只描述已观测行为，不推断摄水量或疾病。"""
from __future__ import annotations

from catcam.database import connect
import time
from datetime import datetime, timedelta
from pathlib import Path


class MonitoringStore:
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        with connect(self.db_path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS monitoring (ts REAL PRIMARY KEY, seconds REAL NOT NULL, usable INTEGER NOT NULL)')
        self._last_ts = None
        self._frame_ts = None
        self._last_quality = None

    def observe(self, now, quality):
        # 仅连续且两端都有有效画面的区间计入覆盖率；重启/断流不补算。
        seconds = 0 if self._last_ts is None or now-self._last_ts > 65 else max(0, min(30, now-self._last_ts))
        usable = bool(quality['usable'] and self._last_quality and self._last_quality['usable'])
        with connect(self.db_path) as c:
            c.execute('INSERT OR REPLACE INTO monitoring VALUES (?,?,?)', (now, seconds, int(usable)))
        self._last_ts, self._last_quality = now, dict(quality)
        self._frame_ts=now

    def touch(self, now):
        """更新采集心跳，不增加数据库采样写入。"""
        self._frame_ts=now

    def status(self, now=None):
        now = time.time() if now is None else now
        if self._frame_ts is None or now-self._frame_ts > 5 or self._last_quality is None:
            return {'usable': False, 'status': 'offline', 'message': '摄像头没有新画面，当前监控中断。'}
        return {**self._last_quality, 'last_frame_ts': self._frame_ts}

    def coverage(self, start, end):
        with connect(self.db_path) as c:
            rows = c.execute('SELECT ts, seconds, usable FROM monitoring WHERE ts > ? AND ts < ?', (start, end+30)).fetchall()
        return sum(max(0, min(ts, end)-max(ts-seconds, start)) for ts,seconds,usable in rows if usable)


def health_summary(stats, monitoring, now=None):
    now = datetime.now() if now is None else now
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = now.timestamp()
    history = []
    for offset in range(14, 0, -1):
        d = start-timedelta(days=offset)
        stop = d+timedelta(days=1)
        coverage = monitoring.coverage(d.timestamp(), stop.timestamp()) if monitoring else 0
        events = stats.events_between(d.timestamp(), stop.timestamp())
        with stats._conn() as c:
            pending_day = c.execute('SELECT COUNT(*) FROM events e LEFT JOIN labels l ON e.clip_name=l.clip_name WHERE e.ts>=? AND e.ts<? AND e.review_candidate=1 AND l.clip_name IS NULL',
                                    (d.timestamp(), stop.timestamp())).fetchone()[0]
        history.append({'date': d.strftime('%Y-%m-%d'), 'count': len(events),
                        'coverage_hours': round(coverage/3600, 2), 'pending_review': pending_day,
                        'comparable': coverage >= 18*3600 and pending_day == 0})
    baseline_days = [d for d in history if d['comparable']]
    baseline = sum(d['count'] for d in baseline_days)/len(baseline_days) if len(baseline_days) >= 3 else None
    events = stats.events_between(start.timestamp(), end)
    coverage = monitoring.coverage(start.timestamp(), end) if monitoring else 0
    elapsed = max(1, end-start.timestamp())
    coverage_ratio = min(1, coverage/elapsed)
    with stats._conn() as c:
        pending = c.execute('SELECT COUNT(*) FROM events e LEFT JOIN labels l ON e.clip_name=l.clip_name WHERE e.ts>=? AND e.ts<? AND e.review_candidate=1 AND l.clip_name IS NULL', (start.timestamp(),end)).fetchone()[0]
        last = c.execute('SELECT MAX(e.ts) FROM events e JOIN labels l ON l.clip_name=e.clip_name WHERE l.is_drinking=1').fetchone()[0]
    status, message = 'insufficient_data', '有效监控满18小时的历史日不足3天，尚不能建立饮水习惯基线。'
    # 当天只与历史相同时间段比较，避免上午与历史整天的总次数相比。
    expected = None
    if baseline is not None:
        expected = sum(stats.count_between((start-timedelta(days=i)).timestamp(),
                      (now-timedelta(days=i)).timestamp()) for i in range(1,15)
                      if history[14-i]['comparable']) / len(baseline_days)
        if coverage_ratio < .75 or pending:
            status, message = 'monitoring_limited', '监控存在缺失或待复核片段，先检查补光、摄像头和标注。'
        elif elapsed < 6*3600 or expected < 2:
            status, message = 'observing', '继续观察今天的饮水记录，当前时段样本较少。'
        elif len(events) < expected*.5:
            status, message = 'decreased', '已确认饮水次数低于历史同时段；请结合食欲、精神和排尿观察。'
        elif len(events) > expected*1.8:
            status, message = 'increased', '已确认饮水次数高于历史同时段；请复核重复片段并观察日常状态。'
        else:
            status, message = 'usual_pattern', '已确认饮水次数接近历史同时段习惯。'
    return {'status': status, 'message': message, 'today_count': len(events), 'pending_review': pending,
            'coverage_hours': round(coverage/3600,2), 'coverage_ratio': round(coverage_ratio,3),
            'baseline_daily_count': baseline, 'baseline_days': len(baseline_days),
            'expected_by_now': expected, 'last_confirmed_ts': last, 'history': history,
            'monitoring': monitoring.status(end) if monitoring else {'status':'offline','usable':False},
            'note': '水量为按次数或喝水时长校准的估值；时长默认取录像的80%，人工记录优先。未计入湿粮等其他水分；健康提示基于完整监控下的习惯变化。持续异常或伴随不适时请咨询兽医。'}
