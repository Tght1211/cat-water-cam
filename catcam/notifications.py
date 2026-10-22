"""Persistent email queue for confirmed drinks, outlet faults and 23:00 summaries."""
from datetime import datetime, timedelta
from html import escape
import json
from catcam.database import connect
import threading
import time

import cv2

from catcam.mailer import _frame_to_jpg


class EmailNotifications:
    def __init__(self, cfg, emailer, stats, feedback, dispenser, monitoring, flow, indicator=None):
        self.cfg, self.emailer, self.stats = cfg, emailer, stats
        self.feedback, self.dispenser, self.monitoring, self.flow = feedback, dispenser, monitoring, flow
        self.indicator = indicator
        self.lock = threading.RLock()
        self.testing = False
        self.last_test = 0
        self.db_path = cfg.db_path
        self.emailer.managed = True
        with self.conn() as c:
            c.execute('CREATE TABLE IF NOT EXISTS email_jobs (key TEXT PRIMARY KEY, kind TEXT, payload TEXT, state TEXT, attempts INTEGER DEFAULT 0, next_try REAL, sent_at REAL, error TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS email_meta (key TEXT PRIMARY KEY, value REAL)')
            c.execute("INSERT OR IGNORE INTO email_meta VALUES ('enabled_since', ?)", (time.time(),))
        self.last_fault = False

    def conn(self):
        return connect(self.db_path)

    def ready(self):
        return bool(self.cfg.mail_enabled and self.cfg.smtp_host and self.cfg.smtp_user and self.cfg.smtp_password and self.cfg.mail_to)

    def status(self):
        with self.lock, self.conn() as c:
            last = c.execute("SELECT sent_at,kind FROM email_jobs WHERE state='sent' ORDER BY sent_at DESC LIMIT 1").fetchone()
            error = c.execute("SELECT error FROM email_jobs WHERE state='pending' AND error!='' ORDER BY next_try DESC LIMIT 1").fetchone()
            pending = c.execute("SELECT COUNT(*) FROM email_jobs WHERE state='pending'").fetchone()[0]
        return {'enabled':self.cfg.mail_enabled, 'configured': bool(self.cfg.smtp_user and self.cfg.smtp_password and self.cfg.mail_to),
                'smtp_host':self.cfg.smtp_host, 'smtp_port':self.cfg.smtp_port, 'smtp_security':self.cfg.smtp_security,
                'smtp_user':self.cfg.smtp_user, 'mail_to':self.cfg.mail_to, 'has_password':bool(self.cfg.smtp_password),
                'pending':pending, 'last_sent_at':last[0] if last else None, 'last_error':error[0] if error else '',
                'schedule':'每天 23:00；周日 23:00 改发本周汇总（电脑本地时间）', 'testing':self.testing}

    def reconfigure(self):
        with self.lock:
            old = self.emailer
            from catcam.mailer import Emailer
            self.emailer = Emailer(self.cfg)
            self.emailer.managed = True
            with self.conn() as c:
                c.execute("UPDATE email_jobs SET next_try=0,error='' WHERE state='pending'")
            if not old.enabled and self.cfg.mail_enabled:
                with self.conn() as c:
                    c.execute("UPDATE email_meta SET value=? WHERE key='enabled_since'", (time.time(),))

    def enqueue(self, key, kind, payload, now):
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO email_jobs (key,kind,payload,state,next_try,error) VALUES(?,?,?,'pending',?,'')", (key,kind,json.dumps(payload),now))

    def discover(self, now):
        if not self.ready():
            return
        with self.conn() as c:
            since = c.execute("SELECT value FROM email_meta WHERE key='enabled_since'").fetchone()[0]
            rows = c.execute('SELECT e.clip_name, MIN(e.ts) FROM events e JOIN labels l ON e.clip_name=l.clip_name WHERE l.is_drinking=1 AND l.ts>=? AND e.candidate_only=0 GROUP BY e.clip_name', (since,)).fetchall()
        # 邮件与喝水模块使用同一份一分钟去重结果，避免短片段各发一封。
        records = {e['clip_name'] for e in self.stats.drinking_records()}
        for clip, ts in rows:
            if clip in records:
                self.enqueue('drink:'+clip, 'drink', {'clip':clip,'ts':ts},now)
        local = datetime.fromtimestamp(now)
        due = local.replace(hour=23,minute=0,second=0,microsecond=0)
        if local < due:
            due -= timedelta(days=1)
        if due.timestamp() >= since:
            weekly = due.weekday() == 6
            start = due.replace(hour=0) - timedelta(days=6 if weekly else 0)
            self.enqueue('summary:'+due.strftime('%Y-%m-%d'), 'weekly' if weekly else 'daily',
                         {'start':start.timestamp(),'end':due.timestamp()},now)
        status = self.flow.status(now)
        if status['alert'] and not self.last_fault:
            incident_start = status.get('fault_since') or status['changed_at']
            self.enqueue('flow:'+str(incident_start), 'flow', {'ts':now, 'incident_start':incident_start},now)
        self.last_fault = status['alert']

        reminder = self.water_reminder(now)
        if reminder['need_water']:
            state = self.dispenser.get_state()
            key = str(state['last_refill_ts'])+':'+str(state['options'].get('last_add_ts',state['last_refill_ts']))
            self.enqueue('water-low:'+key,'water_low',{'ts':now},now)

    def water_reminder(self, now):
        state = self.dispenser.get_state()
        calibration = self.dispenser.calibration(self.stats)
        reading = state['reading']
        remaining = reading['ml'] if reading else None
        if reading:
            for event in self.stats.events_between(reading['ts'],now):
                meta = self.feedback.label_meta(event['clip_name']) or {}
                seconds = meta.get('effective_drinking_seconds')
                remaining -= seconds*calibration['ml_per_second'] if seconds and calibration['ml_per_second'] else calibration['ml_per_drink']
        return self.dispenser.reminder_status(calibration,max(0,remaining) if remaining is not None else None,self.indicator.status(now) if self.indicator else None)

    def build(self, kind, payload):
        images = {}
        gallery = ''
        if kind == 'drink':
            dt = datetime.fromtimestamp(payload['ts'])
            subject = '🐱 已确认一次喝水 · '+dt.strftime('%m月%d日 %H:%M')
            meta = self.feedback.label_meta(payload['clip']) or {}
            seconds = meta.get('effective_drinking_seconds')
            detail = f'发生于 {dt:%Y-%m-%d %H:%M:%S}'
            if seconds:
                detail += f'，喝水时长约 {seconds:.1f} 秒（录像估值或人工标注）。'
            cap = cv2.VideoCapture(str(self.cfg.clips_dir/payload['clip']))
            try:
                # 取录像时间中点，避开靠近前的空画面。
                count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if count > 0 and fps > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, (count / fps) * 500)
                ok, frame = cap.read()
                if ok:
                    images['photo'] = _frame_to_jpg(frame)
            finally:
                cap.release()
        elif kind in ('daily','weekly'):
            start,end = payload['start'],payload['end']
            events = self.stats.events_between(start,end)
            calibration = self.dispenser.calibration(self.stats)
            total_seconds = 0
            estimated_ml = 0
            for event in events:
                meta = self.feedback.label_meta(event['clip_name']) or {}
                seconds = meta.get('effective_drinking_seconds')
                total_seconds += seconds or 0
                estimated_ml += seconds*calibration['ml_per_second'] if seconds and calibration['ml_per_second'] else calibration['ml_per_drink']
            end_dt = datetime.fromtimestamp(end)
            subject = ('🐱 本周喝水汇总' if kind=='weekly' else '🐱 今日喝水汇总') + f' · {end_dt:%m月%d日}'
            hours = self.monitoring.coverage(start,end)/3600 if self.monitoring else 0
            detail = (f'{datetime.fromtimestamp(start):%m月%d日 %H:%M} 至 {end_dt:%m月%d日 %H:%M}：'
                      f'已确认喝水 {len(events)} 次，估算饮水量 {estimated_ml:.0f} ml，'
                      f'已统计喝水时长约 {total_seconds:.1f} 秒。有效监控 {hours:.1f} 小时。'
                      '饮水量为估值；未记录、待复核或监控缺失不代表没有喝水。')
        elif kind == 'water_low':
            subject = '💧 饮水机水量偏低，请准备加水'
            detail = self.water_reminder(time.time())['message']
        elif kind == 'flow':
            subject = '⚠️ 饮水机疑似停止出水'
            diagnosis = self.dispenser.flow_diagnosis(self.stats, time.time())
            detail = '出水口与停止出水参考相符。'+diagnosis['cause_message']
            evidence = self._flow_evidence(payload)
            if len(evidence) < 3:
                raise ValueError('停水提醒需要三张连续现场图片')
            span = evidence[-1][0]-evidence[0][0]
            gallery = f'<p>以下为约 {span:.1f} 秒内连续采集的三张现场画面，请结合出水口变化判断是否真的停水。</p>'
            for i, (ts, jpg) in enumerate(evidence):
                cid = f'flow-{i+1}'
                images[cid] = jpg
                stamp = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                gallery += (f'<figure style="margin:16px 0"><figcaption>第 {i+1} 张 · {stamp}'
                            f'（+{ts-evidence[0][0]:.1f} 秒）</figcaption>'
                            f'<img src="cid:{cid}" alt="第 {i+1} 张停水现场" '
                            'style="max-width:640px;width:100%;height:auto"></figure>')
        else:
            subject = '喵喵水站 · 测试邮件'
            detail = '邮箱连接成功。每次确认喝水都会提醒；每天 23:00 汇总，周日同一时间改发本周汇总。'
        url = escape(self.emailer.lan_url(), quote=True)
        html = f'<html><body style="font-family:Arial,sans-serif;line-height:1.8;padding:24px"><h2>{escape(subject)}</h2><p>{escape(detail)}</p>'
        if gallery:
            html += gallery
        elif images:
            html += '<img src="cid:photo" style="max-width:560px;width:100%">'
        html += f'<p><a href="{url}">打开喵喵水站</a>（需连接同一局域网）</p></body></html>'
        return subject,html,images

    def _flow_evidence(self, payload):
        return getattr(self.flow, 'evidence', lambda _: [])(payload.get('incident_start', payload['ts']))

    def process(self, now):
        if not self.ready():
            return
        with self.conn() as c:
            jobs = c.execute("SELECT key,kind,payload,attempts FROM email_jobs WHERE state='pending' AND next_try<=? ORDER BY next_try LIMIT 5", (now,)).fetchall()
        for key,kind,data,attempts in jobs:
            payload = json.loads(data)
            if kind == 'flow' and 'incident_start' not in payload:
                # 兼容升级前已排队的停水提醒，仍按原事件读取图片。
                payload['incident_start'] = float(key.split(':', 1)[1])
            if kind == 'drink':
                meta = self.feedback.label_meta(payload['clip']) or {}
                records = {e['clip_name'] for e in self.stats.drinking_records()}
                if meta.get('is_drinking') is not True or payload['clip'] not in records:
                    with self.conn() as c:
                        c.execute("UPDATE email_jobs SET state='cancelled' WHERE key=?",(key,))
                    continue
            flow_status = self.flow.status() if kind == 'flow' else None
            obsolete_flow = flow_status is not None and (
                not flow_status['alert'] or
                (flow_status.get('fault_since') or flow_status.get('changed_at')) != payload['incident_start'])
            if obsolete_flow or (kind=='water_low' and not self.water_reminder(now)['need_water']):
                with self.conn() as c:
                    c.execute("UPDATE email_jobs SET state='cancelled' WHERE key=?",(key,))
                continue
            if kind == 'flow' and len(self._flow_evidence(payload)) < 3:
                with self.conn() as c:
                    c.execute("UPDATE email_jobs SET next_try=?,error=? WHERE key=?",
                              (now+10, '正在采集停水现场的三张连续图片', key))
                continue
            try:
                self.emailer.send(*self.build(kind,payload))
                with self.conn() as c:
                    c.execute("UPDATE email_jobs SET state='sent',sent_at=?,error='' WHERE key=?",(now,key))
            except Exception as e:
                from catcam.mailer import mail_error
                error = mail_error(e)
                with self.conn() as c:
                    c.execute("UPDATE email_jobs SET attempts=attempts+1,next_try=?,error=? WHERE key=?",(now+min(3600,60*2**min(attempts,6)),error,key))

    def test(self):
        with self.lock:
            if not self.ready():
                raise ValueError('请先填写邮箱配置并启用提醒')
            if time.time()-self.last_test < 60:
                raise ValueError('测试邮件每分钟最多发送一次')
            self.last_test = time.time()
            self.testing = True
        self.enqueue('test:'+str(time.time()),'test',{},time.time())
        self.testing = False
        return {'queued':True}

    def tick(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            self.discover(now)
        self.process(now)

    def start(self):
        def run():
            while True:
                try:
                    self.tick()
                except Exception:
                    print('邮件提醒队列暂不可用，稍后重试')
                time.sleep(10)
        threading.Thread(target=run,daemon=True,name='email-notifications').start()
