from datetime import datetime
from types import SimpleNamespace
import sqlite3
import time

import numpy as np

from catcam.config import Config
from catcam.dispenser import DispenserStore
from catcam.feedback import FeedbackStore
from catcam.mailer import Emailer, mail_error
from catcam.notifications import EmailNotifications
from catcam.stats import StatsStore


def build(tmp_path):
    cfg=Config(data_dir=str(tmp_path),mail_enabled=True,smtp_user='from@example.com',smtp_password='private',mail_to='to@example.com')
    stats=StatsStore(cfg.db_path)
    feedback=FeedbackStore(cfg.db_path,cfg.training_dir)
    dispenser=DispenserStore(cfg.db_path)
    flow=SimpleNamespace(status=lambda *args: {'alert':False,'changed_at':0})
    service=EmailNotifications(cfg,Emailer(cfg),stats,feedback,dispenser,None,flow)
    return service


def since(service, timestamp):
    with service.conn() as c:
        c.execute("UPDATE email_meta SET value=? WHERE key='enabled_since'",(timestamp,))


def confirm(service,name,now,drinking=True):
    service.stats.record_event(now,name)
    with service.conn() as c:
        c.execute("INSERT OR REPLACE INTO labels(clip_name,is_drinking,ts,source) VALUES(?,?,?,'human')",(name,int(drinking),now))


def test_each_confirmed_drink_and_restart_dedup(tmp_path):
    service=build(tmp_path)
    now=time.time()
    since(service,now-60)
    confirm(service,'a.mp4',now)
    confirm(service,'b.mp4',now+1)
    confirm(service,'c.mp4',now+61)
    confirm(service,'no.mp4',now+1,False)
    sent=[]
    service.emailer.send=lambda *args: sent.append(args)
    service.tick(now+2)
    assert len(sent)==2  # 一分钟内的短片不重复发；相隔一分钟的新记录正常发。
    with service.conn() as c:
        assert {r[0] for r in c.execute("SELECT key FROM email_jobs WHERE kind='drink'")} == {'drink:a.mp4','drink:c.mp4'}
    restarted=build(tmp_path)
    restarted.emailer.send=lambda *args: sent.append(args)
    restarted.tick(now+5)
    assert len(sent)==2
    assert service.status()['has_password']
    assert 'private' not in str(service.status())


def test_retry_and_corrected_label_cancellation(tmp_path):
    service=build(tmp_path)
    now=time.time()
    since(service,now-60)
    confirm(service,'a.mp4',now)
    def fail(*args): raise TimeoutError('private')
    service.emailer.send=fail
    service.tick(now)
    assert service.status()['pending']==1
    assert 'private' not in service.status()['last_error']
    sent=[]
    service.emailer.send=lambda *args: sent.append(args)
    service.tick(now+30)
    assert not sent
    service.tick(now+61)
    assert len(sent)==1
    confirm(service,'b.mp4',now+62)
    service.discover(now+62)
    with service.conn() as c:
        c.execute("UPDATE labels SET is_drinking=0 WHERE clip_name='b.mp4'")
    service.process(now+63)
    assert len(sent)==1


def test_23_schedule_sunday_replaces_daily_and_restart(tmp_path):
    service=build(tmp_path)
    monday=datetime(2026,10,5,0).timestamp()
    sunday=datetime(2026,10,11,23).timestamp()
    since(service,monday)
    service.discover(datetime(2026,10,5,22,59).timestamp())
    with service.conn() as c:
        assert c.execute('SELECT COUNT(*) FROM email_jobs').fetchone()[0]==0
    service.discover(datetime(2026,10,5,23).timestamp())
    service.discover(sunday)
    service.discover(sunday+10)
    with service.conn() as c:
        kinds=[r[0] for r in c.execute('SELECT kind FROM email_jobs')]
    assert kinds==['daily','weekly']
    import json
    with service.conn() as c:
        payload=json.loads(c.execute("SELECT payload FROM email_jobs WHERE kind='weekly'").fetchone()[0])
    assert payload['start']==monday
    assert payload['end']==sunday


def test_disabled_no_queue_and_fault_reminder(tmp_path):
    service=build(tmp_path)
    now=time.time()
    since(service,now-60)
    service.cfg.mail_enabled=False
    confirm(service,'a.mp4',now)
    service.tick(now)
    assert service.status()['pending']==0
    service.cfg.mail_enabled=True
    service.flow.status=lambda *args: {'alert':True,'changed_at':123}
    service.discover(now)
    service.discover(now+1)
    with service.conn() as c:
        assert c.execute("SELECT COUNT(*) FROM email_jobs WHERE kind='flow'").fetchone()[0]==1
    service.flow.status=lambda *args: {'alert':False,'changed_at':124}
    service.emailer.send=lambda *args: None
    service.process(now+2)
    with service.conn() as c:
        assert c.execute("SELECT state FROM email_jobs WHERE kind='flow'").fetchone()[0]=='cancelled'


def test_summary_counts_only_confirmed_and_estimates(tmp_path):
    service=build(tmp_path)
    confirm(service,'yes.mp4',100)
    confirm(service,'no.mp4',110,False)
    subject,html,_=service.build('daily',{'start':1,'end':200})
    assert '今日喝水汇总' in subject
    assert '已确认喝水 1 次' in html
    assert '估算饮水量' in html


def test_smtp_ssl_and_starttls(monkeypatch,tmp_path):
    import smtplib
    service=build(tmp_path)
    calls=[]
    class SMTP:
        def __init__(self,*args,**kwargs): calls.append('connect')
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def login(self,*args): calls.append('login')
        def sendmail(self,*args): calls.append('send')
        def ehlo(self): calls.append('ehlo')
        def starttls(self,**kwargs): calls.append('tls')
    monkeypatch.setattr(smtplib,'SMTP_SSL',SMTP)
    monkeypatch.setattr(smtplib,'SMTP',SMTP)
    service.emailer.send('test','html',{})
    assert calls==['connect','login','send']
    calls.clear()
    service.emailer.security='starttls'
    service.emailer.send('test','html',{})
    assert calls==['connect','ehlo','tls','ehlo','login','send']
    assert '授权码' in mail_error(smtplib.SMTPAuthenticationError(535,b'private'))


def test_drinking_email_photo_comes_from_video_midpoint(tmp_path):
    import cv2
    service=build(tmp_path)
    service.cfg.clips_dir.mkdir(parents=True,exist_ok=True)
    clip=service.cfg.clips_dir/'midpoint.mp4'
    writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,48))
    assert writer.isOpened()
    for i in range(100):
        color=(0,230,0) if 40<=i<60 else (0,0,230) if i<40 else (230,0,0)
        writer.write(np.full((48,64,3),color,np.uint8))
    writer.release()
    _,_,images=service.build('drink',{'ts':time.time(),'clip':clip.name})
    image=cv2.imdecode(np.frombuffer(images['photo'],np.uint8),cv2.IMREAD_COLOR)
    assert image[:,:,1].mean()>200  # 5 秒的绿帧，不是开头红帧或结尾蓝帧。
    assert image[:,:,0].mean()<20 and image[:,:,2].mean()<20


def test_flow_email_waits_for_three_images_then_sends_in_time_order(tmp_path):
    import cv2
    service = build(tmp_path)
    now = time.time()
    since(service, now-1)
    frames = []
    service.flow.status = lambda *args: {'alert': True, 'fault_since': now, 'changed_at': now}
    service.flow.evidence = lambda start: frames if start == now else []
    sent = []
    service.emailer.send = lambda *args: sent.append(args)
    service.tick(now)
    assert not sent
    for i in range(3):
        jpg = cv2.imencode('.jpg',np.full((48,64,3),60+i*60,np.uint8))[1].tobytes()
        frames.append((now+i, jpg))
    service.tick(now+10)
    assert len(sent) == 1
    subject, html, images = sent[0]
    assert '停止出水' in subject
    assert list(images) == ['flow-1', 'flow-2', 'flow-3']
    assert list(images.values()) == [jpg for _, jpg in frames]
    assert html.index('cid:flow-1') < html.index('cid:flow-2') < html.index('cid:flow-3')
    assert '2.0 秒内' in html
    assert all(datetime.fromtimestamp(ts).strftime('%H:%M:%S') in html for ts, _ in frames)
    service.tick(now+20)
    assert len(sent) == 1


def test_recovered_incident_mail_cannot_be_sent_for_a_later_fault(tmp_path):
    service = build(tmp_path)
    now = time.time()
    since(service, now-1)
    service.flow.status = lambda *args: {'alert': True, 'fault_since': now, 'changed_at': now}
    service.discover(now)
    service.flow.status = lambda *args: {'alert': True, 'fault_since': now+100, 'changed_at': now+100}
    service.emailer.send = lambda *args: (_ for _ in ()).throw(AssertionError('不能发送旧故障邮件'))
    service.process(now+100)
    with service.conn() as c:
        assert c.execute("SELECT state FROM email_jobs WHERE kind='flow'").fetchone()[0] == 'cancelled'
