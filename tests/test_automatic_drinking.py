"""整段裁判、喝水列表与邮件共用一次确认和一分钟去重。"""
from types import SimpleNamespace
import time

from fastapi.testclient import TestClient

from catcam.config import Config
from catcam.dispenser import DispenserStore
from catcam.feedback import FeedbackStore
from catcam.judge import Verdict, route_clip
from catcam.mailer import Emailer
from catcam.notifications import EmailNotifications
from catcam.recorder import ClipRecorder
from catcam.session import SessionResult
from catcam.stats import StatsStore
from catcam.web import create_app


def test_local_confirmation_appears_in_records_and_emails_without_human_label(tmp_path):
    cfg = Config(data_dir=str(tmp_path), mail_enabled=True, smtp_user='a@example.com',
                 smtp_password='test', mail_to='b@example.com')
    stats, feedback = StatsStore(cfg.db_path), FeedbackStore(cfg.db_path, cfg.training_dir)
    recorder = ClipRecorder(cfg.clips_dir, 10, 10)
    emailer = Emailer(cfg)
    mail = EmailNotifications(cfg, emailer, stats, feedback, DispenserStore(cfg.db_path),
                              None, SimpleNamespace(status=lambda *a: {'alert': False}))
    now = time.time()
    with mail.conn() as c:
        c.execute("UPDATE email_meta SET value=? WHERE key='enabled_since'", (now-1,))
    judge = SimpleNamespace(version='v13', judge=lambda p: Verdict(True, .98, '连续舔水', 'v13'))
    for offset in (0, 20, 80):
        ts = now + offset
        name = f'clip_{int(ts*1000)}.mp4'
        (cfg.clips_dir/name).touch()
        session = SessionResult(ts, name, None, ts-3, ts+10,
                                presence_start=ts-2, presence_end=ts+7)
        stats.record_event(ts, name, session=session)
        result = route_clip(clip_path=cfg.clips_dir/name, start_ts=ts, photo=None,
                            ai_labeler=None, local_judge=judge, mode='gate',
                            emailer=emailer, stats=stats, feedback=feedback)
        assert result['authority'].drinking
        assert feedback.label_meta(name)['source'] == 'local'
    client = TestClient(create_app(stats, recorder, feedback, lambda: None, cfg.clips_dir))
    data = client.get('/api/clips').json()
    assert len(data['drinking_records']) == stats.count_between(now, now+100) == 2
    assert all(data['labels'].values())
    assert data['sessions'][name]['presence_end'] == now+87
    assert data['sessions'][name]['end_reason'] == 'left'
    sent = []
    mail.build = lambda kind, payload: (payload['clip'], '', {})
    emailer.send = lambda *args: sent.append(args[0])
    mail.tick(now+100)
    mail.tick(now+110)
    assert set(sent) == set(data['drinking_records'])
    assert len(sent) == 2
