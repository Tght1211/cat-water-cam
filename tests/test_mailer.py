import sqlite3
from datetime import datetime

import numpy as np

from catcam.mailer import Emailer, build_drinking_email
from catcam.stats import StatsStore


class _Cfg:
    mail_enabled = True
    smtp_host = "smtp.qq.com"
    smtp_port = 465
    smtp_user = "a@qq.com"
    smtp_password = "secret"
    mail_to = "b@163.com"
    mail_min_interval_seconds = 600.0
    web_port = 8000


def test_throttle_blocks_within_interval():
    e = Emailer(_Cfg())
    assert e.should_send(1000.0) is True
    e._last_sent = 1000.0
    assert e.should_send(1300.0) is False        # 5 分钟 < 10 分钟间隔
    assert e.should_send(1600.0) is True          # 到点放行


def test_disabled_or_missing_creds_never_send():
    cfg = _Cfg()
    cfg.mail_enabled = False
    assert Emailer(cfg).should_send(0.0) is False
    cfg2 = _Cfg()
    cfg2.mail_to = ""
    assert Emailer(cfg2).should_send(0.0) is False


def test_build_email_has_three_inline_images(tmp_path):
    stats = StatsStore(tmp_path / "db.sqlite")
    now = datetime(2026, 6, 22, 9, 0, 0)
    stats.record_event(now.timestamp(), "clip_1.mp4")
    # 计数口径：只数被确认「喝水」的段 → 标 clip_1 为 is_drinking=1
    with sqlite3.connect(tmp_path / "db.sqlite") as c:
        c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES ('clip_1.mp4', 1, NULL)")
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    subject, html, images = build_drinking_email(stats, frame, now, "http://192.168.1.5:8000")
    assert set(images) == {"photo", "week", "month"}
    assert all(isinstance(v, (bytes, bytearray)) and v for v in images.values())
    assert "cid:photo" in html and "cid:week" in html and "cid:month" in html
    assert "192.168.1.5:8000" in html
    assert "今日第 1 次" in subject
