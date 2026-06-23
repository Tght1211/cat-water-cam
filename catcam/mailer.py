from __future__ import annotations

import smtplib
import ssl
from datetime import datetime
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr

import cv2

from catcam.charts import trend_png
from catcam.netutil import lan_ip
from catcam.stats import day_bounds


def _frame_to_jpg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("照片编码失败")
    return buf.tobytes()


def build_drinking_email(stats, frame, now_dt: datetime, lan_url: str):
    """组装「猫来喝水」邮件：(subject, html, {cid: png/jpg 字节})。

    纯组装、不发信，便于单测。内嵌三张图：触发照片、近 7 天、近 30 天趋势。
    """
    start, end = day_bounds(now_dt)
    events = stats.events_between(start, end)
    times = [datetime.fromtimestamp(e["ts"]).strftime("%H:%M:%S") for e in events]
    count = len(events)
    week = stats.daily_counts(now_dt, 7)
    month = stats.daily_counts(now_dt, 30)

    images = {
        "photo": _frame_to_jpg(frame),
        "week": trend_png(week, "Last 7 days"),
        "month": trend_png(month, "Last 30 days"),
    }

    chips = "".join(
        f'<span style="display:inline-block;background:#eef3ff;color:#0a58ff;'
        f'border-radius:8px;padding:3px 10px;margin:3px 4px 0 0;font-size:13px;'
        f'font-family:monospace">{t}</span>'
        for t in times
    ) or '<span style="color:#86868b">—</span>'

    html = f"""<!doctype html><html><body style="margin:0;background:#f5f5f7;
  font-family:-apple-system,Segoe UI,PingFang SC,Microsoft YaHei,sans-serif;color:#1d1d1f">
<div style="max-width:560px;margin:0 auto;padding:24px 16px">
  <div style="font-size:22px;font-weight:700;margin-bottom:4px">🐱 猫咪来喝水啦</div>
  <div style="color:#86868b;font-size:13px;margin-bottom:18px">
    {now_dt.strftime('%Y-%m-%d %H:%M:%S')} · 今日第 <b style="color:#1d1d1f">{count}</b> 次</div>

  <div style="background:#fff;border-radius:16px;overflow:hidden;box-shadow:0 6px 20px rgba(0,0,0,.06);margin-bottom:16px">
    <img src="cid:photo" style="width:100%;display:block">
    <div style="padding:12px 16px;font-size:12px;color:#86868b">触发瞬间画面</div>
  </div>

  <div style="background:#fff;border-radius:16px;padding:16px;box-shadow:0 6px 20px rgba(0,0,0,.06);margin-bottom:16px">
    <div style="font-size:13px;color:#86868b;margin-bottom:8px">今日各次喝水时间</div>
    <div>{chips}</div>
  </div>

  <div style="background:#fff;border-radius:16px;padding:16px 8px;box-shadow:0 6px 20px rgba(0,0,0,.06);margin-bottom:16px">
    <img src="cid:week" style="width:100%;display:block">
    <img src="cid:month" style="width:100%;display:block;margin-top:8px">
  </div>

  <a href="{lan_url}" style="display:block;text-align:center;background:#0071e3;color:#fff;
    text-decoration:none;border-radius:12px;padding:13px;font-size:15px;font-weight:600">
    打开喝水监控平台 →</a>
  <div style="text-align:center;color:#86868b;font-size:12px;margin-top:10px">{lan_url}（需与摄像头同一局域网）</div>
</div></body></html>"""

    subject = f"🐱 猫咪来喝水啦（今日第 {count} 次）"
    return subject, html, images


class Emailer:
    def __init__(self, cfg):
        self.enabled = cfg.mail_enabled
        self.host = cfg.smtp_host
        self.port = cfg.smtp_port
        self.user = cfg.smtp_user
        self.password = cfg.smtp_password
        self.to = cfg.mail_to
        self.min_interval = cfg.mail_min_interval_seconds
        self.web_port = cfg.web_port
        self._last_sent: float | None = None

    def lan_url(self) -> str:
        return f"http://{lan_ip()}:{self.web_port}"

    def should_send(self, now: float) -> bool:
        if not self.enabled or not (self.user and self.password and self.to):
            return False
        if self._last_sent is not None and now - self._last_sent < self.min_interval:
            return False
        return True

    def send(self, subject: str, html: str, images: dict[str, bytes]) -> None:
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"] = formataddr(("猫咪饮水监控", self.user))
        msg["To"] = self.to
        alt = MIMEMultipart("alternative")
        msg.attach(alt)
        alt.attach(MIMEText(html, "html", "utf-8"))
        for cid, data in images.items():
            img = MIMEImage(data)
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=f"{cid}")
            msg.attach(img)
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.host, self.port, context=ctx, timeout=25) as s:
            s.login(self.user, self.password)
            s.sendmail(self.user, [self.to], msg.as_string())

    def notify_drinking(self, stats, frame, now_dt: datetime) -> bool:
        """限流通过则发一封喝水邮件；返回是否真的发了。失败只打印不抛。"""
        now = now_dt.timestamp()
        if not self.should_send(now):
            return False
        try:
            subject, html, images = build_drinking_email(stats, frame, now_dt, self.lan_url())
            self.send(subject, html, images)
            self._last_sent = now
            print(f"已发送喝水提醒邮件 → {self.to}")
            return True
        except Exception as e:  # noqa: BLE001 — 邮件失败绝不能拖垮采集循环
            print(f"邮件发送失败：{e}")
            return False
