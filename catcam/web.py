from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
import math
import hashlib
from datetime import datetime, timedelta
from collections import OrderedDict
from threading import Lock
from pathlib import Path

import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles
from fastapi import Body, FastAPI, HTTPException, Request, Query
from typing import Literal
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from catcam.audio import muted_cmd
from catcam.charts import trend_png
from catcam.dispenser import estimate_remaining, filter_days_left
from catcam.classifier import DrinkingClassifier
from catcam.recorder import ClipRecorder
from catcam.feedback import FeedbackStore
from catcam.stats import StatsStore, bucket_events, day_bounds
from catcam.health import health_summary
from catcam.access import install_access
from catcam.replay import ReplayArchive
from urllib.error import HTTPError, URLError


def clip_duration(path: Path) -> float:
    """读视频头拿时长（秒），不解码全片。供网页展示「总共几秒」。"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return round(frames / fps, 1) if fps > 0 else 0.0


# ============================================================================
# 前端：PC 优先的侧边栏多页应用（vanilla JS、零 CDN、全本地）。
# 信息架构：总览 / 饮水机（核心）/ 喝水记录（只看已确认喝水）/ 趋势 / 实验室（标注+训练+模型）。
# 设计：水主题 aqua glass —— 玻璃拟态卡片 + 双主题 + 手写动画（count-up、波浪水箱、
# 灯箱、toast、骨架屏），全部尊重 prefers-reduced-motion。
# 设计定稿见 docs/superpowers/specs/2026-07-05-web-ui-redesign-design.md
# ============================================================================
STATIC_DIR = Path(__file__).with_name('static')
INDEX_HTML = STATIC_DIR.joinpath('index.html').read_text(encoding='utf-8')
for asset in ('app.css', 'app.js', 'replay.css', 'replay.js'):
    version = hashlib.sha256(STATIC_DIR.joinpath(asset).read_bytes()).hexdigest()[:12]
    INDEX_HTML = INDEX_HTML.replace(f'/static/{asset}', f'/static/{asset}?v={version}')


class FeedbackBody(BaseModel):
    clip: str
    is_drinking: bool
    drinking_seconds: float | None = None
    outcome: str | None = None
    reason: str | None = None


def create_app(
    stats: StatsStore,
    recorder: ClipRecorder,
    feedback: FeedbackStore,
    frame_provider,
    clips_dir: Path,
    trainer=None,
    registry=None,
    active_model=None,
    video_trainer=None,
    video_model_switch=None,
    video_model_clear=None,
    audio_status_provider=None,
    dispenser=None,
    dispenser_low_water_pct: float = 0.2,
    monitoring=None,
    setup_provider=None,
    setup_update=None,
    ai_status_provider=None,
    ai_test_provider=None,
    model_review=None,
    web_password_hash='',
    web_session_secret='',
    access_password_update=None,
    flow_monitor=None,
    flow_calibrate=None,
    mail_notifications=None,
    mail_update=None,
    indicator_monitor=None,
    reminder_update=None,
    replay_archive=None,
    live_status_provider=None,
) -> FastAPI:
    app = FastAPI()
    install_access(app,web_password_hash,web_session_secret,access_password_update)
    app.mount("/static", StaticFiles(directory=Path(__file__).with_name("static")), name="static")
    clips_dir = Path(clips_dir)
    replay_archive = replay_archive or ReplayArchive(recorder, stats, feedback)

    @app.get('/api/live/status')
    def live_status():
        return live_status_provider() if live_status_provider else {'connected': None, 'detection_enabled': None}

    @app.get('/api/replay')
    def replay_catalog():
        try:
            return replay_archive.catalog()
        except (ValueError, OSError, URLError):
            raise HTTPException(status_code=503, detail='无法读取当前摄像头的滚动录像。请检查 WoSee 服务、所选设备和录像开关；未替换为旧识别片段。') from None

    @app.get('/api/replay/media/{source}/{name:path}')
    def replay_media(source: str, name: str, request: Request, archive: str | None = None):
        if source == 'local':
            path = replay_archive.local_path(name)
            if path is None:
                raise HTTPException(status_code=404, detail='录像已被覆盖或不存在')
            return FileResponse(path, media_type='video/mp4')
        if source != 'wosee':
            raise HTTPException(status_code=404, detail='录像来源无效')
        try:
            upstream = replay_archive.open_remote(name, archive, request.headers.get('range'))
        except HTTPError as error:
            code = error.code
            error.close()
            raise HTTPException(status_code=code if code in (404, 409, 416) else 503,
                                detail='录像已覆盖或目录已改变，请刷新回放') from None
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from None
        except (OSError, URLError):
            raise HTTPException(status_code=503, detail='录像服务暂不可用') from None
        async def stream():
            try:
                while True:
                    chunk = await asyncio.to_thread(upstream.read, 65536)
                    if not chunk:
                        break
                    yield chunk
            finally:
                upstream.close()
        headers = {key: upstream.headers[key] for key in ('Content-Length', 'Content-Range', 'Accept-Ranges') if key in upstream.headers}
        headers['Cache-Control'] = 'no-store'
        return StreamingResponse(stream(), status_code=upstream.status, media_type='video/mp4', headers=headers)

    def _estimated_ml(events, calibration):
        total=0
        for event in events:
            meta=feedback.label_meta(event['clip_name']) or {}
            seconds=meta.get('effective_drinking_seconds')
            total+=seconds*calibration['ml_per_second'] if seconds and calibration['ml_per_second'] else calibration['ml_per_drink']
        return total

    def _active_hitrate():
        if registry is None or not registry.active_id():
            return None
        entry=registry.get(registry.active_id()) or {}
        split=(entry.get('report') or {}).get('split') or {}
        exclude=[n for part in ('train','validation','retired_holdout') for n in split.get(part,[])]
        return stats.model_hitrate(registry.active_id(),exclude_names=exclude)

    @app.get('/api/events')
    async def updates(request: Request):
        from catcam.live_updates import DataRevision
        async def generate():
            paths=[stats.db_path,feedback.db_path]
            if dispenser is not None:paths.append(dispenser.db_path)
            revision=DataRevision(paths)
            previous=None
            heartbeat=time.monotonic()
            while not await request.is_disconnected():
                validator=getattr(app.state,'access_session_valid',None)
                if validator and not validator(request.cookies.get('catcam_session')):
                    yield 'event: auth-expired\ndata: {}\n\n'
                    return
                data=await asyncio.to_thread(revision.get)
                flow=flow_monitor.status() if flow_monitor else {}
                quality=monitoring.status() if monitoring else {}
                training=(getattr(video_trainer,'_state',None),getattr(video_trainer,'_phase',None),
                          getattr(video_trainer,'_done',None),getattr(video_trainer,'_total',None),
                          json.dumps(getattr(video_trainer,'_fine_progress',{}),sort_keys=True))
                model_stamp=None
                if registry is not None:
                    try:model_stamp=registry.path.stat().st_mtime_ns
                    except FileNotFoundError:pass
                current={'data':data,'flow':(flow.get('state'),flow.get('alert'),flow.get('changed_at'),
                          json.dumps(flow.get('recovery'),sort_keys=True)),
                         'monitoring':(quality.get('usable'),quality.get('status')),
                         'training':(training,model_stamp,json.dumps(model_review.status(),sort_keys=True) if model_review else None),
                         'clock':datetime.now().strftime('%Y-%m-%d %H:%M')}
                if previous is None:
                    yield 'event: ready\ndata: {}\n\n'
                else:
                    changed=[k for k in current if current[k]!=previous[k]]
                    if current['clock'][:10]!=previous['clock'][:10] and 'data' not in changed:changed.append('data')
                    if changed:yield 'event: change\ndata: '+json.dumps({'topics':changed})+'\n\n'
                previous=current
                if time.monotonic()-heartbeat>=20:
                    yield ': keepalive\n\n'
                    heartbeat=time.monotonic()
                await asyncio.sleep(2)
        return StreamingResponse(generate(),media_type='text/event-stream',
                                 headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse(INDEX_HTML, headers={"Cache-Control": "no-store"})

    @app.get("/api/stats/range")
    def stats_range(days: int = 7):
        days = max(1, min(int(days), 90))
        points = stats.daily_counts(datetime.now(), days)
        return {"days": [{"date": d, "count": c} for d, c in points]}

    @app.get('/api/setup')
    def setup_get():
        return setup_provider() if setup_provider else {'enabled':False}

    @app.post('/api/setup')
    def setup_set(body: dict):
        if setup_update is None:
            raise HTTPException(status_code=400, detail='当前进程未启用场景设置')
        try:
            return setup_update(body)
        except ValueError as e:
            raise HTTPException(status_code=400,detail=str(e))

    @app.post('/api/ai/test')
    def ai_test(body: dict):
        if ai_test_provider is None:
            raise HTTPException(status_code=400, detail='当前进程未启用视觉老师测试')
        try:
            return ai_test_provider(body)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get('/api/ai/status')
    def ai_status():
        return ai_status_provider() if ai_status_provider else {'enabled':False}

    @app.post('/api/model/review')
    def model_recheck():
        return {'started':model_review.start() if model_review else False}

    @app.get('/api/model/review/status')
    def model_review_status():
        return model_review.status() if model_review else {'state':'disabled'}

    @app.get('/api/health')
    def health():
        h = health_summary(stats, monitoring)
        camera = live_status_provider() if live_status_provider else None
        if camera is not None:
            h['camera'] = camera
        preview = camera is not None and camera.get('detection_enabled') is False
        if preview:
            h['status'] = 'preview_only'
            h['message'] = '当前仅预览摄像头，识别、喝水记录及有效监控覆盖统计未启动。'
            h['monitoring'] = {'usable': False, 'status': 'preview' if camera['connected'] else 'offline',
                               'message': '视频已连接，仅预览；识别和监控统计未启动。' if camera['connected'] else '摄像头没有新画面，正在等待恢复。'}
        actions = []
        if preview and camera['connected']:
            actions.append('需要识别喝水时，请启动完整监控。')
        elif not h['monitoring'].get('usable'):
            actions.append('先检查摄像头和补光，当前无法完整观察饮水。')
        if h['pending_review']:
            actions.append(f"先复核{h['pending_review']}段候选录像，再判断饮水次数变化。")
        if dispenser is not None:
            water = _dispenser_payload()
            h['dispenser'] = water
            h['today_estimated_ml'] = water['today_ml']
            h['estimate_calibrated'] = water['calibrated']
            h['baseline_estimated_ml'] = h['baseline_daily_count']*water['ml_per_drink'] if h['baseline_daily_count'] is not None else None
            calibration=dispenser.calibration(stats)
            now=datetime.now()
            start=now.replace(hour=0,minute=0,second=0,microsecond=0)
            elapsed=(now-start).total_seconds()
            comparable=[day for day in h['history'] if day['comparable']]
            if len(comparable)>=3:
                totals=[];by_now=[]
                for day in comparable:
                    day_start=datetime.strptime(day['date'],'%Y-%m-%d')
                    totals.append(_estimated_ml(stats.events_between(day_start.timestamp(),(day_start+timedelta(days=1)).timestamp()),calibration))
                    by_now.append(_estimated_ml(stats.events_between(day_start.timestamp(),day_start.timestamp()+elapsed),calibration))
                h['baseline_estimated_ml']=sum(totals)/len(totals)
                h['expected_estimated_ml_by_now']=sum(by_now)/len(by_now)
                expected=h['expected_estimated_ml_by_now']
                if h['coverage_ratio']>=.75 and h['pending_review']==0 and elapsed>=6*3600 and expected>0:
                    ratio=water['today_ml']/expected
                    h['estimated_amount_change']=round(ratio-1,3)
                    if ratio<.5:
                        actions.append('今日摄水估值低于历史同时段，请先确认记录完整和饮水机供水，再观察小猫状态。')
                    elif ratio>1.8:
                        actions.append('今日摄水估值高于历史同时段，请复核误判并观察食欲、精神和排尿。')
            if water['need_water']:
                actions.append('预估水位偏低，请检查实际供水并及时补水。')
            if water['water_age_hours'] is None or water['water_age_hours'] > 24:
                actions.append('更新实测水位，确认饮水机仍有水。')
            if water['need_filter']:
                actions.append('滤芯已到更换周期，请检查滤芯和清洁情况。')
        h['actions'] = actions
        return h

    @app.get("/api/audio/status")
    def audio_status():
        if audio_status_provider is None:
            return {"enabled": False, "available": False}
        return {"enabled": True, **audio_status_provider()}

    @app.get("/api/stats/trend")
    def stats_trend(days: int = 7):
        """趋势页一次取齐：每日次数 + KPI（总计/活跃天数）+ 环比 + 时段/星期分布。"""
        days = max(1, min(int(days), 90))
        now = datetime.now()
        points = stats.daily_counts(now, days)
        # 当前窗口 [start, end)
        end_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        start_day = end_day - timedelta(days=days)
        evs = stats.events_between(start_day.timestamp(), end_day.timestamp())
        buckets = bucket_events(evs)
        total = len(evs)
        active_days = sum(1 for _, c in points if c > 0)
        prev_start = start_day - timedelta(days=days)
        prev_total = stats.count_between(prev_start.timestamp(), start_day.timestamp())
        return {
            "days": [{"date": d, "count": c} for d, c in points],
            "total": total,
            "active_days": active_days,
            "prev_total": prev_total,
            "hourly": buckets["hourly"],
            "weekday": buckets["weekday"],
        }

    def _dispenser_payload() -> dict:
        st = dispenser.get_state()
        now = time.time()
        calibration = dispenser.calibration(stats)
        mpd = calibration['ml_per_drink']
        refill_ts, refill_ml = st["last_refill_ts"], st["last_refill_ml"]
        drinks_since = stats.count_between(refill_ts, now) if refill_ts > 0 else 0
        water = dispenser.water_summary(now)
        reading = st['reading']
        visits = stats.events_between(reading['ts'],now) if reading else []
        def estimated(events):
            return _estimated_ml(events,calibration)
        remaining = max(0,water['remaining_ml']-estimated(visits)) if water['remaining_ml'] is not None else None
        capacity = st['options'].get('capacity_ml') or refill_ml
        remaining_pct = min(1,remaining/capacity) if remaining is not None and capacity else None
        indicator_status = indicator_monitor.status() if indicator_monitor else None
        reminder = dispenser.reminder_status(calibration,remaining,indicator_status)
        used_cycles=[c for c in calibration['cycles'] if c['used']][-3:]
        period=sum(c['end']-c['start'] for c in used_cycles)
        daily_loss=sum(c['consumed_ml'] for c in used_cycles)/(period/86400) if period>0 else None
        if daily_loss is None:
            daily_loss=water['depletion_ml_per_day']
        water.update(remaining_ml=round(remaining) if remaining is not None else None,
                     remaining_pct=remaining_pct,water_source='estimated',
                     days_until_empty=round(max(0,remaining-(reminder['stop_residual_ml'] or 0))/daily_loss,1) if daily_loss and remaining is not None and reminder['stop_residual_ml'] is not None else None,
                     days_until_low=round(max(0,remaining-reminder['low_ml'])/daily_loss,1) if daily_loss and remaining is not None and reminder['low_ml'] is not None and reminder['ready'] else None,
                     depletion_ml_per_day=daily_loss)
        fdl = (filter_days_left(st["last_filter_change_ts"], st["filter_cycle_days"], now)
               if st["last_filter_change_ts"] > 0 else None)
        # 今日约喝 ml = 今日喝水次数 × 每次 ml
        d0, d1 = day_bounds(datetime.now())
        today_ml = estimated(stats.events_between(d0,min(d1,now)))
        flow_status = flow_monitor.status() if flow_monitor else None
        if flow_status and flow_status['alert']:
            flow_status.update(dispenser.flow_diagnosis(stats, now))
            if indicator_status and indicator_status['state']=='blinking_red' and st['options'].get('indicator_enabled'):
                flow_status.update(cause='likely_empty',cause_message='红灯闪烁且出水口检测到停水，较可能低于设备工作水位。请测量残水后反馈本轮。')
        return {
            "flow": flow_status,
            "indicator": indicator_status,
            "reminder": reminder,
            "options": st["options"],
            "additions": st["additions"],
            "readings": st["readings"],
            "signals": st["signals"],
            "cycle_closed": st["cycle_closed"],
            "capacity_ml": st["options"].get("capacity_ml"),
            "has_refill": refill_ts > 0,
            "has_filter": st["last_filter_change_ts"] > 0,
            "last_refill_ts": refill_ts,
            "last_refill_ml": refill_ml,
            "drinks_since_refill": drinks_since,
            **water,
            "remaining_pct": remaining_pct,
            "ml_per_drink": round(mpd, 1),
            **calibration,
            "today_ml": round(today_ml) if today_ml is not None else None,
            "last_filter_change_ts": st["last_filter_change_ts"],
            "filter_cycle_days": st["filter_cycle_days"],
            "filter_days_left": None if fdl is None else round(fdl, 1),
            "need_water": reminder["need_water"],
            "need_filter": bool(fdl is not None and fdl <= 0),
        }

    @app.get("/api/dispenser")
    def dispenser_get():
        if dispenser is None:
            raise HTTPException(status_code=400, detail="未启用饮水机")
        return _dispenser_payload()

    @app.get('/api/mail')
    def mail_get():
        return mail_notifications.status() if mail_notifications else {'enabled':False,'available':False}

    @app.post('/api/mail')
    def mail_config(body: dict = Body(default={})):
        if mail_update is None:
            raise HTTPException(status_code=400,detail='未启用邮件设置')
        try:
            return mail_update(body)
        except ValueError as e:
            raise HTTPException(status_code=400,detail=str(e))

    @app.post('/api/mail/test')
    def mail_test():
        if mail_notifications is None:
            raise HTTPException(status_code=400,detail='未启用邮件提醒')
        try:
            return mail_notifications.test()
        except ValueError as e:
            raise HTTPException(status_code=400,detail=str(e))

    @app.post('/api/dispenser/flow/dismiss')
    def dispenser_flow_dismiss(body: dict = Body(default={})):
        if flow_monitor is None:
            raise HTTPException(status_code=400,detail='未启用出水监测')
        try:
            return flow_monitor.dismiss(body.get('id'))
        except ValueError as e:
            raise HTTPException(status_code=400,detail=str(e))

    @app.post('/api/dispenser/flow/reference')
    def dispenser_flow_reference(body: dict = Body(default={})):
        if flow_calibrate is None:
            raise HTTPException(status_code=400, detail='未启用出水监测')
        try:
            return flow_calibrate(body.get('kind'))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post('/api/dispenser/flow/roi')
    def dispenser_flow_roi(body: dict = Body(default={})):
        if flow_monitor is None:
            raise HTTPException(status_code=400, detail='未启用出水监测')
        try:
            return flow_monitor.set_roi(body.get('roi'))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post('/api/dispenser/add')
    def dispenser_add(body: dict = Body(default={})):
        if dispenser is None:
            raise HTTPException(status_code=400,detail='未启用饮水机')
        try:
            current = _dispenser_payload()
            initial = body.get('initial_remaining_ml')
            dispenser.add_water(float(body.get('added_ml')),time.time(),current['remaining_ml'],float(initial) if initial is not None else None)
        except (ValueError,TypeError) as e:
            raise HTTPException(status_code=400,detail=str(e))
        return _dispenser_payload()

    @app.post('/api/dispenser/reminder')
    def dispenser_reminder(body: dict = Body(default={})):
        if reminder_update is None:
            raise HTTPException(status_code=400,detail='未启用提醒设置')
        try:
            reminder_update(body)
        except (ValueError,TypeError) as e:
            raise HTTPException(status_code=400,detail=str(e))
        return _dispenser_payload()

    @app.post('/api/dispenser/signal')
    def dispenser_signal(body: dict = Body(default={})):
        if dispenser is None:
            raise HTTPException(status_code=400,detail='未启用饮水机')
        try:
            dispenser.record_signal(body.get('kind'),body.get('remaining_ml'),time.time())
        except (ValueError,TypeError) as e:
            raise HTTPException(status_code=400,detail=str(e))
        return _dispenser_payload()

    @app.post('/api/dispenser/indicator/roi')
    def dispenser_indicator_roi(body: dict = Body(default={})):
        if indicator_monitor is None:
            raise HTTPException(status_code=400,detail='未启用指示灯监测')
        try:
            return indicator_monitor.set_roi(body.get('roi'))
        except ValueError as e:
            raise HTTPException(status_code=400,detail=str(e))

    @app.post("/api/dispenser/refill")
    def dispenser_refill(body: dict = Body(default={})):
        if dispenser is None:
            raise HTTPException(status_code=400, detail="未启用饮水机")
        try:
            ml = float(body.get("ml", 0))
        except (TypeError, ValueError):
            ml = 0.0
        if not math.isfinite(ml) or not 0 < ml <= 20000:
            raise HTTPException(status_code=400, detail="加水后的总量须在0到20000毫升之间")
        now = time.time()
        st = dispenser.get_state()
        prev_drinks = stats.count_between(st["last_refill_ts"], now) if st["last_refill_ts"] > 0 else 0
        dispenser.refill(ml=ml, now=now, prev_drinks=prev_drinks)
        return _dispenser_payload()

    @app.post('/api/dispenser/observe')
    def dispenser_observe(body: dict = Body(default={})):
        if dispenser is None:
            raise HTTPException(status_code=400, detail='未启用饮水机')
        st = dispenser.get_state()
        reading = st['reading']
        now = time.time()
        drinks = stats.count_between(reading['ts'], now) if reading else 0
        try:
            dispenser.observe_water(float(body.get('ml')), now, drinks,
                                    calibrate=body.get('calibrate') is True,
                                    other_loss_ml=float(body.get('other_loss_ml',0)))
        except (ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        return _dispenser_payload()

    @app.post('/api/dispenser/cycle')
    def dispenser_cycle(body: dict = Body(default={})):
        if dispenser is None:
            raise HTTPException(status_code=400,detail='未启用饮水机')
        try:
            dispenser.finish_cycle(time.time(),float(body.get('remaining_ml')),float(body.get('other_loss_ml',0)),stopped_confirmed=body.get('stopped_confirmed') is True)
        except (ValueError,TypeError) as e:
            raise HTTPException(status_code=400,detail=str(e))
        return _dispenser_payload()

    @app.post("/api/dispenser/filter")
    def dispenser_filter():
        if dispenser is None:
            raise HTTPException(status_code=400, detail="未启用饮水机")
        dispenser.mark_filter(time.time())
        return _dispenser_payload()

    @app.post("/api/dispenser/config")
    def dispenser_config(body: dict = Body(default={})):
        if dispenser is None:
            raise HTTPException(status_code=400, detail="未启用饮水机")
        days = body.get("filter_cycle_days")
        if days is not None:
            try:
                d = int(days)
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail="周期要是整数天")
            if d < 1:
                raise HTTPException(status_code=400, detail="周期至少 1 天")
            dispenser.set_cycle(d)
        return _dispenser_payload()

    @app.get("/chart/{span}.png")
    def chart(span: str):
        days = {"week": 7, "month": 30}.get(span)
        if days is None:
            raise HTTPException(status_code=404, detail="unknown span")
        title = "Last 7 days" if days == 7 else "Last 30 days"
        png = trend_png(stats.daily_counts(datetime.now(), days), title)
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    def _unlabeled_count() -> int:
        names = [p.name for p in recorder.list_clips()]
        return sum(1 for n in names if feedback.get_label(n) is None)

    @app.post("/api/train")
    def train():
        if trainer is None:
            return JSONResponse({"started": False, "error": "本入口未启用训练"})
        # 避免重复无效训练：没有「已标注未训练」的新数据就不练。
        states = feedback.label_states()
        if states["untrained"] == 0 and states["trained"] > 0:
            return JSONResponse({"started": False, "error": "暂无新标注，无需重复训练"})
        started = trainer.start()
        return JSONResponse({"started": started,
                             "error": None if started else "已经在训练中"})

    @app.get("/api/train/status")
    def train_status():
        if trainer is None:
            return {"state": "disabled", "detail": "本入口未启用训练", "models": [], "active": None}
        s = trainer.status()
        s["unlabeled"] = _unlabeled_count()  # 待标注（当前还在的视频里没标的）
        if registry is not None:
            s["active_mode"] = registry.active_mode()
            s["hitrate"] = _active_hitrate()
        return s

    @app.post("/api/train_video")
    def train_video(body: dict = Body(default={})):
        if video_trainer is None:
            return JSONResponse({"started": False, "error": "本入口未启用视频训练"})
        rebuild = bool((body or {}).get("rebuild"))   # 从头重建：忽略缓存、强制重抽特征
        backend = body.get('backend', 's3d-temporal-v2')
        if backend not in ('s3d-temporal-v2','videomae-temporal-v1','s3d-legacy','s3d-finetuned-v1'):
            raise HTTPException(status_code=400, detail='不支持的训练主干')
        started = video_trainer.start(rebuild=rebuild, backend=backend,auto_activate=body.get('auto_activate') is True)
        return JSONResponse({"started": started,
                             "error": None if started else "已经在训练中"})

    @app.get("/api/train_video/status")
    def train_video_status():
        if video_trainer is None:
            return {"state": "disabled", "detail": "本入口未启用视频训练"}
        s = video_trainer.status()
        s['label_states'] = feedback.label_states()
        s['unlabeled'] = _unlabeled_count()
        if registry is not None:
            s['active_mode'] = registry.active_mode()
            s['hitrate'] = _active_hitrate()
        return s

    @app.post("/api/model/activate")
    def activate(body: dict):
        if registry is None or active_model is None:
            raise HTTPException(status_code=400, detail="未启用模型管理")
        model_id = body.get("id")  # None = 停用，只用简单模型
        mode = body.get("mode") or "shadow"  # 默认测试模式（不拦截录制）
        if mode not in ("shadow", "gate"):
            raise HTTPException(status_code=400, detail="请选择测试模式或自动判定模式")
        if model_id is None:
            active_model.clear()
            if video_model_clear is not None:
                video_model_clear()
        else:
            entry = registry.get(model_id)
            if entry is None:
                raise HTTPException(status_code=404, detail="没有这个版本")
            if entry and entry.get("base") == "s3d+head":
                # 评估不足在版本页提示；用户仍可主动选择自动判定，人工纠错始终保留。
                if video_model_switch is None:
                    raise HTTPException(status_code=400, detail="当前进程未接入本地视频裁判")
                try:
                    video_model_switch(entry, mode)
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(status_code=500, detail=f"视频模型加载失败：{e}")
                active_model.clear()
            else:
                path = entry.get("path")
                if not path or not Path(path).exists():
                    raise HTTPException(status_code=404, detail="模型文件丢了")
                try:
                    active_model.set(DrinkingClassifier.from_path(path), model_id, mode)
                except Exception as e:  # noqa: BLE001
                    raise HTTPException(status_code=500, detail=f"加载失败：{e}")
                if video_model_clear is not None:
                    video_model_clear()
        registry.set_active(model_id, mode)
        if model_review and model_id:
            model_review.start()
        note = "模型已立即生效，将用于下一段录像" if model_id else "模型已停用"
        return {"active": registry.active_id(), "mode": registry.active_mode(), "note": note}

    @app.get("/api/stats/today")
    def today():
        start, end = day_bounds(datetime.now())
        events = stats.events_between(start, end)
        times = [datetime.fromtimestamp(e["ts"]).strftime("%H:%M:%S") for e in events]
        return {"count": len(events), "times": times}

    def clip_list_data(include_records=True):
        names = [p.name for p in recorder.list_clips()]
        stored = feedback.clip_metadata()
        events = stats.clip_metadata()
        meta = {n: stored['meta'].get(n) for n in names}
        labels = {n: m['is_drinking'] if m else None for n, m in meta.items()}
        prediction_details = {n: dict(events[n]['prediction']) for n in names
                              if 'prediction' in events.get(n, {})}
        thresholds = {entry['id']: ((entry.get('report') or {}).get('threshold', .5))
                      for entry in registry.list()} if registry is not None and prediction_details else {}
        for detail in prediction_details.values():
            version = detail['by']
            detail['threshold'] = thresholds.get(version, .5)
        return {"clips": names, "labels": labels,
                "sessions": {n: events[n]['session'] for n in names if 'session' in events.get(n, {})},
                "durations": {n: round(stored['durations'][n], 1) if n in stored['durations'] else None for n in names},
                "review_candidates": {n: events.get(n, {}).get('review_candidate', True)
                                      or (meta[n] or {}).get('source') == 'human' for n in names},
                "notes": stored['notes'],
                "candidate_only": {n: events.get(n, {}).get('candidate_only', False) for n in names},
                "water_review": {n: events.get(n, {}).get('water_review', False) for n in names},
                "predictions": {n: d['drinking'] for n, d in prediction_details.items()},
                "prediction_details": prediction_details, "meta": meta,
                "drinking_records": [e['clip_name'] for e in reversed(stats.drinking_records())] if include_records else [],
                "min_drinking_interval_seconds": 60,
                "max_clips": recorder.max_clips,
                "retention_days": recorder.retention_days,
                "recording_fps": recorder.fps}

    @app.get("/api/clips")
    def clips():
        return clip_list_data()

    @app.get('/api/clips/page')
    def clip_page(
        filter: Literal['all', 'review', 'yes', 'no', 'mistakes'] = 'review',
        limit: int = Query(default=12, ge=1, le=48),
        cursor: str | None = Query(default=None, pattern=r'^clip_[0-9]+\.mp4$'),
        snapshot: str | None = Query(default=None, pattern=r'^clip_[0-9]+\.mp4$'),
    ):
        # 元数据批量读取，不读取视频。游标避免标注移除后 OFFSET 跳过未看项。
        data = clip_list_data(include_records=False)
        boundary = snapshot or (data['clips'][0] if data['clips'] else 'clip_0.mp4')
        counts = dict.fromkeys(('all', 'review', 'yes', 'no', 'mistakes'), 0)
        matches = []
        for name in data['clips']:
            if name > boundary or not data['review_candidates'][name]:
                continue
            meta = data['meta'][name] or {}
            prediction = data['prediction_details'].get(name)
            if prediction is None and meta.get('source') != 'human' and meta.get('confidence') is not None:
                prediction = {'drinking': meta['is_drinking'], 'confidence': meta['confidence'], 'threshold': .5}
            human = meta.get('source') == 'human'
            pending = not human and (not meta or not prediction
                or prediction['confidence'] is None
                or abs(prediction['confidence'] - prediction.get('threshold', .5)) < .15)
            flags = {'all': True, 'review': pending,
                     'yes': bool(prediction and prediction['drinking']),
                     'no': bool(prediction and not prediction['drinking']),
                     'mistakes': bool(human and prediction and prediction['drinking'] != meta['action_drinking'])}
            for key, value in flags.items():
                counts[key] += int(value)
            if flags[filter] and (cursor is None or name < cursor):
                matches.append(name)
        names = matches[:limit]
        # 只返回本页元信息，不将全量文件名/标注传给浏览器。
        fields = ('labels', 'durations', 'meta', 'notes', 'review_candidates',
                  'candidate_only', 'water_review', 'predictions', 'prediction_details')
        page = {key: {n: data[key][n] for n in names if n in data[key]} for key in fields}
        page.update({key: data[key] for key in ('max_clips', 'retention_days', 'recording_fps')})
        page.update(clips=names, counts=counts, total=counts[filter], snapshot=boundary,
                    next_cursor=names[-1] if len(matches) > limit else None, limit=limit)
        return page

    @app.get('/api/clips/summary')
    def clip_summary(since: str | None = None):
        names=[p.name for p in recorder.list_clips()]
        candidates=stats.review_candidates()
        with feedback._conn() as conn:
            sources=dict(conn.execute('SELECT clip_name,source FROM labels'))
        review=[n for n in names if candidates.get(n,True) or sources.get(n)=='human']
        if since is not None:
            return {'pending_count': sum(sources.get(n) != 'human' for n in review),
                    'new_count': sum(n > since for n in review)}
        return {'review_names':review,'pending_names':[n for n in review if sources.get(n)!='human'],
                'clips':names}

    def existing_clip(name):
        if '/' in name or '\\' in name or '..' in name:
            raise HTTPException(status_code=400,detail='bad clip')
        if not (clips_dir/name).is_file():
            raise HTTPException(status_code=404,detail='录像不存在')

    @app.post('/api/clips/{name}/review')
    def promote_clip(name:str):
        existing_clip(name)
        stats.set_review_candidate(name,True)
        return {'ok':True}

    @app.post('/api/clips/{name}/note')
    def clip_note(name:str,body:dict=Body(...)):
        existing_clip(name)
        try:feedback.set_note(name,body.get('reason',''))
        except ValueError as error:raise HTTPException(status_code=400,detail=str(error))
        return {'ok':True}

    # 按文件身份缓存中间帧；缓存有界，录像覆盖后自动失效。
    thumb_cache = OrderedDict()
    thumb_lock = Lock()
    thumb_decode_locks = [Lock() for _ in range(32)]

    @app.get("/clips/{name}/thumb.jpg")
    def clip_thumb(name: str, request: Request):
        # 用中间帧做封面，比开头的空场/刚入画更有判断价值。
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="bad name")
        path = clips_dir / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        info = path.stat()
        key = (name, info.st_size, info.st_mtime_ns)
        etag = '"'+hashlib.sha256(repr(key).encode()).hexdigest()[:24]+'"'
        headers = {"Cache-Control": "private, no-cache", "ETag": etag}
        if request.headers.get('if-none-match') == etag:
            return Response(status_code=304, headers=headers)
        with thumb_decode_locks[hash(name) % len(thumb_decode_locks)]:
            with thumb_lock:
                content = thumb_cache.get(key)
                if content is not None:
                    thumb_cache.move_to_end(key)
            if content is None:
                cap = cv2.VideoCapture(str(path))
                try:
                    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    if frames > 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 2)
                    ok, frame = cap.read()
                finally:
                    cap.release()
                if not ok:
                    raise HTTPException(status_code=404, detail="no frame")
                h, w = frame.shape[:2]
                if w > 480:
                    frame = cv2.resize(frame, (480, max(1, round(h * 480 / w))))
                ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                if not ok:
                    raise HTTPException(status_code=500, detail="encode failed")
                content = buf.tobytes()
                with thumb_lock:
                    thumb_cache[key] = content
                    while len(thumb_cache) > 128:
                        thumb_cache.popitem(last=False)
        return Response(content=content, media_type="image/jpeg",
                        headers=headers)

    @app.get("/clips/{name}")
    def get_clip(name: str, audio: int = 1, download: int = 0):
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="bad name")
        path = clips_dir / name
        if not path.exists():
            raise HTTPException(status_code=404, detail="not found")
        # audio=0：现场用 ffmpeg 去掉音轨后流式吐出（视频轨 copy，不重编码）。
        # ffmpeg 不可用 → 回退原文件（fail-open，宁可给原始也不 500）。
        if audio == 0 and shutil.which("ffmpeg") is not None:
            proc = subprocess.Popen(muted_cmd(str(path)),
                                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            def _stream():
                try:
                    while True:
                        chunk = proc.stdout.read(65536)
                        if not chunk:
                            break
                        yield chunk
                finally:
                    proc.stdout.close()
                    proc.terminate()

            muted_name = name[:-4] + "_muted.mp4" if name.endswith(".mp4") else name + "_muted.mp4"
            return StreamingResponse(
                _stream(), media_type="video/mp4",
                headers={"Content-Disposition": f'attachment; filename="{muted_name}"'})
        headers = ({"Content-Disposition": f'attachment; filename="{name}"'}
                   if download else None)
        return FileResponse(path, media_type="video/mp4", headers=headers)

    @app.get("/snapshot.jpg")
    def snapshot():
        frame = frame_provider()
        if frame is None:
            raise HTTPException(status_code=503, detail="no frame yet")
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise HTTPException(status_code=500, detail="encode failed")
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    @app.get("/stream.mjpg")
    def stream():
        # MJPEG：单连接持续推帧，<img> 直接当视频放，告别每秒刷快照的卡顿。
        enc = [cv2.IMWRITE_JPEG_QUALITY, 55]
        def gen():
            deadline=time.monotonic()
            while True:
                frame = frame_provider()
                if frame is None:
                    frame=np.full((480,640,3),(24,16,10),dtype=np.uint8)
                    cv2.putText(frame,'Camera reconnecting...', (110,245),cv2.FONT_HERSHEY_SIMPLEX,.8,(190,170,140),1,cv2.LINE_AA)
                if frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame, enc)
                    if ok:
                        chunk = buf.tobytes()
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                               b"Content-Length: " + str(len(chunk)).encode()
                               + b"\r\n\r\n" + chunk + b"\r\n")
                deadline+=1/30
                now=time.monotonic()
                if deadline<now:deadline=now
                time.sleep(max(0,deadline-now))
        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.post("/api/feedback")
    def post_feedback(body: FeedbackBody):
        if "/" in body.clip or "\\" in body.clip or ".." in body.clip:
            raise HTTPException(status_code=400, detail="bad clip")
        path=clips_dir/body.clip
        if not path.exists():
            raise HTTPException(status_code=404,detail='录像不存在')
        if body.drinking_seconds is not None:
            if not math.isfinite(body.drinking_seconds) or body.drinking_seconds <= 0 or body.drinking_seconds > clip_duration(path)+.1:
                raise HTTPException(status_code=400,detail='喝水动作时长须大于0且不超过录像长度')
            if not body.is_drinking:
                raise HTTPException(status_code=400,detail='没喝水片段不能标注喝水时长')
        try:
            feedback.label_clip(path, body.is_drinking, outcome=body.outcome,reason=body.reason)
        except ValueError as e:
            raise HTTPException(status_code=400,detail=str(e))
        if body.drinking_seconds is not None:
            feedback.set_duration(body.clip,body.drinking_seconds)
        stats.set_review_candidate(body.clip,True)
        return JSONResponse({"ok": True,"meta":feedback.label_meta(body.clip)})

    return app
