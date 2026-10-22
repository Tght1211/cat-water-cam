from __future__ import annotations

import threading
import time
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import uvicorn

from catcam.ai_labeler import AILabeler
from catcam.classifier import ActiveModel
from catcam.config import load_config
from catcam.detector import DrinkingDetector
from catcam.feedback import FeedbackStore
from catcam.models import ModelRegistry, install_bundled_video_model
from catcam.audio import AudioRing, mux_audio_into
from catcam.dispenser import DispenserStore
from catcam.framebuffer import FrameBuffer
from catcam.judge import route_clip
from catcam.mailer import Emailer
from catcam.netutil import lan_ip
from catcam import nightvision
from catcam.pipeline import Pipeline
from catcam.recorder import ClipRecorder
from catcam.session import DrinkSession
from catcam.simple import MotionGrayDetector
from catcam.stats import StatsStore
from catcam.trainer import TrainingManager
from catcam.video_trainer import VideoTrainingManager
from catcam.videojudge import DrinkingHead, LocalVideoClipJudge, extractor_for_backend
from catcam.health import MonitoringStore
from catcam.ai_queue import AIQueue
from catcam.model_review import ModelReview
from catcam.vision import CatDetector
from catcam.camera import ReconnectingCapture,FreshMacCapture,mac_camera_name
from catcam.web import create_app
from catcam.flow import FlowMonitor
from catcam.motion import MotionDetector
from catcam.review_routing import ReviewRouter
from catcam.notifications import EmailNotifications
from catcam.source import open_capture
from catcam.replay import ReplayArchive


class LatestFrame:
    """采集线程写、网页/检测线程读的最新帧（含时间戳与是否夜间）。

    预览（MJPEG/快照）只要帧；检测线程还要 now/night。加锁、返回拷贝避免并发改。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._now = None
        self._frame = None
        self._night = False
        self._quality = None
        self._preview = None

    def set(self, now: float, frame, night: bool, quality=None, preview=None) -> None:
        with self._lock:
            self._now = now
            self._frame = frame
            self._night = night
            self._quality = quality
            self._preview = frame if preview is None else preview

    def get_quality(self):
        with self._lock:
            return dict(self._quality) if self._quality else {'usable': False}

    def status(self):
        with self._lock:
            age = max(0, time.time() - self._now) if self._now is not None else None
            connected = self._frame is not None and age is not None and age <= 3
            return {'connected': connected, 'last_frame_age': round(age, 2) if age is not None else None,
                    'width': self._frame.shape[1] if connected else None,
                    'height': self._frame.shape[0] if connected else None,
                    'quality': dict(self._quality) if self._quality else None}

    def clear(self):
        with self._lock:
            self._now = None
            self._frame = None
            self._night = False
            self._quality = None
            self._preview = None

    def get(self):
        """供网页预览：只返回帧拷贝。"""
        with self._lock:
            return None if self._preview is None or time.time()-self._now>3 else self._preview.copy()

    def get_state(self):
        """供检测线程：返回 (now, frame拷贝, night)。"""
        with self._lock:
            if self._frame is None:
                return None
            return self._now, self._frame.copy(), self._night


class Presence:
    """检测线程写、采集线程读的「猫是否在水碗」状态（含当前状态起始时间）。

    采集线程据此跑会话录制状态机；用 since 判断是否在场够久（dwell）。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._in = False
        self._since = None

    def set(self, now: float, in_roi: bool) -> None:
        with self._lock:
            if in_roi and not self._in:
                self._since = now      # 刚进入在场 → 记起始时间
            elif not in_roi:
                self._since = None
            self._in = in_roi

    def get(self):
        with self._lock:
            return self._in, self._since


def _serve_web(app, host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")


class VideoJudgeRuntime:
    """可热切换的本地视频裁判；录制线程每段开始判断前取一次快照。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._judge = None
        self._mode = "shadow"

    def activate(self, entry: dict, mode: str):
        path = Path(entry["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        head = DrinkingHead.load(path)
        judge = LocalVideoClipJudge(extractor_for_backend(head.metadata['backend'], head.metadata.get('roi'),head.metadata.get('weights_path')), head, entry["id"])
        with self._lock:
            self._judge = judge
            self._mode = mode if mode in ("shadow", "gate") else "shadow"

    def clear(self):
        with self._lock:
            self._judge = None
            self._mode = "shadow"

    def snapshot(self):
        with self._lock:
            return self._judge, self._mode


def main(config_path: str = "config.json", preview_only: bool = False) -> None:
    cfg = load_config(config_path)
    if preview_only:
        from catcam.preview import run_preview
        run_preview(cfg)
        return

    cat_detector = CatDetector.from_path(cfg.yolo_model, min(.15,cfg.cat_confidence) if cfg.motion_recording else cfg.cat_confidence)
    stats = StatsStore(cfg.db_path)
    monitoring = MonitoringStore(cfg.db_path)
    recorder = ClipRecorder(
        cfg.clips_dir, cfg.max_clips, cfg.fps, retention_days=cfg.retention_days,
        archive_dirs=(cfg.training_dir / 'videos',),
    )
    recorder.prune()
    feedback = FeedbackStore(cfg.db_path, cfg.training_dir)
    review_router = ReviewRouter(cat_detector,stats,feedback)
    # 裁剪口径：超量时只删被判「没喝」的段（喝水/未判定永不自动删）。
    # 没喝段的训练价值（抽帧 + s3d 特征缓存）已另存，删 mp4 不影响训练。
    recorder.is_deletable = lambda name: feedback.get_label(name) is False
    ai_labeler = AILabeler.from_config(feedback, cfg)
    if ai_labeler is not None:
        print(f"外部 AI 裁判已开启：{cfg.ai_model}（画面帧会上传到所配服务）")
    emailer = Emailer(cfg)
    registry = ModelRegistry(cfg.models_dir / "registry.json")
    bundled = install_bundled_video_model(registry, cfg.models_dir)
    if bundled is not None:
        print(f"已安装内置预训练视频模型：{bundled['id']}（shadow）")
    video_judge_runtime = VideoJudgeRuntime()
    model_review = ModelReview(video_judge_runtime,stats,feedback,recorder,lambda:cfg.auto_confirm_extremes)
    active_entry = registry.get(registry.active_id()) if registry.active_id() else None
    if active_entry and active_entry.get("base") == "s3d+head":
        try:
            video_judge_runtime.activate(active_entry, registry.active_mode())
            print(f"本地视频裁判已加载：{active_entry['id']}（{registry.active_mode()}）")
        except Exception as e:  # noqa: BLE001
            print(f"本地视频裁判加载失败，暂不启用：{e}")
    # 单帧模型不拦截录制；整段视频模型在录完后按生效模式自动确认。
    active_model = ActiveModel()
    trainer = TrainingManager(
        cfg.training_dir, cfg.models_dir, cfg.cls_base_model, cfg.train_epochs, cfg.train_imgsz,
        feedback=feedback, registry=registry,
    )
    # 网页「训练视频模型」按钮用：后台训 s3d+head 小头（与单帧 TrainingManager 并存）。
    def activate_trained_model(entry):
        mode=registry.active_mode()
        if mode=='gate' and not (entry.get('report') or {}).get('auto_label_eligible'):
            return '现有自动判定模型继续运行；新版本的独立评估尚不足以替换。'
        video_judge_runtime.activate(entry,mode)
        active_model.clear()
        registry.set_active(entry['id'],mode)
        model_review.start()
        return '已立即启用，将用于下一段录像。'+('测试模式下请继续复核和纠错。' if mode=='shadow' else '')
    video_trainer = VideoTrainingManager(
        cfg.clips_dir, cfg.training_dir, feedback, registry, cfg.models_dir, roi=cfg.bowl_roi,on_complete=activate_trained_model,
    )
    ai_queue = None
    from catcam.videojudge import read_clip_frames
    from catcam.geometry import ratio_rect_to_pixels, cat_overlaps_bowl
    def candidate_check(path):
        for rgb in read_clip_frames(path, 3):
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not nightvision.assess_quality(frame,cfg.bowl_roi,max(50,cfg.minimum_monitor_brightness))['usable']:
                continue
            h,w=frame.shape[:2]
            bowl=ratio_rect_to_pixels(cfg.bowl_roi,w,h)
            if any(cat_overlaps_bowl(cat,bowl,cfg.min_overlap_ratio) for cat in cat_detector.detect_cats(frame)):
                return True
        return False
    if ai_labeler is not None:
        ai_labeler.candidate_check = candidate_check
        ai_labeler.roi_provider=lambda:cfg.bowl_roi
        ai_queue = AIQueue(ai_labeler, stats)
    # 会话录制要把「凑近过程 + dwell 这几秒」一起补进开头，缓冲就开这么长。
    buffer_seconds = (
        cfg.preroll_seconds if cfg.motion_recording else
        cfg.preroll_seconds + cfg.dwell_seconds if cfg.record_session else cfg.clip_seconds
    )
    frame_buffer = FrameBuffer(buffer_seconds, cfg.fps)
    pipeline = Pipeline(
        cat_detector=cat_detector,
        drinking_detector=DrinkingDetector(cfg.dwell_seconds, cfg.cooldown_seconds),
        frame_buffer=frame_buffer,
        recorder=recorder,
        stats=stats,
        bowl_roi_ratio=cfg.bowl_roi,
        min_overlap_ratio=cfg.min_overlap_ratio,
        presence_detector=MotionGrayDetector(),
        active_model=None,
    )
    session = (
        DrinkSession(
            recorder,
            0.0 if cfg.motion_recording else cfg.dwell_seconds,
            cfg.session_end_grace_seconds,
            cfg.max_session_seconds,
            0.0 if cfg.motion_recording else cfg.cooldown_seconds,
        )
        if cfg.record_session or cfg.motion_recording
        else None
    )
    presence = Presence()
    outlet_occlusion = Presence()
    motion = MotionDetector()
    # 近景常只拍到猫头，YOLO 可能间歇漏框；背景前景保持器让猫停住时仍继续录制。
    bowl_presence = nightvision.LowLightPresenceDetector()

    # 录音（仅会话录制模式）：持续采麦克风进环形缓冲，会话结束把对应时段 mux 进 mp4。
    # 起不来（无权限/无设备）由 AudioRing 内部看门狗兜底 → 无声，不影响录制。
    audio_ring = None
    if cfg.record_audio and session is not None and not cfg.video_source.startswith("wosee:"):
        # 不限长的视频仍使用有界音频缓冲，避免麦克风持续占满内存。
        ring_seconds = cfg.preroll_seconds + (cfg.max_session_seconds or 3600.0) + 10.0
        audio_ring = AudioRing(
            cfg.audio_input_format, cfg.audio_device,
            sample_rate=cfg.audio_sample_rate, max_seconds=ring_seconds,
        )
        audio_ring.start()
        print(f"正在启动录音（{cfg.audio_input_format} {cfg.audio_device}）；"
              f"⚠️ 若无声请在『系统设置→隐私与安全性→麦克风』给终端授权。")

    latest = LatestFrame()
    flow = FlowMonitor(Path(cfg.data_dir) / "dispenser_flow")

    def flow_calibrate(kind):
        state = latest.get_state()
        if state is None:
            raise ValueError('摄像头尚未提供画面，请稍后重试')
        stamp, frame, _ = state
        quality = nightvision.assess_quality(frame, flow.roi, 50)
        return flow.calibrate(kind, frame, quality['usable'] and time.time()-stamp < 2, outlet_occlusion.get()[0])

    dispenser = DispenserStore(
        cfg.db_path, cfg.dispenser_default_ml_per_drink, cfg.filter_cycle_days)
    mail_notifications = EmailNotifications(cfg, emailer, stats, feedback, dispenser, monitoring, flow)

    def update_mail(body):
        enabled = body.get('enabled', cfg.mail_enabled)
        if not isinstance(enabled, bool):
            raise ValueError('启用开关必须是布尔值')
        host = body.get('smtp_host', cfg.smtp_host)
        user = body.get('smtp_user', cfg.smtp_user)
        recipient = body.get('mail_to', cfg.mail_to)
        password = body.get('smtp_password') or cfg.smtp_password
        security = body.get('smtp_security', cfg.smtp_security)
        if any(not isinstance(v, str) or '\r' in v or '\n' in v for v in (host,user,recipient,password,security)):
            raise ValueError('邮箱配置格式无效')
        host,user,recipient = host.strip(),user.strip(),recipient.strip()
        if not host or any(v in host for v in ('/', ':', '@', ' ')) or len(host)>253:
            raise ValueError('请填写邮件服务器主机名，例如 smtp.qq.com')
        if security not in ('ssl','starttls'):
            raise ValueError('请选择 SSL 或 STARTTLS')
        try:
            port = int(body.get('smtp_port',cfg.smtp_port))
        except (TypeError, ValueError):
            raise ValueError('端口必须是整数')
        if not 1<=port<=65535:
            raise ValueError('端口超出范围')
        if enabled and (not password or any('@' not in v or ' ' in v or ',' in v or len(v)>254 for v in (user,recipient))):
            raise ValueError('请填写有效的发件邮箱、授权码和收件邮箱')
        with mail_notifications.lock:
            cfg.mail_enabled, cfg.smtp_host, cfg.smtp_user = enabled,host,user
            cfg.smtp_port, cfg.smtp_security = port,security
            cfg.smtp_password, cfg.mail_to = password,recipient
            path = Path(config_path)
            temp = path.with_suffix('.tmp')
            temp.write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2))
            temp.chmod(0o600)
            temp.replace(path)
            mail_notifications.reconfigure()
        return mail_notifications.status()

    def update_reminder(body):
        dispenser.configure_reminder({**body,'indicator_enabled':False})

    dispenser.configure_reminder({'indicator_enabled':False})

    def setup_state():
        return {'enabled':True,'bowl_roi':cfg.bowl_roi,
                'access_enabled':bool(cfg.web_password_hash),
                'ai_enabled':ai_queue is not None,'has_ai_key':bool(cfg.ai_api_key),
                'ai_base_url':cfg.ai_base_url,'ai_model':cfg.ai_model,'ai_daily_budget':cfg.ai_daily_budget,
                'auto_confirm_extremes':cfg.auto_confirm_extremes}

    def update_setup(body):
        nonlocal ai_queue, ai_labeler
        roi = body.get('bowl_roi',list(cfg.bowl_roi))
        if roi is None or not isinstance(roi,list) or len(roi)!=4:
            raise ValueError('请选择有效的饮水区域')
        try:
            roi=tuple(float(v) for v in roi)
        except (TypeError,ValueError) as e:
            raise ValueError('区域坐标无效') from e
        if not all(0<=v<=1 for v in roi) or roi[2]-roi[0]<.03 or roi[3]-roi[1]<.03:
            raise ValueError('区域过小或超出画面')
        from catcam.ai_service import resolve_settings
        settings=resolve_settings(body,cfg)
        model,base_url,budget=settings['model'],settings['base_url'],settings['limit']
        enabled=body.get('ai_enabled',cfg.ai_label_enabled)
        auto_confirm=body.get('auto_confirm_extremes',cfg.auto_confirm_extremes)
        if not isinstance(auto_confirm,bool):
            raise ValueError('自动确认开关必须是布尔值')
        if not isinstance(enabled,bool):
            raise ValueError('AI开关必须是布尔值')
        key=settings['key']
        if not isinstance(key,str) or enabled and not key:
            raise ValueError('启用外部辅助需要填写API Key')
        ai_changed=any(k in body for k in ('ai_enabled','ai_base_url','ai_api_key','ai_model','ai_daily_budget'))
        cfg.bowl_roi=roi
        cfg.ai_model=model
        cfg.ai_base_url=base_url
        cfg.ai_daily_budget=budget
        cfg.ai_label_enabled=enabled
        cfg.ai_api_key=key
        cfg.auto_confirm_extremes=auto_confirm
        pipeline.bowl_roi_ratio=roi
        video_trainer.roi=roi
        path=Path(config_path)
        tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2))
        tmp.chmod(0o600)
        tmp.replace(path)
        if ai_changed:
            if ai_queue:
                ai_queue.close()
            ai_labeler=AILabeler.from_config(feedback,cfg)
            ai_queue=None
            if ai_labeler:
                ai_labeler.candidate_check=candidate_check
                ai_labeler.roi_provider=lambda:cfg.bowl_roi
                ai_queue=AIQueue(ai_labeler,stats)
        if auto_confirm and 'auto_confirm_extremes' in body:
            model_review.start()
        return setup_state()
    def save_access_password(encoded):
        old=cfg.web_password_hash
        cfg.web_password_hash=encoded
        try:
            path=Path(config_path)
            temp=path.with_suffix('.tmp')
            temp.write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2))
            temp.chmod(0o600)
            temp.replace(path)
        except Exception:
            cfg.web_password_hash=old
            raise

    def test_ai(body):
        from catcam.ai_service import test_service
        return test_service(body,cfg,latest.get(),feedback.db_path)

    app = create_app(
        stats, recorder, feedback, latest.get, cfg.clips_dir, trainer,
        registry=registry, active_model=active_model, video_trainer=video_trainer,
        video_model_switch=video_judge_runtime.activate,
        video_model_clear=video_judge_runtime.clear,
        audio_status_provider=(audio_ring.status if audio_ring is not None else None),
        dispenser=dispenser, dispenser_low_water_pct=cfg.dispenser_low_water_pct,
        flow_monitor=flow, flow_calibrate=flow_calibrate,
        reminder_update=update_reminder,
        mail_notifications=mail_notifications, mail_update=update_mail,
        monitoring=monitoring,
        setup_provider=setup_state, setup_update=update_setup,
        ai_status_provider=lambda:ai_queue.status() if ai_queue else {'enabled':False},
        ai_test_provider=test_ai,
        model_review=model_review,
        web_password_hash=cfg.web_password_hash,web_session_secret=cfg.web_session_secret,
        access_password_update=save_access_password,
        replay_archive=ReplayArchive(recorder, stats, feedback, cfg.video_source),
        live_status_provider=lambda: {**latest.status(), 'detection_enabled': True},
    )
    mail_notifications.start()
    threading.Thread(
        target=_serve_web, args=(app, cfg.web_host, cfg.web_port), daemon=True
    ).start()
    if cfg.web_host == "0.0.0.0":
        print(
            f"网页已启动（局域网）；本机 http://127.0.0.1:{cfg.web_port}"
            f"，同局域网设备 http://{lan_ip()}:{cfg.web_port}"
        )
    else:
        print(
            f"网页已启动（绑定 {cfg.web_host}:{cfg.web_port}）；"
            f"本机访问 http://127.0.0.1:{cfg.web_port}"
        )
    if cfg.record_session:
        print(
            f"录制模式：整段会话（前补 {cfg.preroll_seconds:g}s，"
            f"离开 {cfg.session_end_grace_seconds:g}s 收尾，"
            f"{'不限时长' if cfg.max_session_seconds <= 0 else f'封顶 {cfg.max_session_seconds:g}s'}）"
        )
    if cfg.motion_recording:
        print(f'画面变化即录制：全画面触发、无停留门槛、无冷却，{cfg.frame_width}×{cfg.frame_height} / {cfg.fps} fps；视频滚动保留 {cfg.retention_days:g} 天。')

    source = cfg.video_source if cfg.video_source else cfg.camera_index
    import sys
    import shutil
    factory=cv2.VideoCapture
    if sys.platform=='darwin' and not cfg.video_source and shutil.which('ffmpeg'):
        device=cfg.camera_device_name
        if not device:
            device=mac_camera_name(cfg.camera_index)
            if device:
                cfg.camera_device_name=device
                path=Path(config_path);temp=path.with_suffix('.tmp')
                temp.write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2));temp.chmod(0o600);temp.replace(path)
        source=device or source
        factory=lambda camera:FreshMacCapture(camera,cfg.frame_width,cfg.frame_height,fps=cfg.fps)
    cap = open_capture(cfg) if cfg.video_source.startswith("wosee:") else ReconnectingCapture(
        source,cfg.frame_width,cfg.frame_height,factory=factory,fps=cfg.fps)
    if not cap.isOpened():
        print("摄像头暂时不可用，网页保持运行并等待自动重连。")

    buf_interval = 1.0 / max(1, cfg.fps)  # 回放缓冲/会话写帧按 fps 节奏，保证时长/速度正确

    def _finalize_async(res) -> None:
        stats.set_water_review(res.clip_name,flow.was_stopped(res.timestamp,time.time()))
        # 先合入音频，再由当前生效的本地视频模型判断；都放后台，不阻塞采集。
        def _run():
            feedback.record_video_duration(cfg.clips_dir / res.clip_name)
            if cfg.motion_recording and not review_router.route(cfg.clips_dir / res.clip_name):
                print(f'普通变化录像 {res.clip_name}：已放入回放，不占用待复核队列。')
                return
            if audio_ring is not None and audio_ring.available:
                pcm = audio_ring.slice(res.audio_start, res.audio_end)
                muxed = mux_audio_into(
                    cfg.clips_dir / res.clip_name, pcm, cfg.audio_sample_rate
                )
                audio_ring.record_mux_result(res.clip_name, muxed)
                if muxed:
                    print(f"已为 {res.clip_name} 合入声音。")
                else:
                    print(f"{res.clip_name} 音频合成失败，保留无声视频。")
            judge, mode = video_judge_runtime.snapshot()
            if res.candidate_only:
                print(f'画质不足候选 {res.clip_name}：已保存录像，动作细节不足，保留人工复核。')
                return
            result = route_clip(
                clip_path=cfg.clips_dir / res.clip_name,
                start_ts=res.timestamp,
                photo=res.photo,
                ai_labeler=None,
                local_judge=judge,
                mode=mode,
                emailer=emailer,
                stats=stats,
                feedback=feedback,
                auto_confirm_extremes=cfg.auto_confirm_extremes,
            )
            if ai_queue is not None and mode != 'gate':
                detail=stats.clip_prediction_details().get(res.clip_name,{})
                score=detail.get('confidence')
                threshold=judge.head.threshold if judge else .5
                uncertainty=1-abs(score-threshold) if score is not None else 1
                # 初期充分收集；后期优先不确定项，另留10%高把握样本做抽查。
                import hashlib
                audit=int(hashlib.sha256(res.clip_name.encode()).hexdigest()[:8],16)%10==0
                if feedback.label_states()['labeled']<100 or score is None or abs(score-threshold)<.2 or audit:
                    ai_queue.enqueue(cfg.clips_dir/res.clip_name,uncertainty)
            verdict = result.get("authority")
            if verdict is not None:
                print(f"裁判 {verdict.by} 判断 {res.clip_name}："
                      f"{'喝水' if verdict.drinking else '没喝'}")
        threading.Thread(target=_run, daemon=True).start()

    # 采集线程：全速读相机 → 更新预览（网页流畅）；按 fps 节奏喂回放缓冲，
    # 会话录制也在这里按帧率写帧（writer fps 一致，播放速度才正确）。
    capture_stop=threading.Event()

    def _capture() -> None:
        next_buf = 0.0
        next_prune = 0.0
        last_monitor = 0.0
        last_monitor_state=None
        while not capture_stop.is_set():
            if time.monotonic() >= next_prune:
                recorder.prune()
                next_prune = time.monotonic() + 30
            ok, raw = cap.read()
            if not ok:
                state=latest.get_state()
                if state is not None and time.time()-state[0]>3:
                    presence.set(time.time(),False)
                    motion.reset()
                    frame_buffer.clear()
                    if session is not None:
                        result=session.close()
                        if result is not None:
                            stats.record_event(result.timestamp,result.clip_name,candidate_only=result.candidate_only,review_candidate=not cfg.motion_recording,session=result)
                            _finalize_async(result)
                    latest.clear()
                time.sleep(buf_interval)
                continue
            now = cap.timestamp if getattr(cap, "is_live_stream", False) else time.time()
            height,width=raw.shape[:2]
            scale=min(1.0,cfg.frame_width/width,cfg.frame_height/height)
            if scale<1:
                raw=cv2.resize(raw,(round(width*scale),round(height*scale)),interpolation=cv2.INTER_AREA)
            monitoring.touch(now)
            quality = nightvision.assess_quality(raw,cfg.bowl_roi,max(50,cfg.minimum_monitor_brightness))
            # 暗背景不应否决已经通过检查的饮水区域；录制只依赖局部画质。
            night = quality['brightness'] < cfg.night_brightness_threshold
            monitor_state=(quality['usable'],quality.get('capture_mode'),quality.get('status'))
            if now - last_monitor >= 30 or monitor_state!=last_monitor_state:
                monitoring.observe(now, quality)
                last_monitor = now
                last_monitor_state=monitor_state
            frame=raw
            latest.set(now,frame,night,quality)
            if cfg.motion_recording:
                presence.set(now,motion.observe(now,frame))
            # 水面变化不等于猫遮挡；出水口按自己的区域检查画质和猫框。
            flow.observe(now, raw, True, outlet_occlusion.get()[0])
            if now + .002 >= next_buf:
                next_buf += buf_interval
                if next_buf < now - buf_interval:
                    next_buf = now + buf_interval
                res=None
                if cfg.motion_recording or quality['usable']:
                    pipeline.observe(now,frame)
                    if session is not None:
                        in_roi,since=presence.get()
                        res=session.update(now,frame,in_roi,since,frame_buffer,
                                           candidate_only=not quality['usable'])
                else:
                    presence.set(now,False)
                    frame_buffer.clear()
                    # 结束已有的有效录像，不把不可用画面写进去。
                    if session is not None:
                        res=session.close()
                if res is not None:
                    print(f"录到一段候选： {res.clip_name}（等待判断）")
                    stats.record_event(res.timestamp,res.clip_name,candidate_only=res.candidate_only,review_candidate=not cfg.motion_recording,session=res)
                    _finalize_async(res)

    capture_thread=threading.Thread(target=_capture, daemon=True)
    capture_thread.start()

    def classify_existing():
        for name in stats.unclassified_clips():
            if capture_stop.is_set():return
            path=cfg.clips_dir/name
            if path.is_file():
                try:review_router.route(path)
                except Exception as error:print(f'回放分流失败 {name}：{error}')
        print('已有录像已完成回放与复核分流。')
    threading.Thread(target=classify_existing,daemon=True).start()

    # 主进程退出时关闭相机子进程，避免重启后仍占用设备。
    import signal
    if threading.current_thread() is threading.main_thread():
        def stop_camera_process(signum,frame):
            raise SystemExit(0)
        signal.signal(signal.SIGTERM,stop_camera_process)

    # 检测循环：按自己的节奏取最新帧跑识别；预览/录制不受其拖累。
    # 会话模式：只更新「猫是否在碗」给采集线程的状态机；旧模式：直接定长录制。
    try:
        while True:
            state = latest.get_state()
            if state is None:
                time.sleep(cfg.detect_interval_seconds)
                continue
            now, frame, night = state
            quality = latest.get_quality()
            blocked = not quality.get('usable', False) or time.time()-now>2
            if blocked:
                bowl_presence.reset()
                foreground_present = False
            else:
                foreground_present = bowl_presence.observe(frame, cfg.bowl_roi, now)['present']
            cats = [] if time.time()-now > 2 else cat_detector.detect_cats(frame)
            h, w = frame.shape[:2]
            outlet = ratio_rect_to_pixels(flow.roi, w, h)
            outlet_occlusion.set(now, foreground_present or any(cat_overlaps_bowl(cat, outlet, .15) for cat in cats))
            if cfg.motion_recording:
                time.sleep(cfg.detect_interval_seconds)
                continue
            if session is not None:
                in_roi=False if blocked else (pipeline.cat_in_bowl(frame,night,cats=cats) or foreground_present)
                presence.set(now, in_roi)
            elif not blocked:
                clip = pipeline.detect(now, frame, night=night)
                if clip:
                    print(f"录到一段候选： {clip}（等待人工判断）")
            time.sleep(cfg.detect_interval_seconds)
    finally:
        capture_stop.set()
        capture_thread.join(timeout=4)
        if session is not None:
            res = session.close()
            if res is not None:
                stats.record_event(res.timestamp, res.clip_name,candidate_only=res.candidate_only,review_candidate=not cfg.motion_recording,session=res)
                # 退出时收尾的最后一段：同步合一次声音（守护线程即将随进程消失，来不及异步）。
                if audio_ring is not None and audio_ring.available:
                    mux_audio_into(cfg.clips_dir / res.clip_name,
                                   audio_ring.slice(res.audio_start, res.audio_end),
                                   cfg.audio_sample_rate)
        if audio_ring is not None:
            audio_ring.stop()
        cap.release()
