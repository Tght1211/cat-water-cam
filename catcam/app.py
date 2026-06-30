from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import uvicorn

from catcam.classifier import ActiveModel, DrinkingClassifier
from catcam.ai_labeler import AILabeler
from catcam.config import load_config
from catcam.detector import DrinkingDetector
from catcam.feedback import FeedbackStore
from catcam.models import ModelRegistry
from catcam.framebuffer import FrameBuffer
from catcam.judge import VLMClipJudge, route_clip
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
from catcam.vision import CatDetector
from catcam.web import create_app


class LatestFrame:
    """采集线程写、网页/检测线程读的最新帧（含时间戳与是否夜间）。

    预览（MJPEG/快照）只要帧；检测线程还要 now/night。加锁、返回拷贝避免并发改。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._now = None
        self._frame = None
        self._night = False

    def set(self, now: float, frame, night: bool) -> None:
        with self._lock:
            self._now = now
            self._frame = frame
            self._night = night

    def get(self):
        """供网页预览：只返回帧拷贝。"""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

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


def main(config_path: str = "config.json") -> None:
    cfg = load_config(config_path)

    cat_detector = CatDetector.from_path(cfg.yolo_model, cfg.cat_confidence)
    stats = StatsStore(cfg.db_path)
    recorder = ClipRecorder(cfg.clips_dir, cfg.max_clips, cfg.fps)
    feedback = FeedbackStore(cfg.db_path, cfg.training_dir)
    # 裁剪口径：超量时只删被判「没喝」的段（喝水/未判定永不自动删）。
    # 没喝段的训练价值（抽帧 + s3d 特征缓存）已另存，删 mp4 不影响训练。
    recorder.is_deletable = lambda name: feedback.get_label(name) is False
    ai_labeler = AILabeler.from_config(feedback, cfg)  # 未启用/缺 key 返回 None
    if ai_labeler is not None:
        print(f"AI 自动标注已开启 → {cfg.ai_model}（⚠️ 画面帧会上传外部服务器）")
    # 整段裁判：第一阶段用 VLM（包 ai_labeler）。它是「发邮件 + 记次数」的唯一权威——
    # 判「真喝水」才发、才计入；没启用 AI（无 key）则没有裁判，不会发邮件、次数恒 0。
    judge = VLMClipJudge(ai_labeler) if ai_labeler is not None else None
    if judge is None and cfg.record_session:
        print("⚠️ 未启用 AI 裁判（config 里 ai_api_key 为空）："
              "不会发喝水邮件、今日喝水次数为 0。喝水判定依赖 AI——请填 ai_api_key。")
    emailer = Emailer(cfg)
    registry = ModelRegistry(cfg.models_dir / "registry.json")
    # 本地视频裁判（s3d+head）：仅当 registry 当前生效版本是视频模型时加载。
    # shadow=影子评估（VLM 仍权威）；gate=本地当权威且不再调 VLM。其余情况 = None（第一阶段行为）。
    local_video_judge = None
    local_video_mode = "shadow"
    _active = registry.get(registry.active_id()) if registry.active_id() else None
    if _active and _active.get("base") == "s3d+head":
        try:
            from catcam.videojudge import S3DFeatureExtractor, DrinkingHead, LocalVideoClipJudge
            _head = DrinkingHead.load(_active["path"])
            local_video_judge = LocalVideoClipJudge(S3DFeatureExtractor(), _head, _active["id"])
            local_video_mode = registry.active_mode()
            tip = "本地当权威、不调 VLM" if local_video_mode == "gate" else "影子评估、VLM 仍权威"
            print(f"本地视频裁判已加载：{_active['id']}（{local_video_mode} · {tip}）")
        except Exception as e:  # noqa: BLE001
            print(f"本地视频裁判加载失败，忽略（继续用 VLM）：{e}")
    active_model = ActiveModel()
    # 启动时把上次选中的「生效模型」加载进来（若有）。s3d+head 视频版本不走这里
    # （由上面的 local_video_judge 处理；塞进单帧 active_model 会被 ultralytics 误当 YOLO 加载）。
    active_path = registry.active_path()
    _active_is_video = bool(_active and _active.get("base") == "s3d+head")
    if active_path and Path(active_path).exists() and not _active_is_video:
        try:
            mode = registry.active_mode()
            active_model.set(DrinkingClassifier.from_path(active_path), registry.active_id(), mode)
            tip = "过滤误触" if mode == "gate" else "测试模式，只评估不拦截录制"
            print(f"已启用分类器：{registry.active_id()}（{mode} · {tip}）")
        except Exception as e:  # noqa: BLE001
            print(f"启用分类器失败，改用简单模型：{e}")
    trainer = TrainingManager(
        cfg.training_dir, cfg.models_dir, cfg.cls_base_model, cfg.train_epochs, cfg.train_imgsz,
        feedback=feedback, registry=registry,
    )
    # 网页「训练视频模型」按钮用：后台训 s3d+head 小头（与单帧 TrainingManager 并存）。
    video_trainer = VideoTrainingManager(
        cfg.clips_dir, cfg.training_dir, feedback, registry, cfg.models_dir,
    )
    # 会话录制要把「凑近过程 + dwell 这几秒」一起补进开头，缓冲就开这么长。
    buffer_seconds = (
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
        active_model=active_model,
    )
    session = (
        DrinkSession(
            recorder,
            cfg.dwell_seconds,
            cfg.session_end_grace_seconds,
            cfg.max_session_seconds,
            cfg.cooldown_seconds,
        )
        if cfg.record_session
        else None
    )
    presence = Presence()

    latest = LatestFrame()
    app = create_app(
        stats, recorder, feedback, latest.get, cfg.clips_dir, trainer,
        registry=registry, active_model=active_model, video_trainer=video_trainer,
    )
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
    if emailer.enabled:
        print(f"邮件提醒已开启 → {emailer.to}（最小间隔 {int(emailer.min_interval)}s）")
    if cfg.record_session:
        print(
            f"录制模式：整段会话（前补 {cfg.preroll_seconds:g}s，"
            f"离开 {cfg.session_end_grace_seconds:g}s 收尾，封顶 {cfg.max_session_seconds:g}s）"
        )

    source = cfg.video_source if cfg.video_source else cfg.camera_index
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"打不开视频源： {source!r}")

    buf_interval = 1.0 / max(1, cfg.fps)  # 回放缓冲/会话写帧按 fps 节奏，保证时长/速度正确

    def _judge_async(clip_name: str, start_ts: float, photo) -> None:
        # 会话录完后台编排：按模式决定谁标注/谁判邮件计数/谁影子预测。绝不阻塞采集。
        if judge is None and local_video_judge is None:
            return
        def _run():
            try:
                route_clip(
                    clip_path=cfg.clips_dir / clip_name, start_ts=start_ts, photo=photo,
                    ai_labeler=ai_labeler, local_judge=local_video_judge,
                    mode=local_video_mode, emailer=emailer, stats=stats, feedback=feedback,
                )
            except Exception as e:  # noqa: BLE001
                print(f"AI 裁判流程异常（{clip_name}）：{e}")
        threading.Thread(target=_run, daemon=True).start()

    # 采集线程：全速读相机 → 更新预览（网页流畅）；按 fps 节奏喂回放缓冲，
    # 会话录制也在这里按帧率写帧（writer fps 一致，播放速度才正确）。
    def _capture() -> None:
        last_buf = 0.0
        while True:
            ok, raw = cap.read()
            if not ok:
                time.sleep(buf_interval)
                continue
            now = time.time()
            night = nightvision.is_dark(raw, cfg.night_brightness_threshold)
            frame = nightvision.enhance_lowlight(raw) if night else raw
            latest.set(now, frame, night)
            if now - last_buf >= buf_interval:
                last_buf = now
                pipeline.observe(now, frame)
                if session is not None:
                    in_roi, since = presence.get()
                    res = session.update(now, frame, in_roi, since, frame_buffer)
                    if res is not None:
                        print(f"录到一段候选： {res.clip_name}（等 AI 裁判判是否真喝水）")
                        # 影子模型对这段的预测（不影响判定，仅记下来供评估）。
                        pred = active_model.predict(res.photo)
                        stats.record_event(
                            res.timestamp, res.clip_name,
                            predicted=None if pred is None else int(pred),
                            predicted_by=active_model.active_id,
                        )
                        # 整段裁判 → 判「真喝水」才发邮件 + 计入次数；放后台线程绝不阻塞采集。
                        _judge_async(res.clip_name, res.timestamp, res.photo)

    threading.Thread(target=_capture, daemon=True).start()

    # 检测循环：按自己的节奏取最新帧跑识别；预览/录制不受其拖累。
    # 会话模式：只更新「猫是否在碗」给采集线程的状态机；旧模式：直接定长录制。
    try:
        while True:
            state = latest.get_state()
            if state is None:
                time.sleep(cfg.detect_interval_seconds)
                continue
            now, frame, night = state
            blocked = night and not cfg.record_at_night
            if session is not None:
                in_roi = False if blocked else pipeline.cat_in_bowl(frame, night)
                presence.set(now, in_roi)
            elif not blocked:
                clip = pipeline.detect(now, frame, night=night)
                if clip:
                    print(f"录到一段候选： {clip}（等 AI 裁判）")
                    _judge_async(clip, now, frame.copy())
            time.sleep(cfg.detect_interval_seconds)
    finally:
        if session is not None:
            res = session.close()
            if res is not None:
                stats.record_event(res.timestamp, res.clip_name)
        cap.release()
