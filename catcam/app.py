from __future__ import annotations

import threading
import time
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
from catcam.video_trainer import VideoTrainingManager, extract_and_cache
from catcam.videojudge import DrinkingHead, LocalVideoClipJudge, S3DFeatureExtractor
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
        self._visibility = {"status": "unknown", "can_judge": False, "reason": "尚未收到画面"}

    def set(self, now: float, frame, night: bool, visibility=None) -> None:
        with self._lock:
            self._now = now
            self._frame = frame
            self._night = night
            if visibility is not None:
                self._visibility = dict(visibility)

    def visibility(self):
        with self._lock:
            if self._frame is not None and time.time() - self._now > 5:
                return {"status": "unknown", "can_judge": False, "reason": "摄像头画面已中断"}
            return dict(self._visibility)

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


class VideoJudgeRuntime:
    """可热切换的本地视频裁判；录制线程每段开始判断前取一次快照。"""

    def __init__(self, extractor=None, training_dir=None):
        self._lock = threading.Lock()
        self._judge = None
        self._mode = "shadow"
        self.extractor = extractor or S3DFeatureExtractor()
        self.training_dir = training_dir

    def activate(self, entry: dict, mode: str):
        path = Path(entry["path"])
        if not path.exists():
            raise FileNotFoundError(path)
        head = DrinkingHead.load(path)
        feature_provider = None
        if self.training_dir is not None:
            feature_provider = lambda clip: extract_and_cache(clip, self.training_dir, self.extractor, dim=head.dim)
        judge = LocalVideoClipJudge(self.extractor, head, entry["id"], feature_provider=feature_provider)
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


def main(config_path: str = "config.json") -> None:
    cfg = load_config(config_path)

    cat_detector = CatDetector.from_path(cfg.yolo_model, cfg.cat_confidence)
    stats = StatsStore(cfg.db_path)
    recorder = ClipRecorder(cfg.clips_dir, cfg.max_clips, cfg.fps)
    feedback = FeedbackStore(cfg.db_path, cfg.training_dir)
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
    video_extractor = S3DFeatureExtractor(cfg.video_device)
    video_judge_runtime = VideoJudgeRuntime(video_extractor, cfg.training_dir)
    active_entry = registry.get(registry.active_id()) if registry.active_id() else None
    if active_entry and active_entry.get("base") == "s3d+head":
        try:
            video_judge_runtime.activate(active_entry, registry.active_mode())
            print(f"本地视频裁判已加载：{active_entry['id']}（{registry.active_mode()}）")
        except Exception as e:  # noqa: BLE001
            print(f"本地视频裁判加载失败，暂不启用：{e}")
    # 喝水结论只接受人工标注。保留模型管理对象仅用于兼容已有 API，不加载、不参与录制。
    active_model = ActiveModel()
    trainer = TrainingManager(
        cfg.training_dir, cfg.models_dir, cfg.cls_base_model, cfg.train_epochs, cfg.train_imgsz,
        feedback=feedback, registry=registry,
    )
    # 网页「训练视频模型」按钮用：后台训 s3d+head 小头（与单帧 TrainingManager 并存）。
    video_trainer = VideoTrainingManager(
        cfg.clips_dir, cfg.training_dir, feedback, registry, cfg.models_dir,
        extractor=video_extractor,
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
        active_model=None,
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

    # 录音（仅会话录制模式）：持续采麦克风进环形缓冲，会话结束把对应时段 mux 进 mp4。
    # 起不来（无权限/无设备）由 AudioRing 内部看门狗兜底 → 无声，不影响录制。
    audio_ring = None
    if cfg.record_audio and session is not None:
        ring_seconds = cfg.preroll_seconds + cfg.max_session_seconds + 10.0
        audio_ring = AudioRing(
            cfg.audio_input_format, cfg.audio_device,
            sample_rate=cfg.audio_sample_rate, max_seconds=ring_seconds,
        )
        audio_ring.start()
        print(f"正在启动录音（{cfg.audio_input_format} {cfg.audio_device}）；"
              f"⚠️ 若无声请在『系统设置→隐私与安全性→麦克风』给终端授权。")

    latest = LatestFrame()
    dispenser = DispenserStore(
        cfg.db_path, cfg.dispenser_default_ml_per_drink, cfg.filter_cycle_days)
    app = create_app(
        stats, recorder, feedback, latest.get, cfg.clips_dir, trainer,
        registry=registry, active_model=active_model, video_trainer=video_trainer,
        video_model_switch=video_judge_runtime.activate,
        video_model_clear=video_judge_runtime.clear,
        audio_status_provider=(audio_ring.status if audio_ring is not None else None),
        visibility_status_provider=latest.visibility,
        dispenser=dispenser, dispenser_low_water_pct=cfg.dispenser_low_water_pct,
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
        if audio_ring is not None:
            audio_ring.stop()
        raise RuntimeError(f"打不开视频源： {source!r}")

    buf_interval = 1.0 / max(1, cfg.fps)  # 回放缓冲/会话写帧按 fps 节奏，保证时长/速度正确

    def _finalize_async(res) -> None:
        # 先合入音频，再由当前生效的本地视频模型判断；都放后台，不阻塞采集。
        def _run():
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
            if judge is None:
                return
            result = route_clip(
                clip_path=cfg.clips_dir / res.clip_name,
                start_ts=res.timestamp,
                photo=res.photo,
                ai_labeler=ai_labeler,
                local_judge=judge,
                mode=mode,
                emailer=emailer,
                stats=stats,
                feedback=feedback,
                visibility_check=lambda path: nightvision.clip_is_visible(
                    path, cfg.bowl_roi, cfg.minimum_visibility_brightness),
            )
            verdict = result.get("authority")
            if verdict is not None:
                print(f"裁判 {verdict.by} 判断 {res.clip_name}："
                      f"{'喝水' if verdict.drinking else '没喝'}")
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
            # Keep original evidence for recording/training; enhancement is not night vision.
            frame = raw
            visibility = nightvision.visibility_status(raw, cfg.bowl_roi, cfg.minimum_visibility_brightness,
                                                       cfg.night_brightness_threshold)
            latest.set(now, frame, night, visibility)
            if now - last_buf >= buf_interval:
                last_buf = now
                pipeline.observe(now, frame)
                if session is not None:
                    in_roi, since = presence.get()
                    res = session.update(now, frame, in_roi, since, frame_buffer)
                    if res is not None:
                        print(f"录到一段候选： {res.clip_name}（等待人工判断）")
                        stats.record_event(res.timestamp, res.clip_name)
                        _finalize_async(res)

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
            blocked = (night and not cfg.record_at_night) or not latest.visibility()["can_judge"]
            if night and not blocked:
                frame = nightvision.enhance_lowlight(frame)
            if session is not None:
                in_roi = False if blocked else pipeline.cat_in_bowl(frame, night)
                presence.set(now, in_roi)
            elif not blocked:
                clip = pipeline.detect(now, frame, night=night)
                if clip:
                    print(f"录到一段候选： {clip}（等待人工判断）")
            time.sleep(cfg.detect_interval_seconds)
    finally:
        if session is not None:
            res = session.close()
            if res is not None:
                stats.record_event(res.timestamp, res.clip_name)
                # 退出时收尾的最后一段：同步合一次声音（守护线程即将随进程消失，来不及异步）。
                if audio_ring is not None and audio_ring.available:
                    mux_audio_into(cfg.clips_dir / res.clip_name,
                                   audio_ring.slice(res.audio_start, res.audio_end),
                                   cfg.audio_sample_rate)
        if audio_ring is not None:
            audio_ring.stop()
        cap.release()
