from __future__ import annotations

from catcam.geometry import ratio_rect_to_pixels, cat_overlaps_bowl


class Pipeline:
    def __init__(
        self,
        cat_detector,
        drinking_detector,
        frame_buffer,
        recorder,
        stats,
        bowl_roi_ratio,
        min_overlap_ratio: float,
        presence_detector=None,
    ):
        self.cat_detector = cat_detector
        self.drinking_detector = drinking_detector
        self.frame_buffer = frame_buffer
        self.recorder = recorder
        self.stats = stats
        self.bowl_roi_ratio = bowl_roi_ratio
        self.min_overlap_ratio = min_overlap_ratio
        # 「简单模型」（画面变化 + 灰蓝猫色块）。与 YOLO 认猫取「或」，提高召回、多攒候选。
        self.presence_detector = presence_detector

    def observe(self, now: float, frame) -> None:
        """把帧放进回放缓冲。由采集线程按帧率调用，与检测解耦，保证录制连贯。"""
        self.frame_buffer.add(now, frame)

    def detect(self, now: float, frame, night: bool = False) -> str | None:
        """只做识别 + 触发录制（不碰缓冲）。由检测线程按自己的节奏调用。"""
        height, width = frame.shape[:2]
        bowl = ratio_rect_to_pixels(self.bowl_roi_ratio, width, height)
        cats = self.cat_detector.detect_cats(frame)
        cat_in_roi = any(
            cat_overlaps_bowl(c, bowl, self.min_overlap_ratio) for c in cats
        )
        if not cat_in_roi and self.presence_detector is not None:
            cat_in_roi = self.presence_detector.present(frame, bowl, night)
        event = self.drinking_detector.update(now, cat_in_roi)
        if event is None:
            return None
        clip_path = self.recorder.save_clip(self.frame_buffer.all_frames(), event.timestamp)
        self.stats.record_event(event.timestamp, clip_path.name)
        return clip_path.name

    def process(self, now: float, frame, night: bool = False) -> str | None:
        """缓冲 + 检测一把梭（单线程用法，测试也走这条）。"""
        self.observe(now, frame)
        return self.detect(now, frame, night)
