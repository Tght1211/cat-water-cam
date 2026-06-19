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
    ):
        self.cat_detector = cat_detector
        self.drinking_detector = drinking_detector
        self.frame_buffer = frame_buffer
        self.recorder = recorder
        self.stats = stats
        self.bowl_roi_ratio = bowl_roi_ratio
        self.min_overlap_ratio = min_overlap_ratio

    def process(self, now: float, frame) -> str | None:
        self.frame_buffer.add(now, frame)
        height, width = frame.shape[:2]
        bowl = ratio_rect_to_pixels(self.bowl_roi_ratio, width, height)
        cats = self.cat_detector.detect_cats(frame)
        cat_in_roi = any(
            cat_overlaps_bowl(c, bowl, self.min_overlap_ratio) for c in cats
        )
        event = self.drinking_detector.update(now, cat_in_roi)
        if event is None:
            return None
        clip_path = self.recorder.save_clip(self.frame_buffer.all_frames(), event.timestamp)
        self.stats.record_event(event.timestamp, clip_path.name)
        return clip_path.name
