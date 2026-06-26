import numpy as np
from catcam.detector import DrinkingDetector
from catcam.framebuffer import FrameBuffer
from catcam.recorder import ClipRecorder
from catcam.stats import StatsStore
from catcam.pipeline import Pipeline


class _CatAlwaysInBowl:
    """假认猫器：永远返回一个铺满整张图的猫框。"""
    def detect_cats(self, frame):
        h, w = frame.shape[:2]
        return [(0.0, 0.0, float(w), float(h))]


class _NoCat:
    def detect_cats(self, frame):
        return []


def _build(tmp_path, cat_detector):
    return Pipeline(
        cat_detector=cat_detector,
        drinking_detector=DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0),
        frame_buffer=FrameBuffer(seconds=4.0, fps=5),
        recorder=ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5),
        stats=StatsStore(tmp_path / "s.db"),
        bowl_roi_ratio=(0.0, 0.0, 1.0, 1.0),
        min_overlap_ratio=0.15,
    )


def _frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_no_event_without_cat(tmp_path):
    pipe = _build(tmp_path, _NoCat())
    assert pipe.process(now=0.0, frame=_frame()) is None
    assert pipe.process(now=5.0, frame=_frame()) is None


def test_event_records_clip_and_stat(tmp_path):
    pipe = _build(tmp_path, _CatAlwaysInBowl())
    assert pipe.process(now=0.0, frame=_frame()) is None     # 开始计时
    clip_name = pipe.process(now=3.0, frame=_frame())        # 满 3s 触发
    assert clip_name == "clip_3000.mp4"
    assert (tmp_path / "clips" / clip_name).exists()
    # 计数口径：事件录下了，但要被确认「喝水」(is_drinking=1) 才计入次数
    start, end = 0.0, 1e12
    assert pipe.stats.count_between(start, end) == 0          # 还没确认 → 不计
    import sqlite3
    with sqlite3.connect(tmp_path / "s.db") as c:
        c.execute("INSERT INTO labels (clip_name, is_drinking, ts) VALUES (?, 1, NULL)", (clip_name,))
    assert pipe.stats.count_between(start, end) == 1          # 确认喝水后才计


class _SmallCatBelowThreshold:
    """猫框只占画面一角，与水碗(整帧)的重叠 < min_overlap_ratio(0.15)。"""
    def detect_cats(self, frame):
        return [(0.0, 0.0, 16.0, 16.0)]   # 256 / 3072 ≈ 0.083


class _PartialCatAboveThreshold:
    """猫框部分覆盖，但重叠 >= min_overlap_ratio。"""
    def detect_cats(self, frame):
        return [(0.0, 0.0, 32.0, 32.0)]   # 1024 / 3072 ≈ 0.333


def test_buffer_populated_on_no_event_frames(tmp_path):
    pipe = _build(tmp_path, _NoCat())
    pipe.process(now=0.0, frame=_frame())
    pipe.process(now=1.0, frame=_frame())
    assert len(pipe.frame_buffer.all_frames()) == 2


def test_no_event_when_overlap_below_threshold(tmp_path):
    pipe = _build(tmp_path, _SmallCatBelowThreshold())
    assert pipe.process(now=0.0, frame=_frame()) is None
    # 即便超过停留阈值，重叠不足也不算喝水
    assert pipe.process(now=5.0, frame=_frame()) is None


def test_event_when_overlap_above_threshold(tmp_path):
    pipe = _build(tmp_path, _PartialCatAboveThreshold())
    assert pipe.process(now=0.0, frame=_frame()) is None
    assert pipe.process(now=3.0, frame=_frame()) == "clip_3000.mp4"
