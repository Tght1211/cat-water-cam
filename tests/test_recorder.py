import numpy as np
from catcam.recorder import prune_dir, clip_filename, ClipRecorder


def _touch_clip(d, ms):
    (d / f"clip_{ms}.mp4").write_bytes(b"x")


def test_clip_filename_uses_milliseconds():
    assert clip_filename(1.5) == "clip_1500.mp4"


def test_prune_keeps_only_newest(tmp_path):
    for ms in [1000, 2000, 3000, 4000]:
        _touch_clip(tmp_path, ms)
    prune_dir(tmp_path, max_clips=2)
    remaining = sorted(p.name for p in tmp_path.glob("clip_*.mp4"))
    assert remaining == ["clip_3000.mp4", "clip_4000.mp4"]


def test_prune_noop_when_under_limit(tmp_path):
    _touch_clip(tmp_path, 1000)
    prune_dir(tmp_path, max_clips=10)
    assert len(list(tmp_path.glob("clip_*.mp4"))) == 1


def test_save_clip_writes_file_and_enforces_limit(tmp_path):
    rec = ClipRecorder(clips_dir=tmp_path, max_clips=10, fps=5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frames = [frame for _ in range(10)]
    path = rec.save_clip(frames, timestamp=12.0)
    assert path.exists()
    assert path.stat().st_size > 0
    assert path.name == "clip_12000.mp4"


def test_save_clip_ring_buffer_drops_oldest(tmp_path):
    rec = ClipRecorder(clips_dir=tmp_path, max_clips=3, fps=5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for ts in [1.0, 2.0, 3.0, 4.0]:
        rec.save_clip([frame, frame], timestamp=ts)
    names = sorted(p.name for p in rec.list_clips())
    assert names == ["clip_2000.mp4", "clip_3000.mp4", "clip_4000.mp4"]


def test_list_clips_newest_first(tmp_path):
    rec = ClipRecorder(clips_dir=tmp_path, max_clips=10, fps=5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    for ts in [1.0, 3.0, 2.0]:
        rec.save_clip([frame], timestamp=ts)
    names = [p.name for p in rec.list_clips()]
    assert names == ["clip_3000.mp4", "clip_2000.mp4", "clip_1000.mp4"]


def test_prune_protects_nondeletable_deletes_oldest_deletable(tmp_path):
    for ms in [1000, 2000, 3000, 4000, 5000]:
        _touch_clip(tmp_path, ms)
    # 受保护（喝水/未判定）：最旧的 1000 和 3000；其余可删（没喝）
    protected = {"clip_1000.mp4", "clip_3000.mp4"}
    prune_dir(tmp_path, max_clips=3, is_deletable=lambda n: n not in protected)
    remaining = sorted(p.name for p in tmp_path.glob("clip_*.mp4"))
    # 超 2 段 → 从最旧删可删的：删 2000、4000；留受保护的 1000/3000 + 最新的 5000
    assert remaining == ["clip_1000.mp4", "clip_3000.mp4", "clip_5000.mp4"]


def test_prune_keeps_protected_even_over_limit(tmp_path):
    for ms in [1000, 2000, 3000, 4000]:
        _touch_clip(tmp_path, ms)
    # 全部受保护 → 即便超限也一个不删（喝水/未判定永不删）
    prune_dir(tmp_path, max_clips=1, is_deletable=lambda n: False)
    assert len(list(tmp_path.glob("clip_*.mp4"))) == 4


def test_prune_default_still_deletes_oldest(tmp_path):
    for ms in [1000, 2000, 3000]:
        _touch_clip(tmp_path, ms)
    prune_dir(tmp_path, max_clips=2)   # 无 predicate → 旧行为
    remaining = sorted(p.name for p in tmp_path.glob("clip_*.mp4"))
    assert remaining == ["clip_2000.mp4", "clip_3000.mp4"]
