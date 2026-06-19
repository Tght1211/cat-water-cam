from catcam.recorder import ClipRecorder
from catcam.demo import seed_sample_clips


def test_seed_sample_clips_creates_real_clips(tmp_path):
    rec = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=10, fps=5)
    paths = seed_sample_clips(rec, n=3)
    assert len(paths) == 3
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert len(rec.list_clips()) == 3


def test_seed_respects_ring_buffer_limit(tmp_path):
    rec = ClipRecorder(clips_dir=tmp_path / "clips", max_clips=2, fps=5)
    seed_sample_clips(rec, n=3)
    assert len(rec.list_clips()) == 2
