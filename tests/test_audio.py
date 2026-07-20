"""音频环形缓冲 + mux/muted 命令：全部走「喂合成数据 / 只拼参数」路径，不碰真麦克风或 ffmpeg。"""
from pathlib import Path

from catcam.audio import AudioRing, capture_cmd, mux_cmd, muted_cmd, mux_audio_into


def test_ingest_sets_timeline_and_slices():
    r = AudioRing("avfoundation", ":1", sample_rate=1000, max_seconds=10)
    # 1000 采样 = 2000 字节 = 1 秒，于 t=100.0 到达（覆盖 [99,100]）→ t0=99.0
    r._ingest(100.0, b"\x01\x02" * 1000)
    r._ingest(100.5, b"\x03\x04" * 500)   # 再 0.5 秒，覆盖 [100,100.5]
    # 切 [99.5,100.0] → 第一块后半 500 采样 = 1000 字节，全是 \x01\x02
    s = r.slice(99.5, 100.0)
    assert s == b"\x01\x02" * 500
    # 跨块 [99.5,100.25] → 500(\x01\x02) + 250(\x03\x04) 采样
    s2 = r.slice(99.5, 100.25)
    assert s2 == b"\x01\x02" * 500 + b"\x03\x04" * 250


def test_slice_before_any_data_is_none():
    r = AudioRing("f", "d", sample_rate=1000)
    assert r.slice(0.0, 1.0) is None


def test_ring_trims_oldest_when_over_capacity():
    r = AudioRing("f", "d", sample_rate=1000, max_seconds=1)  # 上限 1000 采样
    r._ingest(1.0, b"aa" * 1000)   # 采样 [0,1000)，t0=0
    r._ingest(2.0, b"bb" * 1000)   # 采样 [1000,2000)，超量→丢最旧 1000
    assert r._base_sample == 1000
    assert r.slice(0.0, 1.0) is None            # 老数据已丢
    assert r.slice(1.0, 2.0) == b"bb" * 1000    # 最近的仍在


def test_slice_clamps_out_of_range():
    r = AudioRing("f", "d", sample_rate=1000, max_seconds=10)
    r._ingest(10.0, b"zz" * 1000)   # [9,10]
    # 请求窗口比现有数据宽 → 夹取到现有范围
    s = r.slice(0.0, 100.0)
    assert s == b"zz" * 1000


def test_status_reports_buffer_and_mux_results():
    r = AudioRing("f", "mic", sample_rate=1000, max_seconds=10)
    r._ingest(10.0, b"zz" * 1500)
    r.available = True
    r.record_mux_result("a.mp4", True)
    r.record_mux_result("b.mp4", False)
    status = r.status()
    assert status["available"] is True
    assert status["buffered_seconds"] == 1.5
    assert status["mux_successes"] == 1 and status["mux_failures"] == 1
    assert status["last_mux_clip"] == "b.mp4"


def test_capture_cmd_has_format_device_and_pcm_output():
    c = capture_cmd("avfoundation", ":1", 16000)
    assert "ffmpeg" in c[0]
    assert "-f" in c and "avfoundation" in c and ":1" in c
    assert "s16le" in c and "16000" in c and c[-1] == "-"


def test_mux_cmd_copies_video_and_encodes_audio():
    c = mux_cmd("v.mp4", "a.raw", "out.mp4", 16000)
    assert "-c:v" in c and "copy" in c
    assert "-c:a" in c and "aac" in c and "-shortest" in c
    assert "v.mp4" in c and "a.raw" in c and "out.mp4" in c


def test_muted_cmd_drops_audio_to_pipe():
    c = muted_cmd("v.mp4")
    assert "-an" in c and "-c:v" in c and "copy" in c
    assert "pipe:1" in c and "v.mp4" in c


def test_mux_audio_into_fail_open_on_empty_pcm(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"not a real mp4")
    assert mux_audio_into(v, b"", 16000) is False   # 空 pcm → 不动、返回 False
    assert v.read_bytes() == b"not a real mp4"       # 原文件没被破坏


def test_mux_audio_into_fail_open_on_timeout(tmp_path, monkeypatch):
    import subprocess
    import catcam.audio as audio

    v = tmp_path / "clip.mp4"
    v.write_bytes(b"video")
    monkeypatch.setattr(audio.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(audio.subprocess, "run", timeout)
    assert mux_audio_into(v, b"\x00\x00", timeout_seconds=0.01) is False
    assert v.read_bytes() == b"video"
