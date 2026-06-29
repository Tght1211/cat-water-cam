import base64
import pytest
from catcam.ai_labeler import encode_frames, build_messages, parse_label


def test_encode_frames_data_url(tmp_path):
    p = tmp_path / "f.jpg"; p.write_bytes(b"\xff\xd8jpegbytes")
    urls = encode_frames([p])
    assert urls[0].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(urls[0].split(",", 1)[1]) == b"\xff\xd8jpegbytes"


def test_build_messages_has_all_images_and_prompt():
    msgs = build_messages(["data:image/jpeg;base64,AAA", "data:image/jpeg;base64,BBB"])
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    texts = [c for c in content if c["type"] == "text"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 2
    assert "喝水" in texts[0]["text"] and "JSON" in texts[0]["text"]


def test_parse_label_plain_json():
    out = parse_label('{"drinking": true, "confidence": 0.9, "reason": "舔水"}')
    assert out == {"drinking": True, "confidence": 0.9, "reason": "舔水"}


def test_parse_label_with_code_fence_and_noise():
    raw = '好的：\n```json\n{"drinking": false, "confidence": 0.6, "reason": "只是凑近"}\n```\n'
    out = parse_label(raw)
    assert out["drinking"] is False and out["confidence"] == 0.6


def test_parse_label_string_bool_and_missing_fields():
    out = parse_label('{"drinking": "yes"}')
    assert out["drinking"] is True and out["confidence"] is None and out["reason"] == ""


def test_parse_label_garbage_raises():
    with pytest.raises(ValueError):
        parse_label("完全不是 json")


def test_parse_label_missing_drinking_raises():
    # 合法 JSON 但缺 drinking 字段 → 当解析失败，别静默写成「没喝」
    with pytest.raises(ValueError):
        parse_label('{"confidence": 0.5, "reason": "说不清"}')


import numpy as np, cv2
from catcam.ai_labeler import AILabeler
from catcam.feedback import FeedbackStore


def _make_clip(path, frames=6):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (16, 16))
    for _ in range(frames):
        w.write(np.full((16, 16, 3), 120, np.uint8))
    w.release()


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]


class _FakeClient:
    def __init__(self, content, fail_times=0):
        self._content = content; self._fail = fail_times; self.calls = 0
        self.chat = type("Chat", (), {"completions": self})()
    def create(self, **kw):  # 充当 client.chat.completions.create
        self.calls += 1
        if self.calls <= self._fail:
            raise RuntimeError("429 rate limit")
        return _FakeResp(self._content)


def _store(tmp_path):
    return FeedbackStore(tmp_path / "db.sqlite", tmp_path / "training")


def test_label_writes_ai_label_and_training_frames(tmp_path):
    clip = tmp_path / "a.mp4"; _make_clip(clip)
    store = _store(tmp_path)
    client = _FakeClient('{"drinking": true, "confidence": 0.77, "reason": "舔水"}')
    lab = AILabeler(store, tmp_path / "training", "m", frames=3, client=client, sleep=lambda s: None)
    out = lab.label(clip)
    assert out["drinking"] is True
    assert store.label_source("a.mp4") == "ai"
    assert store.label_meta("a.mp4")["confidence"] == 0.77
    assert list((tmp_path / "training" / "drinking").glob("*.jpg"))


class _ModelAwareClient:
    """记录每次调用的 model；对 bad_models 里的模型抛错（模拟限流/故障）。"""
    def __init__(self, content, bad_models=()):
        self._content = content; self._bad = set(bad_models); self.models_called = []
        self.chat = type("Chat", (), {"completions": self})()
    def create(self, **kw):
        m = kw.get("model"); self.models_called.append(m)
        if m in self._bad:
            raise RuntimeError(f"429 {m}")
        return _FakeResp(self._content)


def test_label_falls_back_to_next_model(tmp_path):
    clip = tmp_path / "b.mp4"; _make_clip(clip)
    store = _store(tmp_path)
    client = _ModelAwareClient('{"drinking": false}', bad_models=["bad"])
    lab = AILabeler(store, tmp_path / "training", ["bad", "good"], frames=3,
                    client=client, sleep=lambda s: None)
    out = lab.label(clip)
    assert out["drinking"] is False
    assert client.models_called == ["bad", "good"]  # 先试 bad 失败、顺位换 good 成功


def test_label_rotates_start_model_across_calls(tmp_path):
    store = _store(tmp_path)
    client = _ModelAwareClient('{"drinking": true}')  # 都成功
    lab = AILabeler(store, tmp_path / "training", ["a", "b"], frames=3,
                    client=client, sleep=lambda s: None)
    for i in range(2):
        c = tmp_path / f"r{i}.mp4"; _make_clip(c); lab.label(c)
    # 第一次从 a 起、第二次轮换到 b 起 → 分摊各模型额度
    assert client.models_called == ["a", "b"]


def test_label_raises_when_all_models_fail(tmp_path):
    clip = tmp_path / "d.mp4"; _make_clip(clip)
    store = _store(tmp_path)
    client = _ModelAwareClient('{"drinking": false}', bad_models=["x", "y"])
    lab = AILabeler(store, tmp_path / "training", ["x", "y"], frames=3,
                    client=client, sleep=lambda s: None)
    with pytest.raises(RuntimeError):
        lab.label(clip)
    assert store.label_source("d.mp4") is None  # 全失败 → 该段未标注


def test_label_skips_human_labeled(tmp_path):
    clip = tmp_path / "c.mp4"; _make_clip(clip)
    store = _store(tmp_path)
    store.label_clip(clip, True)  # 人工标过
    client = _FakeClient('{"drinking": false}')
    lab = AILabeler(store, tmp_path / "training", "m", frames=3, client=client, sleep=lambda s: None)
    assert lab.label(clip) is None
    assert client.calls == 0  # 没调模型
    assert store.label_source("c.mp4") == "human"  # 未被覆盖


def test_call_retries_pool_after_backoff(tmp_path):
    # 整池第一轮全挂、退避后第二轮成功 → 不该 fail-open 丢掉这段
    clip = tmp_path / "r.mp4"; _make_clip(clip)
    store = _store(tmp_path)
    client = _FakeClient('{"drinking": true, "confidence": 0.9, "reason": "舔水"}', fail_times=1)
    slept = []
    lab = AILabeler(store, tmp_path / "training", "m", frames=3, client=client,
                    sleep=lambda s: slept.append(s), retries=3, retry_base=5.0)
    out = lab.label(clip)
    assert out["drinking"] is True
    assert client.calls == 2            # 第1次失败、退避后第2次成功
    assert slept == [5.0]               # 两轮之间退避一次


def test_call_handles_empty_choices(tmp_path):
    # 某模型返回 choices=None（部分 provider 的错误响应）→ 当失败换下一个，不崩 NoneType
    clip = tmp_path / "e.mp4"; _make_clip(clip)
    store = _store(tmp_path)

    class _NoneChoicesThenGood:
        def __init__(self):
            self.calls = 0
            self.chat = type("Chat", (), {"completions": self})()
        def create(self, **kw):
            self.calls += 1
            if self.calls == 1:
                return type("R", (), {"choices": None})()      # 空 choices
            return _FakeResp('{"drinking": false}')
    client = _NoneChoicesThenGood()
    lab = AILabeler(store, tmp_path / "training", ["a", "b"], frames=3, client=client,
                    sleep=lambda s: None, retries=2)
    out = lab.label(clip)
    assert out["drinking"] is False and client.calls == 2
