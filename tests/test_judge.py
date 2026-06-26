from datetime import datetime
from catcam.judge import Verdict, VLMClipJudge, judge_and_notify


class _FakeLabeler:
    """假的 ai_labeler：按预设返回 dict / None / 抛错，并记录被调的 clip。"""
    def __init__(self, result=None, raises=None):
        self._result = result
        self._raises = raises
        self.labeled = []

    def label(self, clip_path):
        self.labeled.append(str(clip_path))
        if self._raises is not None:
            raise self._raises
        return self._result


class _FakeEmailer:
    def __init__(self):
        self.sent = []

    def notify_drinking(self, stats, frame, now_dt):
        self.sent.append((frame, now_dt))
        return True


def test_judge_returns_verdict_on_drinking():
    lab = _FakeLabeler({"drinking": True, "confidence": 0.9, "reason": "舔水"})
    v = VLMClipJudge(lab).judge("a.mp4")
    assert v == Verdict(drinking=True, confidence=0.9, reason="舔水", by="vlm")
    assert lab.labeled == ["a.mp4"]


def test_judge_returns_notdrinking_verdict():
    lab = _FakeLabeler({"drinking": False, "confidence": 0.7, "reason": "只凑近"})
    v = VLMClipJudge(lab).judge("b.mp4")
    assert v.drinking is False and v.confidence == 0.7


def test_judge_none_when_labeler_skips():
    # ai_labeler 返回 None（人工已标 / 抽帧空）→ 裁判无判定
    assert VLMClipJudge(_FakeLabeler(None)).judge("c.mp4") is None


def test_judge_fail_open_on_exception():
    # 全模型失败 ai_labeler 会 raise → 裁判吞成 None（不发不计、不崩）
    logs = []
    j = VLMClipJudge(_FakeLabeler(raises=RuntimeError("429")), log=logs.append)
    assert j.judge("d.mp4") is None
    assert logs and "d.mp4" in logs[0]


def test_judge_and_notify_emails_only_on_drinking():
    em = _FakeEmailer()
    j = VLMClipJudge(_FakeLabeler({"drinking": True, "confidence": 0.9, "reason": "舔水"}))
    v = judge_and_notify(j, em, stats="S", clip_path="a.mp4",
                         start_ts=1700000000.0, photo="PHOTO")
    assert v.drinking is True
    assert len(em.sent) == 1
    frame, now_dt = em.sent[0]
    assert frame == "PHOTO" and isinstance(now_dt, datetime)


def test_judge_and_notify_silent_on_notdrinking():
    em = _FakeEmailer()
    j = VLMClipJudge(_FakeLabeler({"drinking": False, "confidence": 0.8, "reason": "路过"}))
    judge_and_notify(j, em, "S", "b.mp4", 1700000000.0, "PHOTO")
    assert em.sent == []


def test_judge_and_notify_silent_when_no_verdict():
    em = _FakeEmailer()
    j = VLMClipJudge(_FakeLabeler(None))
    judge_and_notify(j, em, "S", "c.mp4", 1700000000.0, "PHOTO")
    assert em.sent == []


def test_judge_and_notify_no_email_without_photo():
    em = _FakeEmailer()
    j = VLMClipJudge(_FakeLabeler({"drinking": True, "confidence": 0.9, "reason": "舔水"}))
    judge_and_notify(j, em, "S", "a.mp4", 1700000000.0, None)
    assert em.sent == []
