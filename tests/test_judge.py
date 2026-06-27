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


from catcam.judge import route_clip


class _FakeLocalJudge:
    def __init__(self, drinking, version="v5"):
        self._v = drinking; self.version = version; self.judged = []
    def judge(self, clip_path):
        self.judged.append(str(clip_path))
        return Verdict(drinking=self._v, confidence=0.9, reason="", by=self.version)


class _FakeStats:
    def __init__(self): self.preds = []
    def set_prediction(self, clip, pred, by): self.preds.append((clip, pred, by))


class _FakeFeedback:
    def __init__(self): self.machine = []
    def record_machine_label(self, clip, drink, source="local", confidence=None, reason=None):
        self.machine.append((clip, drink, source)); return True


def test_route_shadow_vlm_authority_local_shadow_predicts():
    em = _FakeEmailer(); st = _FakeStats(); fb = _FakeFeedback()
    lab = _FakeLabeler({"drinking": True, "confidence": 0.8, "reason": "舔水"})  # VLM 判喝水
    local = _FakeLocalJudge(False, "v5")                                       # 本地判没喝
    r = route_clip(clip_path="a.mp4", start_ts=1700000000.0, photo="P", ai_labeler=lab,
                   local_judge=local, mode="shadow", emailer=em, stats=st, feedback=fb)
    assert r["emailed"] is True            # VLM 是权威、判喝水 → 发
    assert lab.labeled == ["a.mp4"]        # VLM 写了标注
    assert st.preds == [("a.mp4", 0, "v5")]  # 本地影子预测回填(没喝=0)
    assert fb.machine == []                # shadow 不写机器计数标签


def test_route_gate_local_authority_no_vlm():
    em = _FakeEmailer(); st = _FakeStats(); fb = _FakeFeedback()
    lab = _FakeLabeler({"drinking": False})    # 即便 VLM 会判没喝……
    local = _FakeLocalJudge(True, "v5")        # 本地判喝水（gate 权威）
    r = route_clip(clip_path="b.mp4", start_ts=1700000000.0, photo="P", ai_labeler=lab,
                   local_judge=local, mode="gate", emailer=em, stats=st, feedback=fb)
    assert r["emailed"] is True                       # 本地权威判喝水 → 发
    assert lab.labeled == []                          # gate 不调 VLM（画面不上传）
    assert fb.machine == [("b.mp4", True, "local")]   # 本地写计数标签
    assert st.preds == []                             # gate 不另记影子


def test_route_no_local_is_phase1():
    em = _FakeEmailer(); st = _FakeStats(); fb = _FakeFeedback()
    lab = _FakeLabeler({"drinking": True, "confidence": 0.9, "reason": "舔水"})
    route_clip(clip_path="c.mp4", start_ts=1700000000.0, photo="P", ai_labeler=lab,
               local_judge=None, mode="shadow", emailer=em, stats=st, feedback=fb)
    assert len(em.sent) == 1 and lab.labeled == ["c.mp4"] and st.preds == []


def test_route_gate_notdrinking_silent():
    em = _FakeEmailer(); st = _FakeStats(); fb = _FakeFeedback()
    local = _FakeLocalJudge(False, "v5")
    route_clip(clip_path="d.mp4", start_ts=1700000000.0, photo="P", ai_labeler=None,
               local_judge=local, mode="gate", emailer=em, stats=st, feedback=fb)
    assert em.sent == [] and fb.machine == [("d.mp4", False, "local")]
