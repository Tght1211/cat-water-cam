from catcam.detector import DrinkingDetector, DrinkingEvent


def test_no_event_before_dwell_threshold():
    d = DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0)
    assert d.update(now=0.0, cat_in_roi=True) is None
    assert d.update(now=2.9, cat_in_roi=True) is None


def test_event_fires_after_continuous_dwell():
    d = DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0)
    d.update(now=0.0, cat_in_roi=True)
    evt = d.update(now=3.0, cat_in_roi=True)
    assert isinstance(evt, DrinkingEvent)
    assert evt.timestamp == 3.0


def test_leaving_resets_dwell():
    d = DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0)
    d.update(now=0.0, cat_in_roi=True)
    d.update(now=1.0, cat_in_roi=False)   # 离开，重置
    d.update(now=2.0, cat_in_roi=True)    # 重新开始计时
    assert d.update(now=4.0, cat_in_roi=True) is None   # 仅停留 2s
    assert isinstance(d.update(now=5.0, cat_in_roi=True), DrinkingEvent)


def test_cooldown_blocks_second_event():
    d = DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0)
    d.update(now=0.0, cat_in_roi=True)
    assert isinstance(d.update(now=3.0, cat_in_roi=True), DrinkingEvent)
    # 冷却期内即使一直停留也不再触发
    assert d.update(now=10.0, cat_in_roi=True) is None
    assert d.update(now=62.0, cat_in_roi=True) is None


def test_event_again_after_cooldown():
    d = DrinkingDetector(dwell_seconds=3.0, cooldown_seconds=60.0)
    d.update(now=0.0, cat_in_roi=True)
    d.update(now=3.0, cat_in_roi=True)              # 第一次事件，冷却到 63
    d.update(now=64.0, cat_in_roi=True)             # 冷却结束，重新计时起点
    evt = d.update(now=67.0, cat_in_roi=True)
    assert isinstance(evt, DrinkingEvent)
    assert evt.timestamp == 67.0
