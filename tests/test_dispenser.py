from catcam.dispenser import (
    DispenserStore, estimate_remaining, filter_days_left,
)


def test_estimate_remaining_floors_at_zero():
    assert estimate_remaining(2000, 10, 30) == 1700
    assert estimate_remaining(2000, 100, 30) == 0      # 喝超了 → 夹到 0，不给负数
    assert estimate_remaining(0, 5, 30) == 0


def test_filter_days_left():
    now = 100 * 86400.0
    # 上次换在第 90 天，周期 30 天 → 还剩 20 天
    assert abs(filter_days_left(90 * 86400.0, 30, now) - 20) < 1e-6
    # 超期 → 负数
    assert filter_days_left(60 * 86400.0, 30, now) < 0


def test_store_defaults(tmp_path):
    s = DispenserStore(tmp_path / "d.db", default_ml_per_drink=30.0)
    st = s.get_state()
    assert st["last_refill_ts"] == 0 and st["last_refill_ml"] == 0
    assert st["filter_cycle_days"] == 30              # 由 default_cycle_days 落库
    assert st["ml_per_drink"] == 30.0                 # 未校准 → 用默认


def test_refill_self_calibrates_over_periods(tmp_path):
    s = DispenserStore(tmp_path / "d.db", default_ml_per_drink=30.0, default_cycle_days=30)
    # 第一次蓄水：没有上一周期 → 不校准
    s.refill(ml=2000, now=1000.0, prev_drinks=0)
    assert s.get_state()["ml_per_drink"] == 30.0
    assert s.get_state()["last_refill_ml"] == 2000
    # 第二次蓄水：上一周期加了 2000ml、期间喝了 40 次 → 每次 50ml
    s.refill(ml=1500, now=2000.0, prev_drinks=40)
    assert abs(s.get_state()["ml_per_drink"] - 50.0) < 1e-6
    # 第三次：上周期 1500ml / 30 次 = 50ml；池化 (2000+1500)/(40+30)=50
    s.refill(ml=1000, now=3000.0, prev_drinks=30)
    assert abs(s.get_state()["ml_per_drink"] - 50.0) < 1e-6
    assert s.get_state()["last_refill_ml"] == 1000


def test_refill_ignores_zero_drink_period(tmp_path):
    s = DispenserStore(tmp_path / "d.db", default_ml_per_drink=25.0)
    s.refill(ml=2000, now=1000.0, prev_drinks=0)
    s.refill(ml=2000, now=2000.0, prev_drinks=0)   # 上周期没喝 → 不能除零、不校准
    assert s.get_state()["ml_per_drink"] == 25.0


def test_mark_filter_and_set_cycle(tmp_path):
    s = DispenserStore(tmp_path / "d.db")
    s.mark_filter(now=5000.0)
    assert s.get_state()["last_filter_change_ts"] == 5000.0
    s.set_cycle(45)
    assert s.get_state()["filter_cycle_days"] == 45
