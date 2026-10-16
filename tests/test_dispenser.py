import pytest
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


def test_refill_does_not_assume_previous_tank_was_consumed(tmp_path):
    s = DispenserStore(tmp_path / 'd.db')
    s.refill(2000, 1000)
    s.refill(1500, 2000, prev_drinks=40)
    assert s.get_state()['calib_drinks'] == 0
    assert s.get_state()['ml_per_drink'] == 30


def test_measured_water_calibrates_only_when_explicit(tmp_path):
    s = DispenserStore(tmp_path / 'd.db')
    s.refill(2000, 1000)
    s.observe_water(1500, 5000, prev_drinks=10)
    assert s.get_state()['calib_drinks'] == 0
    s.observe_water(1200, 9000, prev_drinks=10, calibrate=True, other_loss_ml=100)
    assert s.get_state()['ml_per_drink'] == 20
    state = s.water_summary(9000)
    assert state['remaining_ml'] == 1200
    assert state['depletion_ml_per_day'] > 0


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


def test_stopped_flow_cause_uses_confirmed_empty_cycles(tmp_path):
    from catcam.stats import StatsStore
    d=DispenserStore(tmp_path/'s.db')
    stats=StatsStore(tmp_path/'s.db')
    assert d.flow_diagnosis(stats,100)['cause']=='unknown'
    start=1000
    d.refill(2000,start)
    assert d.flow_diagnosis(stats,start+86400)['cause']=='unknown'
    d.finish_cycle(start+5*86400,0)
    d.refill(2000,start+6*86400)
    assert d.flow_diagnosis(stats,start+6*86400+3600)['cause']=='check_fault'
    assert d.flow_diagnosis(stats,start+10.5*86400)['cause']=='likely_empty'
    # Recent measured water contradicts elapsed-time inference.
    d.observe_water(1500,start+10.5*86400,0)
    assert d.flow_diagnosis(stats,start+10.5*86400+60)['cause']=='check_fault'
    # Changing the fill volume scales the time estimate.
    d.refill(1000,start+12*86400)
    assert d.flow_diagnosis(stats,start+14.5*86400)['expected_cycle_days']==2.5


def test_partial_cycle_is_not_empty_reference(tmp_path):
    from catcam.stats import StatsStore
    d=DispenserStore(tmp_path/'s.db')
    stats=StatsStore(tmp_path/'s.db')
    d.refill(2000,1000)
    d.finish_cycle(1000+86400,1000)
    d.refill(2000,1000+2*86400)
    assert d.flow_diagnosis(stats,1000+3*86400)['empty_cycles']==0


def test_partial_additions_keep_cycle_and_account_for_consumption(tmp_path):
    d=DispenserStore(tmp_path/'s.db')
    d.add_water(950,1000,initial_remaining=0)
    d.add_water(1050,1100,estimated_remaining=900)
    state=d.get_state()
    assert state['last_refill_ts']==1000
    assert state['last_refill_ml']==2000
    assert state['reading']['ml']==1950
    assert len(state['additions'])==2
    d.finish_cycle(1200,200,stopped_confirmed=True)
    assert d.get_state()['options']['stop_residual_ml']==200
    d.add_water(950,1300)
    assert d.get_state()['last_refill_ml']==1150
    assert d.get_state()['last_refill_ts']==1300


def test_unmeasured_initial_residual_and_no_default_warning(tmp_path):
    from catcam.stats import StatsStore
    d=DispenserStore(tmp_path/'s.db')
    stats=StatsStore(tmp_path/'s.db')
    d.add_water(950,1000)
    assert not d.get_state()['options']['baseline_known']
    calibration=d.calibration(stats)
    assert d.reminder_status(calibration,10)['low_ml'] is None
    assert not d.reminder_status(calibration,10)['need_water']
    d.record_signal('low',100,1100)
    assert d.get_state()['options']['reminder_low_ml']==100
    assert not d.reminder_status(calibration,50)['need_water']
    ready={'cycles':[{'used':True}]}
    assert d.reminder_status(ready,50)['need_water']
    with pytest.raises(ValueError):d.record_signal('stopped',None,1200)


def test_working_water_threshold_may_leave_ten_percent(tmp_path):
    from catcam.stats import StatsStore
    d=DispenserStore(tmp_path/'s.db')
    stats=StatsStore(tmp_path/'s.db')
    d.refill(2000,1000)
    d.finish_cycle(1000+86400,200,stopped_confirmed=True)
    d.refill(2000,1000+2*86400)
    diagnosis=d.flow_diagnosis(stats,1000+3*86400)
    assert diagnosis['empty_cycles']==1
    assert diagnosis['cause']=='likely_empty'
