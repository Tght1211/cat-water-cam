from datetime import datetime, timedelta
import numpy as np

from catcam.health import MonitoringStore, health_summary
from catcam.stats import StatsStore
from catcam.nightvision import assess_quality


def test_dark_frame_is_not_valid_monitoring():
    result = assess_quality(np.zeros((32,32,3),np.uint8))
    assert not result['usable']
    assert result['status'] == 'unavailable'


def test_monitoring_does_not_fill_camera_outages(tmp_path):
    m = MonitoringStore(tmp_path/'db')
    q = {'usable':True, 'message':'可用', 'status':'normal'}
    m.observe(100,q); m.observe(130,q); m.observe(400,q)
    assert m.coverage(0,500) == 30
    assert m.status(500)['status'] == 'offline'
    m.observe(430,{'usable':False,'status':'unavailable'})
    assert m.coverage(0,500) == 30


def test_no_record_does_not_imply_no_drinking(tmp_path):
    s = StatsStore(tmp_path/'db')
    m = MonitoringStore(tmp_path/'db')
    h = health_summary(s,m,datetime(2026,10,2,12))
    assert h['status'] == 'insufficient_data'
    assert h['baseline_daily_count'] is None
    assert h['coverage_hours'] == 0


def test_health_compares_same_time_and_requires_coverage(tmp_path):
    import sqlite3
    s = StatsStore(tmp_path/'db'); m = MonitoringStore(tmp_path/'db')
    now = datetime(2026,10,2,12)
    with sqlite3.connect(s.db_path) as c:
        for day in range(1,4):
            start = (now-timedelta(days=day)).replace(hour=0)
            for minute in range(1,1441):
                c.execute('INSERT INTO monitoring VALUES (?,?,1)',(start.timestamp()+minute*60,60))
            for hour in (8,10,16,20):
                ts=start.replace(hour=hour).timestamp(); name=f'{day}_{hour}.mp4'
                c.execute('INSERT INTO events (ts,clip_name) VALUES (?,?)',(ts,name))
                c.execute('INSERT INTO labels VALUES (?,1,?)',(name,ts))
        today=now.replace(hour=0)
        for minute in range(1,721):
            c.execute('INSERT INTO monitoring VALUES (?,?,1)',(today.timestamp()+minute*60,60))
    h=health_summary(s,m,now)
    assert h['baseline_daily_count'] == 4
    assert h['expected_by_now'] == 2
    assert h['status'] == 'decreased'
