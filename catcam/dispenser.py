"""饮水机状态：滤芯周期、蓄水记录、喝水量(ml)自校准、剩余水量反推。

不测体积——靠「一整个蓄水周期的总量 ÷ 期间喝水次数」池化反推每次约几 ml，越用越准。
`DispenserStore` 只管自身状态与校准；「期间喝了几次」由 web 层用 StatsStore 算好传进来（单一职责）。
"""
from __future__ import annotations

import sqlite3

from catcam.database import connect
import math
import statistics
import json
from pathlib import Path


def estimate_remaining(refill_ml: float, drinks_since: int, ml_per_drink: float) -> float:
    """剩余水量 = 上次加的量 − 之后喝水次数×每次 ml，夹到 0（不给负数）。"""
    return max(0.0, refill_ml - drinks_since * ml_per_drink)


def filter_days_left(last_change_ts: float, cycle_days: float, now: float) -> float:
    """滤芯还剩几天 = 周期 − 距上次更换的天数（可为负=已超期）。"""
    return cycle_days - (now - last_change_ts) / 86400.0


class DispenserStore:
    def __init__(self, db_path, default_ml_per_drink: float = 30.0, default_cycle_days: int = 30):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_ml_per_drink = default_ml_per_drink
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS dispenser ("
                "id INTEGER PRIMARY KEY CHECK (id = 1), "
                "last_filter_change_ts REAL NOT NULL DEFAULT 0, "
                "filter_cycle_days INTEGER NOT NULL DEFAULT 30, "
                "last_refill_ts REAL NOT NULL DEFAULT 0, "
                "last_refill_ml REAL NOT NULL DEFAULT 0, "
                "calib_sum_ml REAL NOT NULL DEFAULT 0, "
                "calib_drinks INTEGER NOT NULL DEFAULT 0)"
            )
            conn.execute('CREATE TABLE IF NOT EXISTS water_readings (id INTEGER PRIMARY KEY, ts REAL NOT NULL, ml REAL NOT NULL, kind TEXT NOT NULL)')
            # v2 不再将整箱水量当饮水量；旧校准池无法证明有效，保留原表但不再使用。
            conn.execute('CREATE TABLE IF NOT EXISTS water_calibration (id INTEGER PRIMARY KEY, ml REAL NOT NULL, drinks INTEGER NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS water_cycles (id INTEGER PRIMARY KEY, start REAL NOT NULL, end REAL NOT NULL, consumed_ml REAL NOT NULL, other_loss_ml REAL NOT NULL DEFAULT 0)')
            columns = {row[1] for row in conn.execute('PRAGMA table_info(water_cycles)')}
            for name in ('remaining_ml', 'refill_ml'):
                if name not in columns:
                    conn.execute(f'ALTER TABLE water_cycles ADD COLUMN {name} REAL')
            for name, definition in (('stopped_confirmed','INTEGER NOT NULL DEFAULT 0'),('baseline_known','INTEGER NOT NULL DEFAULT 1')):
                if name not in columns:
                    conn.execute(f'ALTER TABLE water_cycles ADD COLUMN {name} {definition}')
            conn.execute('CREATE TABLE IF NOT EXISTS water_additions (id INTEGER PRIMARY KEY, ts REAL NOT NULL, ml REAL NOT NULL, cycle_start REAL NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS dispenser_options (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            conn.execute('CREATE TABLE IF NOT EXISTS water_signals (id INTEGER PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, remaining_ml REAL, cycle_start REAL NOT NULL)')
            # 单行状态：首次插入默认值（周期用配置默认）
            conn.execute(
                "INSERT OR IGNORE INTO dispenser (id, filter_cycle_days) VALUES (1, ?)",
                (default_cycle_days,),
            )
            if not conn.execute('SELECT 1 FROM water_readings LIMIT 1').fetchone():
                row = self._row(conn)
                if row['last_refill_ts'] > 0 and row['last_refill_ml'] > 0:
                    conn.execute("INSERT INTO water_readings(ts,ml,kind) VALUES(?,?,'refill')",
                                 (row['last_refill_ts'], row['last_refill_ml']))

    def _conn(self) -> sqlite3.Connection:
        return connect(self.db_path)

    def _row(self, conn) -> dict:
        cur = conn.execute(
            "SELECT last_filter_change_ts, filter_cycle_days, last_refill_ts, "
            "last_refill_ml, calib_sum_ml, calib_drinks FROM dispenser WHERE id = 1"
        )
        c = cur.fetchone()
        return {
            "last_filter_change_ts": c[0], "filter_cycle_days": c[1],
            "last_refill_ts": c[2], "last_refill_ml": c[3],
            "calib_sum_ml": c[4], "calib_drinks": c[5],
        }

    def ml_per_drink(self, row: dict) -> float:
        """池化估计：累计总量 / 累计喝水次数；没校准过就用默认。"""
        if row["calib_drinks"] > 0:
            return row["calib_sum_ml"] / row["calib_drinks"]
        return self.default_ml_per_drink

    def get_state(self) -> dict:
        with self._conn() as conn:
            row = self._row(conn)
        with self._conn() as conn:
            ml, drinks = conn.execute('SELECT COALESCE(SUM(ml),0), COALESCE(SUM(drinks),0) FROM water_calibration').fetchone()
            reading = conn.execute('SELECT ts, ml, kind FROM water_readings ORDER BY id DESC LIMIT 1').fetchone()
        row['calib_sum_ml'], row['calib_drinks'] = ml, drinks
        row['ml_per_drink'] = self.ml_per_drink(row)
        row['reading'] = {'ts': reading[0], 'ml': reading[1], 'kind': reading[2]} if reading else None
        with self._conn() as c:
            options = {key:json.loads(value) for key,value in c.execute('SELECT key,value FROM dispenser_options')}
            additions = [{'ts':ts,'ml':ml,'cycle_start':cycle} for ts,ml,cycle in c.execute('SELECT ts,ml,cycle_start FROM water_additions ORDER BY id DESC LIMIT 30')]
            closed = bool(c.execute('SELECT 1 FROM water_cycles WHERE start=?',(row['last_refill_ts'],)).fetchone())
            readings = [{'ts':ts,'ml':ml,'kind':kind} for ts,ml,kind in c.execute('SELECT ts,ml,kind FROM water_readings ORDER BY id DESC LIMIT 30')]
            signals = [{'ts':ts,'remaining_ml':ml,'kind':kind} for ts,ml,kind in c.execute('SELECT ts,remaining_ml,kind FROM water_signals ORDER BY id DESC LIMIT 30')]
        row.update(options=options,additions=additions,cycle_closed=closed,readings=readings,signals=signals)
        return row

    def refill(self, ml: float, now: float, prev_drinks: int = 0) -> None:
        """记录加水后的总水量；补水不意味着上一箱已喝完，不自动校准。"""
        if not math.isfinite(ml) or not 0 < ml <= 20000:
            raise ValueError('水量必须在0到20000毫升之间')
        with self._conn() as conn:
            conn.execute('UPDATE dispenser SET last_refill_ts=?, last_refill_ml=? WHERE id=1', (now, ml))
            conn.execute("INSERT INTO water_readings(ts,ml,kind) VALUES(?,?,'refill')", (now,ml))
            conn.execute("INSERT OR REPLACE INTO dispenser_options VALUES ('baseline_known','true')")

    def observe_water(self, ml, now, prev_drinks=0, calibrate=False, other_loss_ml=0):
        """记录实测剩余；只有显式确认独立饮水校准时才更新每次摄水量。"""
        if not math.isfinite(ml) or not 0 <= ml <= 20000:
            raise ValueError('剩余水量必须在0到20000毫升之间')
        if not math.isfinite(other_loss_ml) or other_loss_ml < 0:
            raise ValueError('非饮水损耗必须是非负数')
        with self._conn() as conn:
            prior = conn.execute('SELECT ml FROM water_readings ORDER BY id DESC LIMIT 1').fetchone()
            capacity = self._row(conn)['last_refill_ml']
            if not prior:
                raise ValueError('请先记录加水后的总水量')
            if ml > capacity:
                known = conn.execute("SELECT value FROM dispenser_options WHERE key='baseline_known'").fetchone()
                if known is None or json.loads(known[0]):
                    raise ValueError('剩余量超过加水后的总量，请使用加水记录')
                # The first addition may have had an unmeasured initial residual.
                conn.execute('UPDATE dispenser SET last_refill_ml=? WHERE id=1',(ml,))
            consumed = prior[0]-ml-other_loss_ml
            if calibrate:
                if prev_drinks < 1 or consumed <= 0:
                    raise ValueError('校准需要确认喝水次数和正的饮水消耗量')
                conn.execute('INSERT INTO water_calibration(ml,drinks) VALUES(?,?)', (consumed,prev_drinks))
            conn.execute("INSERT INTO water_readings(ts,ml,kind) VALUES(?,?,'measurement')", (now,ml))

    def water_summary(self, now):
        st = self.get_state()
        reading = st['reading']
        if reading is None:
            return {'remaining_ml': None, 'remaining_pct': None, 'water_source': 'unknown',
                    'water_message': '请先记录加水后的总量。', 'water_reading_ts': None,
                    'water_age_hours': None, 'depletion_ml_per_day': None, 'days_until_low': None}
        with self._conn() as c:
            prev = c.execute('SELECT ts,ml FROM water_readings WHERE ts < ? AND ts >= ? ORDER BY id DESC LIMIT 1',
                             (reading['ts'],st['last_refill_ts'])).fetchone()
        age = max(0,now-reading['ts'])/3600
        rate = None
        if prev and reading['ts']-prev[0] >= 3600 and prev[1] > reading['ml']:
            rate = (prev[1]-reading['ml']) / ((reading['ts']-prev[0])/86400)
        return {'remaining_ml': round(reading['ml']),
                'remaining_pct': min(1,reading['ml']/st['last_refill_ml']) if st['last_refill_ml'] else None,
                'water_source': 'measurement', 'water_reading_ts': reading['ts'], 'water_age_hours': round(age,1),
                'water_message': '水位记录已超过24小时，请重新测量。' if age > 24 else '按当前水量估值叠加新增水量；请通过测量校正。' if reading['kind']=='addition_estimate' else '显示最近一次实测水量；蒸发、泼洒和其他宠物也会影响消耗。',
                'depletion_ml_per_day': round(rate,1) if rate else None,
                'days_until_low': round(max(0,reading['ml']-st['last_refill_ml']*.2)/rate,1) if rate else None}

    def mark_filter(self, now: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE dispenser SET last_filter_change_ts = ? WHERE id = 1", (now,))

    def finish_cycle(self, now, remaining_ml=0, other_loss_ml=0, stopped_confirmed=False):
        with self._conn() as c:
            st = self._row(c)
            if st['last_refill_ts'] <= 0:
                raise ValueError('请先记录本轮加水总量')
            if not all(math.isfinite(v) for v in (remaining_ml,other_loss_ml)) or min(remaining_ml,other_loss_ml) < 0:
                raise ValueError('剩余量和其他损耗必须是有效非负数')
            consumed = st['last_refill_ml']-remaining_ml-other_loss_ml
            if consumed <= 0:
                raise ValueError('本轮有效消耗量须大于0')
            if c.execute('SELECT 1 FROM water_cycles WHERE start=?',(st['last_refill_ts'],)).fetchone():
                raise ValueError('这一轮已经反馈过，请先开始新一轮加水')
            known = c.execute("SELECT value FROM dispenser_options WHERE key='baseline_known'").fetchone()
            known = json.loads(known[0]) if known else True
            c.execute('INSERT INTO water_cycles(start,end,consumed_ml,other_loss_ml,remaining_ml,refill_ml,stopped_confirmed,baseline_known) VALUES(?,?,?,?,?,?,?,?)',
                      (st['last_refill_ts'],now,consumed,other_loss_ml,remaining_ml,st['last_refill_ml'],int(stopped_confirmed),int(known)))
            if stopped_confirmed:
                c.execute("INSERT OR REPLACE INTO dispenser_options VALUES ('stop_residual_ml',?)",(json.dumps(remaining_ml),))
                c.execute("INSERT INTO water_signals(ts,kind,remaining_ml,cycle_start) VALUES(?,'stopped',?,?)",(now,remaining_ml,st['last_refill_ts']))
            c.execute("INSERT INTO water_readings(ts,ml,kind) VALUES(?,?,'measurement')",(now,remaining_ml))

    def calibration(self, stats):
        """按最新人工/机器纠错标签重新计数，历史校准随纠错同步更新。"""
        with self._conn() as c:
            cycles = c.execute('SELECT id,start,end,consumed_ml,baseline_known,remaining_ml,stopped_confirmed FROM water_cycles ORDER BY id').fetchall()
        total_ml,total_drinks = 0,0
        latest = None
        report=[]
        with stats._conn() as c:
            cols={r[1] for r in c.execute('PRAGMA table_info(labels)')}
            has_durations=c.execute("SELECT 1 FROM sqlite_master WHERE name='clip_durations'").fetchone()
            for ident,start,end,ml,known,residual,stopped in cycles:
                events=stats.events_between(start,end)
                pending=c.execute('SELECT COUNT(*) FROM events e LEFT JOIN labels l ON e.clip_name=l.clip_name WHERE e.ts>=? AND e.ts<? AND l.clip_name IS NULL',(start,end)).fetchone()[0]
                count=len(events)
                complete=count > 0 and pending==0 and bool(known)
                # 相邻录像虽合并为一次访问，动作秒数仍需包含全部确认片段；同一录像多次事件只计一次。
                if 'drinking_seconds' in cols:
                    seconds_column='l.drinking_seconds'
                else:
                    seconds_column='NULL'
                video_column='d.video_seconds' if has_durations else 'NULL'
                join='LEFT JOIN clip_durations d ON l.clip_name=d.clip_name' if has_durations else ''
                rows=c.execute(f"SELECT {seconds_column},{video_column} FROM labels l JOIN "
                               "(SELECT clip_name,MIN(ts) ts FROM events GROUP BY clip_name) e ON l.clip_name=e.clip_name "
                               f"{join} WHERE l.is_drinking=1 AND e.ts>=? AND e.ts<?",(start,end)).fetchall()
                durations=[actual if actual is not None else round(video*.8,1) if video else None for actual,video in rows]
                missing=sum(value is None or value<=0 for value in durations)
                total_seconds=sum(value for value in durations if value is not None and value>0)
                latest={'id':ident,'start':start,'end':end,'consumed_ml':ml,'drinks':count,
                        'clips':len(rows),'drinking_seconds':total_seconds,'missing_durations':missing,
                        'pending':pending,'ready':complete and bool(rows) and missing==0,
                        'estimated_seconds_used':any(actual is None for actual,video in rows)}
                report.append({'id':ident,'start':start,'end':end,'consumed_ml':ml,'drinks':count,'pending':pending,'used':complete,'baseline_known':bool(known),'remaining_ml':residual,'stopped_confirmed':bool(stopped),
                               'drinking_seconds':total_seconds,'missing_durations':missing,'duration_ready':latest['ready']})
                if complete:
                    total_ml+=ml;total_drinks+=count
        manual=self.get_state()
        total_ml+=manual['calib_sum_ml'];total_drinks+=manual['calib_drinks']
        return {'ml_per_drink':total_ml/total_drinks if total_drinks else self.default_ml_per_drink,
                'ml_per_second':latest['consumed_ml']/latest['drinking_seconds'] if latest and latest['ready'] else None,
                'duration_calibration_estimated':bool(latest and latest['estimated_seconds_used']),
                'duration_calibration_cycle':latest,
                'calibrated':total_drinks > 0,'calibration_drinks':total_drinks,'cycles':report,
                'assumption':'按本轮水量下降估计，未填写的蒸发、泼洒及其他宠物饮水会计入估值。'}

    def set_cycle(self, days: int) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE dispenser SET filter_cycle_days = ? WHERE id = 1", (int(days),))


    def flow_diagnosis(self, stats, now):
        """Heuristic cause hints from explicitly measured empty cycles, never a diagnosis."""
        state = self.get_state()
        start, fill = state['last_refill_ts'], state['last_refill_ml']
        with self._conn() as c:
            rows = c.execute('SELECT start,end,refill_ml FROM water_cycles WHERE remaining_ml IS NOT NULL AND refill_ml>0 AND (stopped_confirmed=1 OR remaining_ml<=MAX(50,refill_ml*.05)) AND baseline_known=1 AND end>start ORDER BY end DESC LIMIT 5').fetchall()
        result = {'cause':'unknown', 'cause_message':'缺水、断电或水泵故障均有可能，请检查设备。',
                  'empty_cycles':len(rows), 'expected_cycle_days':None, 'elapsed_days':None}
        if not start or not fill:
            result['cause_message'] = '尚未记录加水和耗尽周期，原因待确认。'
            return result
        elapsed = max(0, now-start)/86400
        result['elapsed_days'] = round(elapsed,2)
        if not rows:
            result['cause_message'] = '尚未积累确认耗尽的周期，先检查水箱、供电和水泵。'
            return result
        expected = statistics.median((end-begin)/86400/volume*fill for begin,end,volume in rows)
        result['expected_cycle_days'] = round(expected,2)
        if expected <= 0:
            return result
        reading = state['reading']
        if reading and reading['kind']=='measurement' and now-reading['ts'] < 6*3600 and reading['ml'] > fill*.3:
            result.update(cause='check_fault',cause_message='最近实测水位仍较充足，与缺水判断不符。优先检查供电、水泵和堵塞，并再次核实水位。')
            return result
        if elapsed >= .8*expected:
            result.update(cause='likely_empty',cause_message=f'较可能缺水：距加水已 {elapsed:.1f} 天，接近历史耗尽周期约 {expected:.1f} 天。请检查水箱并确认。')
        elif elapsed < .4*expected:
            result.update(cause='check_fault',cause_message=f'本轮加水仅 {elapsed:.1f} 天，明显早于历史耗尽周期约 {expected:.1f} 天。优先检查供电、水泵和堵塞，同时核实水位。')
        else:
            result['cause_message'] = '未到历史常见耗尽时间，原因暂不确定，请先查看水箱和设备。'
        return result


    def add_water(self, ml, now, estimated_remaining=None, initial_remaining=None):
        if not math.isfinite(ml) or not 0 < ml <= 20000:
            raise ValueError('本次加水量必须大于0且不超过20000毫升')
        if initial_remaining is not None and (not math.isfinite(initial_remaining) or not 0<=initial_remaining<=20000):
            raise ValueError('加水前的实测残水量无效')
        with self._conn() as c:
            state = self._row(c)
            prior = c.execute('SELECT ts,ml,kind FROM water_readings ORDER BY id DESC LIMIT 1').fetchone()
            closed = c.execute('SELECT 1 FROM water_cycles WHERE start=?',(state['last_refill_ts'],)).fetchone()
            new_cycle = not state['last_refill_ts'] or bool(closed)
            if new_cycle:
                base = initial_remaining if initial_remaining is not None else (prior[1] if closed and prior else 0)
                known = initial_remaining is not None or bool(closed and prior)
                start = now
                supplied = base+ml
                c.execute("INSERT OR REPLACE INTO dispenser_options VALUES ('baseline_known',?)",(json.dumps(known),))
            else:
                base = initial_remaining if initial_remaining is not None else max(0,estimated_remaining if estimated_remaining is not None else prior[1])
                start = state['last_refill_ts']
                supplied = state['last_refill_ml']+ml
            if base+ml>20000 or supplied>20000:
                raise ValueError('本轮累计水量超过20000毫升，请先反馈周期')
            c.execute('UPDATE dispenser SET last_refill_ts=?,last_refill_ml=? WHERE id=1',(start,supplied))
            c.execute('INSERT INTO water_additions(ts,ml,cycle_start) VALUES(?,?,?)',(now,ml,start))
            c.execute("INSERT INTO water_readings(ts,ml,kind) VALUES(?,?,'addition_estimate')",(now,base+ml))
            c.execute("INSERT OR REPLACE INTO dispenser_options VALUES ('last_add_ts',?)",(json.dumps(now),))

    def configure_reminder(self, body):
        with self._conn() as c:
            for key in ('capacity_ml','reminder_low_ml'):
                if key in body:
                    value = body[key]
                    if value is not None:
                        value = float(value)
                        if not math.isfinite(value) or not 0<value<=20000:
                            raise ValueError('容量和提醒水量应大于0且不超过20000毫升')
                    c.execute('INSERT OR REPLACE INTO dispenser_options VALUES (?,?)',(key,json.dumps(value)))
            for key in ('reminder_enabled','indicator_enabled'):
                if key in body:
                    if not isinstance(body[key],bool):
                        raise ValueError('提醒开关必须是布尔值')
                    c.execute('INSERT OR REPLACE INTO dispenser_options VALUES (?,?)',(key,json.dumps(body[key])))

    def record_signal(self, kind, ml, now):
        if kind not in ('low','stopped'):
            raise ValueError('请选择红灯常亮或闪烁停水')
        if ml is None:
            raise ValueError('请测量并填写当前残水量，不要默认填零')
        ml = float(ml)
        state = self.get_state()
        if not state['last_refill_ts']:
            raise ValueError('请先记录加水')
        if kind=='stopped':
            self.finish_cycle(now,ml,stopped_confirmed=True)
        else:
            self.observe_water(ml,now)
            with self._conn() as c:
                c.execute("INSERT INTO water_signals(ts,kind,remaining_ml,cycle_start) VALUES(?,'low',?,?)",(now,ml,state['last_refill_ts']))
                c.execute("INSERT OR REPLACE INTO dispenser_options VALUES ('reminder_low_ml',?)",(json.dumps(ml),))
                c.execute("INSERT OR REPLACE INTO dispenser_options VALUES ('reminder_enabled','true')")

    def reminder_status(self, calibration, remaining, indicator=None):
        state = self.get_state()
        options = state['options']
        complete = sum(c['used'] for c in calibration['cycles'])
        low = options.get('reminder_low_ml')
        stop = options.get('stop_residual_ml')
        visual = (indicator or {}).get('state') if options.get('indicator_enabled') else None
        low_active = visual=='steady_red' or (complete>=1 and options.get('reminder_enabled') and low is not None and remaining is not None and remaining<=low)
        return {'low_ml':low,'stop_residual_ml':stop,'enabled':bool(options.get('reminder_enabled')),
                'indicator_enabled':bool(options.get('indicator_enabled')),'complete_cycles':complete,
                'ready':complete>=1,'need_water':bool(low_active),'source':'indicator' if visual=='steady_red' else 'estimated',
                'message':'红灯常亮，饮水机提示水少了' if visual=='steady_red' else '已到您校准的低水量，请准备加水' if low_active else
                          '先完成至少一轮加水、喝水确认和残水测量，再启用软件估算提醒' if complete<1 else
                          '尚未设置提醒水位；看到红灯常亮时可测量并校准' if low is None else '按您设置的水位提醒'}
