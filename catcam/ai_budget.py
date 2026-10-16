"""跨重启的外部API预算：每次HTTP尝试都占预算；账户在其他程序的调用仍由服务端限流。"""
import sqlite3
import time
from datetime import datetime, timezone


class AIBudget:
    def __init__(self, db_path, daily_limit=40, min_interval=60):
        self.db_path = db_path
        self.limit = max(0,int(daily_limit))
        self.interval = max(0,min_interval)
        with sqlite3.connect(db_path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS ai_requests (id INTEGER PRIMARY KEY, ts REAL NOT NULL, day TEXT NOT NULL, model TEXT NOT NULL, clip TEXT, status TEXT NOT NULL)')

    def reserve(self, model, clip=None, now=None):
        now = time.time() if now is None else now
        day = datetime.fromtimestamp(now,timezone.utc).strftime('%Y-%m-%d')
        with sqlite3.connect(self.db_path) as c:
            c.execute('BEGIN IMMEDIATE')
            count = c.execute('SELECT COUNT(*) FROM ai_requests WHERE day=?',(day,)).fetchone()[0]
            last = c.execute('SELECT MAX(ts) FROM ai_requests').fetchone()[0]
            if count >= self.limit or self.interval > 0 and last is not None and now-last < self.interval:
                return None
            return c.execute("INSERT INTO ai_requests(ts,day,model,clip,status) VALUES(?,?,?,?,'reserved')",(now,day,model,clip)).lastrowid

    def finish(self, request_id, status):
        with sqlite3.connect(self.db_path) as c:
            c.execute('UPDATE ai_requests SET status=? WHERE id=?',(status,request_id))

    def status(self):
        day=datetime.now(timezone.utc).strftime('%Y-%m-%d')
        with sqlite3.connect(self.db_path) as c:
            rows=c.execute('SELECT status,COUNT(*) FROM ai_requests WHERE day=? GROUP BY status',(day,)).fetchall()
        counts=dict(rows);used=sum(counts.values())
        return {'day_utc':day,'limit':self.limit,'used':used,'remaining':max(0,self.limit-used),'requests':counts,
                'note':'计入失败和重试；预算为本项目共用，账户其他程序的调用不在此统计。'}
