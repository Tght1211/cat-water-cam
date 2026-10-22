"""先本地筛选再排队标注；仅成功/确定的外部标注进入训练，不覆盖人工。"""
from catcam.database import connect
import threading
import time
from pathlib import Path


class AIQueue:
    def __init__(self,labeler,stats):
        self.labeler,self.stats=labeler,stats
        self.db_path=stats.db_path
        self._wake=threading.Event()
        self._closed=threading.Event()
        with connect(self.db_path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS ai_queue (clip TEXT PRIMARY KEY, path TEXT NOT NULL, priority REAL NOT NULL, status TEXT NOT NULL DEFAULT 'waiting', attempts INTEGER NOT NULL DEFAULT 0, detail TEXT NOT NULL DEFAULT '')")
        threading.Thread(target=self._run,daemon=True).start()

    def enqueue(self,path,priority=1):
        path=Path(path).resolve()
        with connect(self.db_path) as c:
            c.execute('INSERT OR IGNORE INTO ai_queue(clip,path,priority) VALUES(?,?,?)',(path.name,str(path),priority))
        self._wake.set()

    def status(self):
        with connect(self.db_path) as c:
            counts=dict(c.execute('SELECT status,COUNT(*) FROM ai_queue GROUP BY status').fetchall())
            last=c.execute("SELECT detail FROM ai_queue WHERE detail!='' ORDER BY rowid DESC LIMIT 1").fetchone()
        return {'enabled':True,'queue':counts,'budget':self.labeler.budget.status() if self.labeler.budget else None,
                'models':self.labeler.models,'last_detail':last[0] if last else ''}

    def process_one(self):
        if self.labeler.budget:
            budget=self.labeler.budget
            if budget.status()['remaining']==0:
                return False
            with connect(self.db_path) as c:
                last=c.execute('SELECT MAX(ts) FROM ai_requests').fetchone()[0]
            if last is not None and time.time()-last<budget.interval:
                return False
        with connect(self.db_path) as c:
            row=c.execute("SELECT clip,path,attempts FROM ai_queue WHERE status='waiting' ORDER BY priority DESC,rowid LIMIT 1").fetchone()
        if not row:
            return False
        clip,path,attempts=row
        try:
            result=self.labeler.label(path)
            status='labeled' if result else 'review'
            detail='已生成辅助标注' if result else '无可用标注，留待人工复核'
        except Exception as e:
            detail=f'辅助标注未完成：{type(e).__name__}: {e}'
            retryable=getattr(e,'status_code',None) in (429,500,502,503,504) or any(t in str(e) for t in ('provider_unavailable','冷却','429'))
            status='waiting' if retryable and attempts<1 else 'review'
        with connect(self.db_path) as c:
            c.execute('UPDATE ai_queue SET status=?,attempts=attempts+1,detail=? WHERE clip=?',(status,detail,clip))
        return True

    def _run(self):
        while not self._closed.is_set():
            try:
                self.process_one()
            except Exception:
                pass  # 暂时DB冲突不影响采集；下轮重试。
            self._wake.wait(2)
            self._wake.clear()

    def close(self):
        self._closed.set()
        self._wake.set()
