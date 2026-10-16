"""单个事件流的变更指纹；数据库未变化时不重复查询，不将监控采样当业务变更。"""
import hashlib
import json
import sqlite3
from pathlib import Path

TABLES=('events','labels','dispenser','water_readings','water_calibration','water_cycles',
        'water_additions','dispenser_options','water_signals')


class DataRevision:
    def __init__(self,paths):
        self.paths=set(map(Path,paths))
        self.cache={}

    def get(self):
        revisions=[]
        for path in sorted(self.paths):
            stamps=[]
            for file in (path,Path(str(path)+'-wal')):
                try:
                    stat=file.stat();stamps.append((stat.st_mtime_ns,stat.st_size))
                except FileNotFoundError:stamps.append(None)
            stamp=tuple(stamps)
            if path not in self.cache or self.cache[path][0]!=stamp:
                digest=hashlib.sha256()
                with sqlite3.connect(path) as conn:
                    tables={r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    for name in TABLES:
                        if name not in tables:continue
                        digest.update(name.encode())
                        for row in conn.execute(f'SELECT * FROM {name} ORDER BY rowid'):
                            digest.update(json.dumps(row,ensure_ascii=False,default=str).encode())
                self.cache[path]=(stamp,digest.hexdigest())
            revisions.append(self.cache[path][1])
        return tuple(revisions)
