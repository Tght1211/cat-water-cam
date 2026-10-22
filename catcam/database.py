"""短连接在事务结束后立即关闭，避免后台进程耗尽文件句柄。"""
from __future__ import annotations

import sqlite3


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        # SQLite 默认的上下文管理器只提交/回滚，不会关闭连接。
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect(database) -> sqlite3.Connection:
    return sqlite3.connect(database, factory=_ClosingConnection)
