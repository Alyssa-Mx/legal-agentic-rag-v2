"""
轻量 Artifact 存储：把大块文本（如 KB 证据）存在外部，state 只持引用。

设计目的：
- 避免 LangGraph checkpointer 把每一轮的检索原文都序列化进 state 快照
- 同一段证据按 sha1 去重，跨节点 / 跨轮共享

当前实现是进程内 dict + Lock，后续可以替换为 sqlite / Redis 而不动调用方。
"""

from __future__ import annotations

import hashlib
import threading
from typing import Optional


class ArtifactStore:
    """按 sha1 摘要去重的内存 KV 存储。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: dict[str, str] = {}

    def put(self, payload: str) -> str:
        """写入文本，返回引用 ID。空字符串返回空字符串。"""
        if not payload:
            return ""
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        ref = f"art:sha1:{digest}"
        with self._lock:
            if ref not in self._store:
                self._store[ref] = payload
        return ref

    def get(self, ref: Optional[str]) -> str:
        """按引用 ID 取回文本，找不到时返回空字符串。"""
        if not ref:
            return ""
        with self._lock:
            return self._store.get(ref, "")

    def stats(self) -> dict:
        with self._lock:
            total_bytes = sum(len(v) for v in self._store.values())
        return {"entries": len(self._store), "bytes": total_bytes}

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


artifact_store = ArtifactStore()
