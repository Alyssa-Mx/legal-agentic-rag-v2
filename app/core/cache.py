"""
轻量内存 LRU 缓存（线程安全），用于 embedding / 检索结果等可复用计算。

设计目的：
- 避免 e2e 评测和重复 query 反复调用 DashScope API 浪费时间和配额
- 提供 hit/miss 统计，方便优化时验证命中率

使用：
    from app.core.cache import embed_cache, retrieval_cache

    cached = embed_cache.get(("text-embedding-v3", text))
    if cached is None:
        cached = compute_embedding(...)
        embed_cache.put(("text-embedding-v3", text), cached)
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Hashable, Optional


class LRUCache:
    """线程安全的 LRU 缓存。"""

    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = maxsize
        self._cache: "OrderedDict[Hashable, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# 全局缓存实例（按职责分离，便于分别看命中率）
embed_cache = LRUCache(maxsize=4096)        # key=(model_name, text) -> List[float]
retrieval_cache = LRUCache(maxsize=1024)    # key=(query, top_k) -> List[Document]
