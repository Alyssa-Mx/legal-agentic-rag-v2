"""
LLM Token 计数器（线程安全 + 支持快照对比）。

设计目的：
- 把 DashScope chat 返回的 usage 收集起来，按"题"或"会话"做成本归因
- 提供 snapshot/diff 接口，方便在 e2e 评测里按问题切片统计

DashScope OpenAI 兼容模式返回的 usage 结构：
    {
        "prompt_tokens": 123,
        "completion_tokens": 456,
        "total_tokens": 579
    }

使用：
    from app.core.token_counter import token_counter

    before = token_counter.snapshot()
    # ... 跑一段图 ...
    delta = token_counter.diff(before)
    print(delta["prompt_tokens"], delta["completion_tokens"], delta["llm_calls"])
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenSnapshot:
    prompt_tokens: int
    completion_tokens: int
    llm_calls: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
        }


class TokenCounter:
    """全局 token 计数器。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt = 0
        self._completion = 0
        self._calls = 0

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        if prompt_tokens <= 0 and completion_tokens <= 0:
            return
        with self._lock:
            self._prompt += max(0, prompt_tokens)
            self._completion += max(0, completion_tokens)
            self._calls += 1

    def snapshot(self) -> TokenSnapshot:
        with self._lock:
            return TokenSnapshot(
                prompt_tokens=self._prompt,
                completion_tokens=self._completion,
                llm_calls=self._calls,
            )

    def diff(self, baseline: TokenSnapshot) -> dict:
        """返回 baseline → 当前 之间的增量，便于按题/按轮统计。"""
        cur = self.snapshot()
        return {
            "prompt_tokens": cur.prompt_tokens - baseline.prompt_tokens,
            "completion_tokens": cur.completion_tokens - baseline.completion_tokens,
            "total_tokens": cur.total_tokens - baseline.total_tokens,
            "llm_calls": cur.llm_calls - baseline.llm_calls,
        }

    def reset(self) -> None:
        with self._lock:
            self._prompt = 0
            self._completion = 0
            self._calls = 0


token_counter = TokenCounter()
