from __future__ import annotations

import json
import logging
from typing import List

from langchain_core.messages import BaseMessage

from .schemas import MemoryConfig, MemoryContext
from .short_window import ShortTermMemory
from .mid_summarizer import MidTermMemory
from .long_write_gate import LongTermMemory

logger = logging.getLogger(__name__)


# ======================================================
# 4. 记忆系统的统一入口
# ======================================================
class MemoryManager:
    """
    记忆系统的统一门面。

    将三层记忆的实现细节完全封装在各自的子模块中：
        short_window.py   → ShortTermMemory   （滑动窗口）
        mid_summarizer.py → MidTermMemory     （会话摘要）
        long_write_gate.py → LongTermMemory   （向量存储 + 异步写入）

    主流程只需调用三个方法：
        prepare_context()   — 轮前：组装记忆上下文
        post_turn_update()  — 轮后：触发摘要更新与异步长期写入
        get_query_context() — 为 query rewrite 提供记忆背景
    """

    def __init__(
        self, llm, embeddings, config: MemoryConfig = None, session_id: str = "default"
    ) -> None:
        self.config = config or MemoryConfig()
        self._session_id = session_id
        self._short = ShortTermMemory(self.config.SLIDING_WINDOW_SIZE)
        self._mid = MidTermMemory(llm, self.config)
        self._long = LongTermMemory(llm, embeddings, self.config, session_id=session_id)
        self._turn: int = 0

    # ── 属性 ──────────────────────────────────────────────────────────────

    @property
    def conversation_summary(self) -> str:
        return self._mid.summary

    @property
    def turn_count(self) -> int:
        return self._turn

    def restore_summary(self, summary: str) -> None:
        """从外部权威源（state / checkpoint）回填中期摘要。"""
        self._mid.restore(summary)

    # ── 轮前：准备记忆上下文 ────────────────────────────────────────────

    def prepare_context(
        self, all_messages: List[BaseMessage], current_query: str
    ) -> MemoryContext:
        """
        组装本轮的记忆上下文：
        1. 对消息应用滑动窗口
        2. 从长期记忆中语义检索相关片段
        3. 汇总用户画像（结构化存储）
        """
        windowed = self._short.window(all_messages)

        mem_docs = self._long.search(current_query, top_k=self.config.MAX_RETRIEVED_MEMORIES)
        mem_texts = [doc.page_content for doc in mem_docs]

        self._long.refresh_profile()

        return MemoryContext(
            windowed_messages=windowed,
            conversation_summary=self._mid.summary,
            retrieved_memories=mem_texts,
            user_profile=self._long.get_profile(),
        )

    # ── 轮后：同步更新摘要 + 异步写长期记忆 ────────────────────────────

    def post_turn_update(self, recent_messages: List[BaseMessage]) -> None:
        """
        每轮结束后调用：
        1. 递增轮次计数器
        2. 按间隔同步更新会话摘要（中期记忆）
        3. 按间隔异步触发长期记忆写入
        """
        self._turn += 1
        self._mid.maybe_update(recent_messages, self._turn)
        self._long.maybe_enqueue_write(recent_messages, self._turn)

    # ── 为 query rewrite 提供记忆背景 ──────────────────────────────────

    def get_query_context(self, query: str) -> str:
        """返回与当前 query 相关的记忆背景（摘要 + 历史记忆 + 用户画像）。"""
        parts: List[str] = []
        if self._mid.summary:
            parts.append(f"【会话摘要】\n{self._mid.summary}")
        docs = self._long.search(query, top_k=3)
        if docs:
            lines = "\n".join(f"- {d.page_content}" for d in docs)
            parts.append(f"【相关历史记忆】\n{lines}")
        profile = self._long.get_profile()
        if profile:
            lines = "\n".join(f"- {v}" for v in list(profile.values())[:5])
            parts.append(f"【用户信息】\n{lines}")
        return "\n\n".join(parts) if parts else ""

    # ── 格式化 MemoryContext 为 prompt 文本 ────────────────────────────

    def format_context(self, ctx: MemoryContext) -> str:
        """将 MemoryContext 对象格式化为可插入 prompt 的结构化文本。"""
        parts: List[str] = []
        if ctx.conversation_summary:
            parts.append(f"【会话摘要】\n{ctx.conversation_summary}")
        if ctx.retrieved_memories:
            lines = "\n".join(f"- {m}" for m in ctx.retrieved_memories)
            parts.append(f"【历史记忆】\n{lines}")
        if ctx.user_profile:
            lines = "\n".join(f"- {v}" for v in list(ctx.user_profile.values())[:5])
            parts.append(f"【用户画像（个人偏好优先参考）】\n{lines}")
        if ctx.task_state:
            parts.append(f"【任务状态】\n{json.dumps(ctx.task_state, ensure_ascii=False)}")
        return "\n\n".join(parts) if parts else ""
