from typing import List

from langchain_core.messages import BaseMessage


# ======================================================
# 1. 短期记忆 - 滑动窗口
# ======================================================
class ShortTermMemory:
    """
    短期记忆：滑动窗口。

    每次只取最近 N 条消息送入 prompt，防止上下文过长。
    消息本体由 LangGraph MemorySaver（checkpointer）负责跨调用持久化，
    本类只负责截取。
    """

    def __init__(self, window_size: int) -> None:
        self._size = window_size

    def window(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """返回最近 window_size 条消息。"""
        return list(messages[-self._size:]) if len(messages) > self._size else list(messages)
