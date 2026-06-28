import logging
from typing import List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from .schemas import MemoryConfig
from .utils import format_messages  # noqa: F401 — 保留导出供外部兼容

logger = logging.getLogger(__name__)


# ======================================================
# 2. 中期记忆 - 摘要
# ======================================================
class ConversationSummarizer:
    """利用 LLM 将对话历史增量压缩为摘要，保留关键语义。"""

    _PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个对话摘要专家。请将已有摘要和新增对话合并，"
         "生成一段简洁的摘要（不超过 300 字）。\n"
         "重点保留：用户身份/偏好、任务目标、关键结论、重要约束。\n"
         "忽略：寒暄、重复内容、一次性上下文。\n"
         "只输出摘要文本，不要加标题或前缀。"),
        ("human",
         "已有摘要：\n{existing_summary}\n\n"
         "新增对话：\n{new_messages}\n\n"
         "请输出更新后的摘要："),
    ])

    def __init__(self, llm) -> None:
        self._llm = llm

    def summarize(
        self, messages: List[BaseMessage], existing_summary: str = ""
    ) -> str:
        text = format_messages(messages)
        if not text.strip():
            return existing_summary
        try:
            resp = self._llm.invoke(
                self._PROMPT.format_messages(
                    existing_summary=existing_summary or "（无）",
                    new_messages=text,
                )
            )
            result = (getattr(resp, "content", "") or "").strip()
            logger.debug("[Summarizer] 摘要更新完成，长度 %d 字", len(result))
            return result
        except Exception as e:
            logger.warning("[Summarizer] 摘要生成失败，保留旧摘要: %s", e)
            return existing_summary


# ── 中期记忆（摘要触发控制 + 状态维护）──────────────────────────────────────

class MidTermMemory:
    """
    中期记忆：会话摘要。

    封装 ConversationSummarizer 及其触发间隔控制。
    每隔 SUMMARY_INTERVAL 轮自动压缩一次对话历史，防止上下文过长。
    """

    def __init__(self, llm, config: MemoryConfig) -> None:
        self._summarizer = ConversationSummarizer(llm)
        self._summary: str = ""
        self._last_trigger_turn: int = 0
        self._interval: int = config.SUMMARY_INTERVAL

    @property
    def summary(self) -> str:
        return self._summary

    def restore(self, summary: str) -> None:
        """
        从外部（如 LangGraph state / checkpoint）回填摘要。

        用于进程重启或会话恢复后，把"权威源"里的摘要灌回内存缓存，
        避免后续 maybe_update 把空摘要当成已有摘要去合并。
        """
        if summary and not self._summary:
            self._summary = summary
            logger.debug("[MidTerm] 摘要已从外部回填（长度 %d 字）", len(summary))

    def maybe_update(self, messages: List[BaseMessage], current_turn: int) -> bool:
        """
        按间隔触发摘要更新。

        Returns:
            True 表示本次实际更新了摘要，False 表示未到间隔。
        """
        if (current_turn - self._last_trigger_turn) >= self._interval:
            self._summary = self._summarizer.summarize(messages, self._summary)
            self._last_trigger_turn = current_turn
            logger.info("[MidTerm] 会话摘要已更新（第 %d 轮）", current_turn)
            return True
        return False
