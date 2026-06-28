import json
import logging
import re
from typing import Dict, List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from .schemas import MemoryConfig, MemoryEntry
from .utils import format_messages
from .long_store import LongTermMemoryStore

logger = logging.getLogger(__name__)


# ======================================================
# 3. 长期记忆 - 写入门控：LLM 根据对话片段判断哪些信息值得长期记忆
# ======================================================
class MemoryWriteGate:
    """
    LLM 驱动的记忆写入门控。

    从对话片段中自动判断哪些信息值得长期记忆，过滤寒暄/临时内容，
    输出带置信度的 MemoryEntry 列表。
    """

    _PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个记忆筛选专家。分析下面的对话片段，判断是否包含值得长期记住的信息。\n\n"
         "【应该记住的】\n"
         "- 用户身份信息（姓名、专业、角色、所在地）\n"
         "- 稳定偏好（语言风格、格式要求、回答长度偏好）\n"
         "- 任务目标和关键约束\n"
         "- 重要结论和已确认的事实\n"
         "- 用户明确要求记住的内容\n\n"
         "【不应该记住的】\n"
         "- 寒暄和闲聊\n"
         "- 一次性上下文（如“帮我找一下这个”）\n"
         "- 临时性讨论\n"
         "- 已过时的信息\n\n"
         "输出 JSON 数组，每个元素格式：\n"
         "{{\"content\": \"...\", \"type\": \"profile|task|preference|episodic|semantic\", "
         "\"confidence\": 0.0~1.0}}\n"
         "没有值得记住的内容则输出 []。只输出 JSON，不要输出其他内容。"),
        ("human", "对话片段：\n{conversation}\n\n请分析并输出记忆列表："),
    ])

    def __init__(self, llm) -> None:
        self._llm = llm

    def extract(self, messages: List[BaseMessage]) -> List[MemoryEntry]:
        """从对话消息中提取值得长期记忆的信息。"""
        text = format_messages(messages)
        if not text.strip():
            return []
        try:
            resp = self._llm.invoke(self._PROMPT.format_messages(conversation=text))
            raw = (getattr(resp, "content", "") or "").strip()
            entries = self._parse(raw)
            if entries:
                logger.debug(
                    "[WriteGate] 提取到 %d 条记忆候选: %s",
                    len(entries), [e.memory_type for e in entries],
                )
            return entries
        except Exception as e:
            logger.warning("[WriteGate] 记忆提取失败: %s", e)
            return []

    def extract_from_text(self, conversation_text: str) -> List[MemoryEntry]:
        """直接接受格式化文本，供异步任务调用。"""
        if not conversation_text.strip():
            return []
        try:
            resp = self._llm.invoke(self._PROMPT.format_messages(conversation=conversation_text))
            raw = (getattr(resp, "content", "") or "").strip()
            return self._parse(raw)
        except Exception as e:
            logger.warning("[WriteGate] 记忆提取失败（文本模式）: %s", e)
            return []

    @staticmethod
    def _parse(raw: str) -> List[MemoryEntry]:
        raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
        try:
            arr = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
            if not m:
                return []
            try:
                arr = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []

        if not isinstance(arr, list):
            return []

        entries: List[MemoryEntry] = []
        for item in arr:
            if isinstance(item, dict) and "content" in item:
                entries.append(MemoryEntry(
                    content=item["content"],
                    memory_type=item.get("type", "semantic"),
                    confidence=float(item.get("confidence", 0.8)),
                ))
        return entries


# ── 长期记忆编排层（原散落在 manager.py 的写入逻辑）─────────────────────────

class LongTermMemory:
    """
    长期记忆：写入编排 + 存储访问。

    封装 MemoryWriteGate + LongTermMemoryStore，
    负责按间隔触发写入、维护用户画像缓存、对接异步任务队列。
    """

    def __init__(self, llm, embeddings, config: MemoryConfig, session_id: str = "default") -> None:
        self._store = LongTermMemoryStore(embeddings, config, session_id=session_id)
        self._gate = MemoryWriteGate(llm)
        self._profile: Dict[str, str] = {}
        self._write_interval: int = config.WRITE_GATE_INTERVAL
        self._register_async_handler()

    # ── 对外接口 ──────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5):
        """语义检索相关历史记忆。"""
        return self._store.search(query, top_k)

    def refresh_profile(self) -> None:
        """从 store 刷新用户画像缓存（profile + preference）。"""
        for e in self._store.get_structured("profile"):
            self._profile[e.content] = e.content
        for e in self._store.get_structured("preference"):
            self._profile[e.content] = e.content

    def get_profile(self) -> Dict[str, str]:
        return dict(self._profile)

    def get_structured(self, memory_type: str = None):
        """获取当前 session 的结构化记忆条目（profile / preference 等）。"""
        return self._store.get_structured(memory_type)

    def maybe_enqueue_write(
        self, messages: List[BaseMessage], current_turn: int
    ) -> None:
        """按间隔触发长期记忆写入（异步入队）。"""
        if current_turn % self._write_interval == 0:
            conversation_text = format_messages(messages)
            self._enqueue_write(conversation_text, current_turn)

    # ── 异步写入 ──────────────────────────────────────────────────────────

    def _register_async_handler(self) -> None:
        try:
            from app.core.async_worker import async_worker
            async_worker.register("write_long_term_memory", self._async_write_handler)
            logger.debug("[LongTerm] 异步写入处理器注册成功")
        except Exception as e:
            logger.warning("[LongTerm] 异步处理器注册失败，将同步写入: %s", e)

    def _enqueue_write(self, conversation_text: str, turn: int) -> None:
        if not conversation_text.strip():
            return
        try:
            from app.core.async_worker import async_worker
            async_worker.enqueue(
                "write_long_term_memory",
                {"conversation_text": conversation_text, "turn": turn},
            )
            logger.debug("[LongTerm] 长期记忆写入任务已入队（第 %d 轮）", turn)
        except Exception as e:
            logger.warning("[LongTerm] 入队失败，降级为同步写入: %s", e)
            self._async_write_handler({"conversation_text": conversation_text, "turn": turn})

    def _async_write_handler(self, payload: dict) -> None:
        """后台任务处理器：从对话文本提取并写入长期记忆。"""
        text = payload.get("conversation_text", "")
        turn = payload.get("turn", 0)
        entries = self._gate.extract_from_text(text)
        for entry in entries:
            entry.source_turn = turn
            self._store.add(entry)
        if entries:
            types = [e.memory_type for e in entries]
            logger.info("[LongTerm] 异步写入 %d 条长期记忆: %s", len(entries), types)
