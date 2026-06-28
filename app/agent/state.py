from typing import Any, Dict

from langgraph.graph.message import MessagesState

from app.core.artifact_store import artifact_store


# ======================================================
# 1. 骨架层 - 定义整张图共享的数据容器
# ======================================================
class AgenticRAGState(MessagesState):
    """
    Agentic RAG 图的完整状态（ReAct 极简版）。

    在 LangGraph MessagesState（消息列表）基础上扩展：
        memory_context        — 格式化的记忆上下文（摘要 + 历史记忆 + 用户画像）
        conversation_summary  — 中期记忆（会话摘要原文），由 update_memory 写回，
                                 持久化到 checkpoint，重启后 prepare_context 回填给
                                 MemoryManager，避免摘要丢失
        kb_evidence_ref       — 本轮所有工具调用结果累积后写入 ArtifactStore 的 ref
        search_count          — 本轮已经执行过的工具调用次数（含 retrieve / web_search）
                                 用于限制 ReAct 循环最大步数，避免无限调用
        self_check_result     — 回答后自检结果（used_kb / hallucination_risk 等）
        original_question     — 用户本轮原始问题（用于自检与记忆写入定位）

    设计要点：
        - 不再有 route / rewritten_queries / grade_result / kb_fallback_note /
          kb_evidence_warning / rewrite_count / rewrite_reason 这些"控制位"，
          所有"要不要再查、查什么、用什么工具"全部交由 Agent 在 ReAct 循环中
          通过 messages 自主决策
        - kb_evidence 使用 ArtifactStore 指针化，避免 checkpoint 膨胀
    """

    memory_context: str
    conversation_summary: str
    kb_evidence_ref: str
    search_count: int
    self_check_result: Dict[str, Any]
    original_question: str


# ── Artifact 读写辅助 ────────────────────────────────────────────────────────

def put_kb_evidence(text: str) -> str:
    """把 KB 证据写入 ArtifactStore，返回 ref。空文本返回空串。"""
    return artifact_store.put(text)


def get_kb_evidence(state: dict) -> str:
    """从 state 读取 KB 证据原文（自动 deref）。兼容旧字段 kb_evidence。"""
    ref = state.get("kb_evidence_ref")
    if ref:
        return artifact_store.get(ref)
    return state.get("kb_evidence") or ""
