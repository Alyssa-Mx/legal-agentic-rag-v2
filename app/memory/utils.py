from typing import List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def format_messages(messages: List[BaseMessage]) -> str:
    """将消息列表格式化为纯文本（用于摘要 / 门控输入）。"""
    parts: List[str] = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"用户: {m.content}")
        elif isinstance(m, AIMessage) and (m.content or "").strip():
            if not (getattr(m, "tool_calls", None) or []):
                parts.append(f"助手: {m.content}")
    return "\n".join(parts)
