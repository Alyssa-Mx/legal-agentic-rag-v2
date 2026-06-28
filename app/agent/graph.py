import logging
from typing import List, Sequence

from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.config.setting import DEFAULT_CHAT_MODEL
from app.memory.manager import MemoryManager
from app.models.dashscope_chat import build_chat

from .state import AgenticRAGState
from .nodes import AgentNodes

logger = logging.getLogger(__name__)


# ======================================================
# 1. 骨架层 - 系统的图结构（ReAct 极简版，多工具）
# ======================================================
def make_graph(
    tools: Sequence[BaseTool],
    memory_manager: MemoryManager = None,
    max_search_steps: int = 8,
    **legacy_kwargs,
):
    """
    构建 ReAct 风格的 Agentic RAG LangGraph。

    Args:
        tools             — Agent 可调用的全部工具列表（如 vector_search /
                            bm25_search / lookup_article / recall_memory /
                            list_user_profile / web_search）
        memory_manager    — 可选的记忆管理器
        max_search_steps  — 单轮对话内的工具调用上限
        **legacy_kwargs   — 兼容旧调用方（如 grader_mode），会被忽略并 warning

    流程：
        prepare_context   — 记忆摘要等强上下文准备 + 重置 search_count
              ↓
        ┌──> agent       — ReAct 主节点（思考 + 行动；输出回答或工具调用）
        │     ↓ (route_after_agent)
        │     ├→ tools   — ToolNode 自动并行执行所有 tool_calls
        │     │   ↓
        │     └─ collect_evidence — 累加所有 ToolMessage 内容，search_count += N
        │         ↓ (回到 agent)
        │
        └─→ self_check_repair  → update_memory → END
    """
    if legacy_kwargs:
        ignored = ", ".join(legacy_kwargs.keys())
        logger.warning(
            "[Graph] 已忽略历史参数: %s（ReAct 架构下不再使用前置控制节点）",
            ignored,
        )

    if not tools:
        raise ValueError("make_graph 需要至少一个工具")

    tools_list: List[BaseTool] = list(tools)

    llm = build_chat(DEFAULT_CHAT_MODEL)
    llm_with_tools = llm.bind_tools(tools_list)
    tool_node = ToolNode(tools_list)

    nodes = AgentNodes(
        llm=llm,
        llm_with_tools=llm_with_tools,
        tools=tools_list,
        memory_manager=memory_manager,
        max_search_steps=max_search_steps,
    )

    graph = StateGraph(AgenticRAGState)

    graph.add_node("prepare_context", nodes.prepare_context)
    graph.add_node("agent", nodes.agent)
    graph.add_node("tools", tool_node)
    graph.add_node("collect_evidence", nodes.collect_evidence)
    graph.add_node("self_check_repair", nodes.self_check_repair)
    graph.add_node("update_memory", nodes.update_memory)

    graph.add_edge(START, "prepare_context")
    graph.add_edge("prepare_context", "agent")

    graph.add_conditional_edges(
        "agent",
        nodes.route_after_agent,
        {
            "tools": "tools",
            "self_check_repair": "self_check_repair",
        },
    )

    graph.add_edge("tools", "collect_evidence")
    graph.add_edge("collect_evidence", "agent")
    graph.add_edge("self_check_repair", "update_memory")
    graph.add_edge("update_memory", END)

    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    tool_names = ", ".join(t.name for t in tools_list)
    logger.info(
        "[Graph] ReAct Agentic RAG 图编译完成（工具: [%s]，最大调用次数: %d）",
        tool_names, max_search_steps,
    )
    return compiled
