import logging
import time
from typing import Any, Dict, List, Literal, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.agent.inspector import AnswerInspector
from app.agent.state import get_kb_evidence, put_kb_evidence
from app.core.trace import add_event
from app.memory.manager import MemoryManager

logger = logging.getLogger(__name__)


# ======================================================
# 2. 节点层 - ReAct 极简版
#    只保留 4 类节点：prepare_context / agent / collect_evidence /
#    self_check_repair / update_memory（外加 LangGraph 自带的 ToolNode）
# ======================================================


# ── 工具函数 ─────────────────────────────────────────────────────────────────

def _last_human_message(state: dict) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


def _last_ai_answer(state: dict) -> str:
    """取最近一条有内容且无工具调用的 AIMessage。"""
    for m in reversed(state["messages"]):
        if (
            isinstance(m, AIMessage)
            and (m.content or "").strip()
            and not (getattr(m, "tool_calls", None) or [])
        ):
            return m.content
    return ""


# ── 主类 ──────────────────────────────────────────────────────────────────────

class AgentNodes:
    """
    所有 LangGraph 节点方法的集合（ReAct 版）。

    设计要点：
        - 不再持有 QueryRouter / QueryRewriter，路由/改写能力内化到 agent prompt
        - agent 节点接管原 decide_or_call_tools + generate_answer 的全部职责
        - 工具调用次数由 search_count + max_search_steps 控制
    """

    def __init__(
        self,
        llm,
        llm_with_tools,
        tools: Sequence[BaseTool],
        memory_manager: MemoryManager,
        max_search_steps: int = 8,
    ) -> None:
        self._llm = llm
        self._llm_with_tools = llm_with_tools
        self._tools = list(tools)
        self._memory_manager = memory_manager
        self._max_search_steps = max_search_steps
        self._window_size = (
            memory_manager.config.SLIDING_WINDOW_SIZE if memory_manager else 50
        )

        self._inspector = AnswerInspector(llm)
        self._tools_section = self._render_tools_section(self._tools)

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    def _windowed(self, messages: list) -> list:
        n = self._window_size
        msgs = messages[-n:] if len(messages) > n else list(messages)
        return self._drop_orphan_tool_messages(msgs)

    @staticmethod
    def _drop_orphan_tool_messages(messages: list) -> list:
        """
        确保 ToolMessage 前面一定有对应的 tool_calls AIMessage。

        窗口截断后，历史头部可能残留没有前驱 tool_calls 的 ToolMessage，
        DashScope API 会直接报 400。从头扫描，跳过这些孤儿消息直到序列合法。
        """
        start = 0
        for i, m in enumerate(messages):
            if isinstance(m, ToolMessage):
                if i == 0 or not getattr(messages[i - 1], "tool_calls", None):
                    start = i + 1
            else:
                break
        return messages[start:]

    def _build_conv_text(self, messages: list) -> str:
        parts: List[str] = []
        for m in messages:
            if isinstance(m, HumanMessage):
                parts.append(f"用户: {m.content}")
            elif (
                isinstance(m, AIMessage)
                and (m.content or "").strip()
                and not (getattr(m, "tool_calls", None) or [])
            ):
                parts.append(f"助手: {m.content}")
        return "\n".join(parts) if parts else "（无历史）"

    # ─────────────────────────────────────────────────────────────────────────
    # 节点 1：prepare_context
    # 职责：仅做记忆检索 + 重置本轮的 search_count / kb_evidence_ref。
    #       不再做路由决策，也不再生成 kb_query/episodic_query。
    # ─────────────────────────────────────────────────────────────────────────

    def prepare_context(self, state: dict) -> dict:
        user_message = _last_human_message(state)
        add_event("turn_start", question=user_message)

        # 每轮新输入都要清空"上一轮残留"的搜索计数和证据指针，
        # 否则 checkpointer 会把跨轮状态带进来污染 ReAct 循环。
        turn_reset = {
            "original_question": user_message,
            "search_count": 0,
            "kb_evidence_ref": "",
        }

        if not self._memory_manager:
            add_event("memory_loaded", ctx_len=0, enabled=False)
            return {**turn_reset, "memory_context": ""}

        # 进程重启后，state 里的 conversation_summary 是权威，回填给内存缓存
        persisted_summary = (state.get("conversation_summary") or "").strip()
        if persisted_summary:
            self._memory_manager.restore_summary(persisted_summary)

        ctx = self._memory_manager.prepare_context(state["messages"], user_message)
        ctx_text = self._memory_manager.format_context(ctx)

        logger.debug("[Nodes] prepare_context: memory_len=%d", len(ctx_text))
        add_event("memory_loaded", ctx_len=len(ctx_text), enabled=True)
        return {**turn_reset, "memory_context": ctx_text}

    # ─────────────────────────────────────────────────────────────────────────
    # 节点 2：agent —— ReAct 主节点
    # 职责：把"工具说明书 + 任务约束 + 记忆背景 + 已收集证据"装入 system prompt，
    #       让模型自主决定：直接回答 / 调 retrieve_docs / 调 web_search。
    # ─────────────────────────────────────────────────────────────────────────

    def agent(self, state: dict) -> dict:
        memory_text: str = state.get("memory_context") or ""
        kb_evidence: str = get_kb_evidence(state)
        search_count: int = int(state.get("search_count") or 0)
        budget_left = max(0, self._max_search_steps - search_count)

        system = SystemMessage(content=self._build_agent_system(
            memory_text=memory_text,
            kb_evidence=kb_evidence,
            search_count=search_count,
            budget_left=budget_left,
        ))
        msgs = [system] + self._windowed(state["messages"])

        t0 = time.time()
        resp = self._llm_with_tools.invoke(msgs)
        llm_elapsed = round((time.time() - t0) * 1000)

        # 记录本次 ReAct 步骤：Reason（content）+ Action（tool_calls）
        reason_text = (resp.content or "").strip() if hasattr(resp, "content") else ""
        tool_calls_brief: List[Dict[str, Any]] = []
        for tc in (getattr(resp, "tool_calls", None) or []):
            tool_calls_brief.append({
                "name": tc.get("name"),
                "args": tc.get("args", {}),
            })

        # step 编号 = 之前已发生的 ReAct 步骤数 + 1
        prior_steps = self._count_prior_agent_steps(state.get("messages") or [])
        add_event(
            "agent_step",
            step=prior_steps + 1,
            budget_left=budget_left,
            reason=reason_text[:400],
            tool_calls=tool_calls_brief,
            llm_elapsed_ms=llm_elapsed,
        )

        return {"messages": [resp]}

    @staticmethod
    def _count_prior_agent_steps(messages: list) -> int:
        """统计已有的 AIMessage 数量 = 之前已发生的 agent 决策次数。"""
        return sum(
            1 for m in messages
            if isinstance(m, AIMessage)
        )

    @staticmethod
    def _render_tools_section(tools: Sequence[BaseTool]) -> str:
        """把工具列表渲染成 system prompt 里的【可用工具】段落。"""
        if not tools:
            return "（当前无可用工具）"
        lines: List[str] = []
        for i, tool in enumerate(tools, 1):
            lines.append(f"{i}) {tool.name}")
            desc = (tool.description or "").strip()
            if desc:
                lines.append(f"   {desc}")
        return "\n".join(lines)

    def _build_agent_system(
        self,
        memory_text: str,
        kb_evidence: str,
        search_count: int,
        budget_left: int,
    ) -> str:
        parts: List[str] = [
            "你是一个严谨的法律领域 Agentic RAG 助手，采用 ReAct（Reason + Act）范式工作。",
            "",
            "【可用工具】",
            self._tools_section,
            "",
            "【ReAct 工作方式】",
            "- 每一步先在内心做 Reason（分析当前已知 vs 还缺什么），再选择 Action：",
            "  · 若信息足够 → 直接给出最终答案（不调用任何工具）",
            "  · 若信息不足 → 调用一个或多个工具，等待结果后回到 Reason",
            "- 改写 query 是你自己的工作：一次没查到时，主动换关键词或换工具再试",
            "- 何时调用工具、调用哪几个、串行还是并行，全部由你自主判断",
            "",
            "【并行 vs 串行调用】",
            "- 如果一个问题包含多个相互独立的子点（例如「A 和 B 各属什么罪名」"
            "/「同时需要法条 X 和判例 Y」），允许且鼓励在【同一回合】内发起多个 tool_calls，"
            "运行时会并行执行并把结果一起返回给你。",
            "- 如果后一步 query 依赖前一步结果（例如先查到某法律名再查具体条号），则串行。",
            "- 不要为了凑次数而并行，每次工具调用都要有清晰目的。",
            "",
            "【硬约束】",
            "- 禁止猜测、编造法条编号、当事人姓名、金额、日期等关键事实",
            "- 没有证据支撑就明确说「未找到相关依据 / 信息不足」，不要凭空补",
            "- 引用知识库内容时，尽量贴近原文表述",
            f"- 本轮工具调用预算：已用 {search_count} 次，还剩 {budget_left} 次"
            f"（上限 {self._max_search_steps}）；用尽后必须直接基于当前证据给出答案",
        ]

        if budget_left == 0:
            parts.append(
                "- ⚠ 工具次数已用尽，请立即基于现有证据生成最终答案，不要再发起任何工具调用"
            )

        if memory_text:
            parts += [
                "",
                "【会话摘要 / 记忆背景】",
                memory_text,
                "请利用以上信息理解用户指代与省略。如需更详细的历史对话或用户画像，"
                "请主动调用 recall_memory / list_user_profile（若可用）。",
            ]

        if kb_evidence:
            parts += [
                "",
                "【已累积的检索证据（来自本轮前几次工具调用）】",
                kb_evidence,
                "如证据已足以回答用户问题，请直接生成答案，不必再调用工具。",
            ]

        return "\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # 路由：agent → tools / self_check_repair
    # ─────────────────────────────────────────────────────────────────────────

    def route_after_agent(
        self, state: dict
    ) -> Literal["tools", "self_check_repair"]:
        """
        纯路由函数：
        - 若 agent 输出了 tool_calls 且未超预算 → 走 tools
        - 否则（直接答 / 已超预算）→ 走 self_check_repair
        """
        messages = state.get("messages") or []
        if not messages:
            return "self_check_repair"

        last = messages[-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return "self_check_repair"

        search_count = int(state.get("search_count") or 0)
        if search_count >= self._max_search_steps:
            # 兜底：模型仍想调工具但预算用尽，强制收敛
            logger.warning(
                "[Nodes] 工具调用预算已用尽（%d/%d），忽略 agent 的 tool_calls，"
                "强制进入自检",
                search_count, self._max_search_steps,
            )
            return "self_check_repair"

        return "tools"

    # ─────────────────────────────────────────────────────────────────────────
    # 节点 3：collect_evidence
    # 职责：把刚刚 tools 节点产出的 ToolMessage（可能多条，对应多个并行 tool_call）
    #       内容追加到 kb_evidence_ref 中，并把 search_count + N
    # ─────────────────────────────────────────────────────────────────────────

    def collect_evidence(self, state: dict) -> dict:
        """
        累积本批 ToolMessage 内容到 kb_evidence_ref，并增加 search_count。

        注意：tool_observation 事件不在这里写，由 ToolTimingCallbackHandler
        在工具真正执行完时写，这样能拿到精确的单条 elapsed_ms。
        """
        messages = state.get("messages") or []
        new_tool_msgs: List[ToolMessage] = []
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                new_tool_msgs.append(m)
            else:
                break
        new_tool_msgs.reverse()

        if not new_tool_msgs:
            return {}

        prev_count = int(state.get("search_count") or 0)

        previous = get_kb_evidence(state)
        chunks: List[str] = []
        if previous:
            chunks.append(previous)
        for tm in new_tool_msgs:
            tool_name = getattr(tm, "name", "") or "tool"
            content = (tm.content or "").strip()
            if content:
                chunks.append(f"── [{tool_name}] ──\n{content}")

        merged = "\n\n".join(chunks)
        new_count = prev_count + len(new_tool_msgs)

        logger.debug(
            "[Nodes] collect_evidence: +%d 条工具返回，累计搜索次数=%d，证据总长=%d",
            len(new_tool_msgs), new_count, len(merged),
        )
        return {
            "kb_evidence_ref": put_kb_evidence(merged),
            "search_count": new_count,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 节点 4：self_check_repair
    # 职责：回答后自检，高风险时触发轻量修正（语义与原版一致）
    # ─────────────────────────────────────────────────────────────────────────

    def self_check_repair(self, state: dict) -> dict:
        question = (state.get("original_question") or "").strip() or _last_human_message(state)
        answer = _last_ai_answer(state)
        search_count = int(state.get("search_count") or 0)

        # 提前记录 final_answer 事件（即使后续被 repair 覆盖，也保留原始版本痕迹）
        add_event(
            "final_answer",
            answer_len=len(answer),
            search_count=search_count,
        )

        if not answer:
            logger.debug("[Nodes] self_check_repair: 无答案可检查，跳过")
            add_event("self_check", hallucination_risk="n/a", revised=False, used_kb=False, skipped=True)
            return {"self_check_result": {}}

        kb_evidence: str = get_kb_evidence(state)

        # 用户画像文本（从记忆管理器获取）
        profile_text = ""
        if self._memory_manager:
            profile_text = self._memory_manager.get_query_context(question)

        # ReAct 架构下不再有路由决策；只要本轮发生过工具调用，就视为
        # "预期使用 KB"，未发生工具调用则视为闲聊/记忆类问题
        expected_use_kb = int(state.get("search_count") or 0) > 0

        check = self._inspector.check(
            question=question,
            answer=answer,
            kb_evidence=kb_evidence,
            user_profile=profile_text,
            expected_use_kb=expected_use_kb,
        )

        result: dict = {"self_check_result": check}

        revised = False
        if check.get("needs_revision"):
            repaired = self._inspector.repair(
                question=question,
                answer=answer,
                check_result=check,
                kb_evidence=kb_evidence,
                user_profile=profile_text,
            )
            result["messages"] = [AIMessage(content=repaired)]
            revised = True
            logger.info(
                "[Nodes] 答案已修正，幻觉风险: %s，问题: %s",
                check.get("hallucination_risk"),
                check.get("notes", []),
            )

        add_event(
            "self_check",
            hallucination_risk=check.get("hallucination_risk", "n/a"),
            used_kb=bool(check.get("used_kb", False)),
            needs_revision=bool(check.get("needs_revision", False)),
            revised=revised,
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 节点 5：update_memory
    # 职责：同步更新中期摘要，异步写入长期记忆
    # ─────────────────────────────────────────────────────────────────────────

    def update_memory(self, state: dict) -> dict:
        if not self._memory_manager:
            return {}

        messages = state["messages"]
        original_question = (state.get("original_question") or "").strip()
        turn_start = self._find_turn_start(messages, original_question)
        turn_messages = messages[turn_start:]

        self._memory_manager.post_turn_update(turn_messages)

        # 同步可能更新过的中期摘要，让 checkpointer 持久化
        new_summary = self._memory_manager.conversation_summary
        prev_summary = (state.get("conversation_summary") or "").strip()
        summary_updated = new_summary != prev_summary
        add_event("memory_update", summary_updated=summary_updated)
        if summary_updated:
            return {"conversation_summary": new_summary}
        return {}

    @staticmethod
    def _find_turn_start(messages: list, original_question: str) -> int:
        """定位本轮起始消息索引 = original_question 对应的 HumanMessage。"""
        if original_question:
            for i in range(len(messages) - 1, -1, -1):
                m = messages[i]
                if (
                    isinstance(m, HumanMessage)
                    and (m.content or "").strip() == original_question
                ):
                    return i
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                return i
        return 0
