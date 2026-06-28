"""
Agent 检索性能评测脚本（ReAct 多工具 vs 固定流水线 baseline）。

╭─ 评测目标 ────────────────────────────────────────────────────────────────╮
│ 用同一份 50 题测试集 + 同一份 5000 条 KB + 同一套指标公式，              │
│ 衡量"ReAct 多工具 Agent"在法条检索上的成功率，与                          │
│ results/retrieval/abl5_50/emb_bm25_rrf_rerank.json 直接对比。            │
╰──────────────────────────────────────────────────────────────────────────╯

╭─ 评测协议 ────────────────────────────────────────────────────────────────╮
│ 1. Agent 必须先调用至少一次 KB 工具（vector_search / bm25_search /        │
│    lookup_article）才能 submit_ranking                                    │
│ 2. Agent 不需要回答用户问题，只需把它认为最相关的 top-5 article_id        │
│    按相关性降序通过 submit_ranking 工具提交                               │
│ 3. 用 compute_retrieval_metrics（与 evaluate.py 完全一致）计算            │
│    Hit@{1,3,5} / Recall@{1,3,5} / MRR / NDCG@{1,3,5}                     │
╰──────────────────────────────────────────────────────────────────────────╯

使用方式：
    .\\.venv\\Scripts\\Activate.ps1

    # smoke test（5 题，~2 分钟）
    python scripts/evaluate_agent_retrieval.py --limit 5 \\
        --output results/agent_retrieval/smoke.json

    # 全量 50 题（与 baseline 对齐）
    python scripts/evaluate_agent_retrieval.py --limit 50 \\
        --output results/agent_retrieval/agent_v1.json \\
        --baseline results/retrieval/abl5_50/emb_bm25_rrf_rerank.json \\
        --resume

    # 调整 max_steps / chat model
    python scripts/evaluate_agent_retrieval.py --limit 50 \\
        --output results/agent_retrieval/agent_v1_step12.json \\
        --max-steps 12 --chat-model qwen-plus
"""

import os

import argparse
import json
import logging
import math
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from app.agent.eval_tools import SubmissionCollector, make_submit_ranking_tool
from app.config.setting import DEFAULT_CHAT_MODEL
from app.core.callbacks import ToolTimingCallbackHandler
from app.core.trace import add_event, start_trace
from app.models.dashscope_chat import build_chat
from app.retrieval.kb_tools import load_kb_tools

DEFAULT_TEST = "data/lecoqa/test.json"
DEFAULT_BM25 = "data/lecoqa/bm25_index.pkl"
DEFAULT_OUTPUT = "results/agent_retrieval/agent_v1.json"
DEFAULT_BASELINE = "results/retrieval/abl5_50/emb_bm25_rrf_rerank.json"
DEFAULT_K_VALUES = (1, 3, 5)
PRIMARY_K = 5

logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("eval_agent_retrieval")
logger.setLevel(logging.INFO)


# ════════════════════════════════════════════════════════════════════════════
#  指标计算（与 scripts/evaluate.py compute_retrieval_metrics 完全一致）
# ════════════════════════════════════════════════════════════════════════════

def dcg(relevances: List[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def compute_retrieval_metrics(
    retrieved: List[int],
    gold: Set[int],
    k_values: Tuple[int, ...] = DEFAULT_K_VALUES,
) -> Dict[str, float]:
    out: Dict[str, float] = {"retrieved_count": len(retrieved), "gold_count": len(gold)}
    if not gold:
        for k in k_values:
            out[f"hit@{k}"] = 0
            out[f"recall@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
        out["mrr"] = 0.0
        return out
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            mrr = 1.0 / rank
            break
    out["mrr"] = round(mrr, 4)
    for k in k_values:
        topk = retrieved[:k]
        rels = [1 if d in gold else 0 for d in topk]
        ideal = dcg([1] * min(len(gold), k))
        out[f"hit@{k}"] = int(any(rels))
        out[f"recall@{k}"] = round(sum(rels) / len(gold), 4)
        out[f"ndcg@{k}"] = round((dcg(rels) / ideal) if ideal > 0 else 0.0, 4)
    return out


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))


# ════════════════════════════════════════════════════════════════════════════
#  评测专用极简 graph：agent ↔ tools 循环（无 memory / 无 self_check）
# ════════════════════════════════════════════════════════════════════════════

def _render_tools_section(tools: Sequence[BaseTool]) -> str:
    lines: List[str] = []
    for i, tool in enumerate(tools, 1):
        lines.append(f"{i}) {tool.name}")
        desc = (tool.description or "").strip()
        if desc:
            lines.append(f"   {desc}")
    return "\n".join(lines)


def build_eval_system_prompt(
    tools: Sequence[BaseTool],
    max_steps: int,
    require_thought: bool = False,
) -> str:
    """评测专用 system prompt：强制先检索 KB，最后 submit_ranking。

    Args:
        require_thought: True 时要求 agent 在每次发起 tool_calls 之前，
            先在回复正文写一句 Thought（≤50字）。便于 trace 观测推理过程。
    """
    lines: List[str] = [
        "你是一个【法条检索评测 Agent】。你的唯一任务是为用户问题找出最相关的法条并按相关性排序提交。",
        "",
        "【可用工具】",
        _render_tools_section(tools),
        "",
        "【工作流程（严格遵守）】",
        f"1. 你**必须先**调用至少一次检索工具（vector_search / bm25_search / lookup_article），上限 {max_steps} 次。",
        "   - 一个回合可并行调用多个工具（不同关键词 / 不同方法），更高效。",
        "   - 若首轮结果不理想，可换关键词重试，或切换 vector ↔ bm25 工具。",
        "2. 综合所有检索结果，从中**挑选最多 5 条最相关**的法条，按相关性**从高到低**排序。",
        "3. 调用 `submit_ranking(article_ids=[id1, id2, ...], reason=\"...\")` 提交。提交后任务结束。",
        "",
        "【硬规则】",
        "- 严禁未检索就直接 submit_ranking。",
        "- **强制要求**：每一题都【必须】以一次 submit_ranking 调用收尾，否则视为任务失败。",
        "  - 哪怕只找到 1 条相关法条，也用 submit_ranking([那 1 条]) 提交；",
        "  - 哪怕全部检索结果都不太相关，也要用 submit_ranking([前 3-5 条相对最像的]) 提交；",
        "  - 严禁『找不到就放弃』 —— 那是最差的策略。",
        "- ranking 中的 article_id 必须来自你检索结果中看到的 `(article_id=X)` 字段，**禁止编造**。",
        "- 不要回答用户的法律问题，**只输出 ranking**（通过 submit_ranking 工具）。",
        "- 选择 ranking 时考虑：法条与问题的语义匹配度、是否直接回答了问题的核心、出处权威性。",
        "",
        "【关于 lookup_article 失败】",
        "- 如果 lookup_article 返回『该法条不在 KB 中』的警告，**立即停止对同一条号的重复尝试**！",
        "  KB 仅是采样语料，你记忆中的某些具体条号确实可能不在里面。",
        "  此时立刻切换到 vector_search/bm25_search，用【场景描述】而非【条号】去查。",
    ]
    if require_thought:
        lines += [
            "",
            "【Thought 要求（必须遵守）】",
            "- 每次发起 tool_calls 之前，**必须**先在回复正文（content）里用一句话",
            "  （≤50 字）写明你这一步打算做什么、为什么，格式为：",
            "      Thought: <你的推理>",
            "  示例：",
            "      Thought: 题目涉及多式联运赔偿，先用 vector 找语义相关条文，并用 bm25 补关键词召回",
            "      Thought: vector 漏掉了第264条，用 lookup_article 精确补一条",
            "      Thought: 已有 5 条强相关法条，准备 submit_ranking 收尾",
            "- 写完 Thought 之后**同一回合内**再发起对应的 tool_calls；",
            "  不要只写 Thought 不调工具（除非已 submit_ranking 完毕）。",
        ]
    lines += [
        "",
        "【关于挑选的策略建议】",
        "- 若 vector_search 和 bm25_search 同时命中某条法条，说明它高度相关。",
        "- 若结果集中只剩 3 条强相关，**宁可只提交 3 条**也不要凑数。",
        "- 排序时把『直接命中问题核心』的法条放最前，把『略相关』的放后面。",
        "",
        "【关于法条长度】",
        "- 检索结果若看到 `[...本条共 N 字，已截断 X 字；如需全文请调用 lookup_article(query=\"...\")]`，",
        "  说明该法条被截断显示。如果它看起来很相关、但截断处恰好可能含关键信息，",
        "  请用 lookup_article 取全文再判断；否则保持原判，不必每条都展开。",
    ]
    return "\n".join(lines)


def make_eval_graph(
    llm_with_tools,
    tools: Sequence[BaseTool],
    max_steps: int,
    collector: "SubmissionCollector",
    require_thought: bool = False,
    force_submit_retries: int = 2,
):
    """
    构造评测专用的 ReAct graph，加入三阶段 submit 强制机制：

    1. **自由探索阶段** (rounds < max_steps)
       - 自由选择 vector/bm25/lookup/submit 工具
       - budget_left ≤ 2 时给"倒计时提示"（柔性鼓励 submit）
    2. **强制 submit 阶段** (max_steps ≤ rounds < max_steps + force_submit_retries + 1)
       - 每轮在 system msg 里附"违规警告"，要求 agent 必须且只能 submit_ranking
       - 共给 (1 + force_submit_retries) 次机会（默认 3 次）
    3. **graph 兜底阶段** (rounds ≥ max_steps + force_submit_retries + 1)
       - graph 直接构造一个 submit_ranking 调用注入，使用 raw pool 前 5 条
       - 由 ToolNode 调用真正的 submit_ranking 工具，写入 collector

    Args:
        collector: 必需。让 agent_node 能感知"agent 是否已 submit"，避免重复 force_submit。
        force_submit_retries: 强制阶段的"额外重试"次数（不算首次）。共 1 + N 次机会。
    """
    system_text = build_eval_system_prompt(tools, max_steps, require_thought=require_thought)
    system_msg = SystemMessage(content=system_text)

    def count_tool_call_rounds(messages: list) -> int:
        """统计 AIMessage 中出现过的 tool_calls 回合数（不计具体并行个数）。"""
        return sum(
            1 for m in messages
            if isinstance(m, AIMessage) and (getattr(m, "tool_calls", None) or [])
        )

    def agent_node(state: dict) -> dict:
        # —— 0) 如果已经 submit，直接结束，避免 agent 多余探索 ——
        if collector.submitted:
            add_event(
                "agent_step",
                step=count_tool_call_rounds(state["messages"]) + 1,
                budget_left=-1,
                reason="(already submitted, graph ends)",
                tool_calls=[],
            )
            return {"messages": [AIMessage(content="（任务已完成，无需再调用工具。）")]}

        rounds = count_tool_call_rounds(state["messages"])
        budget_left = max(0, max_steps - rounds)
        # 强制阶段的"第几次警告"：1, 2, ..., (1 + force_submit_retries)
        force_attempt = rounds - max_steps + 1  # rounds=max_steps → attempt=1
        max_force_attempts = 1 + force_submit_retries

        # —— 3) graph 兜底：所有强制机会用完仍未 submit ——
        if force_attempt > max_force_attempts:
            _, raw_ids = extract_tool_calls_and_retrieved(state["messages"])
            top5 = raw_ids[:PRIMARY_K]
            fallback_call = {
                "id": "graph_fallback_submit_001",
                "name": "submit_ranking",
                "args": {
                    "article_ids": top5,
                    "reason": (
                        f"[GRAPH FALLBACK] agent 在 {max_force_attempts} 次强制提示后"
                        f"仍未调用 submit_ranking，graph 自动提交 raw pool 前 {len(top5)} 条。"
                    ),
                },
            }
            fallback_msg = AIMessage(content="", tool_calls=[fallback_call])
            add_event(
                "agent_step",
                step=rounds + 1,
                budget_left=-1,
                reason=f"(GRAPH FALLBACK: auto-submit raw pool top-{len(top5)} {top5})",
                tool_calls=[{"name": "submit_ranking", "args": fallback_call["args"]}],
            )
            return {"messages": [fallback_msg]}

        # —— 1) / 2) 拼 prompt（区分自由阶段 / 强制阶段） ——
        msgs: List[Any] = [system_msg]

        if force_attempt >= 1:
            # ── 强制 submit 阶段 ──
            _, raw_ids = extract_tool_calls_and_retrieved(state["messages"])
            pool_hint = raw_ids[:10]
            if force_attempt == 1:
                urgency = SystemMessage(content=(
                    f"⚠️【强制 submit 阶段（第 1/{max_force_attempts} 次警告）】\n"
                    f"你已经用完 {max_steps} 轮探索预算。\n"
                    f"**本轮你必须且只能调用 submit_ranking**，"
                    f"禁止再调用 vector_search/bm25_search/lookup_article（一律视为违规）。\n"
                    f"从已检索到的候选 article_id 中挑最相关的 3-5 个提交。\n"
                    f"可参考的候选池（按召回顺序）：{pool_hint}"
                ))
            elif force_attempt == 2:
                urgency = SystemMessage(content=(
                    f"🚨【强制 submit 阶段（第 2/{max_force_attempts} 次警告 — 你已违规 1 次）】\n"
                    f"你上一轮违反了规则！没有调用 submit_ranking。\n"
                    f"**这是你的第 2 次机会**。请立即调用 submit_ranking，"
                    f"再违规将进入最后警告。\n"
                    f"候选池：{pool_hint}"
                ))
            else:  # force_attempt == max_force_attempts (3)
                urgency = SystemMessage(content=(
                    f"❌【最后一次警告（第 {force_attempt}/{max_force_attempts} 次）】\n"
                    f"你已连续 {force_attempt - 1} 次违规未提交！\n"
                    f"**这是最后的机会**。如果本轮再不调用 submit_ranking，\n"
                    f"graph 将强制使用 {pool_hint[:PRIMARY_K]} 兜底替你提交（你将失去自主选择权）。\n"
                    f"候选池：{pool_hint}"
                ))
            msgs.append(urgency)
        elif budget_left <= 2:
            # 自由探索阶段的"倒计时提示"
            urgency = SystemMessage(content=(
                f"【倒计时提示】你只剩 {budget_left} 轮工具调用机会。"
                "**本轮建议直接发起 submit_ranking 调用**，"
                "从已经检索到的所有候选里挑相对最相关的 3-5 条提交。"
            ))
            msgs.append(urgency)

        msgs.extend(state["messages"])
        resp = llm_with_tools.invoke(msgs)
        # —— 写 trace：把 Reason（thinking 优先 / content）+ Act（tool_calls）落到 agent_step ——
        tool_calls = []
        for tc in (getattr(resp, "tool_calls", None) or []):
            tool_calls.append({
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            })
        additional = getattr(resp, "additional_kwargs", None) or {}
        reasoning = additional.get("reasoning_content", "")
        # 优先用 thinking 模型的 reasoning_content，否则用 content（thought 模式下也写在 content）
        reason_text = (reasoning or resp.content or "").strip()
        reason_source = "reasoning_content" if reasoning else "content"
        # 标注当前阶段（自由 / 强制第 N 次）
        phase = "free" if force_attempt < 1 else f"force_{force_attempt}/{max_force_attempts}"
        add_event(
            "agent_step",
            step=rounds + 1,
            budget_left=budget_left,
            phase=phase,
            reason=reason_text,
            reason_source=reason_source,
            tool_calls=tool_calls,
        )
        return {"messages": [resp]}

    def route(state: dict) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and (getattr(last, "tool_calls", None) or []):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(list(tools)))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ════════════════════════════════════════════════════════════════════════════
#  Tool-call 解析：从消息流提取 (tool_name, args, returned_article_ids)
# ════════════════════════════════════════════════════════════════════════════

ARTICLE_ID_PATTERN = __import__("re").compile(r"article_id\s*=\s*(\d+)")


def extract_tool_calls_and_retrieved(messages: list) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    从消息流提取：
      - 每次 tool call 的 {name, args, returned_ids}
      - 所有 KB 工具返回的 article_id（按出现顺序去重）
    """
    tool_calls: List[Dict[str, Any]] = []
    raw_ids_in_order: List[int] = []
    seen_ids: Set[int] = set()

    # 1) 收集 AIMessage 的 tool_calls：name + args
    pending: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, AIMessage) and (getattr(m, "tool_calls", None) or []):
            for tc in m.tool_calls:
                pending.append({
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                    "returned_ids": [],
                })
        elif isinstance(m, ToolMessage):
            # 用 tool_call_id 关联回 pending（容错：若拿不到 id 就匹配同名首个未填充）
            tc_id = getattr(m, "tool_call_id", "")
            matched = None
            for tc in pending:
                if tc.get("id") == tc_id and not tc.get("_filled"):
                    matched = tc
                    break
            if matched is None:
                for tc in pending:
                    if not tc.get("_filled"):
                        matched = tc
                        break
            content = m.content or ""
            ids_in_msg = [int(x) for x in ARTICLE_ID_PATTERN.findall(content)]
            if matched is not None:
                matched["returned_ids"] = ids_in_msg
                matched["_filled"] = True
            # KB 检索工具返回的 ID 按顺序累计（去重）
            if matched and matched["name"] in {"vector_search", "bm25_search", "lookup_article"}:
                for x in ids_in_msg:
                    if x not in seen_ids:
                        seen_ids.add(x)
                        raw_ids_in_order.append(x)

    for tc in pending:
        tc.pop("_filled", None)
    return pending, raw_ids_in_order


# ════════════════════════════════════════════════════════════════════════════
#  单题评测
# ════════════════════════════════════════════════════════════════════════════

def evaluate_one(
    sample: Dict[str, Any],
    kb_tools: List[BaseTool],
    chat_model: str,
    max_steps: int,
    recursion_limit: int = 40,
    save_trace: bool = False,
    require_thought: bool = False,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    question = sample.get("问题", "") or ""
    query_id = sample.get("query_id")
    gold_ids: List[int] = list(sample.get("match_id") or [])
    gold_names: List[str] = list(sample.get("match_name") or [])

    # —— 每题构造独立的 collector + submit_ranking 工具 ——
    collector = SubmissionCollector()
    submit_tool = make_submit_ranking_tool(collector, top_k=PRIMARY_K)
    tools: List[BaseTool] = list(kb_tools) + [submit_tool]

    # thinking 模式下单次响应慢得多，加 timeout
    llm = build_chat(chat_model, enable_thinking=enable_thinking, timeout=180 if enable_thinking else 60)
    llm_with_tools = llm.bind_tools(tools)
    graph = make_eval_graph(
        llm_with_tools, tools,
        max_steps=max_steps,
        collector=collector,
        require_thought=require_thought,
    )

    trace = start_trace(thread_id=f"eval_qid_{query_id}")
    trace.add("turn_start", question=question, query_id=query_id, gold_count=len(gold_ids))
    # 显式持有 trace 引用：ToolNode 走线程池，子线程不继承 contextvars
    handler = ToolTimingCallbackHandler(trace=trace)

    t0 = time.perf_counter()
    error_msg = ""
    final_messages: list = []
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={
                "callbacks": [handler],
                "recursion_limit": recursion_limit,
            },
        )
        final_messages = result.get("messages", [])
    except Exception as e:  # 网络 / 配额 / recursion limit
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning("[query_id=%s] 异常: %s", query_id, error_msg)
        if logger.isEnabledFor(logging.DEBUG):
            traceback.print_exc()
    elapsed = round(time.perf_counter() - t0, 3)

    tool_calls, retrieved_ids = extract_tool_calls_and_retrieved(final_messages)
    submitted_ranking = collector.ranking if collector.submitted else None

    # —— 最终用于打分的 ranking ——
    # 优先用 agent 提交的；若 agent 没提交，fallback 用"检索过的去重保序前 K 个"
    if submitted_ranking is not None and len(submitted_ranking) > 0:
        scoring_ranking = submitted_ranking
        scoring_source = "submitted"
    elif retrieved_ids:
        scoring_ranking = retrieved_ids[:PRIMARY_K]
        scoring_source = "fallback_retrieved"
    else:
        scoring_ranking = []
        scoring_source = "empty"

    metrics = compute_retrieval_metrics(scoring_ranking, set(gold_ids))

    # 同时算 raw_retrieved（不靠 agent 排序）作为参考：能反映 KB 召回能力上限
    metrics_raw_retrieved = compute_retrieval_metrics(retrieved_ids, set(gold_ids))

    kb_calls = [tc for tc in tool_calls if tc["name"] in {"vector_search", "bm25_search", "lookup_article"}]
    rec: Dict[str, Any] = {
        "query_id": query_id,
        "question": question,
        "gold_ids": gold_ids,
        "gold_names": gold_names,
        "submitted_ranking": submitted_ranking,
        "scoring_ranking": scoring_ranking,
        "scoring_source": scoring_source,
        "raw_retrieved_ids": retrieved_ids,
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"], "returned_ids": tc["returned_ids"]}
            for tc in tool_calls
        ],
        "tool_call_count": len(tool_calls),
        "kb_call_count": len(kb_calls),
        "submitted": collector.submitted,
        "submit_reason": collector.reason,
        "latency_s": elapsed,
        "metrics": metrics,
        "metrics_raw_retrieved": metrics_raw_retrieved,
        "error": error_msg,
    }
    # trace_text 始终生成（便于打印），但只在 save_trace 时写入持久化 record
    trace_text = trace.render_text(max_preview=160)
    if save_trace:
        rec["trace_text"] = trace_text
        rec["trace_events"] = trace.export()["events"]
    else:
        rec["trace_text"] = trace_text  # 临时字段，main() 打印后会被丢弃
    return rec


# ════════════════════════════════════════════════════════════════════════════
#  汇总
# ════════════════════════════════════════════════════════════════════════════

def aggregate(per_question: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(per_question)
    if n == 0:
        return {"n": 0}

    def mean(key_path: List[str]) -> float:
        vals = []
        for r in per_question:
            cur = r
            for k in key_path:
                cur = cur.get(k, 0) if isinstance(cur, dict) else 0
            try:
                vals.append(float(cur))
            except (TypeError, ValueError):
                vals.append(0.0)
        return round(sum(vals) / n, 4) if vals else 0.0

    lat = [r.get("latency_s", 0.0) for r in per_question]
    summary = {
        "n": n,
        # —— 主指标：用 agent 提交的 ranking（fallback 时用 raw retrieved 前 K） ——
        "hit@1": mean(["metrics", "hit@1"]),
        "recall@1": mean(["metrics", "recall@1"]),
        "ndcg@1": mean(["metrics", "ndcg@1"]),
        "hit@3": mean(["metrics", "hit@3"]),
        "recall@3": mean(["metrics", "recall@3"]),
        "ndcg@3": mean(["metrics", "ndcg@3"]),
        "hit@5": mean(["metrics", "hit@5"]),
        "recall@5": mean(["metrics", "recall@5"]),
        "ndcg@5": mean(["metrics", "ndcg@5"]),
        "mrr": mean(["metrics", "mrr"]),
        # —— 参考指标：仅看检索回的所有 ID（不靠 agent 排序），反映工具召回上限 ——
        "raw_hit@5": mean(["metrics_raw_retrieved", "hit@5"]),
        "raw_recall@5": mean(["metrics_raw_retrieved", "recall@5"]),
        "raw_mrr": mean(["metrics_raw_retrieved", "mrr"]),
        # —— 系统行为 ——
        "submit_rate": round(sum(1 for r in per_question if r["submitted"]) / n, 4),
        "avg_tool_calls": round(sum(r["tool_call_count"] for r in per_question) / n, 2),
        "avg_kb_calls": round(sum(r["kb_call_count"] for r in per_question) / n, 2),
        "avg_retrieved_pool": round(sum(len(r["raw_retrieved_ids"]) for r in per_question) / n, 2),
        "errors": sum(1 for r in per_question if r.get("error")),
        # —— 延迟 ——
        "lat_avg": round(sum(lat) / n, 3),
        "lat_p50": round(percentile(lat, 50), 3),
        "lat_p95": round(percentile(lat, 95), 3),
    }
    return summary


# ════════════════════════════════════════════════════════════════════════════
#  I/O + 断点续跑
# ════════════════════════════════════════════════════════════════════════════

def load_existing(path: str) -> Tuple[Dict, List[Dict]]:
    if not os.path.exists(path):
        return {}, []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("config", {}), payload.get("per_question", [])


def save_results(
    path: str,
    config: Dict[str, Any],
    per_question: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "variant": "agent_react_multitool",
        "label": "ReAct 多工具 Agent（vector + bm25 + lookup + submit）",
        "config": config,
        "summary": summary,
        "per_question": per_question,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════════════
#  对比报告（vs baseline JSON）
# ════════════════════════════════════════════════════════════════════════════

def render_compare_table(agent_summary: Dict[str, Any], baseline_path: str) -> str:
    if not (baseline_path and os.path.exists(baseline_path)):
        return f"\n（未找到 baseline: {baseline_path}，跳过对比）\n"
    with open(baseline_path, encoding="utf-8") as f:
        bl = json.load(f).get("summary", {})

    rows = [
        ("Hit@1",     "hit@1"),
        ("Recall@1",  "recall@1"),
        ("NDCG@1",    "ndcg@1"),
        ("Hit@3",     "hit@3"),
        ("Recall@3",  "recall@3"),
        ("NDCG@3",    "ndcg@3"),
        ("Hit@5",     "hit@5"),
        ("Recall@5",  "recall@5"),
        ("NDCG@5",    "ndcg@5"),
        ("MRR",       "mrr"),
    ]
    lines: List[str] = []
    lines.append("")
    lines.append("=" * 76)
    lines.append("ReAct Agent  vs  Hybrid Baseline（emb+bm25+rrf+rerank）")
    lines.append("=" * 76)
    lines.append(f"{'指标':<10} | {'Baseline':>10} | {'Agent':>10} | {'Δ':>10} | {'相对':>8}")
    lines.append("-" * 60)
    for label, key in rows:
        b = float(bl.get(key, 0.0))
        a = float(agent_summary.get(key, 0.0))
        delta = a - b
        rel = (delta / b * 100) if b else 0.0
        marker = "  ↑" if delta > 1e-4 else ("  ↓" if delta < -1e-4 else "  ·")
        lines.append(
            f"{label:<10} | {b:>10.4f} | {a:>10.4f} | {delta:>+10.4f} | {rel:>+7.1f}%{marker}"
        )
    lines.append("-" * 60)
    lines.append(
        f"主指标(Hit@{PRIMARY_K})：baseline={bl.get(f'hit@{PRIMARY_K}', 0):.3f} "
        f"→ agent={agent_summary.get(f'hit@{PRIMARY_K}', 0):.3f}"
    )
    # 行为
    lines.append("")
    lines.append("Agent 行为指标：")
    lines.append(f"  提交率              : {agent_summary.get('submit_rate', 0)*100:.1f}%")
    lines.append(f"  平均工具调用数      : {agent_summary.get('avg_tool_calls', 0):.2f}")
    lines.append(f"  平均 KB 调用数      : {agent_summary.get('avg_kb_calls', 0):.2f}")
    lines.append(f"  平均检索回的候选数  : {agent_summary.get('avg_retrieved_pool', 0):.2f}")
    lines.append(f"  错误样本数          : {agent_summary.get('errors', 0)}")
    lines.append(f"  延迟 P50 / P95      : {agent_summary.get('lat_p50', 0):.2f}s / {agent_summary.get('lat_p95', 0):.2f}s")
    lines.append("")
    lines.append("【参考】纯候选池指标（不用 agent 排序，反映工具召回上限）：")
    lines.append(
        f"  Hit@5={agent_summary.get('raw_hit@5', 0):.4f}  "
        f"Recall@5={agent_summary.get('raw_recall@5', 0):.4f}  "
        f"MRR={agent_summary.get('raw_mrr', 0):.4f}"
    )
    lines.append("=" * 76)
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
#  主流程
# ════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--bm25", default=DEFAULT_BM25)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--baseline", default=DEFAULT_BASELINE, help="用于对比的 baseline JSON 路径")
    ap.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    ap.add_argument("--max-steps", type=int, default=10, help="Agent 最大工具调用回合数")
    ap.add_argument("--recursion-limit", type=int, default=60)
    ap.add_argument(
        "--trunc-chars", type=int, default=None,
        help="法条字数截断阈值（如 200）。超过的截断 + 提示 lookup_article 取全文。"
             "None=不截断（baseline）。lookup_article 永远全文。",
    )
    ap.add_argument(
        "--default-k", type=int, default=10,
        help="vector_search / bm25_search 默认 k（当 LLM 不显式传 k 时用）。建议截断时改用 10",
    )
    ap.add_argument(
        "--rerank", action="store_true",
        help="对 vector_search / bm25_search 启用 reranker 精排：先召回 pool_size 条 -> reranker -> 返回 top-k",
    )
    ap.add_argument(
        "--rerank-pool", type=int, default=20,
        help="启用 --rerank 时每个工具的召回池大小（默认 20）",
    )
    ap.add_argument(
        "--rerank-model", default=None,
        help="reranker 模型名（默认走 setting.DEFAULT_RERANK_MODEL = qwen3-rerank）",
    )
    ap.add_argument("--resume", action="store_true", help="跳过 output 已有的 query_id")
    ap.add_argument("--sleep", type=float, default=0.0, help="题间隔，秒（防 API 限流）")
    ap.add_argument("--flush-every", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--trace",
        action="store_true",
        help="每题完成后打印 ReAct 时间线（Reason→Act→Observation），并把结构化 events 写入结果 JSON",
    )
    ap.add_argument(
        "--thought",
        action="store_true",
        help="要求 agent 在每次发起 tool_calls 前先写一句 Thought (≤50 字)，让 trace 中的 Reason 可见。"
             "副作用：可能轻微影响 agent 行为，仅作诊断模式建议使用。",
    )
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="开启 hybrid 模型的深度思考（reasoning_content）模式，如 qwen-plus-2025-07-28、"
             "qwen3.5/3.7 系列。响应里的 reasoning_content 会作为 trace 的 Reason。"
             "单次响应明显更慢，但推理质量更高。tool_choice 会被锁定为 auto。",
    )
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if not os.path.exists(args.test):
        print(f"[ERR] 测试集不存在: {args.test}")
        return 1
    if not os.path.exists(args.bm25):
        print(f"[ERR] BM25 索引不存在: {args.bm25}\n请先运行 scripts/build_kb.py")
        return 1

    with open(args.test, encoding="utf-8") as f:
        test_data = json.load(f)
    if args.limit:
        test_data = test_data[: args.limit]
    print(f"[INFO] 测试集: {args.test}  样本数: {len(test_data)}")

    print(f"[INFO] 加载 KB（共享 evaluate.py 的 5000 条 Chroma + BM25）...")
    reranker = None
    if args.rerank:
        from app.config.setting import DEFAULT_RERANK_MODEL
        from app.models.dashscope_reranker import QwenDashScopeReranker
        rerank_model = args.rerank_model or DEFAULT_RERANK_MODEL
        reranker = QwenDashScopeReranker(model_name=rerank_model)
        print(f"[INFO] 启用 Reranker: {rerank_model}  pool_size={args.rerank_pool}")

    kb_tools = load_kb_tools(
        bm25_path=args.bm25,
        default_k=args.default_k,
        trunc_chars=args.trunc_chars,
        reranker=reranker,
        rerank_pool_size=args.rerank_pool,
    )
    print(
        f"[INFO] KB 工具: {[t.name for t in kb_tools]}  "
        f"default_k={args.default_k}  trunc_chars={args.trunc_chars}  "
        f"rerank={'ON' if reranker else 'OFF'}"
    )

    # —— 断点续跑 ——
    config = {
        "limit": args.limit,
        "chat_model": args.chat_model,
        "max_steps": args.max_steps,
        "default_k": args.default_k,
        "trunc_chars": args.trunc_chars,
        "rerank": bool(args.rerank),
        "rerank_pool": args.rerank_pool if args.rerank else None,
        "rerank_model": (args.rerank_model or "qwen3-rerank") if args.rerank else None,
        "require_thought": args.thought,
        "enable_thinking": args.thinking,
        "test_set": args.test,
        "bm25_path": args.bm25,
        "primary_k": PRIMARY_K,
        "k_values": list(DEFAULT_K_VALUES),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    existing_cfg, existing = ([], [])
    if args.resume:
        existing_cfg, existing = load_existing(args.output)
        if existing:
            print(f"[INFO] 续跑：已有 {len(existing)} 条结果，将跳过对应 query_id")

    done_ids = {r["query_id"] for r in existing if r.get("query_id") is not None}
    per_question: List[Dict[str, Any]] = list(existing)

    # 主循环
    for i, sample in enumerate(test_data, 1):
        qid = sample.get("query_id")
        if qid in done_ids:
            continue
        question = (sample.get("问题") or "").strip()
        print(f"\n[{i}/{len(test_data)}] query_id={qid}  Q: {question[:50]}{'...' if len(question) > 50 else ''}")

        t_start = time.perf_counter()
        rec = evaluate_one(
            sample=sample,
            kb_tools=kb_tools,
            chat_model=args.chat_model,
            max_steps=args.max_steps,
            recursion_limit=args.recursion_limit,
            save_trace=args.trace,
            require_thought=args.thought,
            enable_thinking=args.thinking,
        )
        elapsed = time.perf_counter() - t_start
        per_question.append(rec)

        m = rec["metrics"]
        print(
            f"        submit={rec['submitted']} src={rec['scoring_source']:>18s}  "
            f"tools={rec['tool_call_count']}(kb={rec['kb_call_count']})  "
            f"pool={len(rec['raw_retrieved_ids'])}  "
            f"hit@5={m['hit@5']} recall@5={m['recall@5']:.2f} mrr={m['mrr']:.3f}  "
            f"{elapsed:.1f}s"
        )
        if rec.get("error"):
            print(f"        [ERR] {rec['error']}")
        if args.trace and rec.get("trace_text"):
            print(rec["trace_text"])
        else:
            # 不开 --trace 时不要把 trace_text 写到结果文件
            rec.pop("trace_text", None)

        # 增量 flush
        if i % args.flush_every == 0 or i == len(test_data):
            summary = aggregate(per_question)
            save_results(args.output, config, per_question, summary)

        if args.sleep > 0:
            time.sleep(args.sleep)

    # —— 最终保存 + 对比 ——
    summary = aggregate(per_question)
    save_results(args.output, config, per_question, summary)
    print(f"\n[INFO] 结果已写入: {args.output}")

    compare_text = render_compare_table(summary, args.baseline)
    print(compare_text)

    # 也把对比表写到 .txt 旁边
    txt_path = os.path.splitext(args.output)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(compare_text)
    print(f"[INFO] 对比表已写入: {txt_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
