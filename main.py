import logging
import os
import getpass
import argparse

from langchain_core.messages import HumanMessage, AIMessage

from app.retrieval.loader import load_demo_docs
from app.retrieval.kb_tools import build_kb_tools
from app.retrieval.web_search import WebSearchTool
from app.agent.memory_tools import build_memory_tools
from app.models.dashscope_chat import build_chat
from app.models.dashscope_embeddings import build_embeddings
from app.config.setting import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL
from app.memory.manager import MemoryManager
from app.agent.graph import make_graph
from app.core.callbacks import ToolTimingCallbackHandler
from app.core.metrics import metrics
from app.core.trace import start_trace, current_trace

# --------------- 测试时可直接在此填写 API Key（填写后下面的交互提示会跳过）---------------
# 注意：不要将真实 Key 提交到 git，测试完改回空字符串或注释掉。
# os.environ["DASHSCOPE_API_KEY"] = "your-key-here"
# os.environ["SERPER_API_KEY"] = "your-key-here"

'''
.\\.venv\\Scripts\\Activate.ps1
python main.py
'''


def setup_logging() -> None:
    """配置日志系统：全局 INFO；app.agent 包单独 DEBUG，便于观察 ReAct 决策。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("app.agent").setLevel(logging.DEBUG)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


def setup_api_keys() -> None:
    """启动时若未设置环境变量，则提示用户输入 API Key。"""
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("未检测到 DASHSCOPE_API_KEY，请填写（输入时不显示）：")
        key = getpass.getpass("DashScope API Key: ").strip()
        if key:
            os.environ["DASHSCOPE_API_KEY"] = key
        else:
            raise SystemExit("未填写 DashScope API Key，无法继续。")

    if not os.environ.get("SERPER_API_KEY"):
        choice = input("是否设置 Serper API Key（联网搜索）？留空跳过，输入 y 则填写: ").strip().lower()
        if choice == "y":
            key = getpass.getpass("Serper API Key: ").strip()
            if key:
                os.environ["SERPER_API_KEY"] = key


def run_cli(max_search_steps: int = 8, show_trace: bool = True) -> None:
    setup_logging()
    setup_api_keys()

    docs = load_demo_docs()
    llm = build_chat(DEFAULT_CHAT_MODEL)
    embeddings = build_embeddings(DEFAULT_EMBED_MODEL)
    thread_id = "user-session-1"
    memory_mgr = MemoryManager(llm, embeddings, session_id=thread_id)

    # ── 组装 Agent 工具集 ─────────────────────────────────────────────────
    # KB 类：vector_search / bm25_search / lookup_article
    # 记忆类：recall_memory / list_user_profile（memory_mgr 为 None 时跳过）
    # 联网类：web_search
    tools = [
        *build_kb_tools(docs),
        *build_memory_tools(memory_mgr),
        WebSearchTool(),
    ]

    app = make_graph(
        tools=tools,
        memory_manager=memory_mgr,
        max_search_steps=max_search_steps,
    )
    base_config = {"configurable": {"thread_id": thread_id}}

    print("=" * 60)
    print("法律智能问答 Agent（ReAct 架构，基于 LeCoQA 数据集）")
    print(f"可用工具（共 {len(tools)} 个）：{', '.join(t.name for t in tools)}")
    print(f"工具调用上限：每轮 {max_search_steps} 次（支持并行调用）")
    print("输入 quit / exit / q 退出；输入 stats 查看本次统计")
    print("=" * 60)
    print("\n💡 提示：Agent 会自主决定何时调用工具、调用哪些、串行还是并行。\n")

    try:
        while True:
            try:
                question = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if not question:
                continue

            if question.lower() in ("quit", "exit", "q"):
                print("再见！")
                break

            if question.lower() == "stats":
                summary = metrics.summary()
                if summary:
                    print("\n── 本次会话统计 ──")
                    for k, v in summary.items():
                        print(f"  {k}: {v}")
                else:
                    print("（暂无统计数据）")
                continue

            # 本轮指标采集
            turn_used_kb = False
            turn_search_count = 0
            turn_repaired = False
            turn_hallucination_risk = "n/a"
            answer = ""

            # 开启本轮的 ReAct trace + 工具计时 callback
            trace = start_trace(thread_id=thread_id) if show_trace else None
            tool_timer = ToolTimingCallbackHandler(trace=trace)
            run_config = {**base_config, "callbacks": [tool_timer]}

            for chunk in app.stream(
                {"messages": [HumanMessage(content=question)]},
                config=run_config,
            ):
                for node_name, update in chunk.items():
                    if not update:
                        continue

                    # 收集指标信号
                    if node_name == "collect_evidence":
                        if update.get("kb_evidence_ref"):
                            turn_used_kb = True
                        turn_search_count = int(update.get("search_count") or turn_search_count)

                    if node_name == "self_check_repair":
                        check = update.get("self_check_result") or {}
                        turn_hallucination_risk = check.get("hallucination_risk", "n/a")
                        if update.get("messages"):
                            turn_repaired = True

                    # 显示节点输出
                    if node_name in ("prepare_context", "update_memory", "collect_evidence"):
                        continue

                    msgs = update.get("messages")
                    if not msgs:
                        continue

                    print("\n" + "─" * 70)
                    print(f"[{node_name}]")
                    last_msg = msgs[-1]

                    if isinstance(last_msg, AIMessage):
                        if (last_msg.content or "").strip():
                            print(last_msg.content)
                            answer = last_msg.content
                        elif getattr(last_msg, "tool_calls", None):
                            for tc in last_msg.tool_calls:
                                print(f"  → 调用工具: {tc.get('name')}({tc.get('args', {})})")
                    elif hasattr(last_msg, "pretty_print"):
                        last_msg.pretty_print()
                    else:
                        print(last_msg)

            metrics.record({
                "question_len": len(question),
                "answer_len": len(answer),
                "used_kb": turn_used_kb,
                "search_count": turn_search_count,
                "answer_revised": turn_repaired,
                "hallucination_risk": turn_hallucination_risk,
            })

            # 打印本轮 ReAct 轨迹（便于调试 / 复盘）
            if trace is not None:
                print()
                print(trace.render_text())

    finally:
        summary = metrics.summary()
        if summary:
            print("\n── 会话结束统计 ──")
            for k, v in summary.items():
                print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG CLI (ReAct 架构)")
    parser.add_argument(
        "--max-search-steps",
        type=int,
        default=8,
        help="单轮对话内 Agent 可调用的工具最大次数（默认 8）",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="关闭每轮末尾的 ReAct 轨迹打印",
    )
    args = parser.parse_args()
    run_cli(max_search_steps=args.max_search_steps, show_trace=not args.no_trace)
