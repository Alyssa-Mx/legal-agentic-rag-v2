"""
LangChain callback handlers，用于不侵入工具实现的方式追踪运行时数据。

ToolTimingCallbackHandler:
    在每个工具的 on_tool_start / on_tool_end 上挂钩，记录精确的单条耗时，
    并把 tool_observation 事件写入 trace。

    设计要点：
    - LangGraph 的 ToolNode 用 ThreadPoolExecutor 并行执行工具，子线程
      不继承 contextvars，所以 handler 不能依赖 current_trace()，必须
      显式持有 trace 引用（每轮 invoke 时新建一个 handler 实例）。
    - 用 run_id 匹配 start/end，自然支持并行（每个工具调用 run_id 独立）。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from app.core.trace import Trace, current_trace

logger = logging.getLogger(__name__)


class ToolTimingCallbackHandler(BaseCallbackHandler):
    """
    给每次工具调用记录精确耗时，写入 trace 的 tool_observation 事件。

    Args:
        trace: 显式持有的 Trace 引用（推荐：避免子线程拿不到 ContextVar）。
               如不传，则在每次事件时 fallback 到 current_trace()。
    """

    def __init__(self, trace: Optional[Trace] = None) -> None:
        super().__init__()
        self._trace = trace
        # run_id -> (tool_name, t0, args_brief)
        self._inflight: Dict[UUID, Tuple[str, float, Dict[str, Any]]] = {}

    def _t(self) -> Optional[Trace]:
        return self._trace or current_trace()

    @staticmethod
    def _resolve_name(serialized: Optional[dict], kwargs: dict) -> str:
        if serialized and isinstance(serialized, dict):
            name = serialized.get("name")
            if name:
                return str(name)
        return str(kwargs.get("name") or "tool")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        name = self._resolve_name(serialized, kwargs)
        # inputs 是 dict（来自 tool_call.args），input_str 是序列化后的字符串
        if isinstance(inputs, dict) and inputs:
            args = inputs
        else:
            args = {"input": (input_str or "")[:200]}
        self._inflight[run_id] = (name, time.time(), args)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        info = self._inflight.pop(run_id, None)
        if info is None:
            return
        name, t0, args = info
        elapsed_ms = round((time.time() - t0) * 1000)

        # output 可能是 str（工具直接返回字符串）或 ToolMessage（langgraph 包装后），
        # 后者要从 .content 取真正的字符串，否则会显示成 "content='...' name='...'"
        if isinstance(output, str):
            out_str = output
        elif hasattr(output, "content"):
            inner = getattr(output, "content")
            out_str = inner if isinstance(inner, str) else str(inner)
        else:
            out_str = str(output)
        preview = out_str.replace("\n", " ")

        trace = self._t()
        if trace is not None:
            trace.add(
                "tool_observation",
                tool=name,
                elapsed_ms=elapsed_ms,
                size=len(out_str),
                preview=preview[:120],
                args=args,
            )
        logger.debug(
            "[ToolTiming] %s done in %dms (out=%dB)", name, elapsed_ms, len(out_str)
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        info = self._inflight.pop(run_id, None)
        if info is None:
            return
        name, t0, args = info
        elapsed_ms = round((time.time() - t0) * 1000)

        trace = self._t()
        if trace is not None:
            trace.add(
                "tool_observation",
                tool=name,
                elapsed_ms=elapsed_ms,
                size=0,
                preview=f"[ERROR] {error}",
                args=args,
                error=True,
            )
        logger.warning("[ToolTiming] %s FAILED in %dms: %s", name, elapsed_ms, error)
