"""
ReAct 链路追踪：完整记录每轮请求的 Reason → Act → Observation 步骤。

使用方式（线程/asyncio 安全，基于 contextvars）：

    from app.core.trace import start_trace, current_trace, add_event

    # 主流程入口（如 main.py 每轮对话开始时）
    trace = start_trace(thread_id="user-session-1")

    app.invoke(...)  # 节点内部用 add_event(...) 写事件

    print(trace.render_text())   # 人类可读的 ReAct 时间线
    json.dump(trace.export(), f) # 结构化导出

事件类型一览：

    turn_start          用户输入
    memory_loaded       prepare_context 完成
    agent_step          一次进入 agent 节点的 Reason + Action
    tool_observation    单个工具的执行结果（含耗时）
    final_answer        Agent 输出最终答案
    self_check          自检结果
    memory_update       记忆写入完成
"""
from __future__ import annotations

import contextvars
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  Trace 数据类
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Trace:
    """单轮 ReAct 请求的链路追踪对象。"""

    thread_id: str = "default"
    _events: List[Dict[str, Any]] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)

    def now_ms(self) -> int:
        return round((time.time() - self._start_time) * 1000)

    def add(self, event: str, **data: Any) -> None:
        entry = {"event": event, "t_ms": self.now_ms(), **data}
        self._events.append(entry)
        logger.debug("[Trace:%s] %s %s", self.thread_id, event, data)

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def export(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "duration_ms": self.now_ms(),
            "events": self.events,
        }

    # ── 人类可读渲染 ─────────────────────────────────────────────────────────

    def render_text(self, max_preview: int = 100) -> str:
        """渲染为终端友好的 ReAct 时间线。"""
        lines = [
            f"━━━━━━━━━━━━━━━━ ReAct Trace [{self.thread_id}] "
            f"total={self.now_ms()}ms ━━━━━━━━━━━━━━━━"
        ]
        for ev in self._events:
            t = ev["t_ms"]
            name = ev["event"]

            if name == "turn_start":
                q = (ev.get("question") or "").replace("\n", " ")
                lines.append(f"  [{t:>5}ms] [Q] {q[:max_preview]}")

            elif name == "memory_loaded":
                lines.append(
                    f"  [{t:>5}ms] [MEM] memory_ctx_len={ev.get('ctx_len', 0)}"
                )

            elif name == "agent_step":
                step = ev.get("step")
                reason = (ev.get("reason") or "").replace("\n", " ").strip()
                calls = ev.get("tool_calls") or []
                budget_left = ev.get("budget_left", "?")
                lines.append(
                    f"  [{t:>5}ms] [AGENT step={step} budget_left={budget_left}]"
                )
                if reason:
                    lines.append(f"           [Reason] {reason[:max_preview]}")
                if calls:
                    for c in calls:
                        args_s = _short_args(c.get("args", {}), max_len=max_preview)
                        lines.append(f"           [Act]    {c['name']}({args_s})")
                else:
                    lines.append("           [Act]    (no tool call -> final answer)")

            elif name == "tool_observation":
                tool = ev.get("tool", "?")
                preview = (ev.get("preview") or "").replace("\n", " ")
                elapsed = ev.get("elapsed_ms", "?")
                lines.append(
                    f"  [{t:>5}ms] [OBS]    {tool} -> {preview[:max_preview]} "
                    f"({elapsed}ms, {ev.get('size', 0)}B)"
                )

            elif name == "final_answer":
                lines.append(
                    f"  [{t:>5}ms] [ANSWER] len={ev.get('answer_len', 0)} "
                    f"search_count={ev.get('search_count', 0)}"
                )

            elif name == "self_check":
                lines.append(
                    f"  [{t:>5}ms] [CHECK]  risk={ev.get('hallucination_risk', 'n/a')} "
                    f"revised={ev.get('revised', False)} "
                    f"used_kb={ev.get('used_kb', 'n/a')}"
                )

            elif name == "memory_update":
                lines.append(f"  [{t:>5}ms] [MEM-W] summary_updated={ev.get('summary_updated', False)}")

            else:
                lines.append(f"  [{t:>5}ms] {name} {ev}")

        lines.append("━" * 80)
        return "\n".join(lines)


def _short_args(args: Dict[str, Any], max_len: int = 80) -> str:
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        try:
            sv = json.dumps(v, ensure_ascii=False)
        except Exception:
            sv = str(v)
        if len(sv) > 50:
            sv = sv[:47] + "..."
        parts.append(f"{k}={sv}")
    s = ", ".join(parts)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


# ════════════════════════════════════════════════════════════════════════════
#  全局当前 trace（ContextVar：协程/线程安全）
# ════════════════════════════════════════════════════════════════════════════

_current: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar(
    "current_trace", default=None
)


def start_trace(thread_id: str = "default") -> Trace:
    """开启新 trace 并设为当前。"""
    t = Trace(thread_id=thread_id)
    _current.set(t)
    return t


def current_trace() -> Optional[Trace]:
    """获取当前 trace，可能为 None（未开启 trace 时）。"""
    return _current.get()


def clear_trace() -> None:
    """清除当前 trace。"""
    _current.set(None)


def add_event(event: str, **data: Any) -> None:
    """便捷函数：往当前 trace 写事件。无 trace 时 no-op。"""
    t = _current.get()
    if t is not None:
        t.add(event, **data)
