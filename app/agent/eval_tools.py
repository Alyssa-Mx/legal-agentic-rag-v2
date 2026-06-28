"""
评测专用的工具：submit_ranking。

让 Agent 在检索结束后，**提交一个 article_id 的有序列表**（最多 5 个），
而不是给出文字回答。脚本通过 SubmissionCollector 在外部捕获该提交。

设计要点：
    - 每题构造一个新的 collector + 工具实例，互不污染
    - 调用 submit_ranking 会把 ranking 写入 collector.ranking 并被记录已提交
    - 工具返回一个确认字符串给 Agent，让它知道"任务结束"
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubmissionCollector:
    """单题的提交结果收集器。"""

    def __init__(self) -> None:
        self.ranking: Optional[List[int]] = None
        self.submitted: bool = False
        self.submit_count: int = 0
        self.reason: str = ""

    def reset(self) -> None:
        self.ranking = None
        self.submitted = False
        self.submit_count = 0
        self.reason = ""


class _SubmitRankingArgs(BaseModel):
    article_ids: List[int] = Field(
        ...,
        description=(
            "按相关性降序的 article_id 整数列表。最多 5 个；"
            "所有 ID 必须来自你之前调用 vector_search / bm25_search / lookup_article "
            "时看到的 (article_id=X) 字段。"
        ),
    )
    reason: str = Field(
        "",
        description="可选：简短说明你的排序依据（≤80 字），便于事后复盘。",
    )


def make_submit_ranking_tool(collector: SubmissionCollector, top_k: int = 5) -> BaseTool:
    """
    构造一个绑定到指定 collector 的 submit_ranking 工具。

    Args:
        collector: 单题的 SubmissionCollector（外部捕获结果）
        top_k:    提交上限，超过的尾部会被截断
    """
    # 用闭包绑定 collector / top_k，避免在 BaseTool 子类上声明字段
    # （pydantic v2 禁止字段名以下划线开头）
    _collector = collector
    _top_k = top_k

    def _do_submit(article_ids: List[int], reason: str = "") -> str:
        normalized: List[int] = []
        for x in article_ids:
            try:
                normalized.append(int(x))
            except (TypeError, ValueError):
                continue
        truncated = normalized[:_top_k]
        _collector.ranking = truncated
        _collector.reason = (reason or "").strip()
        _collector.submitted = True
        _collector.submit_count += 1
        logger.debug("[submit_ranking] 提交 %d 个 article_id: %s", len(truncated), truncated)
        if not truncated:
            return "已收到提交，但 ranking 为空。任务结束。"
        return (
            f"已提交 {len(truncated)} 个 article_id: {truncated}。"
            f"任务结束，无需再调用任何工具。"
        )

    class SubmitRankingTool(BaseTool):
        name: str = "submit_ranking"
        description: str = (
            "提交你认为最相关的法条排序（最多 5 个 article_id，按相关性降序）。"
            "调用后任务即完成，无需再做其他事。"
            "ID 必须来自你之前检索返回的 (article_id=X) 字段，不要编造。"
        )
        args_schema: Type[BaseModel] = _SubmitRankingArgs

        def _run(self, article_ids: List[int], reason: str = "") -> str:
            return _do_submit(article_ids, reason)

        async def _arun(self, article_ids: List[int], reason: str = "") -> str:
            return _do_submit(article_ids, reason)

    return SubmitRankingTool()
