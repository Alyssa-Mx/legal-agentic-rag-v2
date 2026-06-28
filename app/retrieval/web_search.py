# =============================================================================
# 工具：Web Search（Serper）
# =============================================================================

import os
from typing import Any, List, Optional, Type

import requests
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class WebSearchArgs(BaseModel):
    query: str = Field(..., description="搜索关键词/问题（尽量简短明确）")
    k: int = Field(5, description="返回结果条数（建议 3-10）")


class WebSearchTool(BaseTool):
    """联网搜索工具（Serper）。

    注意：如果不设置 SERPER_API_KEY，会自动禁用（返回提示信息）。
    """

    name: str = "web_search"
    description: str = (
        "联网搜索工具：用于查询本地知识库之外的最新信息/百科/新闻/公开资料。"
        "输入 query，输出若干条搜索结果摘要（含标题/链接/片段）。"
    )
    args_schema: Type[BaseModel] = WebSearchArgs

    serper_api_key: Optional[str] = Field(default=None, exclude=True)

    def __init__(self, serper_api_key: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")

    def _run(self, query: str, k: int = 5) -> str:
        if not self.serper_api_key:
            return (
                "web_search 未启用：未检测到 SERPER_API_KEY。\n"
                "如需联网搜索，请设置环境变量 SERPER_API_KEY。"
            )

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": k}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
        except Exception as e:
            return f"[web_search error] request failed: {e}"

        if resp.status_code >= 400:
            return f"[web_search error] {resp.status_code}: {resp.text}"

        data = resp.json()
        organic = (data.get("organic") or [])[:k]

        lines: List[str] = []
        for i, item in enumerate(organic, 1):
            title = item.get("title", "")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            lines.append(f"[{i}] {title}\n{link}\n{snippet}")

        return "\n\n".join(lines) if lines else "No results."

    async def _arun(self, query: str, k: int = 5) -> str:
        return self._run(query=query, k=k)
