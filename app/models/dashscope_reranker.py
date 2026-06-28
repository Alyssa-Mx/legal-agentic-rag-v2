import logging
from typing import Any, Dict, List, Optional

import requests

from app.config.setting import (
    DEFAULT_RERANK_MODEL,
    RERANK_TOP_N,
    RERANK_URL,
    require_env,
)

logger = logging.getLogger(__name__)


class QwenDashScopeReranker:
    """调用 DashScope Rerank API（qwen3-rerank）对候选文档进行精排。"""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        api_key: Optional[str] = None,
        rerank_url: str = RERANK_URL,
        timeout: int = 60,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or require_env("DASHSCOPE_API_KEY")
        self.rerank_url = rerank_url
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = RERANK_TOP_N,
    ) -> List[tuple]:
        """
        对候选文档进行精排。

        Returns:
            [(原始索引, 相关性分数), ...] 按分数降序排列；
            失败时降级返回原始顺序（分数均为 0.0）。
        """
        if not documents:
            return []

        fallback = [(i, 0.0) for i in range(min(top_n, len(documents)))]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }

        try:
            resp = requests.post(
                self.rerank_url, headers=headers, json=payload, timeout=self.timeout
            )
        except Exception as e:
            logger.warning("[Reranker] 请求失败，降级返回原始顺序: %s", e)
            return fallback

        if resp.status_code >= 400:
            logger.warning("[Reranker] HTTP %d，降级返回原始顺序: %s", resp.status_code, resp.text[:200])
            return fallback

        data = resp.json()
        results = data.get("results") or data.get("output", {}).get("results")
        if results is None:
            logger.warning("[Reranker] 响应格式异常，降级返回原始顺序: %s", str(data)[:200])
            return fallback

        return [(item["index"], item["relevance_score"]) for item in results]
