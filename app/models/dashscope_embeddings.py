from typing import List, Optional

import requests
from langchain_core.embeddings import Embeddings

from app.config.setting import (
    DASHSCOPE_BASE_URL,
    DEFAULT_EMBED_MODEL,
    require_env,
)
from app.core.cache import embed_cache


class QwenDashScopeEmbeddings(Embeddings):
    """不依赖 openai SDK 的 DashScope Embeddings。"""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        api_key: Optional[str] = None,
        base_url: str = DASHSCOPE_BASE_URL,
        timeout: int = 60,
    ):
        self.model_name = model_name
        self.api_key = api_key or require_env("DASHSCOPE_API_KEY")
        self.base_url = base_url
        self.timeout = timeout

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_name, "input": inputs}

        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code >= 400:
            raise RuntimeError(f"DashScope embeddings error: {resp.status_code} {resp.text}")

        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量 embed 文档块。先查缓存命中部分，未命中的按 batch_size=10 调 API。
        重建索引时同 chunk 会大量复用，缓存收益显著。
        """
        results: List[Optional[List[float]]] = [None] * len(texts)
        miss_indices: List[int] = []
        miss_texts: List[str] = []

        for i, text in enumerate(texts):
            cached = embed_cache.get((self.model_name, text))
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        # DashScope embedding API 单次最多 10 条
        batch_size = 10
        for start in range(0, len(miss_texts), batch_size):
            batch = miss_texts[start: start + batch_size]
            vectors = self._embed(batch)
            for offset, vec in enumerate(vectors):
                idx = miss_indices[start + offset]
                results[idx] = vec
                embed_cache.put((self.model_name, texts[idx]), vec)

        return [vec for vec in results if vec is not None]

    def embed_query(self, text: str) -> List[float]:
        cached = embed_cache.get((self.model_name, text))
        if cached is not None:
            return cached
        vec = self._embed([text])[0]
        embed_cache.put((self.model_name, text), vec)
        return vec


def build_embeddings(model_name: str = DEFAULT_EMBED_MODEL) -> QwenDashScopeEmbeddings:
    return QwenDashScopeEmbeddings(model_name=model_name)
