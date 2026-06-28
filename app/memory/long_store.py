import logging
import time
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from .schemas import MemoryConfig, MemoryEntry

logger = logging.getLogger(__name__)


# ======================================================
# 3. 长期记忆 - 存储
# ======================================================
class LongTermMemoryStore:
    """
    长期记忆存储层：基于 Chroma 向量库。

    双写策略：
        - 所有记忆写入向量库（持久化，支持语义检索）
        - profile / preference 类型同时写入结构化字典（快速读取用户画像）

    重启恢复：初始化时从向量库重新加载结构化记忆，保证内存与磁盘一致。
    """

    def __init__(self, embeddings, config: MemoryConfig, session_id: str = "default") -> None:
        self._config = config
        self._session_id = session_id
        self._vectorstore = Chroma(
            collection_name=config.MEMORY_COLLECTION,
            embedding_function=embeddings,
            persist_directory=config.MEMORY_CHROMA_DIR,
        )
        self._structured: Dict[str, MemoryEntry] = {}
        self._reload_structured()

    def _reload_structured(self) -> None:
        """启动时从向量库恢复当前 session 的 profile / preference 条目到内存字典。"""
        try:
            raw = self._vectorstore.get(
                where={"$and": [
                    {"memory_type": {"$in": ["profile", "preference"]}},
                    {"session_id": self._session_id},
                ]},
                include=["documents", "metadatas"],
            )
            docs = raw.get("documents") or []
            metas = raw.get("metadatas") or []
            for content, meta in zip(docs, metas):
                entry = MemoryEntry(
                    content=content,
                    memory_type=meta.get("memory_type", "profile"),
                    timestamp=meta.get("timestamp", time.time()),
                    confidence=meta.get("confidence", 1.0),
                    source_turn=meta.get("source_turn", 0),
                )
                key = f"{entry.memory_type}:{hash(content) & 0xFFFFFFFF:08x}"
                self._structured[key] = entry
            if self._structured:
                logger.info("[LongStore] 从向量库恢复 %d 条结构化记忆", len(self._structured))
        except Exception as e:
            logger.warning("[LongStore] 恢复结构化记忆失败（首次启动属正常）: %s", e)

    def add(self, entry: MemoryEntry) -> None:
        """写入一条记忆：同步写向量库，profile/preference 额外写结构化字典。"""
        if entry.memory_type in ("profile", "preference"):
            key = f"{entry.memory_type}:{hash(entry.content) & 0xFFFFFFFF:08x}"
            self._structured[key] = entry

        doc = Document(
            page_content=entry.content,
            metadata={
                "memory_type": entry.memory_type,
                "timestamp": entry.timestamp,
                "confidence": entry.confidence,
                "source_turn": entry.source_turn,
                "session_id": self._session_id,
            },
        )
        self._vectorstore.add_documents([doc])
        logger.debug("[LongStore] 写入记忆 type=%s len=%d", entry.memory_type, len(entry.content))

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """语义检索当前 session 的相关记忆，按相关性阈值过滤。"""
        try:
            results = self._vectorstore.similarity_search_with_relevance_scores(
                query, k=top_k,
                filter={"session_id": self._session_id},
            )
            threshold = self._config.MEMORY_RELEVANCE_THRESHOLD
            filtered = [doc for doc, score in results if score >= threshold]
            logger.debug(
                "[LongStore] 语义检索 query='%.30s' 召回 %d/%d 条（阈值 %.2f）",
                query, len(filtered), len(results), threshold,
            )
            return filtered
        except Exception as e:
            logger.warning("[LongStore] 语义检索失败: %s", e)
            return []

    def get_structured(self, memory_type: Optional[str] = None) -> List[MemoryEntry]:
        """获取当前 session 的结构化记忆（按写入时间倒序）。"""
        entries = list(self._structured.values())
        if memory_type:
            entries = [e for e in entries if e.memory_type == memory_type]
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)
