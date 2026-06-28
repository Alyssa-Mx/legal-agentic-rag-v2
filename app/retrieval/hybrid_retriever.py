import logging
from typing import Any, Dict, List

import jieba
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import create_retriever_tool
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from rank_bm25 import BM25Okapi

from app.config.setting import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COARSE_TOP_K,
    BM25_TOP_K,
    RERANK_TOP_N,
    RERANK_INPUT_TOP_K,
    RRF_K,
    KB_CONFIDENCE_THRESHOLD,
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MODEL,
)
from app.core.cache import retrieval_cache
from app.models.dashscope_embeddings import build_embeddings
from app.models.dashscope_reranker import QwenDashScopeReranker

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """中英文混合分词（用于 BM25）。使用 jieba 切词后过滤空白。"""
    return [t for t in jieba.lcut(text) if t.strip()]


def rrf_fusion(
    doc_lists: List[List[Document]],
    k: int = RRF_K,
) -> List[Document]:
    """
    Reciprocal Rank Fusion：将多路召回结果按排名融合去重。

    RRF_score(doc) = Σ 1 / (k + rank_i)
    其中 rank_i 是该文档在第 i 路召回中的排名（从 1 开始）。
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in doc_map:
                doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.__getitem__, reverse=True)
    result = []
    for key in sorted_keys:
        doc = doc_map[key]
        doc.metadata["rrf_score"] = scores[key]
        result.append(doc)
    return result


class HybridRerankRetriever(BaseRetriever):
    """
    三阶段混合检索器：
        1) BM25 稀疏召回   — 精确关键词匹配
        2) Embedding 稠密召回 — 语义相似度
        3) RRF 融合去重   → Reranker 精排 → 置信度门控过滤
    """

    vectorstore: Any = None
    reranker: Any = None
    bm25_index: Any = None
    bm25_corpus_docs: List[Document] = Field(default_factory=list)

    coarse_top_k: int = COARSE_TOP_K
    bm25_top_k: int = BM25_TOP_K
    rerank_top_n: int = RERANK_TOP_N
    rerank_input_top_k: int = RERANK_INPUT_TOP_K   # 融合后截到这么多再 rerank（0 = 不截）
    rrf_k: int = RRF_K
    confidence_threshold: float = KB_CONFIDENCE_THRESHOLD

    # ── 消融开关（默认全开，等同原行为，老调用方不受影响）────────────
    use_emb: bool = True
    use_bm25: bool = True
    use_rerank: bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # ── 整条流水线结果缓存（按 query + 关键参数 + 消融开关）─────────
        cache_key = (
            query,
            self.coarse_top_k,
            self.bm25_top_k,
            self.rerank_top_n,
            self.rerank_input_top_k,
            self.confidence_threshold,
            self.use_emb,
            self.use_bm25,
            self.use_rerank,
        )
        cached = retrieval_cache.get(cache_key)
        if cached is not None:
            logger.debug("[HybridRetriever] 检索缓存命中 query='%.40s'", query)
            return list(cached)

        # ── 通道 1：Embedding 稠密召回（可关闭）────────────────────────────
        emb_docs: List[Document] = []
        if self.use_emb and self.vectorstore is not None:
            emb_docs = self.vectorstore.similarity_search(query, k=self.coarse_top_k)

        # ── 通道 2：BM25 稀疏召回（可关闭）──────────────────────────────
        bm25_docs: List[Document] = []
        if self.use_bm25 and self.bm25_index is not None:
            query_tokens = tokenize(query)
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            top_indices = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[: self.bm25_top_k]
            bm25_docs = [self.bm25_corpus_docs[i] for i in top_indices if bm25_scores[i] > 0]

        logger.debug(
            "[HybridRetriever] Embedding 召回 %d 篇，BM25 召回 %d 篇 "
            "(use_emb=%s, use_bm25=%s)",
            len(emb_docs), len(bm25_docs), self.use_emb, self.use_bm25,
        )

        # ── 融合：双通道走 RRF；单通道用该通道顺序 ───────────────────────
        if self.use_emb and emb_docs and self.use_bm25 and bm25_docs:
            merged = rrf_fusion([emb_docs, bm25_docs], k=self.rrf_k)
        elif self.use_bm25 and bm25_docs:
            merged = list(bm25_docs)
        else:
            merged = list(emb_docs)

        if not merged:
            logger.warning("[HybridRetriever] 融合结果为空，query='%.40s'", query)
            retrieval_cache.put(cache_key, [])
            return []

        # ── Reranker 输入截断：让 RRF 排序真正起作用 ─────────────────────
        # 不截断的话 reranker 看到所有融合结果（~30 篇），会把 RRF 排序覆盖掉，
        # 等于浪费 RRF。截到 rerank_input_top_k 后，RRF 排名实际决定"哪些候选
        # 进入 reranker 候选池"，BM25 召回的"emb 漏掉的稀有相关条款"也能保留。
        pre_rerank_count = len(merged)
        if self.use_rerank and self.rerank_input_top_k and self.rerank_input_top_k > 0:
            merged = merged[: self.rerank_input_top_k]

        # ── Reranker 精排（可关闭）──────────────────────────────────────
        if self.use_rerank and self.reranker is not None:
            logger.debug(
                "[HybridRetriever] 融合后 %d 篇 → 截到 %d 篇送入 Reranker",
                pre_rerank_count, len(merged),
            )
            candidates = [doc.page_content for doc in merged]
            reranked_pairs = self.reranker.rerank(
                query=query,
                documents=candidates,
                top_n=self.rerank_top_n,
            )
            reranked_docs: List[Document] = []
            filtered_count = 0
            for idx, score in reranked_pairs:
                doc = merged[idx]
                doc.metadata["rerank_score"] = score
                if score >= self.confidence_threshold:
                    reranked_docs.append(doc)
                else:
                    filtered_count += 1
            scores_str = [f"{s:.4f}" for _, s in reranked_pairs]
            logger.info(
                "[HybridRetriever] 精排完成：%d 篇通过门控（阈值 %.2f），%d 篇被过滤。分数: %s",
                len(reranked_docs), self.confidence_threshold, filtered_count, scores_str,
            )
            result = reranked_docs
        else:
            # 无 reranker：直接截 top-N（按融合或 emb 顺序），同样应用门控对齐配置
            # 注意：无 rerank 分数时 confidence_threshold 不可比，消融评测里会显式置 0
            result = merged[: self.rerank_top_n]
            logger.debug(
                "[HybridRetriever] 跳过 Reranker，直接取前 %d 篇（共 %d 篇候选）",
                len(result), len(merged),
            )

        retrieval_cache.put(cache_key, list(result))
        return result


def build_retriever_tool(docs: List[Document]):
    """构建 BM25 索引 + Chroma 向量库 + Reranker，封装为混合检索 Tool。"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    # ── Embedding 向量库（稠密通道）──────────────────────────────────────
    embeddings = build_embeddings(DEFAULT_EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
    )

    # ── BM25 索引（稀疏通道）────────────────────────────────────────────
    tokenized_corpus = [tokenize(doc.page_content) for doc in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    logger.info("[BM25] 索引构建完成，共 %d 个文档块", len(tokenized_corpus))

    # ── Reranker ────────────────────────────────────────────────────────
    reranker = QwenDashScopeReranker(model_name=DEFAULT_RERANK_MODEL)

    retriever = HybridRerankRetriever(
        vectorstore=vectordb,
        reranker=reranker,
        bm25_index=bm25_index,
        bm25_corpus_docs=chunks,
        coarse_top_k=COARSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        rerank_top_n=RERANK_TOP_N,
        rrf_k=RRF_K,
        confidence_threshold=KB_CONFIDENCE_THRESHOLD,
    )

    return create_retriever_tool(
        retriever=retriever,
        name="retrieve_docs",
        description="从本地知识库中检索与问题相关的文档片段。输入应是简洁的检索 query。",
    )
