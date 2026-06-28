"""单库版分治检索：多子问题各自 hybrid 检索后合并（无 8 桶）。"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from app.retrieval.hybrid_retriever import HybridRerankRetriever


def _doc_key(doc: Document) -> str:
    aid = doc.metadata.get("article_id")
    if aid is not None:
        return f"id:{aid}"
    return f"content:{doc.page_content[:80]}"


def docs_to_article_ids(docs: List[Document]) -> List[int]:
    ids: List[int] = []
    seen = set()
    for d in docs:
        aid = d.metadata.get("article_id")
        if aid is None:
            continue
        aid = int(aid)
        if aid in seen:
            continue
        seen.add(aid)
        ids.append(aid)
    return ids


def retrieve_plan_single(
    retriever: HybridRerankRetriever,
    sub_queries: List[str],
    *,
    per_query_top_n_multi: int = 2,
    final_top_n: int = 5,
) -> Tuple[List[Document], List[int]]:
    """
    每个子问题独立走 HybridRerankRetriever（emb+BM25+RRF+rerank），
    按 (hit_count, max_rerank_score) 合并去重，最终截取前 final_top_n 篇。
    """
    if not sub_queries:
        return [], []

    per_query_top_n = final_top_n if len(sub_queries) == 1 else per_query_top_n_multi
    saved_top_n = retriever.rerank_top_n

    per_query_docs: List[List[Document]] = []
    try:
        for q in sub_queries:
            retriever.rerank_top_n = per_query_top_n
            per_query_docs.append(list(retriever.invoke(q)))
    finally:
        retriever.rerank_top_n = saved_top_n

    agg: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for docs in per_query_docs:
        for doc in docs:
            key = _doc_key(doc)
            score = float(doc.metadata.get("rerank_score") or 0.0)
            if key not in agg:
                agg[key] = {"doc": doc, "hit_count": 1, "max_score": score}
                order.append(key)
            else:
                agg[key]["hit_count"] += 1
                if score > agg[key]["max_score"]:
                    agg[key]["max_score"] = score
                    agg[key]["doc"] = doc

    ranked_keys = sorted(
        order,
        key=lambda k: (agg[k]["hit_count"], agg[k]["max_score"]),
        reverse=True,
    )
    result: List[Document] = []
    for k in ranked_keys:
        doc = agg[k]["doc"]
        doc.metadata["hit_count"] = agg[k]["hit_count"]
        result.append(doc)

    result = result[:final_top_n]
    return result, docs_to_article_ids(result)
