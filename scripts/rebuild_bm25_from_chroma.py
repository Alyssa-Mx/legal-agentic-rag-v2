"""
从已有的 Chroma 向量库反查所有文档，重新构建 BM25 索引并 pickle 保存。

用途：当 Chroma 已构建好但 bm25_index.pkl 丢失时，避免重做 embedding。
保证 BM25 与 Chroma 基于同一批文档（同样的 article_id）。
"""
import os

import argparse
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config.setting import CHROMA_COLLECTION, CHROMA_DIR, DEFAULT_EMBED_MODEL
from app.models.dashscope_embeddings import build_embeddings
from app.retrieval.hybrid_retriever import tokenize


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroma", default=CHROMA_DIR)
    ap.add_argument("--collection", default=CHROMA_COLLECTION)
    ap.add_argument("--out", default="data/lecoqa/bm25_index.pkl")
    args = ap.parse_args()

    print(f"[load] Chroma dir={args.chroma}, collection={args.collection}")
    embeddings = build_embeddings(DEFAULT_EMBED_MODEL)
    vectordb = Chroma(
        collection_name=args.collection,
        embedding_function=embeddings,
        persist_directory=args.chroma,
    )
    raw = vectordb._collection.get(include=["documents", "metadatas"])
    docs_text = raw["documents"]
    metas = raw["metadatas"]
    print(f"[load] 从 Chroma 取出 {len(docs_text)} 条文档")

    docs = [
        Document(page_content=t, metadata=m or {})
        for t, m in zip(docs_text, metas)
    ]

    print("[bm25] 分词中...")
    tokenized = [tokenize(d.page_content) for d in docs]
    bm25 = BM25Okapi(tokenized)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump((bm25, docs), f)
    print(f"[bm25] 已保存: {args.out}  (共 {len(docs)} 条)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
