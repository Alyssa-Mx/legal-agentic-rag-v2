"""
一次性脚本：把 LeCoQA 法律条文语料写入 Chroma 向量库，并持久化 BM25 索引。

使用方式：
    python scripts/build_kb.py                            # 使用默认路径
    python scripts/build_kb.py --corpus data/lecoqa/corpus.json

只需运行一次。之后 evaluate.py 直接加载已有索引，无需重新 embedding。
"""


# 你说的场景：50 条测试题 + 总共 5000 条语料
# python scripts/build_kb.py --sample-test 50 --sample-corpus 5000

import argparse
import json
import os
import pickle
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma

from app.config.setting import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    DEFAULT_EMBED_MODEL,
)
from app.models.dashscope_embeddings import build_embeddings
from app.retrieval.loader import load_lecoqa_corpus
from app.retrieval.hybrid_retriever import tokenize

# ── 默认路径 ──────────────────────────────────────────────────────────────────
DEFAULT_CORPUS   = "data/lecoqa/corpus.jsonl"
DEFAULT_TEST     = "data/lecoqa/test.json"
DEFAULT_BM25_OUT = "data/lecoqa/bm25_index.pkl"
EMBED_BATCH_SIZE = 10   # DashScope embedding API 单次上限 10 条


def sample_corpus(all_docs, test_path, sample_test: int, sample_corpus: int, seed: int = 42):
    """
    从完整语料中采样，保证测试题的参考法条一定在库里。

    步骤：
      1. 读取测试集，取前 sample_test 条，收集其 match_id（必选集合）
      2. 按 match_id 从 all_docs 中找出必选文档
      3. 从剩余文档中随机补齐，总量不超过 sample_corpus 条
    """
    # 读测试集，收集必选 match_id
    with open(test_path, encoding="utf-8") as f:
        test_data = json.load(f)
    sampled_tests = test_data[:sample_test]
    required_ids: set = set()
    for item in sampled_tests:
        required_ids.update(item.get("match_id", []))
    print(f"[Sample] 取前 {sample_test} 条测试题，涉及 {len(required_ids)} 个必选法条")

    # 按 id 分类
    id_to_doc = {doc.metadata["article_id"]: doc for doc in all_docs}
    required_docs = [id_to_doc[i] for i in required_ids if i in id_to_doc]
    missing = required_ids - id_to_doc.keys()
    if missing:
        print(f"[Sample] 警告：{len(missing)} 个 match_id 在语料中找不到: {list(missing)[:5]}...")

    # 随机补齐
    remaining = [doc for doc in all_docs if doc.metadata["article_id"] not in required_ids]
    fill_count = max(0, sample_corpus - len(required_docs))
    random.seed(seed)
    random.shuffle(remaining)
    sampled_docs = required_docs + remaining[:fill_count]

    print(f"[Sample] 必选 {len(required_docs)} 条 + 随机补齐 {len(remaining[:fill_count])} 条 = 共 {len(sampled_docs)} 条")
    return sampled_docs, sampled_tests


def build_chroma(docs, embeddings, chroma_dir, collection):
    """分批写入 Chroma，避免单次 API 请求超时。"""
    total = len(docs)
    print(f"\n[Chroma] 开始写入，共 {total} 条法律条文，批大小 {EMBED_BATCH_SIZE}")
    vectordb = None
    for i in range(0, total, EMBED_BATCH_SIZE):
        batch = docs[i: i + EMBED_BATCH_SIZE]
        if vectordb is None:
            vectordb = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=chroma_dir,
                collection_name=collection,
            )
        else:
            vectordb.add_documents(batch)
        done = min(i + EMBED_BATCH_SIZE, total)
        print(f"  已写入 {done}/{total} ({done/total*100:.1f}%)", end="\r")
        time.sleep(0.1)   # 轻微限速，防止 API 限流
    print(f"\n[Chroma] 写入完成，持久化目录: {chroma_dir}")
    return vectordb


def build_bm25(docs, out_path):
    """构建 BM25 索引并用 pickle 持久化，供 evaluate.py 复用。"""
    print(f"\n[BM25] 开始分词，共 {len(docs)} 条...")
    tokenized = [tokenize(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump((bm25, docs), f)
    print(f"[BM25] 索引已保存: {out_path}")
    return bm25


def main():
    parser = argparse.ArgumentParser(description="构建 LeCoQA 法律知识库")
    parser.add_argument("--corpus",        default=DEFAULT_CORPUS,    help="语料路径（JSON / JSONL）")
    parser.add_argument("--test",          default=DEFAULT_TEST,      help="测试集路径（供采样时读取 match_id）")
    parser.add_argument("--bm25",          default=DEFAULT_BM25_OUT,  help="BM25 索引输出路径")
    parser.add_argument("--chroma",        default=CHROMA_DIR,        help="Chroma 持久化目录")
    parser.add_argument("--collection",    default=CHROMA_COLLECTION, help="Chroma collection 名")
    parser.add_argument("--need-chunk",    action="store_true",       help="是否对文档分块（法律条文通常不需要）")
    parser.add_argument("--chunk-size",    type=int, default=500,     help="chunk 大小，默认 500（--need-chunk 时生效）")
    parser.add_argument("--chunk-overlap", type=int, default=80,      help="chunk 重叠，默认 80（--need-chunk 时生效）")
    parser.add_argument("--sample-test",   type=int, default=None,    help="只取前 N 条测试题（None = 全量）")
    parser.add_argument("--sample-corpus", type=int, default=None,    help="知识库最大条数，必选法条 + 随机补齐（None = 全量）")
    args = parser.parse_args()

    # ── 加载语料 ──────────────────────────────────────────────────────────────
    print(f"[Loader] 加载语料: {args.corpus}")
    docs = load_lecoqa_corpus(args.corpus)
    print(f"[Loader] 共加载 {len(docs)} 条法律条文")

    # ── 采样（可选）──────────────────────────────────────────────────────────
    if args.sample_test is not None or args.sample_corpus is not None:
        n_test   = args.sample_test   or len(docs)
        n_corpus = args.sample_corpus or len(docs)
        docs, _ = sample_corpus(docs, args.test, n_test, n_corpus)

    # ── 分块（可选）──────────────────────────────────────────────────────────
    if args.need_chunk:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        docs = splitter.split_documents(docs)
        print(f"[Splitter] chunk 后共 {len(docs)} 个片段（size={args.chunk_size}, overlap={args.chunk_overlap}）")
    else:
        print("[Splitter] 跳过分块（每条法律条文作为完整片段）")
    if not docs:
        print("ERROR: 语料为空，请检查文件路径和格式。")
        sys.exit(1)

    # ── 构建 Chroma 向量库 ────────────────────────────────────────────────────
    embeddings = build_embeddings(DEFAULT_EMBED_MODEL)
    build_chroma(docs, embeddings, args.chroma, args.collection)

    # ── 构建 BM25 索引 ────────────────────────────────────────────────────────
    build_bm25(docs, args.bm25)

    print("\n✓ 知识库构建完成，可运行 scripts/evaluate.py 开始评估。")


if __name__ == "__main__":
    main()
