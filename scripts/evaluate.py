"""
检索消融评估脚本（单变体独立跑 + 改写预计算 + 断点续跑）。

═══════════════════════════════════════════════════════════════════════════
  设计原则
═══════════════════════════════════════════════════════════════════════════

1. 一变体一文件：每个变体单独跑，结果存到独立 JSON。中断只丢一个变体。
2. 改写预计算：先把所有题的 kb_query 算好存盘，4 个变体复用同一份 query，
   保证组件消融不被改写非确定性污染。
3. 断点续跑：变体内部逐题 flush，--resume 时跳过已完成 query_id。
4. TopK 公平性：所有变体统一 coarse_top_k=20, rerank_top_n=5（用户最终
   能看到的 K）。confidence_threshold 强制 0，避免门控让 ③④ 少返回。
   每题输出里记录 candidate_pool_size，供"③ rerank 看 20 篇 vs ④ 看 ~30 篇
   是否公平"这类问题的复盘。

═══════════════════════════════════════════════════════════════════════════
  4 个变体
═══════════════════════════════════════════════════════════════════════════

  ① emb                       Embedding 单通道，按相似度直接前 K
  ② emb_bm25_rrf              Embedding ⊕ BM25 → RRF 融合，按 RRF 分数前 K
  ③ emb_rerank                Embedding 单通道 → Reranker 精排前 N
  ④ emb_bm25_rrf_rerank       完整流水线（线上配置）

═══════════════════════════════════════════════════════════════════════════
  使用方式
═══════════════════════════════════════════════════════════════════════════

  .\\.venv\\Scripts\\Activate.ps1

  # 0) 预计算改写后的 kb_query（一次性，4 个变体共享）
  python scripts/evaluate.py --prepare-queries --limit 50 --output-dir results/abl_50

  # 1) 一个一个变体跑（中断不怕）
  python scripts/evaluate.py --variant emb                  --limit 50 --output-dir results/abl_50
  python scripts/evaluate.py --variant emb_bm25_rrf         --limit 50 --output-dir results/abl_50
  python scripts/evaluate.py --variant emb_rerank           --limit 50 --output-dir results/abl_50
  python scripts/evaluate.py --variant emb_bm25_rrf_rerank  --limit 50 --output-dir results/abl_50

  # 中途断了？同一条命令加 --resume，自动跳过已完成的 query_id
  python scripts/evaluate.py --variant emb_bm25_rrf_rerank --limit 50 --output-dir results/abl_50 --resume

  # 2) 跑完后聚合：对比表 + 边际增益 + Win/Lose
  python scripts/evaluate.py --compare results/abl_50

  # 关闭改写（默认开）：用于跑"无改写 vs 有改写"的正交对比
  python scripts/evaluate.py --variant emb_bm25_rrf_rerank --limit 50 --output-dir results/abl5_50_noWR --no-rewrite
"""

import os
import argparse
import io
import json
import math
import os
import pickle
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.vectorstores import Chroma

from app.config.setting import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    COARSE_TOP_K,
    BM25_TOP_K,
    RERANK_TOP_N,
    RERANK_INPUT_TOP_K,
    RRF_K,
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MODEL,
    DEFAULT_CHAT_MODEL,
)
from app.core.cache import retrieval_cache
from app.models.dashscope_embeddings import build_embeddings
from app.models.dashscope_reranker import QwenDashScopeReranker
from app.retrieval.hybrid_retriever import HybridRerankRetriever


# ── 内联的最小化 query 改写器（仅供检索消融评测使用，不进 agent 主流程） ──
# 主流程已切换为 ReAct，agent 自主写 query；这里仅为评测脚本对照实验保留
# 「按统一规则把口语问题改写为检索 query」的能力。
_KB_REWRITE_PROMPT = (
    "你是一个 query 改写专家。请把下面的用户问题改写成更适合知识库检索的短 query。\n"
    "要求：去口语化、保留核心关键词与法律实体、不引入新事实、简洁清晰。\n"
    "只输出改写后的 query 一行，不要任何解释或前后缀。\n\n"
    "原始问题：{question}\n\n改写后的 query："
)


class _EvalQueryRewriter:
    """评测专用的极简改写器（只做 kb_query 改写）。"""

    def __init__(self, llm) -> None:
        self._llm = llm

    def rewrite(self, question: str, memory_context: str = "") -> Dict[str, str]:
        prompt = _KB_REWRITE_PROMPT.format(question=question)
        try:
            resp = self._llm.invoke(prompt)
            kb_query = (getattr(resp, "content", "") or "").strip()
            kb_query = kb_query.splitlines()[0].strip() if kb_query else question
            return {"kb_query": kb_query or question}
        except Exception:
            return {"kb_query": question}


DEFAULT_TEST = "data/lecoqa/test.json"
DEFAULT_BM25 = "data/lecoqa/bm25_index.pkl"
# K 值跟 RERANK_TOP_N 联动（用户实际能看到的输出篇数 = RERANK_TOP_N）。
DEFAULT_K_VALUES = (1, 3, RERANK_TOP_N)
PRIMARY_K = RERANK_TOP_N         # 主表/边际增益/Win-Lose 用的"对外指标 K"
QUERIES_FILE = "_queries.json"


# ════════════════════════════════════════════════════════════════════════════
#  Section 1: 变体定义
# ════════════════════════════════════════════════════════════════════════════

VARIANT_SPECS = {
    "emb": {
        "label": "① emb",
        "use_emb": True, "use_bm25": False, "use_rerank": False,
    },
    "bm25": {
        "label": "①b bm25",
        "use_emb": False, "use_bm25": True, "use_rerank": False,
    },
    "emb_bm25_rrf": {
        "label": "② emb + bm25 + rrf",
        "use_emb": True, "use_bm25": True,  "use_rerank": False,
    },
    "emb_rerank": {
        "label": "③ emb + rerank",
        "use_emb": True, "use_bm25": False, "use_rerank": True,
    },
    "emb_bm25_rrf_rerank": {
        "label": "④ emb + bm25 + rrf + rerank",
        "use_emb": True, "use_bm25": True,  "use_rerank": True,
    },
}
VARIANT_ORDER = ["emb", "bm25", "emb_bm25_rrf", "emb_rerank", "emb_bm25_rrf_rerank"]


def build_retriever(
    bm25_path: str,
    variant: str,
    rerank_input_top_k: int = RERANK_INPUT_TOP_K,
) -> HybridRerankRetriever:
    spec = VARIANT_SPECS[variant]
    use_emb = spec.get("use_emb", True)
    vectordb = None
    if use_emb:
        embeddings = build_embeddings(DEFAULT_EMBED_MODEL)
        vectordb = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
    reranker = QwenDashScopeReranker(model_name=DEFAULT_RERANK_MODEL) if spec["use_rerank"] else None
    with open(bm25_path, "rb") as f:
        bm25_index, bm25_corpus_docs = pickle.load(f)
    return HybridRerankRetriever(
        vectorstore=vectordb,
        reranker=reranker,
        bm25_index=bm25_index,
        bm25_corpus_docs=bm25_corpus_docs,
        coarse_top_k=COARSE_TOP_K,
        bm25_top_k=BM25_TOP_K,
        rerank_top_n=RERANK_TOP_N,
        rerank_input_top_k=rerank_input_top_k,
        rrf_k=RRF_K,
        confidence_threshold=0.0,        # 消融评测：禁用门控
        use_emb=use_emb,
        use_bm25=spec["use_bm25"],
        use_rerank=spec["use_rerank"],
    )


# ════════════════════════════════════════════════════════════════════════════
#  Section 2: 检索指标 Hit@K / Recall@K / MRR / NDCG@K + 延迟分位
# ════════════════════════════════════════════════════════════════════════════

def dcg(relevances: List[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def compute_retrieval_metrics(
    retrieved: List[int], gold: Set[int], k_values: Tuple[int, ...] = DEFAULT_K_VALUES,
) -> Dict[str, float]:
    out: Dict[str, float] = {"retrieved_count": len(retrieved), "gold_count": len(gold)}
    if not gold:
        for k in k_values:
            out[f"hit@{k}"] = 0
            out[f"recall@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
        out["mrr"] = 0.0
        return out
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in gold:
            mrr = 1.0 / rank
            break
    out["mrr"] = round(mrr, 4)
    for k in k_values:
        topk = retrieved[:k]
        rels = [1 if d in gold else 0 for d in topk]
        ideal = dcg([1] * min(len(gold), k))
        out[f"hit@{k}"] = int(any(rels))
        out[f"recall@{k}"] = round(sum(rels) / len(gold), 4)
        out[f"ndcg@{k}"] = round((dcg(rels) / ideal) if ideal > 0 else 0.0, 4)
    return out


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return float(s[f] * (c - k) + s[c] * (k - f))


# ════════════════════════════════════════════════════════════════════════════
#  Section 3: I/O —— 改写预计算 + 单变体增量保存 + 断点续跑
# ════════════════════════════════════════════════════════════════════════════

def queries_path(output_dir: str) -> str:
    return os.path.join(output_dir, QUERIES_FILE)


def variant_path(output_dir: str, variant: str) -> str:
    return os.path.join(output_dir, f"{variant}.json")


def load_queries(output_dir: str) -> Optional[Dict[str, str]]:
    path = queries_path(output_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("queries", {})


def save_queries(output_dir: str, queries: Dict[str, str], rewrite_mode: str, limit: Optional[int]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = queries_path(output_dir)
    payload = {
        "rewrite_mode": rewrite_mode,
        "limit": limit,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "queries": queries,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_variant_results(output_dir: str, variant: str) -> Tuple[Dict, List[Dict]]:
    path = variant_path(output_dir, variant)
    if not os.path.exists(path):
        return {}, []
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("config", {}), payload.get("per_question", [])


def save_variant_results(
    output_dir: str, variant: str,
    config: Dict, per_question: List[Dict],
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = variant_path(output_dir, variant)
    payload = {
        "variant": variant,
        "label": VARIANT_SPECS[variant]["label"],
        "config": config,
        "summary": aggregate(per_question),
        "per_question": per_question,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ════════════════════════════════════════════════════════════════════════════
#  Section 4: 改写预计算（一次性）
# ════════════════════════════════════════════════════════════════════════════

def cmd_prepare_queries(args) -> None:
    """把所有题的 kb_query 预计算好存盘，4 个变体后续直接复用。"""
    print(f"[Prepare] 加载测试集: {args.test}")
    with open(args.test, encoding="utf-8") as f:
        test_data = json.load(f)
    if args.limit:
        test_data = test_data[: args.limit]
    print(f"[Prepare] 共 {len(test_data)} 条")

    if args.no_rewrite:
        queries = {str(s.get("query_id", i)): s["问题"] for i, s in enumerate(test_data)}
        save_queries(args.output_dir, queries, rewrite_mode="off", limit=args.limit)
        print(f"[Prepare] 不改写，原 query 已存到 {queries_path(args.output_dir)}")
        return

    from app.models.dashscope_chat import build_chat
    rewriter = _EvalQueryRewriter(build_chat(DEFAULT_CHAT_MODEL))
    print(f"[Prepare] 用 {DEFAULT_CHAT_MODEL} 改写 kb_query ...")

    queries: Dict[str, str] = {}
    # 支持续跑：已有的不重算
    existing = load_queries(args.output_dir) or {}
    for i, sample in enumerate(test_data):
        qid = str(sample.get("query_id", i))
        if qid in existing and existing[qid]:
            queries[qid] = existing[qid]
            continue
        question = sample["问题"]
        try:
            rw = rewriter.rewrite(question, memory_context="")
            kb_query = rw.get("kb_query") or question
        except Exception as e:
            print(f"  [WARN] qid={qid} 改写失败: {e}，用原题兜底")
            kb_query = question
        queries[qid] = kb_query

        if (i + 1) % 10 == 0 or (i + 1) == len(test_data):
            print(f"  改写进度 {i+1}/{len(test_data)}")
            save_queries(args.output_dir, queries, rewrite_mode="kb_query", limit=args.limit)
        time.sleep(args.sleep)

    save_queries(args.output_dir, queries, rewrite_mode="kb_query", limit=args.limit)
    print(f"[Prepare] 完成，{len(queries)} 条 kb_query 已存到 {queries_path(args.output_dir)}")


# ════════════════════════════════════════════════════════════════════════════
#  Section 5: 单变体跑（resume + flush）
# ════════════════════════════════════════════════════════════════════════════

def retrieve_with_pool(
    retriever: HybridRerankRetriever, query: str,
) -> Tuple[List[int], int, float]:
    """
    返回（按相关性降序的 article_id, rerank 输入池大小, 耗时秒）。
    pool_size 通过给 retriever 加一个临时记录变量取，无 rerank 时 = 输出长度。
    """
    t0 = time.time()
    try:
        docs = retriever.invoke(query)
    except Exception as e:
        print(f"  [WARN] 检索失败: {e}")
        return [], 0, time.time() - t0
    elapsed = time.time() - t0
    ids: List[int] = []
    seen: Set[int] = set()
    for d in docs:
        aid = d.metadata.get("article_id")
        if aid is None:
            continue
        aid = int(aid)
        if aid in seen:
            continue
        seen.add(aid)
        ids.append(aid)
    # pool_size 作为 sanity-check 信号：当前实现下若 use_rerank=True，
    # rerank 输入 = RRF 融合后或 emb top-K 的实际大小。我们用 len(docs) 作近似上限。
    return ids, len(docs), elapsed


def cmd_run_variant(args) -> None:
    variant = args.variant
    if variant not in VARIANT_SPECS:
        print(f"[ERROR] 未知变体: {variant}（可选: {list(VARIANT_SPECS)}）")
        sys.exit(1)

    # ── 加载 query：优先用预计算的；否则即时算/不改写 ───────────────────
    queries_map = load_queries(args.output_dir)
    if queries_map is None:
        print(f"[Variant {variant}] 未找到 {QUERIES_FILE}，将跟参数即时计算 query")
        queries_map = {}
    else:
        rw_mode = "kb_query" if not args.no_rewrite else "off"
        print(f"[Variant {variant}] 使用预计算的 query（{len(queries_map)} 条）")

    # ── 加载测试集 ───────────────────────────────────────────────────────
    print(f"[Variant {variant}] 加载测试集: {args.test}")
    with open(args.test, encoding="utf-8") as f:
        test_data = json.load(f)
    if args.limit:
        test_data = test_data[: args.limit]
    print(f"[Variant {variant}] 共 {len(test_data)} 条测试题")

    # ── 续跑 ──────────────────────────────────────────────────────────────
    existing_config, results = ({}, [])
    done_ids: Set[str] = set()
    if args.resume:
        existing_config, results = load_variant_results(args.output_dir, variant)
        done_ids = {str(r.get("query_id")) for r in results if r.get("query_id") is not None}
        if results:
            print(f"[Variant {variant}] 续跑：已有 {len(results)} 条，将跳过这些 query_id")

    # ── 构建 retriever ────────────────────────────────────────────────────
    rit = args.rerank_input_top_k if args.rerank_input_top_k is not None else RERANK_INPUT_TOP_K
    print(f"[Variant {variant}] 构建 retriever（BM25: {args.bm25}, "
          f"rerank_input_top_k={rit if rit > 0 else '不截断'}）...")
    retriever = build_retriever(args.bm25, variant, rerank_input_top_k=rit)

    # ── 若需即时改写且没有预计算 ─────────────────────────────────────────
    rewriter = None
    if not args.no_rewrite and not queries_map:
        from app.models.dashscope_chat import build_chat
        rewriter = _EvalQueryRewriter(build_chat(DEFAULT_CHAT_MODEL))
        print(f"[Variant {variant}] ⚠️  即时改写（建议先跑 --prepare-queries）")

    config = {
        "limit": args.limit,
        "rewrite_mode": "off" if args.no_rewrite else "kb_query",
        "coarse_top_k": COARSE_TOP_K,
        "bm25_top_k": BM25_TOP_K,
        "rerank_top_n": RERANK_TOP_N,
        "rerank_input_top_k": rit,
        "rrf_k": RRF_K,
        "embed_model": DEFAULT_EMBED_MODEL,
        "rerank_model": DEFAULT_RERANK_MODEL,
        "test_set": args.test,
        "started_at": existing_config.get("started_at") or datetime.now().isoformat(timespec="seconds"),
    }

    # ── 逐题 ──────────────────────────────────────────────────────────────
    consecutive_errors = 0
    early_stopped = False
    try:
        for i, sample in enumerate(test_data):
            qid = str(sample.get("query_id", i))
            if qid in done_ids:
                continue
            question = sample["问题"]
            gold_ids = set(sample["match_id"])

            # 决定 query
            if args.no_rewrite:
                query = question
            elif qid in queries_map:
                query = queries_map[qid]
            elif rewriter is not None:
                try:
                    rw = rewriter.rewrite(question, memory_context="")
                    query = rw.get("kb_query") or question
                except Exception:
                    query = question
                queries_map[qid] = query  # 顺手缓存
            else:
                query = question

            # 每题前清缓存：让本题在本变体的延迟是真实首跑值
            retrieval_cache.clear()

            print(f"  [{i+1}/{len(test_data)}] qid={qid}  {question[:36]}...", end="", flush=True)
            ranked_ids, pool_size, lat = retrieve_with_pool(retriever, query)
            metrics = compute_retrieval_metrics(ranked_ids, gold_ids)

            results.append({
                "query_id": qid,
                "question": question,
                "query_used": query,
                "gold_ids": list(gold_ids),
                "gold_count": len(gold_ids),
                "ranked_ids": ranked_ids,
                "retrieved_count": len(ranked_ids),
                "candidate_pool_size": pool_size,
                "metrics": metrics,
                "latency_s": round(lat, 3),
            })

            print(f"  hit@{PRIMARY_K}={metrics[f'hit@{PRIMARY_K}']}  "
                  f"recall@{PRIMARY_K}={metrics[f'recall@{PRIMARY_K}']:.2f}  "
                  f"mrr={metrics['mrr']:.2f}  pool={pool_size}  lat={lat:.2f}s")

            if not ranked_ids:
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    print(f"\n[!] 连续 5 次空召回，提前终止")
                    early_stopped = True
                    break
            else:
                consecutive_errors = 0

            if (i + 1) % args.flush_every == 0:
                save_variant_results(args.output_dir, variant, config, results)

            time.sleep(args.sleep)

    except KeyboardInterrupt:
        print(f"\n[!] 用户中断，已完成 {len(results)} 条")
        early_stopped = True
    finally:
        save_variant_results(args.output_dir, variant, config, results)
        # 顺便把改写过的 query 增量持久化（防止下次还要重新改写）
        if rewriter is not None and queries_map:
            existing_q = load_queries(args.output_dir) or {}
            existing_q.update(queries_map)
            save_queries(args.output_dir, existing_q, rewrite_mode="kb_query", limit=args.limit)
        print(f"\n[Variant {variant}] 结果已保存: {variant_path(args.output_dir, variant)}（{len(results)} 条）")

    # ── 单变体小结：屏幕 + .txt ──────────────────────────────────────────
    txt_path = os.path.join(args.output_dir, f"{variant}.txt")
    capture_and_save(txt_path, lambda: print_single_variant_summary(variant, results, config))
    if early_stopped:
        sys.exit(2)


# ════════════════════════════════════════════════════════════════════════════
#  Section 6: 聚合 / 打印
# ════════════════════════════════════════════════════════════════════════════

def aggregate(records: List[Dict]) -> Dict:
    n = len(records)
    if n == 0:
        return {"n": 0}
    out = {"n": n}
    for k in DEFAULT_K_VALUES:
        out[f"hit@{k}"] = round(sum(r["metrics"][f"hit@{k}"] for r in records) / n, 4)
        out[f"recall@{k}"] = round(sum(r["metrics"][f"recall@{k}"] for r in records) / n, 4)
        out[f"ndcg@{k}"] = round(sum(r["metrics"][f"ndcg@{k}"] for r in records) / n, 4)
    out["mrr"] = round(sum(r["metrics"]["mrr"] for r in records) / n, 4)
    lats = [r["latency_s"] for r in records]
    out["lat_p50"] = round(percentile(lats, 50), 3)
    out["lat_p95"] = round(percentile(lats, 95), 3)
    out["lat_avg"] = round(sum(lats) / n, 3)
    pools = [r.get("candidate_pool_size", 0) for r in records]
    out["pool_avg"] = round(sum(pools) / n, 1)
    return out


def print_single_variant_summary(variant: str, records: List[Dict], config: Optional[Dict] = None) -> None:
    a = aggregate(records)
    if a["n"] == 0:
        print("（无结果）")
        return
    label = VARIANT_SPECS[variant]["label"]
    print(f"\n{'='*72}")
    print(f"  {label}  ({a['n']} 条)")
    print(f"{'='*72}")
    if config:
        print(f"  rewrite_mode  : {config.get('rewrite_mode', '?')}")
        print(f"  retriever cfg : coarse_top_k={config.get('coarse_top_k')}, "
              f"bm25_top_k={config.get('bm25_top_k')}, "
              f"rerank_input_top_k={config.get('rerank_input_top_k')}, "
              f"rerank_top_n={config.get('rerank_top_n')}")
        print(f"  test_set      : {config.get('test_set')}")
        print(f"  started_at    : {config.get('started_at')}")
        print(f"  {'-'*68}")
    for k in DEFAULT_K_VALUES:
        print(f"  Hit@{k}: {a[f'hit@{k}']:.3f}    Recall@{k}: {a[f'recall@{k}']:.3f}    NDCG@{k}: {a[f'ndcg@{k}']:.3f}")
    print(f"  MRR    : {a['mrr']:.3f}")
    print(f"  延迟    : P50={a['lat_p50']:.2f}s  P95={a['lat_p95']:.2f}s  avg={a['lat_avg']:.2f}s")
    print(f"  候选池  : 平均 {a['pool_avg']:.1f} 篇")
    print(f"{'='*72}\n")


def capture_and_save(txt_path: str, render_fn: Callable[[], None]) -> None:
    """
    执行 render_fn（里面用 print 输出格式化文本），同时把内容打到屏幕和文件。
    用 StringIO + redirect_stdout 捕获，避免重写每个 print 函数返回字符串。
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        render_fn()
    text = buf.getvalue()
    sys.stdout.write(text)            # 屏幕显示
    sys.stdout.flush()
    os.makedirs(os.path.dirname(os.path.abspath(txt_path)), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"# 生成时间: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(text)
    sys.stdout.write(f"[Output] 文字摘要已保存: {txt_path}\n")


def cmd_compare(args) -> None:
    """读 output_dir 下的所有 *.json 变体结果，输出对比表。"""
    out_dir = args.compare
    if not os.path.isdir(out_dir):
        print(f"[ERROR] 目录不存在: {out_dir}")
        sys.exit(1)

    # 按预定顺序读取存在的变体
    loaded: Dict[str, Dict] = {}
    for v in VARIANT_ORDER:
        p = variant_path(out_dir, v)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            loaded[v] = json.load(f)

    if not loaded:
        print(f"[ERROR] {out_dir} 下没有任何变体结果（找不到 emb.json / emb_bm25_rrf.json ...）")
        sys.exit(1)

    variants = list(loaded.keys())

    # 对齐题数：取所有变体共有的 query_id 子集
    common_ids: Optional[Set[str]] = None
    for v in variants:
        ids = {r["query_id"] for r in loaded[v]["per_question"]}
        common_ids = ids if common_ids is None else (common_ids & ids)
    if not common_ids:
        print("[ERROR] 各变体之间没有公共 query_id，无法对比")
        sys.exit(1)

    # 截取公共子集，重新算 summary
    per_variant: Dict[str, List[Dict]] = {}
    summary: Dict[str, Dict] = {}
    for v in variants:
        recs = [r for r in loaded[v]["per_question"] if r["query_id"] in common_ids]
        per_variant[v] = recs
        summary[v] = aggregate(recs)

    def _render_all() -> None:
        print(f"[Compare] 找到 {len(variants)} 个变体: {', '.join(variants)}")
        print(f"[Compare] 公共题数: {len(common_ids)}")
        print_main_table(summary)
        print_full_metric_table(summary)
        print_marginal_lift(summary, variants)
        print_winlose(per_variant, variants)
        print_buckets(per_variant, variants)
        print()

    txt_path = os.path.join(out_dir, "_compare.txt")
    capture_and_save(txt_path, _render_all)


def print_main_table(agg: Dict[str, Dict]) -> None:
    K = PRIMARY_K
    print(f"\n{'='*102}")
    print(f"  {'变体':<32} {f'Hit@{K}':>7} {f'Recall@{K}':>10} {f'NDCG@{K}':>8} {'MRR':>7} "
          f"{'P50':>7} {'P95':>7} {'pool':>7}")
    print(f"  {'-'*100}")
    for v, a in agg.items():
        if a.get("n", 0) == 0:
            continue
        label = VARIANT_SPECS[v]["label"]
        print(f"  {label:<32} {a[f'hit@{K}']:>7.3f} {a[f'recall@{K}']:>10.3f} {a[f'ndcg@{K}']:>8.3f} "
              f"{a['mrr']:>7.3f} {a['lat_p50']:>5.2f}s {a['lat_p95']:>5.2f}s {a['pool_avg']:>7.1f}")
    print(f"{'='*102}")


def print_full_metric_table(agg: Dict[str, Dict]) -> None:
    print(f"\n  ── 全 K 指标矩阵 ──")
    header = f"  {'变体':<32}"
    for k in DEFAULT_K_VALUES:
        header += f" {f'Hit@{k}':>7} {f'Rec@{k}':>7} {f'NDCG@{k}':>8}"
    header += f" {'MRR':>7}"
    print(header)
    print(f"  {'-'*(len(header)-2)}")
    for v, a in agg.items():
        if a.get("n", 0) == 0:
            continue
        row = f"  {VARIANT_SPECS[v]['label']:<32}"
        for k in DEFAULT_K_VALUES:
            row += f" {a[f'hit@{k}']:>7.3f} {a[f'recall@{k}']:>7.3f} {a[f'ndcg@{k}']:>8.3f}"
        row += f" {a['mrr']:>7.3f}"
        print(row)


def print_marginal_lift(agg: Dict[str, Dict], variants: List[str]) -> None:
    if len(variants) < 2:
        return
    K = PRIMARY_K
    print(f"\n  ── 边际增益（相对前一个变体）──")
    print(f"  {'对比':<46} {f'ΔHit@{K}':>9} {f'ΔRecall@{K}':>11} {'ΔMRR':>8} {'ΔP95':>9}")
    print(f"  {'-'*88}")
    sign = lambda x: f"+{x:.3f}" if x >= 0 else f"{x:.3f}"
    base = variants[0]
    for cur in variants[1:]:
        a, b = agg[cur], agg[base]
        if a.get("n", 0) == 0 or b.get("n", 0) == 0:
            continue
        d_hit = a[f"hit@{K}"] - b[f"hit@{K}"]
        d_rec = a[f"recall@{K}"] - b[f"recall@{K}"]
        d_mrr = a["mrr"] - b["mrr"]
        d_lat = a["lat_p95"] - b["lat_p95"]
        cmp_label = f"{VARIANT_SPECS[cur]['label']} vs {VARIANT_SPECS[base]['label']}"
        print(f"  {cmp_label:<46} {sign(d_hit):>9} {sign(d_rec):>11} {sign(d_mrr):>8}  {sign(d_lat)+'s':>8}")
        base = cur

    if len(variants) >= 2:
        first, last = variants[0], variants[-1]
        a, b = agg[last], agg[first]
        d_hit = a[f"hit@{K}"] - b[f"hit@{K}"]
        d_rec = a[f"recall@{K}"] - b[f"recall@{K}"]
        d_mrr = a["mrr"] - b["mrr"]
        d_lat = a["lat_p95"] - b["lat_p95"]
        print(f"  {'-'*88}")
        print(f"  {'累计 ' + VARIANT_SPECS[last]['label'] + ' vs ' + VARIANT_SPECS[first]['label']:<46} "
              f"{sign(d_hit):>9} {sign(d_rec):>11} {sign(d_mrr):>8}  {sign(d_lat)+'s':>8}")


def print_winlose(per_variant: Dict[str, List[Dict]], variants: List[str]) -> None:
    K = PRIMARY_K
    print(f"\n  ── Win/Lose 矩阵（按 hit@{K}；行胜列）──")
    header = f"  {'':<32}" + "".join(f" {VARIANT_SPECS[v]['label'][:8]:>10}" for v in variants)
    print(header)
    by_qid = {v: {r["query_id"]: r for r in per_variant[v]} for v in variants}
    qids = list(by_qid[variants[0]].keys())
    for vi in variants:
        row = f"  {VARIANT_SPECS[vi]['label']:<32}"
        for vj in variants:
            if vi == vj:
                row += f" {'—':>10}"
                continue
            wins = sum(
                1 for q in qids
                if by_qid[vi].get(q, {}).get("metrics", {}).get(f"hit@{K}") == 1
                and by_qid[vj].get(q, {}).get("metrics", {}).get(f"hit@{K}") == 0
            )
            row += f" {wins:>10}"
        print(row)


def print_buckets(per_variant: Dict[str, List[Dict]], variants: List[str]) -> None:
    K = PRIMARY_K
    print(f"\n  ── 分桶 Hit@{K}（单法规 vs 多法规）──")
    print(f"  {'变体':<32} {f'单法规 Hit@{K}':>16} {f'多法规 Hit@{K}':>16}")
    print(f"  {'-'*70}")
    for v in variants:
        recs = per_variant[v]
        single = [r for r in recs if r["gold_count"] == 1]
        multi = [r for r in recs if r["gold_count"] > 1]
        s_hit = sum(r["metrics"][f"hit@{K}"] for r in single) / len(single) if single else 0
        m_hit = sum(r["metrics"][f"hit@{K}"] for r in multi) / len(multi) if multi else 0
        print(f"  {VARIANT_SPECS[v]['label']:<32} "
              f"{s_hit:>10.3f} ({len(single):>3}) "
              f"{m_hit:>10.3f} ({len(multi):>3})")


# ════════════════════════════════════════════════════════════════════════════
#  Section 7: 入口
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="LeCoQA 检索消融评估")
    parser.add_argument("--test", default=DEFAULT_TEST, help="测试集 JSON 路径")
    parser.add_argument("--bm25", default=DEFAULT_BM25, help="BM25 索引 pkl 路径")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条")
    parser.add_argument("--output-dir", default=None,
                        help="结果目录（每变体一个 JSON），跑变体时必须给")
    parser.add_argument("--sleep", type=float, default=0.2, help="每题间隔秒")

    # 三种模式互斥：--prepare-queries / --variant / --compare
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-queries", action="store_true",
                      help="预计算所有题的 kb_query，存到 <output-dir>/_queries.json")
    mode.add_argument("--variant", default=None, choices=list(VARIANT_SPECS.keys()),
                      help="跑指定的单个变体，结果存到 <output-dir>/<variant>.json")
    mode.add_argument("--compare", default=None, metavar="DIR",
                      help="读 DIR 下所有变体结果，输出对比表")

    parser.add_argument("--no-rewrite", action="store_true",
                        help="禁用改写，直接用原题作 query（默认开改写）")
    parser.add_argument("--resume", action="store_true",
                        help="续跑：跳过 <output-dir>/<variant>.json 已完成的 query_id")
    parser.add_argument("--flush-every", type=int, default=5,
                        help="每 N 条写一次磁盘（默认 5）")
    parser.add_argument("--rerank-input-top-k", type=int, default=None,
                        help=f"RRF 融合后送给 reranker 的最大候选数（默认读 setting={RERANK_INPUT_TOP_K}；"
                             "传 0 表示不截断；传 20 可让 ③④ reranker 输入池严格对齐）")

    args = parser.parse_args()

    if args.compare:
        cmd_compare(args)
        return

    if not args.output_dir:
        parser.error("--prepare-queries / --variant 模式必须给 --output-dir")

    if args.prepare_queries:
        cmd_prepare_queries(args)
        return

    cmd_run_variant(args)


if __name__ == "__main__":
    main()
