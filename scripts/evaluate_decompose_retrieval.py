"""
方案 B：单库 + LLM 拆解 + 分治 hybrid 检索（无 8 桶、无域路由）。

默认每次 run **重新调用 LLM 拆解**（不固定 plan）。仅当显式传入 --plan-cache 时才复用/写入缓存。

单次（--repeats 1）:
    python scripts/evaluate_decompose_retrieval.py --limit 50 \\
        --output results/compare_50/arm_b_once.json

稳定性（--repeats 3，每题 3 次自由拆解）:
    python scripts/evaluate_decompose_retrieval.py --limit 50 --repeats 3 \\
        --output results/compare_50/arm_b_stability_50x3.json --resume
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config.setting import (
    BM25_TOP_K,
    COARSE_TOP_K,
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBED_MODEL,
    DEFAULT_RERANK_MODEL,
    RERANK_INPUT_TOP_K,
    RERANK_TOP_N,
    RRF_K,
)
from app.models.dashscope_chat import build_chat
from app.retrieval.decompose import LegalQueryDecomposer
from app.retrieval.plan_retrieve import retrieve_plan_single
from scripts.evaluate import (
    DEFAULT_BM25,
    DEFAULT_TEST,
    PRIMARY_K,
    build_retriever,
    compute_retrieval_metrics,
    percentile,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("eval_decompose")
logger.setLevel(logging.INFO)


def _std(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    return math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))


def load_plan_cache(path: Optional[str]) -> Dict[str, List[str]]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in raw.items():
        if isinstance(v, list) and v and isinstance(v[0], str):
            out[k] = v
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out[k] = [str(p.get("query", "")).strip() for p in v if p.get("query")]
    return out


def save_plan_cache(path: str, cache: Dict[str, List[str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_report(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_report(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def done_run_keys(runs: List[dict]) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for r in runs:
        rid = r.get("run_id", "")
        if rid.startswith("q") and "_run" in rid:
            try:
                qid = int(rid.split("_run")[0][1:])
                run_idx = int(rid.split("_run")[1])
                out.add((qid, run_idx))
            except ValueError:
                pass
    return out


def done_question_ids(per_question: List[dict]) -> Set[int]:
    return {r["query_id"] for r in per_question if r.get("query_id") is not None}


def build_per_question_stability(runs_for_q: List[dict]) -> dict:
    hits = [r["hit@5"] for r in runs_for_q]
    recalls = [r["recall@5"] for r in runs_for_q]
    mrrs = [r["mrr"] for r in runs_for_q]
    return {
        "hit@5": {"values": hits, "mean": round(sum(hits) / len(hits), 4), "std": round(_std(hits), 4)},
        "recall@5": {"values": recalls, "mean": round(sum(recalls) / len(recalls), 4)},
        "mrr": {"values": mrrs, "mean": round(sum(mrrs) / len(mrrs), 4)},
        "pass@3": int(any(hits)),
        "always_hit@3": int(all(hits)),
        "never_hit@3": int(not any(hits)),
        "flaky@3": int(any(hits) and not all(hits)),
        "sub_queries_per_run": [r.get("n_sub_queries", 0) for r in runs_for_q],
    }


def build_global_stability_summary(
    per_q: Dict[int, dict], runs: List[dict], n_questions: int, runs_per_q: int,
) -> dict:
    n = len(per_q)
    if n == 0:
        return {}
    pass3 = sum(s["pass@3"] for s in per_q.values()) / n
    always = sum(s["always_hit@3"] for s in per_q.values()) / n
    never = sum(s["never_hit@3"] for s in per_q.values()) / n
    flaky = sum(s["flaky@3"] for s in per_q.values()) / n
    mean_hit = sum(s["hit@5"]["mean"] for s in per_q.values()) / n
    mean_recall = sum(s["recall@5"]["mean"] for s in per_q.values()) / n
    mean_mrr = sum(s["mrr"]["mean"] for s in per_q.values()) / n

    def _run_mean(key: str) -> float:
        if not runs:
            return 0.0
        return round(sum(float(r.get(key, 0)) for r in runs) / len(runs), 4)

    lats = [r["latency_s"] for r in runs if r.get("latency_s") is not None]
    return {
        "n_questions": n_questions,
        "runs_per_question": runs_per_q,
        "completed_questions": n,
        "completed_runs": len(runs),
        "total_runs_target": n_questions * runs_per_q,
        "mean_hit@5": round(mean_hit, 4),
        "mean_recall@5": round(mean_recall, 4),
        "mean_mrr": round(mean_mrr, 4),
        "mean_ndcg@5": _run_mean("ndcg@5"),
        "pass@3_rate": round(pass3, 4),
        "always_hit@3_rate": round(always, 4),
        "never_hit@3_rate": round(never, 4),
        "flaky@3_rate": round(flaky, 4),
        "lat_avg_s": round(sum(lats) / len(lats), 2) if lats else 0,
        "avg_sub_queries": round(
            sum(r.get("n_sub_queries", 0) for r in runs) / len(runs), 2,
        ) if runs else 0,
    }


def rebuild_stability_summaries(report: dict, runs_per_q: int) -> None:
    by_qid: Dict[int, List[dict]] = {}
    for r in report.get("runs", []):
        by_qid.setdefault(r["query_id"], []).append(r)
    per_q = {}
    for qid, rs in sorted(by_qid.items()):
        rs_sorted = sorted(rs, key=lambda x: x["run_id"])
        if len(rs_sorted) >= runs_per_q:
            per_q[qid] = build_per_question_stability(rs_sorted[:runs_per_q])
    report["per_question_summary"] = {str(k): v for k, v in per_q.items()}
    report["summary"] = build_global_stability_summary(
        per_q, report.get("runs", []),
        report["config"].get("n_questions", len(per_q)),
        runs_per_q,
    )


def run_one_sample(
    sample: dict,
    retriever,
    decomposer: LegalQueryDecomposer,
    plan_cache: Dict[str, List[str]],
    use_plan_cache: bool,
    per_query_top_n: int,
    final_top_n: int,
) -> dict:
    question = sample["问题"]
    gold_ids = set(sample.get("match_id", []))

    if use_plan_cache and question in plan_cache:
        sub_queries = plan_cache[question]
        plan_source = "cache"
    else:
        sub_queries = decomposer.decompose(question)
        plan_source = "llm_fresh"

    t0 = time.time()
    error = None
    try:
        _docs, retrieved_ids = retrieve_plan_single(
            retriever, sub_queries,
            per_query_top_n_multi=per_query_top_n,
            final_top_n=final_top_n,
        )
    except Exception as e:
        logger.exception("检索失败 q=%s", question[:40])
        retrieved_ids = []
        error = str(e)

    metrics = compute_retrieval_metrics(retrieved_ids, gold_ids)
    return {
        "question": question,
        "gold_ids": list(gold_ids),
        "gold_names": sample.get("match_name", []),
        "sub_queries": sub_queries,
        "plan_source": plan_source,
        "n_sub_queries": len(sub_queries),
        "retrieved_ids": retrieved_ids,
        "latency_s": round(time.time() - t0, 3),
        "hit@5": metrics["hit@5"],
        "recall@5": metrics["recall@5"],
        "mrr": metrics["mrr"],
        "ndcg@5": metrics.get("ndcg@5", 0),
        "metrics": metrics,
        "error": error,
    }


def run_single_mode(args, test_data: List[dict]) -> int:
    use_cache = bool(args.plan_cache)
    plan_cache = load_plan_cache(args.plan_cache) if use_cache else {}
    plan_dirty = False

    report = load_report(args.output) if args.resume else {}
    per_question: List[dict] = list(report.get("per_question", [])) if report else []
    done = done_question_ids(per_question)

    retriever = build_retriever(args.bm25, "emb_bm25_rrf_rerank")
    decomposer = LegalQueryDecomposer(build_chat(DEFAULT_CHAT_MODEL))

    config = _build_config(args, existing=report.get("config", {}), repeats=1, use_cache=use_cache)

    for i, sample in enumerate(test_data):
        qid = sample.get("query_id", i)
        if qid in done:
            continue

        result = run_one_sample(
            sample, retriever, decomposer, plan_cache, use_cache,
            args.per_query_top_n, args.final_top_n,
        )
        if use_cache and sample["问题"] not in plan_cache:
            plan_cache[sample["问题"]] = result["sub_queries"]
            plan_dirty = True

        rec = {"query_id": qid, **result}
        per_question.append(rec)
        m = result["metrics"]
        print(
            f"  [{i+1}/{len(test_data)}] qid={qid} subs={result['n_sub_queries']} "
            f"hit@5={m['hit@5']} recall@5={m['recall@5']:.2f} mrr={m['mrr']:.3f}"
        )
        if (i + 1) % args.flush_every == 0:
            save_report(args.output, {
                "variant": "decompose_plan_single_collection",
                "label": "B: 拆解 + 分治 hybrid（单库，top-5）",
                "config": config, "summary": _single_summary(per_question), "per_question": per_question,
            })
        if args.sleep > 0:
            time.sleep(args.sleep)

    if plan_dirty and args.plan_cache:
        save_plan_cache(args.plan_cache, plan_cache)

    payload = {
        "variant": "decompose_plan_single_collection",
        "label": "B: 拆解 + 分治 hybrid（单库，top-5）",
        "config": config,
        "summary": _single_summary(per_question),
        "per_question": per_question,
    }
    save_report(args.output, payload)
    _print_single_summary(per_question, args.output)
    return 0


def run_stability_mode(args, test_data: List[dict]) -> int:
    use_cache = bool(args.plan_cache)
    if use_cache:
        print("[WARN] --repeats>1 且指定了 --plan-cache：各 run 将复用同一拆解，非「自由拆解」模式")

    plan_cache = load_plan_cache(args.plan_cache) if use_cache else {}
    report = load_report(args.output) if args.resume else {}
    if not report:
        report = {
            "variant": "decompose_plan_single_collection",
            "label": "B: 拆解 + 分治 hybrid（单库，top-5，每题多次自由拆解）",
            "config": _build_config(args, repeats=args.repeats, use_cache=use_cache),
            "runs": [],
            "per_question_summary": {},
            "summary": {},
        }
    else:
        report.setdefault("runs", [])

    finished = done_run_keys(report["runs"]) if args.resume else set()
    retriever = build_retriever(args.bm25, "emb_bm25_rrf_rerank")
    decomposer = LegalQueryDecomposer(build_chat(DEFAULT_CHAT_MODEL))

    total = len(test_data) * args.repeats
    done_count = len(finished)

    print(
        f"[INFO] 方案 B 稳定性 | n={len(test_data)} × {args.repeats} | "
        f"拆解模式={'plan_cache' if use_cache else 'fresh_each_run'}"
    )

    for si, sample in enumerate(test_data, 1):
        qid = sample.get("query_id", si - 1)
        for run_idx in range(1, args.repeats + 1):
            if (qid, run_idx) in finished:
                continue
            run_id = f"q{qid}_run{run_idx}"
            print(f"[{si}/{len(test_data)}] qid={qid} run={run_idx}/{args.repeats}...", end=" ", flush=True)

            result = run_one_sample(
                sample, retriever, decomposer, plan_cache, use_cache,
                args.per_query_top_n, args.final_top_n,
            )
            row = {"query_id": qid, "run_id": run_id, "run_idx": run_idx, **result}
            report["runs"].append(row)
            done_count += 1
            print(
                f"subs={result['n_sub_queries']} hit@5={result['hit@5']} "
                f"({result['latency_s']:.1f}s) [{done_count}/{total}]"
            )
            rebuild_stability_summaries(report, args.repeats)
            report["config"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            save_report(args.output, report)
            if args.sleep > 0:
                time.sleep(args.sleep)

    rebuild_stability_summaries(report, args.repeats)
    report["config"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_report(args.output, report)
    _print_stability_summary(report.get("summary", {}), args.output)
    return 0


def _build_config(args, existing: Optional[dict] = None, repeats: int = 1, use_cache: bool = False) -> dict:
    existing = existing or {}
    return {
        "limit": args.limit,
        "repeats": repeats,
        "test_set": args.test,
        "bm25_path": args.bm25,
        "plan_cache": args.plan_cache if use_cache else None,
        "decompose_mode": "plan_cache" if use_cache else "fresh_each_run",
        "per_query_top_n_multi": args.per_query_top_n,
        "final_top_n": args.final_top_n,
        "pipeline": "decompose + per-subquery HybridRerankRetriever + hit_count merge",
        "coarse_top_k": COARSE_TOP_K,
        "bm25_top_k": BM25_TOP_K,
        "rerank_input_top_k": RERANK_INPUT_TOP_K,
        "rerank_top_n": RERANK_TOP_N,
        "rrf_k": RRF_K,
        "embed_model": DEFAULT_EMBED_MODEL,
        "rerank_model": DEFAULT_RERANK_MODEL,
        "chat_model": DEFAULT_CHAT_MODEL,
        "n_questions": args.limit,
        "started_at": existing.get("started_at") or datetime.now().isoformat(timespec="seconds"),
    }


def _single_summary(per_question: List[dict]) -> dict:
    n = len(per_question)
    if not n:
        return {"n": 0}
    keys = ["hit@1", "recall@1", "ndcg@1", "hit@3", "recall@3", "ndcg@3", "hit@5", "recall@5", "ndcg@5", "mrr"]
    s = {"n": n}
    for key in keys:
        s[key] = round(sum(r["metrics"][key] for r in per_question) / n, 4)
    lats = [r.get("latency_s", 0) for r in per_question]
    s["lat_avg"] = round(sum(lats) / n, 3)
    s["avg_sub_queries"] = round(sum(r.get("n_sub_queries", 0) for r in per_question) / n, 2)
    return s


def _print_single_summary(per_question: List[dict], path: str) -> None:
    s = _single_summary(per_question)
    print(f"\n[DONE] n={s.get('n')}  Hit@5={s.get('hit@5')}  Recall@5={s.get('recall@5')}  "
          f"MRR={s.get('mrr')}  → {path}")


def _print_stability_summary(s: dict, path: str) -> None:
    print("\n" + "=" * 60)
    print("方案 B 稳定性汇总（每题自由拆解）")
    print("=" * 60)
    print(f"  mean Hit@5    : {s.get('mean_hit@5')}")
    print(f"  mean Recall@5 : {s.get('mean_recall@5')}")
    print(f"  pass@3 率     : {s.get('pass@3_rate')}")
    print(f"  always@3 率   : {s.get('always_hit@3_rate')}")
    print(f"  never@3 率    : {s.get('never_hit@3_rate')}")
    print(f"  flaky@3 率    : {s.get('flaky@3_rate')}")
    print(f"  → {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="方案 B：拆解 + 单库分治检索评测")
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--bm25", default=DEFAULT_BM25)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=1, help="每题重复次数（>1 时每次重新拆解，除非指定 plan-cache）")
    ap.add_argument("--output", default="results/compare_50/arm_b_decompose.json")
    ap.add_argument("--plan-cache", default=None,
                    help="可选：固定子问题缓存（默认不用；稳定性横评请保持为空）")
    ap.add_argument("--per-query-top-n", type=int, default=2)
    ap.add_argument("--final-top-n", type=int, default=PRIMARY_K)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--flush-every", type=int, default=5)
    args = ap.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("[ERR] DASHSCOPE_API_KEY 未设置")
        return 1
    if not os.path.exists(args.test):
        print(f"[ERR] 测试集不存在: {args.test}")
        return 1
    if not os.path.exists(args.bm25):
        print(f"[ERR] BM25 不存在: {args.bm25}")
        return 1

    with open(args.test, encoding="utf-8") as f:
        test_data = json.load(f)[: args.limit]

    if args.repeats > 1:
        return run_stability_mode(args, test_data)
    return run_single_mode(args, test_data)


if __name__ == "__main__":
    raise SystemExit(main())
