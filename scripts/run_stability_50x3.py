"""
全量 50 题 × 每题 3 次稳定性评测。

指标说明：
  - hit@5_run{i}     : 第 i 次单次 Hit@5
  - mean_hit@5       : 3 次 Hit@5 均值（期望命中率）
  - pass@3           : 3 次中至少 1 次 Hit@5=1（稳定性 pass 率，类似 code eval 的 pass@n）
  - always_hit@3     : 3 次全部 Hit@5=1
  - never_hit@3      : 3 次全部 Hit@5=0
  - submit_rate@3    : 3 次中至少 1 次成功 submit

支持 --resume：跳过 output 里已完成的 (query_id, run_idx)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config.setting import DEFAULT_CHAT_MODEL
from app.retrieval.kb_tools import load_kb_tools
from scripts.evaluate_agent_retrieval import evaluate_one

DEFAULT_TEST = ROOT / "data/lecoqa/test.json"
DEFAULT_OUT = ROOT / "results/agent_retrieval/stability_50x3.json"


def summarize_run(rec: dict, run_id: str) -> dict:
    gold = set(rec["gold_ids"])
    pool = rec.get("raw_retrieved_ids") or []
    pool_set = set(pool)
    sub = rec.get("submitted_ranking") or []
    m = rec.get("metrics") or {}
    row = {
        "run_id": run_id,
        "hit@5": rec["metrics"]["hit@5"],
        "recall@5": rec["metrics"]["recall@5"],
        "mrr": rec["metrics"]["mrr"],
        "submitted": rec.get("submitted", False),
        "scoring_source": rec.get("scoring_source"),
        "pool_size": len(pool),
        "gold_in_pool": sorted(gold & pool_set),
        "gold_in_submit": sorted(gold & set(sub)),
        "submitted_ranking": sub,
        "tool_calls": rec.get("tool_call_count", 0),
        "latency_s": rec.get("latency_s"),
    }
    for k in (
        "hit@1", "recall@1", "ndcg@1",
        "hit@3", "recall@3", "ndcg@3",
        "ndcg@5",
    ):
        if k in m:
            row[k] = m[k]
    return row


def load_report(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def done_keys(runs: List[dict]) -> Set[Tuple[int, int]]:
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


def build_per_question_summary(runs_for_q: List[dict]) -> dict:
    hits = [r["hit@5"] for r in runs_for_q]
    recalls = [r["recall@5"] for r in runs_for_q]
    mrrs = [r["mrr"] for r in runs_for_q]
    submits = [r["submitted"] for r in runs_for_q]
    return {
        "hit@5": {
            "values": hits,
            "mean": round(sum(hits) / len(hits), 4),
            "std": round(_std(hits), 4),
        },
        "recall@5": {
            "values": recalls,
            "mean": round(sum(recalls) / len(recalls), 4),
        },
        "mrr": {
            "values": mrrs,
            "mean": round(sum(mrrs) / len(mrrs), 4),
        },
        "pass@3": int(any(hits)),
        "always_hit@3": int(all(hits)),
        "never_hit@3": int(not any(hits)),
        "submit_any": int(any(submits)),
        "submit_all": int(all(submits)),
        "pool_size": [r["pool_size"] for r in runs_for_q],
    }


def build_global_summary(
    per_q: Dict[int, dict],
    runs: List[dict],
    n_questions: int,
    runs_per_q: int,
) -> dict:
    n = len(per_q)
    if n == 0:
        return {}
    pass3 = sum(s["pass@3"] for s in per_q.values()) / n
    always = sum(s["always_hit@3"] for s in per_q.values()) / n
    never = sum(s["never_hit@3"] for s in per_q.values()) / n
    mean_hit = sum(s["hit@5"]["mean"] for s in per_q.values()) / n
    mean_recall = sum(s["recall@5"]["mean"] for s in per_q.values()) / n
    mean_mrr = sum(s["mrr"]["mean"] for s in per_q.values()) / n
    unstable = [
        qid for qid, s in per_q.items()
        if s["hit@5"]["std"] > 0 or (min(s["hit@5"]["values"]) != max(s["hit@5"]["values"]))
    ]
    latencies = [r["latency_s"] for r in runs if r.get("latency_s") is not None]
    submits = [bool(r.get("submitted")) for r in runs]

    def _run_mean(key: str) -> float:
        if not runs:
            return 0.0
        return round(sum(float(r.get(key, 0)) for r in runs) / len(runs), 4)

    return {
        "n_questions": n_questions,
        "runs_per_question": runs_per_q,
        "completed_questions": n,
        "completed_runs": len(runs),
        "total_runs_target": n_questions * runs_per_q,
        "mean_hit@1": _run_mean("hit@1"),
        "mean_recall@1": _run_mean("recall@1"),
        "mean_ndcg@1": _run_mean("ndcg@1"),
        "mean_hit@3": _run_mean("hit@3"),
        "mean_recall@3": _run_mean("recall@3"),
        "mean_ndcg@3": _run_mean("ndcg@3"),
        "mean_hit@5": round(mean_hit, 4),
        "mean_recall@5": round(mean_recall, 4),
        "mean_ndcg@5": _run_mean("ndcg@5"),
        "mean_mrr": round(mean_mrr, 4),
        "submit_rate": round(sum(submits) / len(runs), 4) if runs else 0.0,
        "pass@3_rate": round(pass3, 4),
        "always_hit@3_rate": round(always, 4),
        "never_hit@3_rate": round(never, 4),
        "unstable_question_count": len(unstable),
        "unstable_query_ids": sorted(unstable),
        "lat_avg_s": round(sum(latencies) / len(latencies), 2) if latencies else 0,
    }


def _std(vals: list) -> float:
    if len(vals) <= 1:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5


def rebuild_summaries(report: dict, runs_per_q: int) -> None:
    by_qid: Dict[int, List[dict]] = {}
    for r in report.get("runs", []):
        qid = r["query_id"]
        by_qid.setdefault(qid, []).append(r)
    per_q = {}
    for qid, rs in sorted(by_qid.items()):
        rs_sorted = sorted(rs, key=lambda x: x["run_id"])
        if len(rs_sorted) >= runs_per_q:
            per_q[qid] = build_per_question_summary(rs_sorted[:runs_per_q])
    report["per_question_summary"] = {str(k): v for k, v in per_q.items()}
    report["summary"] = build_global_summary(
        {int(k): v for k, v in per_q.items()},
        report.get("runs", []),
        report["config"].get("n_questions", len(per_q)),
        runs_per_q,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="50 题 × 3 次稳定性评测")
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--runs", type=int, default=3, help="每题重复次数")
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--default-k", type=int, default=10)
    ap.add_argument("--trunc-chars", type=int, default=200)
    ap.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    ap.add_argument("--rerank", action="store_true", help="vector/bm25 工具启用 reranker 精排")
    ap.add_argument("--no-bm25", action="store_true", help="禁用 bm25_search 工具（仅 vector + lookup）")
    ap.add_argument("--rerank-pool", type=int, default=30,
                    help="reranker 召回池大小（仅在 --rerank 时生效，默认 30=B3 甜点）")
    ap.add_argument("--rerank-model", default=None, help="默认走 DEFAULT_RERANK_MODEL")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.3, help="每次 run 间隔秒数")
    args = ap.parse_args()

    out_path = Path(args.output)
    with open(args.test, encoding="utf-8") as f:
        samples = json.load(f)[: args.limit]

    report = load_report(out_path) if args.resume else {}
    if not report:
        report = {
            "config": {
                "test_set": args.test,
                "n_questions": len(samples),
                "runs_per_question": args.runs,
                "max_steps": args.max_steps,
                "default_k": args.default_k,
                "trunc_chars": args.trunc_chars,
                "rerank": bool(args.rerank),
                "bm25_enabled": not args.no_bm25,
                "rerank_pool": args.rerank_pool if args.rerank else None,
                "rerank_model": (args.rerank_model or "qwen3-rerank") if args.rerank else None,
                "chat_model": args.chat_model,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            },
            "runs": [],
            "per_question_summary": {},
            "summary": {},
        }

    finished = done_keys(report.get("runs", [])) if args.resume else set()
    total_target = len(samples) * args.runs
    done_count = len(finished)

    rerank_tag = f"ON pool={args.rerank_pool}" if args.rerank else "OFF"
    bm25_tag = "OFF" if args.no_bm25 else "ON"
    print(
        f"[INFO] 样本={len(samples)}  每题{args.runs}次  "
        f"已完成={done_count}/{total_target}  "
        f"max_steps={args.max_steps} k={args.default_k} trunc={args.trunc_chars}  "
        f"rerank={rerank_tag}  bm25={bm25_tag}"
    )
    print(f"[INFO] 输出: {out_path}")

    reranker = None
    if args.rerank:
        from app.config.setting import DEFAULT_RERANK_MODEL
        from app.models.dashscope_reranker import QwenDashScopeReranker
        reranker = QwenDashScopeReranker(model_name=args.rerank_model or DEFAULT_RERANK_MODEL)

    kb_tools = load_kb_tools(
        default_k=args.default_k, trunc_chars=args.trunc_chars,
        reranker=reranker, rerank_pool_size=args.rerank_pool,
    )
    if args.no_bm25:
        kb_tools = [t for t in kb_tools if t.name != "bm25_search"]
        print(f"[INFO] 已禁用 bm25_search，KB 工具: {[t.name for t in kb_tools]}")
    t_all = time.perf_counter()

    for si, sample in enumerate(samples, 1):
        qid = sample["query_id"]
        q_short = (sample.get("问题") or "")[:40]
        for run_idx in range(1, args.runs + 1):
            if (qid, run_idx) in finished:
                continue
            run_id = f"q{qid}_run{run_idx}"
            print(
                f"[{si}/{len(samples)}] qid={qid} run={run_idx}/{args.runs}  {q_short}...",
                end=" ",
                flush=True,
            )
            t0 = time.perf_counter()
            try:
                rec = evaluate_one(
                    sample=sample,
                    kb_tools=kb_tools,
                    chat_model=args.chat_model,
                    max_steps=args.max_steps,
                    save_trace=False,
                )
            except Exception as e:
                print(f"ERR {e}")
                row = {
                    "query_id": qid,
                    "run_id": run_id,
                    "error": str(e),
                    "hit@5": 0,
                    "recall@5": 0.0,
                    "mrr": 0.0,
                    "submitted": False,
                }
                report["runs"].append(row)
                save_report(out_path, report)
                time.sleep(args.sleep)
                continue

            summary = summarize_run(rec, run_id)
            row = {"query_id": qid, **summary}
            report["runs"].append(row)
            elapsed = time.perf_counter() - t0
            done_count += 1
            print(
                f"hit@5={summary['hit@5']} recall={summary['recall@5']:.2f} "
                f"submit={summary['submitted']} pool={summary['pool_size']} "
                f"({elapsed:.1f}s) [{done_count}/{total_target}]"
            )
            rebuild_summaries(report, args.runs)
            report["config"]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            save_report(out_path, report)
            time.sleep(args.sleep)

    rebuild_summaries(report, args.runs)
    report["config"]["finished_at"] = datetime.now().isoformat(timespec="seconds")
    report["config"]["total_elapsed_s"] = round(time.perf_counter() - t_all, 1)
    save_report(out_path, report)

    s = report.get("summary", {})
    print("\n" + "=" * 60)
    print("全量稳定性汇总")
    print("=" * 60)
    print(f"  完成 runs     : {s.get('completed_runs')}/{s.get('total_runs_target')}")
    print(f"  mean Hit@5    : {s.get('mean_hit@5')}")
    print(f"  mean Recall@5 : {s.get('mean_recall@5')}")
    print(f"  mean MRR      : {s.get('mean_mrr')}")
    print(f"  pass@3 率     : {s.get('pass@3_rate')}  (3次中至少1次命中)")
    print(f"  always@3 率   : {s.get('always_hit@3_rate')}  (3次全命中)")
    print(f"  never@3 率    : {s.get('never_hit@3_rate')}  (3次全失败)")
    print(f"  不稳定题数    : {s.get('unstable_question_count')}  qids={s.get('unstable_query_ids')}")
    print(f"  平均延迟      : {s.get('lat_avg_s')}s/run")
    print(f"\n[INFO] 结果: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
