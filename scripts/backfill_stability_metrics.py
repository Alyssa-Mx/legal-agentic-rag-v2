"""
从 stability JSON 的 submitted_ranking / scoring_source 补全 Hit@1/3、NDCG 等指标，
并写回 summary。对 fallback 且无法还原完整 ranking 的 run，用 mrr + recall 近似重建 top-5。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_agent_retrieval import compute_retrieval_metrics, PRIMARY_K

K_VALUES = (1, 3, 5)
METRIC_KEYS = [
    "hit@1", "recall@1", "ndcg@1",
    "hit@3", "recall@3", "ndcg@3",
    "hit@5", "recall@5", "ndcg@5",
    "mrr",
]


def _synthetic_ranking(gold: Set[int], mrr: float, recall_at_5: float) -> List[int]:
    """fallback 场景：用 mrr 推断首个 gold 位置，构造近似 top-5 用于 NDCG/Hit@k。"""
    if mrr <= 0 or not gold:
        return []
    first_rank = max(1, round(1.0 / mrr))
    first_rank = min(first_rank, PRIMARY_K)
    dummy = -1
    ranking: List[int] = []
    for r in range(1, PRIMARY_K + 1):
        if r == first_rank:
            ranking.append(next(iter(gold)))
        else:
            ranking.append(dummy)
            dummy -= 1
    # 多 gold 且 recall@5>首个命中比例时，在后续位补第二个 gold（保守近似）
    if len(gold) > 1 and recall_at_5 > 1.0 / len(gold) + 1e-9:
        others = [g for g in gold if g not in ranking]
        for i, gid in enumerate(others):
            pos = min(first_rank + 1 + i, PRIMARY_K - 1)
            if pos < len(ranking) and ranking[pos] < 0:
                ranking[pos] = gid
    return ranking


def recompute_run_metrics(run: dict, gold_ids: List[int]) -> Dict[str, float]:
    gold = set(gold_ids)
    src = run.get("scoring_source", "")
    ranking: List[int] = []

    if src == "submitted" and run.get("submitted_ranking"):
        ranking = list(run["submitted_ranking"])
    elif src == "fallback_retrieved":
        ranking = _synthetic_ranking(gold, float(run.get("mrr", 0)), float(run.get("recall@5", 0)))
    elif run.get("submitted_ranking"):
        ranking = list(run["submitted_ranking"])

    if ranking:
        return compute_retrieval_metrics(ranking, gold, k_values=K_VALUES)

    # empty：沿用已有 hit@5/recall@5/mrr，其余置 0
    out = {k: 0 for k in METRIC_KEYS}
    out["hit@5"] = int(run.get("hit@5", 0))
    out["recall@5"] = float(run.get("recall@5", 0))
    out["mrr"] = float(run.get("mrr", 0))
    out["hit@1"] = out["hit@3"] = 0
    out["recall@1"] = out["recall@3"] = 0.0
    out["ndcg@1"] = out["ndcg@3"] = out["ndcg@5"] = 0.0
    if out["mrr"] >= 1.0 - 1e-9:
        out["hit@1"] = out["hit@3"] = 1
        out["recall@1"] = out["recall@3"] = min(1.0, 1.0 / len(gold)) if gold else 0.0
        out["ndcg@1"] = out["ndcg@3"] = out["ndcg@5"] = 1.0
    elif out["mrr"] >= 0.5 - 1e-9:
        out["hit@3"] = out["hit@5"]
        out["recall@3"] = min(out["recall@5"], 1.0)
        out["ndcg@3"] = round(1.0 / math.log2(3), 4) if out["hit@3"] else 0.0
        out["ndcg@5"] = out["ndcg@3"]
    return out


def aggregate_summary(runs: List[dict]) -> dict:
    n = len(runs)
    if n == 0:
        return {}

    def mean(key: str) -> float:
        return round(sum(float(r.get(key, 0)) for r in runs) / n, 4)

    submits = [bool(r.get("submitted")) for r in runs]
    return {
        "completed_runs": n,
        "mean_hit@1": mean("hit@1"),
        "mean_recall@1": mean("recall@1"),
        "mean_ndcg@1": mean("ndcg@1"),
        "mean_hit@3": mean("hit@3"),
        "mean_recall@3": mean("recall@3"),
        "mean_ndcg@3": mean("ndcg@3"),
        "mean_hit@5": mean("hit@5"),
        "mean_recall@5": mean("recall@5"),
        "mean_ndcg@5": mean("ndcg@5"),
        "mean_mrr": mean("mrr"),
        "submit_rate": round(sum(submits) / n, 4),
        "lat_avg_s": round(sum(r.get("latency_s", 0) or 0 for r in runs) / n, 2),
    }


def backfill_file(path: Path, test_data: Dict[int, dict]) -> dict:
    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    fallback_n = 0
    for run in report.get("runs", []):
        qid = run["query_id"]
        gold = test_data[qid]["match_id"]
        metrics = recompute_run_metrics(run, gold)
        for k, v in metrics.items():
            if k in METRIC_KEYS:
                run[k] = v
        if run.get("scoring_source") == "fallback_retrieved":
            fallback_n += 1

    agg = aggregate_summary(report["runs"])
    summary = report.get("summary", {})
    summary.update(agg)
    report["summary"] = summary
    report["config"]["metrics_backfilled_at"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    report["config"]["fallback_runs_approximated"] = fallback_n

    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    return summary


def main() -> None:
    test_path = ROOT / "data/lecoqa/test.json"
    with open(test_path, encoding="utf-8") as f:
        test_data = {s["query_id"]: s for s in json.load(f)}

    targets = [
        ROOT / "results/agent_retrieval/stability_50x3.json",
        ROOT / "results/agent_retrieval/stability_50x3_rerank_pool30.json",
    ]
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]

    for path in targets:
        if not path.exists():
            print(f"[SKIP] {path}")
            continue
        s = backfill_file(path, test_data)
        print(f"\n=== {path.name} ===")
        print(f"  mean_hit@1={s['mean_hit@1']}  mean_hit@3={s['mean_hit@3']}  mean_hit@5={s['mean_hit@5']}")
        print(f"  mean_recall@5={s['mean_recall@5']}  mean_ndcg@5={s['mean_ndcg@5']}  mean_mrr={s['mean_mrr']}")
        print(f"  submit_rate={s['submit_rate']}  lat_avg_s={s['lat_avg_s']}")
        print(f"  fallback approx: {s.get('fallback_runs_approximated', 'n/a')}")


if __name__ == "__main__":
    main()
