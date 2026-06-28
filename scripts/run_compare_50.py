"""
三方案 50 题横向对比（按约定跑法）：

  A — 普通 RAG 原题直检：跑 **1 次**（确定性流水线）
  B — 拆解 + 分治：跑 **3 次**，每次 LLM **重新拆解**（不用 plan_cache）
  C — ReAct Agent + rerank：跑 **3 次**（已有稳定性脚本）

用法：
    python scripts/run_compare_50.py --arms a,b,c --resume
    python scripts/run_compare_50.py --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

DEFAULT_OUT = "results/compare_50"
ARM_A = "arm_a_naive_rag.json"
ARM_B = "arm_b_stability_50x3.json"
ARM_C = "arm_c_stability_50x3.json"
ARM_D = "arm_d_emb_only.json"
ARM_E = "arm_e_bm25_only.json"


def _py() -> str:
    return sys.executable


def run_cmd(cmd: List[str], desc: str) -> int:
    print(f"\n{'='*72}\n  {desc}\n{'='*72}\n  {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def load_arm_a(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    s = d.get("summary", {})
    return {
        "mode": "single",
        "hit@5": s.get("hit@5"),
        "recall@5": s.get("recall@5"),
        "mrr": s.get("mrr"),
        "ndcg@5": s.get("ndcg@5"),
        "lat_avg": s.get("lat_avg"),
        "n": s.get("n"),
    }


def load_arm_stability(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    s = d.get("summary", {})
    return {
        "mode": "stability_x3",
        "mean_hit@5": s.get("mean_hit@5"),
        "mean_recall@5": s.get("mean_recall@5"),
        "mean_mrr": s.get("mean_mrr"),
        "mean_ndcg@5": s.get("mean_ndcg@5"),
        "pass@3": s.get("pass@3_rate"),
        "always@3": s.get("always_hit@3_rate"),
        "never@3": s.get("never_hit@3_rate"),
        "flaky@3": s.get("flaky@3_rate"),
        "lat_avg": s.get("lat_avg_s"),
        "n_questions": s.get("n_questions"),
        "runs": s.get("completed_runs"),
    }


def _copy_abl_variant(abl_dir: str, variant: str, dst: str, label: str) -> None:
    src = os.path.join(abl_dir, f"{variant}.json")
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        data["label"] = label
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def print_report(out_dir: str) -> None:
    a = load_arm_a(os.path.join(out_dir, ARM_A))
    b = load_arm_stability(os.path.join(out_dir, ARM_B))
    c = load_arm_stability(os.path.join(out_dir, ARM_C))
    d = load_arm_a(os.path.join(out_dir, ARM_D))
    e = load_arm_a(os.path.join(out_dir, ARM_E))

    print(f"\n{'='*80}")
    print("  50 题对比（A/D/E×1，B/C×3）")
    print(f"{'='*80}")
    print(f"{'方案':<28} {'Hit@5':>8} {'Recall@5':>9} {'MRR':>7} {'pass@3':>8} {'always@3':>9} {'never@3':>8}")
    print("-" * 80)

    singles = [
        ("A hybrid+rerank", a),
        ("D emb-only", d),
        ("E bm25-only", e),
    ]
    for label, data in singles:
        if data:
            print(
                f"{label:<28} {_fmt(data.get('hit@5')):>8} {_fmt(data.get('recall@5')):>9} "
                f"{_fmt(data.get('mrr')):>7} {'—':>8} {'—':>9} {'—':>8}"
            )
        else:
            print(f"{label:<28}  （缺失）")

    for label, data in [("B 拆解（3×自由拆解）", b), ("C ReAct（3×Agent）", c)]:
        if data:
            print(
                f"{label:<28} {_fmt(data.get('mean_hit@5')):>8} {_fmt(data.get('mean_recall@5')):>9} "
                f"{_fmt(data.get('mean_mrr')):>7} {_fmt(data.get('pass@3')):>8} "
                f"{_fmt(data.get('always@3')):>9} {_fmt(data.get('never@3')):>8}"
            )
        else:
            print(f"{label:<28}  （结果缺失）")

    out_path = os.path.join(out_dir, "comparison_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "protocol": {
                "A": "emb+bm25+RRF+rerank, original question, 1 run",
                "D": "emb only, original question, 1 run",
                "E": "bm25 only, original question, 1 run",
                "B": "decompose fresh each run, 3 runs per question",
                "C": "ReAct agent + rerank pool=30, 3 runs per question",
            },
            "A": a, "B": b, "C": c, "D": d, "E": e,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 汇总: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="三方案 50 题横评编排")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--arms", default="a,b,c")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--rerank-pool", type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    arms = [x.strip().lower() for x in args.arms.split(",") if x.strip()]
    resume = ["--resume"] if args.resume else []

    if args.report_only:
        print_report(args.output_dir)
        return 0

    if "a" in arms:
        abl_dir = os.path.join(args.output_dir, "abl_a")
        os.makedirs(abl_dir, exist_ok=True)
        rc = run_cmd(
            [_py(), "scripts/evaluate.py", "--variant", "emb_bm25_rrf_rerank",
             "--no-rewrite", "--limit", str(args.limit), "--output-dir", abl_dir] + resume,
            "A：普通 RAG 原题直检（1 次）",
        )
        if rc != 0:
            return rc
        src = os.path.join(abl_dir, "emb_bm25_rrf_rerank.json")
        dst = os.path.join(args.output_dir, ARM_A)
        if os.path.exists(src):
            with open(src, encoding="utf-8") as f:
                data = json.load(f)
            data["label"] = "A: 普通 RAG 单次（原题直检）"
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    if "d" in arms:
        abl_dir = os.path.join(args.output_dir, "abl_emb")
        os.makedirs(abl_dir, exist_ok=True)
        rc = run_cmd(
            [_py(), "scripts/evaluate.py", "--variant", "emb",
             "--no-rewrite", "--limit", str(args.limit), "--output-dir", abl_dir] + resume,
            "D：普通 RAG 仅 Embedding（1 次）",
        )
        if rc != 0:
            return rc
        _copy_abl_variant(abl_dir, "emb", os.path.join(args.output_dir, ARM_D),
                          "D: 普通 RAG 仅 Embedding（原题直检）")

    if "e" in arms:
        abl_dir = os.path.join(args.output_dir, "abl_bm25")
        os.makedirs(abl_dir, exist_ok=True)
        rc = run_cmd(
            [_py(), "scripts/evaluate.py", "--variant", "bm25",
             "--no-rewrite", "--limit", str(args.limit), "--output-dir", abl_dir] + resume,
            "E：普通 RAG 仅 BM25（1 次）",
        )
        if rc != 0:
            return rc
        _copy_abl_variant(abl_dir, "bm25", os.path.join(args.output_dir, ARM_E),
                          "E: 普通 RAG 仅 BM25（原题直检）")

    if "b" in arms:
        rc = run_cmd(
            [_py(), "scripts/evaluate_decompose_retrieval.py",
             "--limit", str(args.limit),
             "--repeats", "3",
             "--output", os.path.join(args.output_dir, ARM_B)] + resume,
            "B：拆解 + 分治（50×3，每次自由拆解）",
        )
        if rc != 0:
            return rc

    if "c" in arms:
        rc = run_cmd(
            [_py(), "scripts/run_stability_50x3.py",
             "--limit", str(args.limit),
             "--rerank", "--rerank-pool", str(args.rerank_pool),
             "--output", os.path.join(args.output_dir, ARM_C)] + resume,
            "C：ReAct Agent + rerank（50×3）",
        )
        if rc != 0:
            return rc

    print_report(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
