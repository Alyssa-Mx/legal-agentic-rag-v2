"""从 compare_50 各 arm JSON 汇总完整指标表 → comparison_full.json + comparison_full.md"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(ROOT, "results", "compare_50")

ARMS = {
    "A": {
        "label": "A hybrid+RRF+rerank",
        "file": "arm_a_naive_rag.json",
        "mode": "single",
        "protocol": "emb+bm25+RRF+rerank，原题×1",
    },
    "B": {
        "label": "B 拆解分治",
        "file": "arm_b_stability_50x3.json",
        "mode": "stability_x3",
        "protocol": "拆解 fresh×3，单库 hybrid+rerank",
    },
    "C": {
        "label": "C ReAct Agent",
        "file": "arm_c_stability_50x3.json",
        "mode": "stability_x3",
        "protocol": "ReAct+rerank pool=30×3",
    },
    "D": {
        "label": "D emb-only",
        "file": "arm_d_emb_only.json",
        "mode": "single",
        "protocol": "仅 embedding，原题×1，无 rerank",
    },
    "E": {
        "label": "E bm25-only",
        "file": "arm_e_bm25_only.json",
        "mode": "single",
        "protocol": "仅 BM25，原题×1，无 rerank",
    },
}

METRIC_KEYS_SINGLE = [
    "hit@1", "recall@1", "ndcg@1",
    "hit@3", "recall@3", "ndcg@3",
    "hit@5", "recall@5", "ndcg@5",
    "mrr",
    "lat_avg", "lat_p50", "lat_p95", "pool_avg",
]

METRIC_KEYS_STABILITY = [
    "mean_hit@1", "mean_recall@1", "mean_ndcg@1",
    "mean_hit@3", "mean_recall@3", "mean_ndcg@3",
    "mean_hit@5", "mean_recall@5", "mean_ndcg@5",
    "mean_mrr",
    "pass@3_rate", "always_hit@3_rate", "never_hit@3_rate", "flaky@3_rate",
    "submit_rate", "avg_sub_queries",
    "unstable_question_count", "lat_avg_s",
    "completed_runs", "total_runs_target",
]


def _mean_from_b_runs(data: dict) -> Dict[str, float]:
    runs = data.get("runs") or []
    if not runs:
        return {}
    keys = [
        "hit@1", "recall@1", "ndcg@1",
        "hit@3", "recall@3", "ndcg@3",
        "hit@5", "recall@5", "ndcg@5", "mrr",
    ]
    out: Dict[str, float] = {}
    n = len(runs)
    for k in keys:
        out[f"mean_{k}"] = round(
            sum(r.get("metrics", {}).get(k, r.get(k, 0)) for r in runs) / n, 4
        )
    return out


def load_arm_metrics(arm_id: str, meta: dict) -> Dict[str, Any]:
    path = os.path.join(OUT_DIR, meta["file"])
    if not os.path.exists(path):
        return {"missing": True, "source_file": meta["file"]}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    s = dict(data.get("summary") or {})
    s["source_file"] = f"results/compare_50/{meta['file']}"
    s["label"] = data.get("label") or meta["label"]
    s["protocol"] = meta["protocol"]
    s["mode"] = meta["mode"]
    if arm_id == "B":
        s.update(_mean_from_b_runs(data))
    return s


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if isinstance(v, (int, float)):
        return f"{v:.4f}" if isinstance(v, float) else str(v)
    return str(v)


def build_markdown(arms: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        "# 50 题横评 — 完整指标对比表",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "## 数据来源（逐方案原始 JSON）",
        "",
        "| 方案 | 原始结果文件 |",
        "|------|----------------|",
    ]
    for aid, meta in ARMS.items():
        m = arms.get(aid, {})
        lines.append(f"| **{aid}** {meta['label']} | `{m.get('source_file', meta['file'])}` |")
    lines += [
        "",
        "本表由 `scripts/build_compare_full_report.py` 从上述 JSON 的 `summary`（B 的 @1/@3 由 `runs[].metrics` 补算）聚合生成。",
        "机器可读副本：`results/compare_50/comparison_full.json`",
        "",
        "---",
        "",
        "## 表 1 — 检索质量 @K（Hit / Recall / NDCG）+ MRR",
        "",
        "| 指标 | A | B | C | D | E |",
        "|------|-----|-----|-----|-----|-----|",
    ]

    rows = [
        ("Hit@1", "hit@1", "mean_hit@1"),
        ("Recall@1", "recall@1", "mean_recall@1"),
        ("NDCG@1", "ndcg@1", "mean_ndcg@1"),
        ("Hit@3", "hit@3", "mean_hit@3"),
        ("Recall@3", "recall@3", "mean_recall@3"),
        ("NDCG@3", "ndcg@3", "mean_ndcg@3"),
        ("Hit@5", "hit@5", "mean_hit@5"),
        ("Recall@5", "recall@5", "mean_recall@5"),
        ("NDCG@5", "ndcg@5", "mean_ndcg@5"),
        ("MRR", "mrr", "mean_mrr"),
    ]
    for label, sk, mk in rows:
        cells = [label]
        for aid in "ABCDE":
            m = arms[aid]
            if m.get("missing"):
                cells.append("—")
            elif m.get("mode") == "single":
                cells.append(_fmt(m.get(sk)))
            else:
                cells.append(_fmt(m.get(mk)))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 表 2 — 稳定性（B/C ×3；A/D/E 不适用填 —）",
        "",
        "| 指标 | A | B | C | D | E |",
        "|------|-----|-----|-----|-----|-----|",
    ]
    stab_rows = [
        ("pass@3", "pass@3_rate"),
        ("always@3", "always_hit@3_rate"),
        ("never@3", "never_hit@3_rate"),
        ("flaky@3", "flaky@3_rate"),
        ("unstable 题数", "unstable_question_count"),
        ("unstable qids", "unstable_query_ids"),
        ("submit_rate", "submit_rate"),
        ("avg_sub_queries", "avg_sub_queries"),
        ("completed_runs", "completed_runs"),
    ]
    for label, key in stab_rows:
        cells = [label]
        for aid in "ABCDE":
            m = arms[aid]
            if m.get("missing") or m.get("mode") == "single":
                cells.append("—")
            else:
                cells.append(_fmt(m.get(key)))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 表 3 — 延迟与候选池",
        "",
        "| 指标 | A | B | C | D | E |",
        "|------|-----|-----|-----|-----|-----|",
    ]
    lat_rows = [
        ("lat_avg (s)", "lat_avg", "lat_avg_s", "lat_avg_s"),
        ("lat_p50 (s)", "lat_p50", None, None),
        ("lat_p95 (s)", "lat_p95", None, None),
        ("pool_avg", "pool_avg", None, None),
    ]
    for label, sk, mk_avg, _ in lat_rows:
        cells = [label]
        for aid in "ABCDE":
            m = arms[aid]
            if m.get("missing"):
                cells.append("—")
            elif m.get("mode") == "single":
                cells.append(_fmt(m.get(sk)))
            elif label.startswith("lat_avg"):
                cells.append(_fmt(m.get(mk_avg)))
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    arms = {aid: load_arm_metrics(aid, meta) for aid, meta in ARMS.items()}
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/build_compare_full_report.py",
        "source_note": "各 arm 指标来自 results/compare_50/arm_*.json 的 summary 字段；B 的 mean_hit@1/3 等由 runs[].metrics 现场均值补算",
        "arms": arms,
    }
    json_path = os.path.join(OUT_DIR, "comparison_full.json")
    md_path = os.path.join(OUT_DIR, "comparison_full.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(arms))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
