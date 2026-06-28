"""
检查当前 BM25 索引是否覆盖测试集前 N 题的 gold 法条。

用法：
    python scripts/check_kb_coverage.py --limit 309
    python scripts/check_kb_coverage.py --limit 309 --bm25 data/lecoqa/bm25_index.pkl
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_TEST = "data/lecoqa/test.json"
DEFAULT_BM25 = "data/lecoqa/bm25_index.pkl"


def main() -> int:
    ap = argparse.ArgumentParser(description="检查 KB 对测试集 gold 的覆盖")
    ap.add_argument("--test", default=DEFAULT_TEST)
    ap.add_argument("--bm25", default=DEFAULT_BM25)
    ap.add_argument("--limit", type=int, default=309)
    args = ap.parse_args()

    with open(args.test, encoding="utf-8") as f:
        test_data = json.load(f)[: args.limit]

    gold_ids = set()
    for item in test_data:
        gold_ids.update(item.get("match_id", []))

    if not os.path.exists(args.bm25):
        print(f"[ERR] BM25 索引不存在: {args.bm25}")
        print(f"      请先运行: python scripts/build_kb.py --sample-test {args.limit} --sample-corpus 5000")
        return 1

    with open(args.bm25, "rb") as f:
        _, corpus_docs = pickle.load(f)
    indexed = {int(d.metadata["article_id"]) for d in corpus_docs if d.metadata.get("article_id") is not None}

    missing = sorted(gold_ids - indexed)
    print(f"[INFO] 测试题: 前 {args.limit} 题")
    print(f"[INFO] 唯一 gold 法条: {len(gold_ids)}")
    print(f"[INFO] 索引内法条: {len(indexed)}")
    print(f"[INFO] gold 覆盖: {len(gold_ids) - len(missing)}/{len(gold_ids)}")
    if missing:
        print(f"[WARN] 缺失 {len(missing)} 个 gold（评测会不公平）:")
        print(f"       示例: {missing[:10]}")
        print(
            f"\n建议重建 KB:\n"
            f"  python scripts/build_kb.py --sample-test {args.limit} --sample-corpus 5000"
        )
        return 2
    print("[OK] 全部 gold 已在索引中")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
