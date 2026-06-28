# 实验设计与结果（50 题横评）

## 实验目标

在 **同一 KB**（50 题 gold 全覆盖的 5000 条语料采样）、**同一测试集**（`test.json` 前 50 题）、**同一计分协议**（top-5 vs `match_id`）下，对比五种检索架构，支撑从固定 RAG 到 ReAct Agent 的演进结论。

## 方案定义

| 代号 | 名称 | 协议 | 脚本 |
|------|------|------|------|
| **A** | 朴素 Hybrid RAG | 原题 × **1 次**；emb+BM25+RRF+rerank | `evaluate.py` → `emb_bm25_rrf_rerank` |
| **B** | 拆解分治 | LLM 拆 1–4 子 query × **3 次**（每次重新拆解） | `evaluate_decompose_retrieval.py` |
| **C** | ReAct Agent | ReAct + rerank pool=30 × **3 次** | `run_stability_50x3.py` |
| **D** | Embedding only | 原题 × 1；无 BM25 / 无 rerank | `run_compare_50.py --arms d` |
| **E** | BM25 only | 原题 × 1；无 embedding / 无 rerank | `run_compare_50.py --arms e` |

### 方案 B 融合策略（失败路径记录）

每个子 query 独立 `HybridRerankRetriever`，多 query 时每路 `per_query_top_n=2`，合并键：

```python
sorted(keys, key=lambda k: (hit_count, max_rerank_score), reverse=True)
```

**问题**：rerank 分跨子 query 不可比；过度拆解（subs=4 占 44%）引入噪声，Hit@5 低于 A。

### 方案 C 要点

- 工具：`vector_search` + `bm25_search` + `lookup_article` + `submit_ranking`
- 无 Grader、无 RRF 融合
- `submit_rate ≈ 98%`；未 submit 时 fallback 检索序 top-5

### C 消融：去掉 BM25

`run_stability_50x3.py --no-bm25` → Hit@5 降 **1.3pt**（0.887 → 0.873），BM25 有小幅正向贡献。

## 主结果（归档于 `results/compare_50/`）

完整表见 [`comparison_full.md`](../results/compare_50/comparison_full.md)。

### 检索质量（Hit@5）

| 方案 | Hit@5 | MRR | 备注 |
|------|-------|-----|------|
| A | 0.820 | 0.714 | 基线 |
| B | 0.773 | 0.687 | 低于 A |
| **C** | **0.887** | **0.849** | **最优** |
| D | 0.800 | 0.672 | emb-only |
| E | 0.680 | 0.580 | bm25-only |

**排序：C > A > D > B > E**

### 稳定性（B/C ×3）

| 指标 | B | C |
|------|---|---|
| pass@3 | 0.80 | **0.90** |
| always@3 | 0.74 | **0.86** |
| never@3 | 0.20 | **0.10** |

### 延迟（均值）

| A | B | C |
|---|---|---|
| 0.88s | 1.78s | 11.02s |

C 以约 **12× 延迟** 换取 +6.7pt Hit@5。

## 复现实验

```bash
# 1. 构建 KB（需 corpus.jsonl，见 data/lecoqa/README.md）
python scripts/build_kb.py --sample-test 50 --sample-corpus 5000

# 2. 一键横评（耗 API，A/D/E 各 50 次，B/C 各 150 次）
python scripts/run_compare_50.py --arms a,b,c,d,e

# 3. 仅重生成对比表（已有 JSON 时）
python scripts/build_compare_full_report.py
python scripts/run_compare_50.py --report-only
```

## 主要结论

1. **单次固定检索有 ceiling**（A=0.82）；需要多轮，但应由 Agent 循环而非 Grader 闸门驱动。
2. **人为融合排序（B）可证伪**：拆解 + hit_count 合并低于简单 RAG。
3. **通道应解耦而非 RRF 绑死**：D 部分指标优于 A，故 C 让 Agent 自选 vector/bm25。
4. **评测驱动迭代**：pass@3 暴露 B/C 稳定性差异；never@3 约 5 题为生活化 query 语义鸿沟死局。
