# 架构说明

## 设计演进

| 阶段 | 形态 | 核心问题 |
|------|------|----------|
| 固定流水线 | 原题 → hybrid RRF → rerank | 单轮检索、硬融合 |
| 8 节点图（早期版本） | 检索 → **Grader** → 重写 → 生成 | 中间闸门损害自主性 |
| 拆解分治（方案 B） | 2–4 子 query + **hit_count 融合** | 跨 query 分数不可比 |
| **ReAct Agent（方案 C）** | 三工具 + 多轮循环 + submit | 当前主架构 |

## 方案 C：ReAct 循环

```
prepare_context → agent ↔ tools（可并行）→ collect_evidence → … → self_check → END
```

评测专用图（`evaluate_agent_retrieval.py`）在工具集中额外加入 `submit_ranking`，并在步数用尽时 fallback 到检索序 top-5。

### 三个 KB 工具

| 工具 | 职责 |
|------|------|
| `vector_search` | 语义召回（DashScope `text-embedding-v3`） |
| `bm25_search` | 关键词 / 法条名精确匹配 |
| `lookup_article` | 法律名 + 条号直查；补交叉引用；配合截断拉全文 |

**不做 pipeline 内 RRF**：多路结果原样返回，由 Agent 在多轮中自行取舍。

### 关键参数（推荐评测配置）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_steps` | 10 | Agent 工具调用轮次上限 |
| `default_k` | 10 | 单次检索返回条数 |
| `trunc_chars` | 200 | 召回展示截断；超长法条提示 lookup |
| `rerank_pool` | 30 | vector/bm25 先召回 pool 再 rerank（评测 C 方案） |

## 方案 A / B 流水线（对照组）

- **A**：`HybridRerankRetriever` — emb + BM25 → RRF → rerank → top-5（`scripts/evaluate.py` 变体 `emb_bm25_rrf_rerank`）
- **B**：`LegalQueryDecomposer` + `retrieve_plan_single()` — 每子 query 独立 hybrid，按 `(hit_count, max_rerank_score)` 合并（`app/retrieval/plan_retrieve.py`）

## 代码入口

| 用途 | 路径 |
|------|------|
| 交互式问答 | `main.py` |
| 方案 C 评测 | `scripts/evaluate_agent_retrieval.py` |
| A–E 横评编排 | `scripts/run_compare_50.py` |
| KB 工具实现 | `app/retrieval/kb_tools.py` |
| ReAct 图 | `app/agent/graph.py` |
