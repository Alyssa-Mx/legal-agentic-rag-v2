# 评估体系

本项目包含 **两层评估**，对应简历中「检索 gold 对齐」与「端到端 Agent 三维评估」：

## 1. 检索评测（现仓库主协议）

**脚本**：`scripts/evaluate_agent_retrieval.py`、`scripts/run_stability_50x3.py`

**任务**：Agent 不生成答案，通过 `submit_ranking` 提交 top-5 `article_id`，与 LeCoQA `match_id` 对比。

**指标**：

| 指标 | 含义 |
|------|------|
| Hit@K | top-K 是否至少命中一条 gold |
| Recall@K | top-K 命中 gold 数 / gold 总数 |
| MRR | 首个 gold 排名倒数 |
| NDCG@K | 位置折扣排序质量 |
| pass@3 / never@3 | 同题跑 3 次的稳定性 |

**打分来源优先级**：

1. `submitted` — Agent 调用 `submit_ranking`
2. `fallback_retrieved` — 步数用尽时检索序 top-5
3. `empty` — 全 0

A/B/C/D/E 横评共用 `compute_retrieval_metrics`，保证方案间可比。

## 2. 端到端 Agent 评估（设计文档）

完整设计见 [`docs/EVALUATION_DESIGN.md`](EVALUATION_DESIGN.md)（早期 8 节点架构下实现，**方法论仍适用**）。

### 三维框架

| 维度 | 关注点 | 代表指标 |
|------|--------|----------|
| **任务成功** | 检索 + 答案是否达标 | Hit@K、LLM-as-Judge 四维 |
| **轨迹质量** | 工具选对了吗、有编造吗 | tool_choice_rate、**source 幻觉** |
| **工程可用性** | 成本与稳定性 | 延迟、pass@k |

### 核心原则

> Agent 只有在 **最终结果正确且路径合规** 时，才算真正成功。

- **faithfulness**（Judge）：答案事实是否来自 KB 证据 — 区分「RAG 有效」与「模型先验蒙对」
- **source 幻觉**（规则）：答案中《X法》第Y条 是否在 `kb_evidence` 中出现
- **pass@k**：随机 Agent 必须重复跑 k 次，避免 lucky pass

### 对抗集

`data/eval/probe_set.json` — 10 题 probe，用于边界行为探测（工具幻觉、参数错误等）。

## 3. 与简历表述的对应

| 简历 bullet | 本仓库材料 |
|-------------|------------|
| Hit@5 0.820→0.887 | `results/compare_50/comparison_full.md` |
| pass@3 0.90 | `arm_c_stability_50x3.json` |
| 三维评估体系 | `docs/EVALUATION_DESIGN.md` + 本节 |
| A–E 方案对比 | `docs/EXPERIMENTS.md` |

## 4. 复现检索评测（推荐配置）

```bash
python scripts/evaluate_agent_retrieval.py \
  --limit 50 --max-steps 10 --default-k 10 --trunc-chars 200 \
  --rerank --rerank-pool 30 \
  --output results/agent_retrieval/run.json

python scripts/run_stability_50x3.py \
  --limit 50 --runs 3 --rerank --rerank-pool 30 \
  --output results/agent_retrieval/stability.json
```
