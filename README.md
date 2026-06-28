# Legal Agentic RAG v2

面向 **LeCoQA 法律法条检索** 的 Agentic RAG 系统：从 Hybrid RAG 基线出发，经固定流水线、问题拆解、多路融合等路径的系统对比，收敛为 **ReAct 自主 Agent + 解耦检索工具**，在 50 题评测上将 **Hit@5 从 0.820 提升至 0.887（+6.7pt）**，**pass@3 达 0.90**。

---

## 核心结论（50 题横评）

| 方案 | 描述 | Hit@5 | pass@3 |
|------|------|-------|--------|
| **A** | 原题直检，emb+BM25+RRF+rerank | 0.820 | — |
| **B** | LLM 拆解 2–4 子 query + hit_count 融合 | 0.773 | 0.80 |
| **C** | **ReAct Agent，三工具自选，无硬融合** | **0.887** | **0.90** |
| D | Embedding only | 0.800 | — |
| E | BM25 only | 0.680 | — |

完整指标表：[`results/compare_50/comparison_full.md`](results/compare_50/comparison_full.md)

详细实验设计：[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)

---

## 架构亮点

```
用户问题 → ReAct Agent（≤10 轮）
              ├─ vector_search    语义召回
              ├─ bm25_search      关键词匹配
              ├─ lookup_article   条号直查 / 全文补链
              └─ submit_ranking   提交 top-5 法条（评测模式）
```

- **移除** 中间 Grader 闸门与 pipeline 内 RRF 融合  
- **交给 Agent**：query 改写、通道选择、多轮取证、最终排序  
- **200 字证据截断** + lookup 按需拉全文，覆盖长法条与「详见第 X 条」引用链  

架构说明：[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## 运行示例（真实评测样本）

> 以下流程依据 `results/compare_50/arm_c_stability_50x3.json` 的真实记录整理：**问题文本、gold 法条、提交排序、工具调用次数、命中情况均为实跑数据**；工具入参（检索 query 文本）为按 ReAct 决策还原的示意。

### 示例 1 — 简单题，一次语义检索即命中（`query_id=1475`）

**用户问题**：承运人是多式联运，他把货物运输分成很多区段，把货物弄丢了，应该怎么要求赔偿？  
**Gold**：`376`（《民法典》第八百三十九条 · 多式联运）

```text
[agent]  思考：合同纠纷，从多个角度做语义召回
  ├─ vector_search(query="多式联运 货物丢失 区段承运人 赔偿责任")
  ├─ vector_search(query="多式联运经营人 全程运输责任")
  ├─ bm25_search(query="多式联运")
  └─ lookup_article(name="民法典", article_no="839")   # 核对条文全文
       ← 候选池 11 条，(article_id=376) 民法典第839条 稳定排第 1
[agent]  证据充分，按相关性排序提交
  └─ submit_ranking(article_ids=[376, 371, 40831, 11847, 40842])
       ← 提交成功，任务结束

工具调用 5 次（4 次 KB + 1 次 submit）| scoring_source=submitted | Hit@5=1 | MRR=1.0 | 6.9s
```

### 示例 2 — 复杂多 gold 题，多通道检索（`query_id=454`）

**用户问题**：医务人员严重不负责任、不及时救治导致病人死亡，是否构成医疗事故罪？  
**Gold**：`55222`（刑法第三百三十五条 · 医疗事故罪）、`55290`（刑法第三百九十七条 · 玩忽职守罪）

```text
[agent]  思考：刑事罪名认定，语义 + 关键词双通道并行
  ├─ vector_search(query="医务人员 严重不负责任 致就诊人死亡 医疗事故罪")
  ├─ bm25_search(query="医疗事故罪")              # 与上一步并行
  └─ vector_search(query="国家机关工作人员 玩忽职守 重大损失")
       ← 候选池 19 条，含 (article_id=55222) 刑法第335条
[agent]  提交最相关条文
  └─ submit_ranking(article_ids=[55222, 47561])
       ← 提交成功

工具调用 4 次（3 次 KB + 1 次 submit）| scoring_source=submitted | Hit@5=1 | Recall@5=0.5 | 8.0s
```

> **真实局限**：本题 gold 有 2 条，Agent 只召回并提交了 `55222`，漏掉「玩忽职守罪」`55290` → Recall@5 仅 0.5。这类刑民/多罪名交叉题是 `never@3` 死局的典型来源（见 `docs/EXPERIMENTS.md`）。

### 示例 3 — 步数兜底机制（`query_id=544`，3 次运行中的 run2）

**用户问题**：我可以使用别人的注册商标描述我的商品或服务特征吗？  
**Gold**：`10939`（商标法第五十九条 · 正当使用）

```text
run1 / run3：vector_search → submit_ranking([10939])     → submitted,  Hit@5=1
run2       ：vector_search 命中 10939，但 Agent 以纯文本结尾、未调用 submit
              └─ 评测图兜底：注入检索序 top-5 作为 ranking
                 → scoring_source=fallback_retrieved，仍 Hit@5=1
```

> 体现「**10 轮内自由探索 → 强制 submit → graph 兜底**」三级保障：即使 Agent 忘记提交，也用其检索序 top-5 计分，避免一次行为抖动判 0 分。

---

## 评估体系

| 层级 | 内容 | 文档 |
|------|------|------|
| 检索评测 | Hit@K / MRR / NDCG + pass@3 稳定性 | [`docs/EVALUATION.md`](docs/EVALUATION.md) |
| 端到端 Agent 评估 | 任务成功 + 轨迹（source 幻觉）+ 工程指标 | [`docs/EVALUATION_DESIGN.md`](docs/EVALUATION_DESIGN.md) |

---

## 快速开始

### 环境

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # 填入 DASHSCOPE_API_KEY
```

依赖：**DashScope API**（Qwen 对话 + text-embedding-v3 + qwen3-rerank）。

### 数据与知识库

1. 仓库已含 `data/lecoqa/test.json`（评测集）。  
2. 下载 LeCoQA `corpus.jsonl` 放到 `data/lecoqa/`（见 [`data/lecoqa/README.md`](data/lecoqa/README.md)）。  
3. 构建 KB（与实验对齐：50 题 gold + 5000 条语料采样）：

```bash
python scripts/build_kb.py --sample-test 50 --sample-corpus 5000
```

### 运行

**交互式问答（ReAct Agent）**

```bash
python main.py
```

**方案 C 检索评测（smoke）**

```bash
python scripts/evaluate_agent_retrieval.py --limit 5 \
  --max-steps 10 --default-k 10 --trunc-chars 200 \
  --rerank --rerank-pool 30 \
  --output results/agent_retrieval/smoke.json
```

**复现 A–E 横评（耗 API）**

```bash
python scripts/run_compare_50.py --arms a,b,c,d,e --limit 50
python scripts/build_compare_full_report.py
```

---

## 仓库结构

```
legal-agentic-rag-v2/
├── app/                    # Agent、检索工具、模型封装
│   ├── agent/              # LangGraph ReAct 图
│   └── retrieval/          # kb_tools, hybrid_retriever, decompose
├── scripts/                # 构建 KB、A–E 评测、横评编排
├── data/
│   ├── lecoqa/test.json    # 评测集（已含）
│   └── eval/probe_set.json # 对抗 probe（评估设计配套）
├── results/compare_50/     # 归档实验结果（JSON + 对比表）
├── docs/                   # 架构 / 实验 / 评估文档
└── main.py                 # CLI 演示
```

---

## 演进路径（为何这样设计）

1. **方案 A**：单次 hybrid RAG → 有 ceiling，且混合 RRF 有时不如纯 embedding。  
2. **Grader 节点（早期）**：二/三分类无法完备处理「弱相关证据」→ 抑制 rewrite，损害自主性。  
3. **方案 B**：多子 query + `(hit_count, rerank_score)` 融合 → **低于 A**，跨 query 分数不可比。  
4. **方案 C**：三工具解耦 + ReAct 多轮 → Agent 自行融合证据，指标最优。  

---

## 引用与数据

- 数据集：**LeCoQA** — 参见 [LeCoQA 官方仓库](https://github.com/oneal2000/LeCoQA) 获取 `corpus.jsonl` 与引用信息。  
- 模型：阿里云 DashScope（Qwen 系列）。

---

## License

本项目采用 [MIT License](LICENSE)。
