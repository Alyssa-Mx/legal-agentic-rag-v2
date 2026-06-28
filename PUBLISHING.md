# GitHub 发布包说明

本目录 `e:\legal-agentic-rag-v2\` 为 **v2 GitHub 发布包**（建议仓库名 `legal-agentic-rag-v2`，与旧版 `legal-agentic-rag` 区分），由 `agent4-4 -2026-06-26-13` 整理复制，未修改原工作区。

## 已包含

| 类别 | 内容 |
|------|------|
| 代码 | 完整 `app/`、`main.py`、11 个评测/构建脚本 |
| 数据 | `data/lecoqa/test.json`、`data/eval/probe_set.json` |
| 结果 | `results/compare_50/` 下 A–E 主结果 + C−BM25 消融 + 对比表 |
| 文档 | `README.md`、`docs/ARCHITECTURE.md`、`EXPERIMENTS.md`、`EVALUATION.md`、`EVALUATION_DESIGN.md` |
| 配置 | `requirements.txt`、`.gitignore`、`.env.example` |

## 故意不包含（需在 README / data README 中说明）

| 类别 | 原因 |
|------|------|
| `conversation_notes/`、`cursor日志/`、`EXP_*.md`、`AGENTS.md` | 个人研究笔记 / Cursor 配置 |
| `chroma_db/`、`bm25_index.pkl`、`corpus.jsonl`（~24MB） | 体积大，API 可重建 |
| `scripts/_tmp_*`、诊断脚本 | 一次性实验 |
| `deep-searcher-master/`、`LeCoQA/` clone | 第三方 / 冗余 |
| `results/agent_retrieval/` 中间跑分 | 仅保留 compare_50 归档 |
| 硬编码 `DASHSCOPE_API_KEY` | 已从复制脚本中 **删除** |

## 上传前检查清单

- [ ] 在 `.env` 填入 API Key，**勿提交 .env**
- [ ] 确认无其他密钥泄露（`grep -r "sk-" .`）
- [ ] 可选：添加 `LICENSE`
- [ ] 本地 `git init` → push 到新 GitHub repo

## 与原仓库关系

- 后续开发仍在 `agent4-4 -2026-06-26-13` 进行  
- 本目录为 **对外展示快照**；重大实验完成后可再次同步复制
