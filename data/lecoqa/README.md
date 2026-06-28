# LeCoQA 数据说明

本仓库 **已包含** 评测用测试集：

| 文件 | 说明 |
|------|------|
| `test.json` | LeCoQA 测试集（309 题）；50 题横评取前 50 条 |

**未包含**（体积大 / 可重建，见 `.gitignore`）：

| 文件 | 大小约 | 用途 |
|------|--------|------|
| `corpus.jsonl` | ~24 MB | 全库法条语料，构建 KB 必需 |
| `bm25_index.pkl` | ~5 MB | BM25 索引，由 `build_kb.py` 生成 |
| `chroma_db/` | 视采样而定 | 向量库，由 `build_kb.py` 生成 |

## 获取语料

1. 从 [LeCoQA 官方仓库](https://github.com/oneal2000/LeCoQA) 下载 `corpus.jsonl`（与 `test.json` 中 `match_id` 对应）。
2. 将 `corpus.jsonl` 放到本目录 `data/lecoqa/`。

## 构建知识库（与论文实验对齐：50 题 + 5000 条语料）

```bash
# 需已设置 DASHSCOPE_API_KEY（embedding API）
python scripts/build_kb.py --sample-test 50 --sample-corpus 5000
```

默认输出：

- Chroma：`./chroma_db/`（根目录，见 `app/config/setting.py`）
- BM25：`data/lecoqa/bm25_index.pkl`

构建完成后即可运行 `scripts/run_compare_50.py` 或各方案评测脚本。
