# ═══════════════════════════════════════════════════════════════════════════════
# 配置：API、RAG 检索参数（记忆配置在 app.memory.schemas）
# ═══════════════════════════════════════════════════════════════════════════════

import os

DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_CHAT_MODEL = "qwen-plus-2025-07-28"
DEFAULT_EMBED_MODEL = "text-embedding-v3"
CHROMA_DIR = "./chroma_db"
CHROMA_COLLECTION = "agentic_rag_demo"
RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
DEFAULT_RERANK_MODEL = "qwen3-rerank"
COARSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60                       # RRF 公式 score=Σ 1/(k+rank) 里的常数（原论文推荐 60）
RERANK_INPUT_TOP_K = 30          # RRF 融合后送给 reranker 的最大候选数（防止 reranker 输入池过大且让 RRF 排序真正起作用；0 表示不截断）
RERANK_TOP_N = 5                 # reranker 最终输出篇数（用户可见的 K）
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
KB_CONFIDENCE_THRESHOLD = 0.1   # Reranker 分数低于此阈值的文档将被过滤


def require_env(name: str) -> str:
    """读取环境变量，不存在则抛错。"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"缺少环境变量：{name}")
    return value


# 记忆系统配置见 app.memory.schemas.MemoryConfig