"""
知识库相关的 Agent 工具集合（向量检索 / BM25 检索 / 法条精确定位）。

设计原则：
    - 不再使用混合检索（experiments 显示纯向量检索效果更好）
    - 三个工具职责清晰，让 Agent 在 ReAct 中自主选择：
        vector_search   — 语义匹配，默认首选
        bm25_search     — 精确关键词匹配，专有名词/罕见词更准
        lookup_article  — 已知法律名 + 条号时的直查
    - 所有工具复用同一份 Chroma + BM25 索引，避免重复构建
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Type

import jieba
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

import pickle

from app.config.setting import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_EMBED_MODEL,
)
from app.models.dashscope_embeddings import build_embeddings

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  公共辅助：分词 + 中文数字转换 + 文档格式化
# ════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    """中英文混合分词（BM25 用）。"""
    return [t for t in jieba.lcut(text) if t.strip()]


_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def _cn_to_int(s: str) -> int:
    """
    中文数字串转整数：'五百八十四' → 584，'一千二百三十' → 1230，'一万零五' → 10005。
    解析失败返回 0。
    """
    if not s:
        return 0
    total, section, cur = 0, 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            cur = _CN_DIGITS[ch]
        elif ch == "十":
            section += (cur or 1) * 10
            cur = 0
        elif ch == "百":
            section += (cur or 1) * 100
            cur = 0
        elif ch == "千":
            section += (cur or 1) * 1000
            cur = 0
        elif ch == "万":
            section = (section + cur) * 10000
            total += section
            section, cur = 0, 0
        elif ch == "亿":
            section = (section + cur) * 100000000
            total += section
            section, cur = 0, 0
    return total + section + cur


_ARABIC_NUM_RE = re.compile(r"(\d+)\s*条")
_CN_NUM_RE = re.compile(r"第([零〇一二两三四五六七八九十百千万亿]+)条")
_BARE_NUM_RE = re.compile(r"(\d+)")


def _extract_article_number(text: str) -> Optional[int]:
    """从 '民法典第五百八十四条' / '民法典 584 条' / '民法典 584' 中提取条号。"""
    m = _ARABIC_NUM_RE.search(text)
    if m:
        return int(m.group(1))
    m = _CN_NUM_RE.search(text)
    if m:
        n = _cn_to_int(m.group(1))
        return n if n > 0 else None
    m = _BARE_NUM_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def _extract_law_keyword(text: str) -> str:
    """从查询中剥离条号部分，剩下的视为"法律名关键词"。"""
    cleaned = _CN_NUM_RE.sub("", text)
    cleaned = _ARABIC_NUM_RE.sub("", cleaned)
    cleaned = re.sub(r"\d+", "", cleaned)
    cleaned = re.sub(r"[第条款项的\s,，。;；:：]+", "", cleaned)
    return cleaned.strip()


def _rerank_docs(reranker: Any, query: str, docs: List[Document], top_n: int) -> List[Document]:
    """
    使用 reranker 对召回候选做精排，返回相关性前 top_n 篇。
    若 reranker 失败/降级，则保持原顺序截取前 top_n。
    """
    if reranker is None or not docs:
        return docs[:top_n]
    try:
        contents = [d.page_content for d in docs]
        ranked = reranker.rerank(query=query, documents=contents, top_n=top_n)
        if not ranked:
            return docs[:top_n]
        out: List[Document] = []
        for idx, _score in ranked:
            if 0 <= idx < len(docs):
                out.append(docs[idx])
        return out if out else docs[:top_n]
    except Exception as e:
        logger.warning("[_rerank_docs] reranker 异常，回退原顺序: %s", e)
        return docs[:top_n]


def _format_docs(
    docs: List[Document],
    limit: int = 5,
    trunc_chars: Optional[int] = None,
) -> str:
    """
    把文档列表格式化成 Agent 可读的字符串。

    格式（不截断）：
        [1] (article_id=839) 中华人民共和国民法典第八百三十九条
        当事人一方不履行...

    截断模式（trunc_chars=200）：超过阈值的条文截断前 trunc_chars 字 +
    在尾部加 escape hint，让 Agent 知道可调用 lookup_article 取全文：
        [3] (article_id=1327) 监察法第五十五条
        监察机关在调查中可以依法采取下列措施……（前 200 字）
        [...本条共 612 字，已截断；如需全文，请调用 lookup_article("监察法第五十五条")]

    article_id 标签让 Agent 可以在后续步骤精确引用具体法条（评测时也需要
    Agent 提交 article_id 列表作为 ranking）。
    """
    if not docs:
        return "（无结果）"
    parts: List[str] = []
    for i, doc in enumerate(docs[:limit], 1):
        meta = doc.metadata or {}
        title = meta.get("article_name") or meta.get("source") or ""
        article_id = meta.get("article_id")
        if article_id is not None:
            header = f"[{i}] (article_id={article_id}) {title}".rstrip()
        else:
            header = f"[{i}] {title}" if title else f"[{i}]"
        body = doc.page_content.strip()
        if trunc_chars is not None and len(body) > trunc_chars:
            kept = body[:trunc_chars].rstrip()
            remaining = len(body) - trunc_chars
            # 用 article_name 作为 lookup_article 的精确 query 提示
            hint_name = title or f"article_id={article_id}"
            body = (
                f"{kept}……\n"
                f"  [...本条共 {len(body)} 字，已截断 {remaining} 字；"
                f"如需全文请调用 lookup_article(query=\"{hint_name}\")]"
            )
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


# ════════════════════════════════════════════════════════════════════════════
#  工具 1：vector_search —— 纯向量语义检索
# ════════════════════════════════════════════════════════════════════════════

class _VectorSearchArgs(BaseModel):
    query: str = Field(..., description="语义检索 query，可以是自然语言问题或关键词短语")
    k: int = Field(10, description="返回的文档数量，建议 8-12；若召回过窄可调到 15")


class VectorSearchTool(BaseTool):
    name: str = "vector_search"
    description: str = (
        "向量语义检索：在本地法律知识库中按【语义相似度】查找相关条款。"
        "适用场景：用自然语言提问、不知道精确法条号、需要找意思相近的内容时。"
        "实测在法律语料上效果优于混合检索，应作为默认首选检索工具。"
    )
    args_schema: Type[BaseModel] = _VectorSearchArgs

    vectorstore: Any = Field(default=None, exclude=True)
    default_k: int = Field(default=10, exclude=True)
    trunc_chars: Optional[int] = Field(default=None, exclude=True)
    reranker: Any = Field(default=None, exclude=True)
    rerank_pool_size: int = Field(default=20, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vectorstore: Any,
        default_k: int = 10,
        trunc_chars: Optional[int] = None,
        reranker: Any = None,
        rerank_pool_size: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.vectorstore = vectorstore
        self.default_k = default_k
        self.trunc_chars = trunc_chars
        self.reranker = reranker
        self.rerank_pool_size = rerank_pool_size

    def _run(self, query: str, k: Optional[int] = None) -> str:
        k = k or self.default_k
        # 若启用 reranker，则先召回更宽的候选池（pool_size），再精排取 top-k
        pool_k = max(k, self.rerank_pool_size) if self.reranker is not None else k
        try:
            docs = self.vectorstore.similarity_search(query, k=pool_k)
        except Exception as e:
            logger.warning("[vector_search] 调用失败: %s", e)
            return f"[vector_search error] {e}"
        if self.reranker is not None and len(docs) > 1:
            docs = _rerank_docs(self.reranker, query, docs, top_n=k)
        logger.debug(
            "[vector_search] query='%.40s' pool=%d -> 返回 %d 条 (rerank=%s)",
            query, pool_k, len(docs), self.reranker is not None,
        )
        return _format_docs(docs, limit=k, trunc_chars=self.trunc_chars)

    async def _arun(self, query: str, k: Optional[int] = None) -> str:
        return self._run(query=query, k=k)


# ════════════════════════════════════════════════════════════════════════════
#  工具 2：bm25_search —— 纯稀疏关键词检索
# ════════════════════════════════════════════════════════════════════════════

class _BM25SearchArgs(BaseModel):
    query: str = Field(..., description="关键词检索 query，越精准越好（如专有名词、罕见术语）")
    k: int = Field(10, description="返回的文档数量，建议 8-12；若召回过窄可调到 15")


class BM25SearchTool(BaseTool):
    name: str = "bm25_search"
    description: str = (
        "BM25 稀疏关键词检索：基于词频的精确匹配。"
        "适用场景：query 含罕见专有名词、人名、机构名、法条编号片段等需要"
        "【字面匹配】的情况；当 vector_search 召回不相关时可作为补充。"
        "不擅长理解同义词或语义相近的内容。"
    )
    args_schema: Type[BaseModel] = _BM25SearchArgs

    bm25_index: Any = Field(default=None, exclude=True)
    corpus_docs: List[Document] = Field(default_factory=list, exclude=True)
    default_k: int = Field(default=10, exclude=True)
    trunc_chars: Optional[int] = Field(default=None, exclude=True)
    reranker: Any = Field(default=None, exclude=True)
    rerank_pool_size: int = Field(default=20, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        bm25_index: Any,
        corpus_docs: List[Document],
        default_k: int = 10,
        trunc_chars: Optional[int] = None,
        reranker: Any = None,
        rerank_pool_size: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.bm25_index = bm25_index
        self.corpus_docs = corpus_docs
        self.default_k = default_k
        self.trunc_chars = trunc_chars
        self.reranker = reranker
        self.rerank_pool_size = rerank_pool_size

    def _run(self, query: str, k: Optional[int] = None) -> str:
        k = k or self.default_k
        if self.bm25_index is None or not self.corpus_docs:
            return "[bm25_search] BM25 索引未初始化。"
        pool_k = max(k, self.rerank_pool_size) if self.reranker is not None else k
        try:
            tokens = _tokenize(query)
            if not tokens:
                return "（query 分词后为空）"
            scores = self.bm25_index.get_scores(tokens)
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pool_k]
            docs = [self.corpus_docs[i] for i in top_idx if scores[i] > 0]
        except Exception as e:
            logger.warning("[bm25_search] 调用失败: %s", e)
            return f"[bm25_search error] {e}"
        if self.reranker is not None and len(docs) > 1:
            docs = _rerank_docs(self.reranker, query, docs, top_n=k)
        logger.debug(
            "[bm25_search] query='%.40s' pool=%d -> 返回 %d 条 (rerank=%s)",
            query, pool_k, len(docs), self.reranker is not None,
        )
        return _format_docs(docs, limit=k, trunc_chars=self.trunc_chars)

    async def _arun(self, query: str, k: Optional[int] = None) -> str:
        return self._run(query=query, k=k)


# ════════════════════════════════════════════════════════════════════════════
#  工具 3：lookup_article —— 按法律名 + 条款号精确定位
# ════════════════════════════════════════════════════════════════════════════

class _LookupArticleArgs(BaseModel):
    query: str = Field(
        ...,
        description=(
            "形如『民法典 584 条』『刑法第二百三十二条』『中华人民共和国民法典第八百三十九条』"
            "的精确法条引用；支持阿拉伯数字、中文数字、混合表达"
        ),
    )
    k: int = Field(3, description="返回匹配文档数量上限，通常 1-3")


class LookupArticleTool(BaseTool):
    name: str = "lookup_article"
    description: str = (
        "按法律名 + 条款号【精确定位】法条原文。"
        "适用场景：用户已经明确给出某部法律的某一条款（如『民法典 584 条』），"
        "此时直查比 vector_search / bm25_search 都更准。"
        "支持阿拉伯数字与中文数字（如『第五百八十四条』）混用。"
    )
    args_schema: Type[BaseModel] = _LookupArticleArgs

    index: Dict[Tuple[str, int], List[Document]] = Field(default_factory=dict, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, docs: List[Document], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.index = self._build_index(docs)
        logger.info("[lookup_article] 法条索引构建完成，共 %d 个 (法名, 条号) 键", len(self.index))

    @staticmethod
    def _build_index(docs: List[Document]) -> Dict[Tuple[str, int], List[Document]]:
        idx: Dict[Tuple[str, int], List[Document]] = {}
        for doc in docs:
            name = (doc.metadata or {}).get("article_name", "") or ""
            if not name:
                continue
            art_num = _extract_article_number(name)
            if art_num is None:
                continue
            # 法律名 = 去掉「第XXX条」之后的部分（如 "中华人民共和国民法典"）
            law_name = _CN_NUM_RE.sub("", name)
            law_name = _ARABIC_NUM_RE.sub("", law_name)
            law_name = re.sub(r"第?\d+条?$", "", law_name).strip()
            idx.setdefault((law_name, art_num), []).append(doc)
        return idx

    def _run(self, query: str, k: int = 3) -> str:
        art_num = _extract_article_number(query)
        if art_num is None:
            return f"[lookup_article] 未能从 '{query}' 中识别出条款号；请调用 vector_search 试试。"
        law_kw = _extract_law_keyword(query)

        matches: List[Document] = []
        for (law_name, num), docs in self.index.items():
            if num != art_num:
                continue
            if law_kw and law_kw not in law_name:
                continue
            matches.extend(docs)

        if not matches:
            return (
                f"⚠️ [lookup_article] 该法条不在当前 KB 中："
                f"'{query}' →（解析为：法律='{law_kw or '任意'}', 条号={art_num}）。\n"
                f"【重要】当前 KB 仅含约 5000 条采样法条，不一定包含你想到的所有法条。"
                f"**严禁对同一条号反复尝试 lookup_article**——它真的不存在，再查也不会出现。\n"
                f"正确做法：改用 vector_search 描述【问题场景/法律事实】（不要带条号），"
                f"让语义检索从 KB 实际拥有的法条里找最相关的。"
            )
        logger.debug(
            "[lookup_article] query='%.40s' → 法律='%s' 条号=%d 命中 %d 条",
            query, law_kw, art_num, len(matches),
        )
        return _format_docs(matches, limit=k)

    async def _arun(self, query: str, k: int = 3) -> str:
        return self._run(query=query, k=k)


# ════════════════════════════════════════════════════════════════════════════
#  工厂函数：一键构建三件套
# ════════════════════════════════════════════════════════════════════════════

def build_kb_tools(
    docs: List[Document],
    default_k: int = 10,
    trunc_chars: Optional[int] = None,
    reranker: Any = None,
    rerank_pool_size: int = 20,
) -> List[BaseTool]:
    """
    用一份文档构建三个 KB 工具：vector_search / bm25_search / lookup_article。

    所有工具共享同一份 Chroma 向量库 + 同一份 BM25 索引，避免重复 embedding。

    Args:
        docs:             原始法条文档
        default_k:        vector_search / bm25_search 默认 k（当 LLM 不显式传 k 时用）
        trunc_chars:      单条法条字数截断阈值。超过的会被截断 + 提示 lookup_article 取全文。
                          None=不截断（全文返回）。lookup_article 永远返回全文。
        reranker:         可选；提供时 vector/bm25 工具会先召回 pool 再精排取前 k
        rerank_pool_size: 启用 reranker 时的召回池大小（默认 20）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    embeddings = build_embeddings(DEFAULT_EMBED_MODEL)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
    )

    tokenized_corpus = [_tokenize(d.page_content) for d in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    logger.info("[kb_tools] BM25 索引构建完成，共 %d 个文档块", len(tokenized_corpus))

    lookup_docs = docs if docs else chunks

    return [
        VectorSearchTool(
            vectorstore=vectordb, default_k=default_k, trunc_chars=trunc_chars,
            reranker=reranker, rerank_pool_size=rerank_pool_size,
        ),
        BM25SearchTool(
            bm25_index=bm25_index, corpus_docs=chunks,
            default_k=default_k, trunc_chars=trunc_chars,
            reranker=reranker, rerank_pool_size=rerank_pool_size,
        ),
        LookupArticleTool(docs=lookup_docs),
    ]


def load_kb_tools(
    bm25_path: str = "data/lecoqa/bm25_index.pkl",
    chroma_dir: str = CHROMA_DIR,
    collection: str = CHROMA_COLLECTION,
    default_k: int = 10,
    trunc_chars: Optional[int] = None,
    reranker: Any = None,
    rerank_pool_size: int = 20,
) -> List[BaseTool]:
    """
    从已构建好的 Chroma + BM25 pickle 加载三个 KB 工具（不重新 embedding）。

    Args:
        default_k:        vector_search / bm25_search 默认 k
        trunc_chars:      单条法条字数截断阈值（lookup_article 不受影响，永远全文）
        reranker:         可选；提供时 vector/bm25 工具会先召回 pool 再精排取前 k
        rerank_pool_size: 启用 reranker 时的召回池大小
    """
    embeddings = build_embeddings(DEFAULT_EMBED_MODEL)
    vectordb = Chroma(
        collection_name=collection,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

    with open(bm25_path, "rb") as f:
        bm25_index, bm25_corpus_docs = pickle.load(f)
    logger.info(
        "[kb_tools] 已加载 BM25 索引: %d 个 chunk，Chroma: %s, default_k=%d, "
        "trunc_chars=%s, reranker=%s, rerank_pool=%d",
        len(bm25_corpus_docs), chroma_dir, default_k, trunc_chars,
        "ON" if reranker is not None else "OFF", rerank_pool_size,
    )

    return [
        VectorSearchTool(
            vectorstore=vectordb, default_k=default_k, trunc_chars=trunc_chars,
            reranker=reranker, rerank_pool_size=rerank_pool_size,
        ),
        BM25SearchTool(
            bm25_index=bm25_index, corpus_docs=bm25_corpus_docs,
            default_k=default_k, trunc_chars=trunc_chars,
            reranker=reranker, rerank_pool_size=rerank_pool_size,
        ),
        LookupArticleTool(docs=bm25_corpus_docs),
    ]
