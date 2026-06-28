import json
from typing import List

from langchain_core.documents import Document


def load_demo_docs() -> List[Document]:
    """离线 demo 文档（可替换为你的真实语料加载逻辑）。"""
    texts = [
        "RAG combines retrieval with generation to reduce hallucinations.",
        "BM25 is a sparse retrieval method based on inverted index and term statistics.",
        "Dense retrieval uses embeddings to find semantically similar passages.",
        "A reranker improves precision by scoring query-document pairs.",
        "In DPR, queries and passages are encoded separately, then compared by dot product.",
    ]
    return [Document(page_content=t, metadata={"source": f"demo_doc_{i}"}) for i, t in enumerate(texts)]


def load_lecoqa_corpus(corpus_path: str) -> List[Document]:
    """
    加载 LeCoQA 法律条文语料库。

    预期格式（corpus.json）：
        [{"id": 376, "title": "中华人民共和国民法典第八百三十九条", "contents": "条款正文..."}, ...]

    每条法律条文作为一个独立 Document，metadata 保留 article_id 和 article_name，
    供评估时与测试集的 match_id / match_name 精确对比。

    注意：法律条文本身已是最小粒度，不需要再 chunk。
    """
    with open(corpus_path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # JSON 数组格式
            data = json.load(f)
        else:
            # JSONL 格式（每行一个对象）
            data = [json.loads(line) for line in f if line.strip()]

    docs: List[Document] = []
    for item in data:
        # 兼容 content / contents / text 三种字段名（LeCoQA 实际使用 "content"）
        text = item.get("content") or item.get("contents") or item.get("text") or ""
        if not text.strip():
            continue
        docs.append(Document(
            page_content=text.strip(),
            metadata={
                "article_id": item["id"],
                "article_name": item.get("name") or item.get("title") or "",
                "source": "lecoqa",
            },
        ))
    return docs
