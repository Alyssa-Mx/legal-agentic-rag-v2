"""法律问题拆解（单库检索评测用，无域路由）。"""
from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

MAX_SUB_QUERIES = 4

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是法律问题分析器。请按【两步】处理用户问题：\n\n"
     "── 第一步：判断问题复杂度 ──\n"
     "  · simple  — 单一概念 / 单一要件 / 单一程序问题。例如：\n"
     "             '什么是合伙合同' / 'X 由谁承担责任' / 'X 的诉讼时效是多久'\n"
     "             '盗窃罪如何量刑' / '高铁占座违法吗' / '放弃诉权协议是否有效'\n"
     "             这类问题通常一句话就能命中一条法律条文，**不需要拆解**。\n"
     "  · complex — 涉及多个法律要件 / 多个法律领域 / 需要从多角度解构。例如：\n"
     "             '医生失职致人死亡构成什么罪？' （要件 + 罪名 + 量刑 + 因果）\n"
     "             '房东不退押金怎么办？' （依据 + 违约 + 救济 + 举证）\n"
     "             '盗窃和抢劫有什么区别？' （盗窃构成要件 + 抢劫构成要件 + 区别）\n\n"
     "── 第二步：根据复杂度输出 sub_queries ──\n"
     "  · simple  → 只输出 **1 个** sub_query：把口语化的原问题改写为关键词式法律"
     "    query（去口语化、提取法律术语），但**必须保留原问题的关键实体**（人物身份"
     "    / 行为 / 物品 / 时间等），不要删减事实，不要添加用户没问到的内容\n"
     "  · complex → 输出 **2-4 个** 互补的 sub_queries，每个简短关键词式\n\n"
     "【拆解维度参考（仅 complex 用）】\n"
     "  · 构成要件 — 行为是否符合 X 法律关系的构成要件？\n"
     "  · 适用法律 — 哪部法律 / 哪一编 / 哪一章规范该情形？\n"
     "  · 责任后果 — 违反后承担什么民事 / 行政 / 刑事责任？\n"
     "  · 抗辩例外 — 是否有免责 / 减责 / 时效等特殊情形？\n"
     "  · 程序救济 — 维权途径、诉讼时效、举证责任？\n\n"
     "【硬性规则】\n"
     "1) 倾向于 simple — 不确定时优先判 simple（少拆，避免噪声）\n"
     "2) sub_queries 元素必须是简短检索 query，不能是问句\n"
     "3) 不要引入用户没问到的实体或事实\n"
     "4) complexity = 'simple' 时 sub_queries 列表长度必须 = 1\n\n"
     "只输出 JSON（注意 complexity 是必填字段）：\n"
     '{{"complexity": "simple|complex", "sub_queries": ["..."], "reason": "...(≤30字)"}}'),
    ("human",
     "用户原始问题：{question}\n\n"
     "对话背景：\n{context}\n\n"
     "请输出拆解结果 JSON："),
])


class LegalQueryDecomposer:
    """把用户问题拆成 1–4 个子检索 query（与 agent4-3-1 QueryRewriter.decompose 对齐）。"""

    def __init__(self, llm) -> None:
        self._llm = llm
        self._prompt = _DECOMPOSE_PROMPT

    def decompose(
        self,
        user_message: str,
        memory_context: str = "",
        fallback_query: Optional[str] = None,
    ) -> List[str]:
        fallback = (fallback_query or user_message or "").strip()
        try:
            resp = self._llm.invoke(
                self._prompt.format_messages(
                    question=user_message,
                    context=memory_context or "（无背景）",
                )
            )
            raw = (getattr(resp, "content", "") or "").strip()
            raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE)
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not m:
                logger.warning("[Decompose] 输出无 JSON，回退原问题")
                return [fallback]
            obj = json.loads(m.group(0))
            sub = obj.get("sub_queries") or []
            if not isinstance(sub, list):
                return [fallback]

            cleaned: List[str] = []
            seen = set()
            for q in sub:
                if not isinstance(q, str):
                    continue
                q = q.strip()
                if not q or q in seen:
                    continue
                cleaned.append(q)
                seen.add(q)
                if len(cleaned) >= MAX_SUB_QUERIES:
                    break
            if not cleaned:
                return [fallback]

            complexity = (obj.get("complexity") or "").strip().lower()
            if complexity == "simple" and len(cleaned) > 1:
                cleaned = cleaned[:1]

            return cleaned
        except Exception as e:
            logger.warning("[Decompose] 失败 (%s)，回退原问题", e)
            return [fallback]
