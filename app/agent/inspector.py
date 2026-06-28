import json
import logging
import re
from typing import Any, Dict

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_DEFAULT_CHECK: Dict[str, Any] = {
    "used_kb": True,
    "profile_consistent": True,
    "hallucination_risk": "low",
    "needs_revision": False,
    "notes": [],
}


# ======================================================
# 3. 功能模块层 - 回答生成后的质检：LLM 评估答案质量，判断是否存在问题
#    check() — 评估：（每轮）把这些东西一起送给 LLM 审查：用户问题 + 刚生成的答案 + 知识库证据 + 用户画像 + 路由是否要求用KB
#    repair() — 修正：（当 needs_revision=True）用低温重新生成一次：原始问题 + 原始答案 + 自检发现的问题 + KB证据 + 用户画像 → 输出修正后的答案
#    输出: {"used_kb": true, "profile_consistent": true, "hallucination_risk": "low|medium|high", "needs_revision": true, "notes": ["..."]}
# ======================================================
class AnswerInspector:
    """
    回答后自检器。

    流程：
        1. check()  — 用 LLM 评估答案质量（KB 使用、画像一致性、幻觉风险）
        2. repair() — 若 needs_revision=True，触发轻量修正（temperature=0.1）

    工业实践：低风险先打分记录，高风险才触发二次修正，避免每轮都多一次 LLM 调用。
    """

    _CHECK_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个回答质量审查员。评估助手的回答，判断是否存在以下问题。\n\n"
         "只输出 JSON，格式：\n"
         "{{\"used_kb\": bool, "
         "\"profile_consistent\": bool, "
         "\"hallucination_risk\": \"low|medium|high\", "
         "\"needs_revision\": bool, "
         "\"notes\": [\"问题描述1\", ...]}}"),
        ("human",
         "用户问题：{question}\n\n"
         "助手回答：{answer}\n\n"
         "知识库证据：\n{kb_evidence}\n\n"
         "用户画像：\n{user_profile}\n\n"
         "路由是否要求使用知识库：{expected_use_kb}\n\n"
         "【评估要点】\n"
         "1. used_kb: 答案是否实际利用了知识库证据（而非凭空生成）\n"
         "2. profile_consistent: 答案是否符合用户画像中的偏好和身份\n"
         "3. hallucination_risk: 答案中是否包含知识库里没有的内容（低/中/高）\n"
         "4. needs_revision: 是否需要修正（hallucination_risk=high 或 used_kb=false 且 expected_use_kb=true 时）\n\n"
         "请输出评估 JSON："),
    ])

    _REPAIR_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个回答修正专家。根据自检报告和证据，对原始回答进行最小化修正。\n\n"
         "修正原则：\n"
         "- 删除无据可查的内容，如实说明信息不足\n"
         "- 利用知识库证据补充准确信息\n"
         "- 保持与用户画像一致\n"
         "- 保持简洁，不过度扩展\n\n"
         "直接输出修正后的回答，不要加前缀或解释。"),
        ("human",
         "用户问题：{question}\n\n"
         "原始回答：{answer}\n\n"
         "自检发现的问题：\n{issues}\n\n"
         "知识库证据：\n{kb_evidence}\n\n"
         "用户画像：\n{user_profile}\n\n"
         "请输出修正后的回答："),
    ])

    def __init__(self, llm) -> None:
        self._llm = llm

    def check(
        self,
        question: str,
        answer: str,
        kb_evidence: str,
        user_profile: str,
        expected_use_kb: bool,
    ) -> Dict[str, Any]:
        try:
            resp = self._llm.invoke(
                self._CHECK_PROMPT.format_messages(
                    question=question,
                    answer=answer,
                    kb_evidence=kb_evidence or "（无知识库证据）",
                    user_profile=user_profile or "（无用户画像）",
                    expected_use_kb=expected_use_kb,
                )
            )
            raw = (getattr(resp, "content", "") or "").strip()
            raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.IGNORECASE)
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if m:
                result = {**_DEFAULT_CHECK, **json.loads(m.group(0))}
                logger.debug("[Inspector] 自检结果: %s", result)
                return result
        except Exception as e:
            logger.warning("[Inspector] 自检失败，使用默认结果: %s", e)

        return dict(_DEFAULT_CHECK)

    def repair(
        self,
        question: str,
        answer: str,
        check_result: Dict[str, Any],
        kb_evidence: str,
        user_profile: str,
    ) -> str:
        issues = "\n".join(check_result.get("notes") or []) or "答案质量不足，存在潜在幻觉"
        try:
            resp = self._llm.invoke(
                self._REPAIR_PROMPT.format_messages(
                    question=question,
                    answer=answer,
                    issues=issues,
                    kb_evidence=kb_evidence or "（无知识库证据）",
                    user_profile=user_profile or "（无用户画像）",
                )
            )
            repaired = (getattr(resp, "content", "") or "").strip()
            logger.info(
                "[Inspector] 答案修正完成，原长 %d → 新长 %d（风险: %s）",
                len(answer), len(repaired), check_result.get("hallucination_risk"),
            )
            return repaired or answer
        except Exception as e:
            logger.warning("[Inspector] 修正失败，返回原始答案: %s", e)
            return answer
