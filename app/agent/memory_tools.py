"""
让 Agent 主动调用的记忆相关工具。

设计目的：
    在 ReAct 架构下，把"记忆使用权"还给 Agent。原本 prepare_context 会
    强制把召回的历史记忆和用户画像塞进 system prompt，现在改为：
        - 默认 system prompt 只携带【会话摘要】（强 context）
        - 当 Agent 判断"这个问题可能与历史交互相关"时，主动调
          recall_memory / list_user_profile 拉详细信息

    这样让"何时使用记忆"成为模型自己的决策，与"何时检索 KB"对称。
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
#  工具 1：recall_memory —— 从长期记忆中语义检索历史片段
# ════════════════════════════════════════════════════════════════════════════

class _RecallMemoryArgs(BaseModel):
    query: str = Field(..., description="检索历史记忆的 query，描述你想找的历史信息")
    k: int = Field(3, description="返回的记忆条数，建议 2-5")


class RecallMemoryTool(BaseTool):
    name: str = "recall_memory"
    description: str = (
        "从长期记忆中按语义检索相关的历史对话片段。"
        "适用场景：用户问题含『上次/之前/还记得吗/我们讨论过』等指代时；"
        "或当前问题与之前某个话题可能相关、需要确认时。"
        "返回的是过往对话中提炼的事实/事件片段，不是用户偏好（偏好请用 list_user_profile）。"
    )
    args_schema: Type[BaseModel] = _RecallMemoryArgs

    memory_manager: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, memory_manager: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.memory_manager = memory_manager

    def _run(self, query: str, k: int = 3) -> str:
        if self.memory_manager is None:
            return "[recall_memory] 记忆系统未启用。"
        try:
            docs = self.memory_manager._long.search(query, top_k=k)
        except Exception as e:
            logger.warning("[recall_memory] 调用失败: %s", e)
            return f"[recall_memory error] {e}"
        if not docs:
            return "（未检索到与该 query 相关的历史记忆）"
        lines = [f"- {d.page_content}" for d in docs]
        logger.debug("[recall_memory] query='%.40s' 返回 %d 条记忆", query, len(docs))
        return "【相关历史记忆】\n" + "\n".join(lines)

    async def _arun(self, query: str, k: int = 3) -> str:
        return self._run(query=query, k=k)


# ════════════════════════════════════════════════════════════════════════════
#  工具 2：list_user_profile —— 读取用户画像（结构化偏好/身份）
# ════════════════════════════════════════════════════════════════════════════

class _ListUserProfileArgs(BaseModel):
    category: Optional[str] = Field(
        None,
        description="可选过滤：'profile'(身份背景) 或 'preference'(偏好习惯)；不传则返回全部",
    )


class ListUserProfileTool(BaseTool):
    name: str = "list_user_profile"
    description: str = (
        "读取当前用户的画像信息（结构化的身份背景和个人偏好条目）。"
        "适用场景：当回答需要个性化（如默认回复语言、专业方向、过往咨询主题），"
        "或需要确认用户身份/角色（如『我是租客还是房东』）时调用。"
        "可选参数 category='profile'/'preference' 进一步过滤。"
    )
    args_schema: Type[BaseModel] = _ListUserProfileArgs

    memory_manager: Any = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, memory_manager: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.memory_manager = memory_manager

    def _run(self, category: Optional[str] = None) -> str:
        if self.memory_manager is None:
            return "[list_user_profile] 记忆系统未启用。"
        try:
            if category in ("profile", "preference"):
                entries = list(self.memory_manager._long.get_structured(category))
            else:
                entries = (
                    list(self.memory_manager._long.get_structured("profile"))
                    + list(self.memory_manager._long.get_structured("preference"))
                )
        except Exception as e:
            logger.warning("[list_user_profile] 调用失败: %s", e)
            return f"[list_user_profile error] {e}"
        if not entries:
            return "（暂无用户画像信息）"
        lines = [f"- [{e.memory_type}] {e.content}" for e in entries]
        logger.debug("[list_user_profile] category=%s 返回 %d 条", category, len(entries))
        return "【用户画像】\n" + "\n".join(lines)

    async def _arun(self, category: Optional[str] = None) -> str:
        return self._run(category=category)


# ════════════════════════════════════════════════════════════════════════════
#  工厂函数
# ════════════════════════════════════════════════════════════════════════════

def build_memory_tools(memory_manager: Any) -> List[BaseTool]:
    """构建记忆相关工具列表。memory_manager 为 None 时返回空列表。"""
    if memory_manager is None:
        return []
    return [
        RecallMemoryTool(memory_manager=memory_manager),
        ListUserProfileTool(memory_manager=memory_manager),
    ]
