import time
from typing import Any, Dict, List

from pydantic import BaseModel, Field


# ======================================================
# 5. 只定义数据类，不包含业务逻辑
# ======================================================

# 配置
class MemoryConfig:
    """记忆系统可调参数"""

    SLIDING_WINDOW_SIZE: int = 20       # 滑动窗口保留最近 N 条消息
    SUMMARY_INTERVAL: int = 10          # 每隔 N 轮更新一次会话摘要
    MEMORY_CHROMA_DIR: str = "./memory_db"
    MEMORY_COLLECTION: str = "conversation_memory"
    MAX_RETRIEVED_MEMORIES: int = 5     # 每轮最多召回的历史记忆条数
    MEMORY_RELEVANCE_THRESHOLD: float = 0.3
    WRITE_GATE_INTERVAL: int = 2        # 每隔 N 轮执行一次写入门控

    def __init__(self, **overrides):
        for key, val in overrides.items():
            if hasattr(self, key):
                setattr(self, key, val)


# 数据结构
class MemoryEntry(BaseModel):
    """一条长期记忆"""

    content: str
    memory_type: str = Field(
        default="semantic",
        description="profile / task / preference / episodic / semantic",
    )
    timestamp: float = Field(default_factory=time.time)
    confidence: float = 1.0
    source_turn: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    """每轮对话组装好的记忆上下文"""

    windowed_messages: List[Any] = Field(default_factory=list)
    conversation_summary: str = ""
    retrieved_memories: List[str] = Field(default_factory=list)
    user_profile: Dict[str, str] = Field(default_factory=dict)
    task_state: Dict[str, Any] = Field(default_factory=dict)
