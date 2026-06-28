# =============================================================================
# DashScope OpenAI-Compatible Chat
# =============================================================================

import json
from typing import Any, Dict, List, Optional, Sequence, Type

import requests
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config.setting import (
    DASHSCOPE_BASE_URL,
    DEFAULT_CHAT_MODEL,
    require_env,
)
from app.core.token_counter import token_counter


def to_openai_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, Any]]:
    """把 LangChain messages 转成 OpenAI ChatCompletions 兼容格式。"""
    out: List[Dict[str, Any]] = []

    for m in messages:
        cls = m.__class__.__name__

        if cls == "SystemMessage":
            out.append({"role": "system", "content": m.content})

        elif cls == "HumanMessage":
            out.append({"role": "user", "content": m.content})

        elif cls == "AIMessage":
            msg: Dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                msg["tool_calls"] = []
                for tc in tool_calls:
                    args = tc.get("args", {}) if isinstance(tc, dict) else {}
                    msg["tool_calls"].append(
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": json.dumps(args, ensure_ascii=False),
                            },
                        }
                    )
            # 透传 reasoning_content（thinking 模式必需，否则精度会下降）
            additional = getattr(m, "additional_kwargs", None) or {}
            rc = additional.get("reasoning_content")
            if rc:
                msg["reasoning_content"] = rc
            out.append(msg)

        elif cls == "ToolMessage":
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": getattr(m, "tool_call_id", ""),
                    "content": m.content or "",
                }
            )

        else:
            out.append({"role": "user", "content": getattr(m, "content", str(m))})

    return out


def pydantic_schema(model_cls: Optional[Type[BaseModel]]) -> Dict[str, Any]:
    """兼容 pydantic v1 / v2 生成 JSON schema。"""
    if model_cls is None:
        return {"type": "object", "properties": {}}

    if hasattr(model_cls, "model_json_schema"):  # pydantic v2
        return model_cls.model_json_schema()

    if hasattr(model_cls, "schema"):  # pydantic v1
        return model_cls.schema()

    return {"type": "object", "properties": {}}


def tool_to_openai_tool(tool: Any) -> Dict[str, Any]:
    """把 LangChain Tool 转成 OpenAI tools 兼容格式。"""
    name = getattr(tool, "name", None) or tool.__class__.__name__
    desc = getattr(tool, "description", "") or ""
    args_schema = getattr(tool, "args_schema", None)
    parameters = pydantic_schema(args_schema)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": parameters,
        },
    }


class QwenDashScopeChat(BaseChatModel):
    """不依赖 openai SDK 的 DashScope ChatModel（OpenAI Compatible Mode）。

    enable_thinking:
        True 时开启 hybrid 模型的深度思考模式（如 qwen-plus-2025-07-28、qwen3.5/3.7
        系列）。响应里会带 reasoning_content，自动塞到 AIMessage.additional_kwargs
        并在下一轮自动回传（thinking 模式下必须回传，否则精度下降）。
        注意：thinking 模式下 tool_choice 仅支持 auto / none，不支持 required。
    """

    model_name: str
    api_key: str
    base_url: str = DASHSCOPE_BASE_URL
    temperature: float = 0.2
    timeout: int = 60
    enable_thinking: bool = False

    # bind_tools：把工具信息存到实例上，供 _generate 透传
    def bind_tools(self, tools: Sequence[Any], tool_choice: Optional[str] = "auto", **kwargs: Any):
        new = QwenDashScopeChat(
            model_name=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            timeout=self.timeout,
            enable_thinking=self.enable_thinking,
        )
        new._lc_tools = [tool_to_openai_tool(t) for t in tools]
        # thinking 模式下 tool_choice 只支持 auto / none
        allowed = {"auto", "none"} if self.enable_thinking else {"auto", "none", "required"}
        tc = tool_choice or "auto"
        if isinstance(tc, str) and tc not in allowed:
            tc = "auto"
        new._lc_tool_choice = tc
        return new

    @property
    def _llm_type(self) -> str:
        return "qwen-dashscope-openai-compatible"

    def _chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """带瞬时错误重试的 HTTP 调用（指数退避 1s/2s/4s，最多 3 次）。

        重试的错误：
            - 网络 ConnectionError / Timeout（如 ConnectionResetError）
            - 5xx 服务端错误
            - 429 限流
        不重试：4xx 客户端错误（参数/权限问题）
        """
        import time as _time
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        max_retries = 3
        last_exc: Optional[BaseException] = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                # 5xx / 429 → 重试
                if resp.status_code >= 500 or resp.status_code == 429:
                    last_exc = RuntimeError(
                        f"DashScope chat/completions error: {resp.status_code} {resp.text[:200]}"
                    )
                    if attempt < max_retries - 1:
                        _time.sleep(2 ** attempt)
                        continue
                # 其他 4xx 不重试，直接抛
                raise RuntimeError(
                    f"DashScope chat/completions error: {resp.status_code} {resp.text}"
                )
            except (requests.ConnectionError, requests.Timeout) as e:
                last_exc = e
                if attempt < max_retries - 1:
                    _time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"DashScope connection error after {max_retries} retries: {e}")
        # 理论不会到这里
        if last_exc:
            raise RuntimeError(f"DashScope failed after {max_retries} retries: {last_exc}")
        raise RuntimeError("DashScope chat/completions unknown failure")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = getattr(self, "_lc_tools", None)
        tool_choice = getattr(self, "_lc_tool_choice", None)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": to_openai_messages(messages),
            "temperature": self.temperature,
        }
        if stop:
            payload["stop"] = stop

        if tools:
            payload["tools"] = tools
            allowed_tc = {"auto", "none"} if self.enable_thinking else {"auto", "none", "required"}
            tc = tool_choice or "auto"
            if isinstance(tc, str) and tc not in allowed_tc:
                tc = "auto"
            payload["tool_choice"] = tc

        # 开启深度思考（hybrid 模型，如 qwen-plus-2025-07-28 / qwen3.5/3.7 系列）
        if self.enable_thinking:
            payload["enable_thinking"] = True

        data = self._chat_completions(payload)
        msg = data["choices"][0]["message"]

        content = msg.get("content") or ""
        reasoning_content = msg.get("reasoning_content") or ""
        tool_calls_raw = msg.get("tool_calls") or []

        tool_calls: List[Dict[str, Any]] = []
        for tc in tool_calls_raw:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            arguments = fn.get("arguments", "{}")
            try:
                args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
            except Exception:
                args = {"_raw": arguments}
            tool_calls.append({"id": tc.get("id", ""), "name": name, "args": args})

        # 抽取 usage 喂进全局 counter，并保留在 response_metadata 便于按消息追溯
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        if prompt_tokens or completion_tokens:
            token_counter.add(prompt_tokens, completion_tokens)

        response_metadata = {"model_name": self.model_name}
        if usage:
            response_metadata["token_usage"] = usage

        additional_kwargs: Dict[str, Any] = {}
        if reasoning_content:
            additional_kwargs["reasoning_content"] = reasoning_content

        # 注意：即使没有 tool_calls，也保持 tool_calls 字段一致，避免分支逻辑出错
        ai = AIMessage(
            content=content,
            tool_calls=tool_calls,
            additional_kwargs=additional_kwargs,
            response_metadata=response_metadata,
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])

def build_chat(
    model_name: str = DEFAULT_CHAT_MODEL,
    enable_thinking: bool = False,
    timeout: int = 60,
) -> QwenDashScopeChat:
    return QwenDashScopeChat(
        model_name=model_name,
        api_key=require_env("DASHSCOPE_API_KEY"),
        base_url=DASHSCOPE_BASE_URL,
        temperature=0.2,
        timeout=timeout,
        enable_thinking=enable_thinking,
    )