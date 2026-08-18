"""Self-contained chat layer (OpenAI-compatible ``/chat/completions``).

Provides the same adapter semantics as YUSU's ``LangChainChatAdapter``
(``call`` / ``call_collect`` / ``GeneralResponse``) without the LangChain
dependency: a thin async httpx client with SSE streaming. Env contract:

- ``YUSU_LLM_BASE_URL``
- ``YUSU_LLM_API_KEY``
- ``YUSU_LLM_MODEL``
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from yusu_kb.utils.http import _normalize_endpoint
from yusu_kb.utils.logger import logger


class ChatModelError(Exception):
    """Model call failure (URL/model context attached)."""


def _describe_error(exc: Exception) -> str:
    """Format an exception for model-call error messages.

    Some transport-level exceptions (e.g. httpx.ReadError) carry an empty
    str(); the type name keeps the cause visible so upstream retry logic
    can classify the failure as transient.
    """
    detail = str(exc)
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


class GeneralResponse:
    def __init__(self, content, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        # Raw ``choices[0].message.tool_calls`` list (OpenAI shape) when the
        # model requests tool use; None otherwise.
        self.tool_calls = tool_calls
        self.is_full = False


class OpenAIChatAdapter:
    def __init__(
        self,
        model,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        # 默认 300s 与图谱抽取 EXTRACTION_STREAM_CHUNK_TIMEOUT 对齐：sensenova 等
        # 服务在高峰期排队/首 token 延迟可能超过 120s，非流式 call 的硬超时与流式
        # chunk 间隔超时都会误杀长请求（KG-9 补充：adapter 层同样需要放宽）。
        timeout: float = 300.0,
        transport=None,
        headers: dict | None = None,
    ):
        self.model = model
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        # Test seam: an httpx transport to inject (e.g. MockTransport).
        self._transport = transport
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if headers:
            self.headers.update({k: v for k, v in headers.items() if k not in ("Authorization", "Content-Type")})

    def _client(self) -> httpx.AsyncClient:
        kwargs = {"timeout": self.timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    @staticmethod
    def _normalize_messages(message):
        if isinstance(message, str):
            return [{"role": "user", "content": message}]
        if isinstance(message, (list, tuple)):
            return list(message)
        raise ValueError(f"Unsupported message type: {type(message).__name__}")

    async def _post(self, payload: dict[str, Any], *, stream: bool = False) -> httpx.Response:
        payload = dict(payload)
        payload["stream"] = stream
        try:
            async with self._client() as client:
                response = await client.post(self.base_url, json=payload, headers=self.headers)
                response.raise_for_status()
                return response
        except Exception as e:
            err = f"Error calling model: {_describe_error(e)}, URL: {self.base_url}, Model: {self.model_name}"
            logger.error(err)
            raise ChatModelError(err) from e

    async def call(self, message, stream=False, **model_kwargs):
        messages = self._normalize_messages(message)
        payload = {"model": self.model, "messages": messages, **model_kwargs}
        try:
            if stream:
                return self._stream_response(payload)
            response = await self._post(payload, stream=False)
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            # 兼容两种字段名：OpenAI 惯例 reasoning_content 与部分厂商（如商汤）的 reasoning
            reasoning = choice["message"].get("reasoning_content") or choice["message"].get("reasoning")
            # 透传 tool_calls（OpenAI 形状 list[dict]，含 id/type/function{name,arguments}）
            tool_calls = choice["message"].get("tool_calls")
            return GeneralResponse(content, reasoning_content=reasoning, tool_calls=tool_calls)
        except Exception as e:
            err = f"Error calling model: {_describe_error(e)}, URL: {self.base_url}, Model: {self.model_name}"
            logger.error(err)
            raise ChatModelError(err) from e

    async def _stream_response(self, payload: dict[str, Any]):
        payload = dict(payload)
        payload["stream"] = True
        try:
            async with self._client() as client, client.stream(
                "POST", self.base_url, json=payload, headers=self.headers
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    # choices 可能为空（如仅含 usage 的收尾帧），此时跳过该帧
                    choices = chunk.get("choices") or [{}]
                    delta = choices[0].get("delta", {}) or {}
                    text = delta.get("content") or ""
                    if text:
                        yield GeneralResponse(text)
                    # 兼容两种字段名：OpenAI 惯例 reasoning_content 与部分厂商（如商汤）的 reasoning
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if reasoning:
                        yield GeneralResponse("", reasoning_content=reasoning)
        except Exception as e:
            err = f"Error calling model: {_describe_error(e)}, URL: {self.base_url}, Model: {self.model_name}"
            logger.error(err)
            raise ChatModelError(err) from e

    async def call_collect(self, message, **model_kwargs) -> GeneralResponse:
        """流式收集完整内容后返回单条响应（长生成不受非流式超时影响）。"""
        messages = self._normalize_messages(message)
        try:
            parts: list[str] = []
            async for chunk in self._stream_response({"model": self.model, "messages": messages, **model_kwargs}):
                if chunk.content:
                    parts.append(chunk.content)
            return GeneralResponse("".join(parts))
        except Exception as e:
            err = f"Error calling model: {_describe_error(e)}, URL: {self.base_url}, Model: {self.model_name}"
            logger.error(err)
            raise ChatModelError(err) from e


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _build_chat_model_from_spec(default_spec: str) -> OpenAIChatAdapter:
    """Build the chat adapter from a ``provider_id:model_id`` spec via ModelCache."""
    from yusu_kb.models.providers.cache import model_cache

    info = model_cache.get_model_info(default_spec)
    if info is None:
        available = [item.spec for item in model_cache.get_all_specs("chat")[:10]]
        raise ValueError(f"未找到模型: '{default_spec}'。可用 chat 模型 ({len(model_cache.get_all_specs('chat'))}): {available}")
    if info.model_type != "chat":
        raise ValueError(f"模型 '{default_spec}' 的类型是 {info.model_type}，不是 chat")
    base_url = _normalize_endpoint(info.base_url, "chat/completions")
    return OpenAIChatAdapter(
        info.model_id,
        model_name=info.model_id,
        base_url=base_url,
        api_key=info.api_key,
        headers=dict(info.headers or {}),
    )


def create_chat_model(*, default_spec: str | None = None) -> OpenAIChatAdapter:
    """两级解析：AppConfig 默认 spec 优先（经 ModelCache），无配置回退 env 路径。

    ``default_spec`` 为 ``provider_id:model_id``；为 ``None`` 时行为与旧版一致
    （读取 ``YUSU_LLM_*`` 环境变量）。
    """
    if default_spec:
        return _build_chat_model_from_spec(default_spec)
    base_url = _env_value("YUSU_LLM_BASE_URL")
    if not base_url:
        raise ValueError("YUSU_LLM_BASE_URL 未配置")
    model = _env_value("YUSU_LLM_MODEL")
    if not model:
        raise ValueError("YUSU_LLM_MODEL 未配置")
    api_key = _env_value("YUSU_LLM_API_KEY") or ""
    base_url = _normalize_endpoint(base_url, "chat/completions")
    return OpenAIChatAdapter(
        model,
        model_name=model,
        base_url=base_url,
        api_key=api_key,
    )


async def test_chat_model_status() -> dict:
    try:
        model = create_chat_model()
        test_messages = [{"role": "user", "content": "Say 1"}]
        response = await model.call(test_messages, stream=False)
        if response and response.content:
            return {"status": "available", "message": "连接正常"}
        return {"status": "unavailable", "message": "响应无效"}
    except Exception as e:  # noqa: BLE001 - 状态探测须返回结果而非抛出
        logger.error(f"测试模型状态失败: {e}")
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    pass