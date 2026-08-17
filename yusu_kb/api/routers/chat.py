"""Chat streaming endpoint: retrieval context + optional KB tool loop.

SSE event types:
- ``sources``: retrieved chunks (each with a ``citation_source``)
- ``tool``: a KB tool call was executed (name/args/summary, summary ≤200 chars)
- ``delta``: answer text
- ``reasoning``: optional reasoning text
- ``error`` / ``done``
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from yusu_kb.api.deps import ChatModelDep, ManagerDep, verify_api_key
from yusu_kb.knowledge.base import KBNotFoundError
from yusu_kb.knowledge.tools import (
    _kb_citation_source,
    execute_kb_tool,
    get_kb_tool_definitions,
)
from yusu_kb.utils.logger import logger

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
    dependencies=[Depends(verify_api_key)],
)

MAX_TOOL_ROUNDS = 3
_TOOL_SUMMARY_LIMIT = 200

# 引用提示词：回答中引用资料编号，前端把 <cite>n</cite> 渲染为角标 [n]。
SOURCE_CITE_PROMPT = (
    "引用要求：回答中引用参考资料时，请在对应句末使用 <cite>编号</cite> 标注资料序号"
    "（例如 <cite>1</cite> 表示该句依据第 1 条资料）。仅当陈述确实由对应编号资料支撑时才标注；"
    "同时引用多条资料时连续标注（如 <cite>1</cite><cite>2</cite>）。不要编造编号或内容。"
)


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str = Field(..., min_length=1)


class ChatStreamRequest(BaseModel):
    kb_id: str
    messages: list[ChatMessage] = Field(..., min_length=1)
    query_params: dict[str, Any] | None = None
    tools_enabled: bool = True
    system_prompt: str = ""


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _build_context(results: list[dict]) -> str:
    parts = []
    for index, chunk in enumerate(results):
        content = chunk.get("content") or ""
        if content:
            parts.append(f"[{index + 1}] {content}")
    return "\n\n".join(parts)


def _build_system_prompt(context: str, custom_prompt: str) -> str:
    """组装系统提示词：当前日期 + 基础身份 + 用户自定义提示词 + 引用要求。"""
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    parts = [
        f"当前日期：{today}",
        "你是语溯知识库问答助手，负责基于知识库资料回答用户问题。",
    ]
    if custom_prompt:
        parts.append(custom_prompt)
    if context:
        parts.append("参考资料：\n" + context)
    else:
        parts.append("当前知识库中没有检索到相关资料，请如实告知用户并尝试基于常识作答。")
    parts.append(SOURCE_CITE_PROMPT)
    return "\n\n".join(parts)


def _model_rejects_tools(exc: Exception) -> bool:
    """判定异常是否表明模型/端点不支持 tools 参数（据此降级旧路径）。"""
    message = str(exc).lower()
    markers = ("tool", "function", "unsupported", "not support", "not supported", "400", "422")
    return any(marker in message for marker in markers)


def _truncate_tool_summary(result: Any, limit: int = _TOOL_SUMMARY_LIMIT) -> str:
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


async def _stream_answer(chat_model, messages: list[dict]) -> AsyncIterator[dict]:
    """流式生成最终回答（不带 tools 参数，避免模型继续请求工具）。"""
    stream = await chat_model.call(messages, stream=True)
    async for part in stream:
        if part.content:
            yield {"type": "delta", "content": part.content}
        if part.reasoning_content:
            yield {"type": "reasoning", "content": part.reasoning_content}


async def _run_tool_loop(chat_model, messages: list[dict], kb_manager) -> AsyncIterator[dict]:
    """≤3 轮 KB 工具调用循环；模型不支持 tools 时降级为普通流式回答。

    - 模型直接给出 content（无 tool_calls）→ 单条 delta 事件
    - 达到轮数上限仍请求工具 → 基于工具结果流式生成最终回答
    """
    definitions = get_kb_tool_definitions()
    try:
        response = await chat_model.call(messages, stream=False, tools=definitions)
    except Exception as exc:  # 不支持 tools 按异常消息判定后降级
        if not _model_rejects_tools(exc):
            raise
        logger.warning(f"Chat model does not support tools, falling back to plain answer: {exc}")
        async for event in _stream_answer(chat_model, messages):
            yield event
        return

    calls = response.tool_calls or []
    rounds = 0
    while calls and rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        messages.append(
            {"role": "assistant", "content": response.content or "", "tool_calls": calls}
        )
        for call in calls:
            name = call.get("function", {}).get("name", "")
            try:
                arguments = json.loads(call.get("function", {}).get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            result = await execute_kb_tool(name, arguments, kb_manager=kb_manager)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id") or "",
                    "content": result,
                }
            )
            yield {
                "type": "tool",
                "name": name,
                "args": arguments,
                "summary": _truncate_tool_summary(result),
            }
        if rounds >= MAX_TOOL_ROUNDS:
            break
        response = await chat_model.call(messages, stream=False, tools=definitions)
        calls = response.tool_calls or []

    if calls:
        # 轮数已用尽仍请求工具：基于已有工具结果流式生成最终回答
        async for event in _stream_answer(chat_model, messages):
            yield event
    elif response.content:
        yield {"type": "delta", "content": response.content}
    elif response.reasoning_content:
        yield {"type": "reasoning", "content": response.reasoning_content}


@router.post("/stream")
async def chat_stream(
    payload: ChatStreamRequest,
    manager: ManagerDep,
    chat_model: ChatModelDep,
) -> StreamingResponse:
    """Retrieve top-k chunks, optionally run KB tools, and stream an LLM answer."""
    user_text = next((m.content for m in reversed(payload.messages) if m.role == "user"), "")
    if not user_text:
        raise HTTPException(status_code=400, detail="messages 必须包含 user 消息")

    async def event_stream() -> AsyncIterator[str]:
        try:
            kb = await manager._require_kb_for_database(payload.kb_id)
        except KBNotFoundError as exc:
            yield _sse({"type": "error", "message": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 - stream errors surface as SSE events
            yield _sse({"type": "error", "message": str(exc)})
            return

        try:
            results = await kb.aquery(user_text, payload.kb_id, **payload.query_params or {})
            output = kb.build_search_output(payload.kb_id, results)
            chunks = output.get("results", []) if isinstance(output, dict) else []
            # 与 knowledge/tools.py 一致的来源标识，前端按 citation_source 去重编号
            for chunk in chunks:
                if "citation_source" not in chunk:
                    chunk["citation_source"] = _kb_citation_source(
                        payload.kb_id,
                        str(chunk.get("file_id") or ""),
                        chunk_id=str(chunk.get("id") or ""),
                    )
            yield _sse({"type": "sources", "chunks": chunks})

            system_prompt = _build_system_prompt(_build_context(chunks), payload.system_prompt)
            messages = [{"role": "system", "content": system_prompt}] + [
                {"role": m.role, "content": m.content} for m in payload.messages
            ]

            if payload.tools_enabled:
                async for event in _run_tool_loop(chat_model, messages, manager):
                    yield _sse(event)
            else:
                async for event in _stream_answer(chat_model, messages):
                    yield _sse(event)
            yield _sse({"type": "done"})
        except Exception as exc:  # noqa: BLE001 - stream errors surface as SSE events
            logger.error(f"Chat stream failed for kb {payload.kb_id}: {exc}")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
