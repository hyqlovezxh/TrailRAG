"""KB tools for the chat tool loop (ported from YUSU ``agents/toolkits/kbs/tools.py``).

Adaptations:
- No langgraph ``ToolRuntime`` -> tools take an explicit ``kb_manager`` argument
  (defaults to the global manager from ``api.deps``), which keeps them callable
  from tests and from the chat route.
- Tools never raise: failures are returned as error strings (YUSU semantics).
- ``query_kb`` goes through ``kb.aquery`` and attaches ``citation_source``
  (``kb://{kb_id}/{file_id}?chunk={chunk_id}``) to every result item.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, Field

from yusu_kb.knowledge.base import KBNotFoundError
from yusu_kb.utils.logger import logger

# query_kb 工具结果的 token 预算上限（约 48K 字符）。
# 为 system prompt + 对话历史 + 用户问题留余量，避免总长超模型窗口被 provider 截断。
_QUERY_KB_MAX_OUTPUT_TOKENS = 12000

_OMITTED_MARKER = "[内容因长度限制已省略]"


def _estimate_tokens(text: str) -> int:
    """近似 token 计数：英文单词 + 数字 + CJK 单字。"""
    if not text:
        return 0
    parts = re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text)
    return max(1, len(parts)) if text.strip() else 0


def _apply_token_budget(results: list[dict[str, Any]], max_tokens: int) -> None:
    """按 token 预算截断检索结果：高 rank 保留完整，低 rank 截断尾部。

    - 从 rank 0 开始累加 token 数，超出预算时截断当前 chunk 的 content
    - 后续 chunk 标记为省略（保留检索可达性，不丢弃 chunk 元数据）
    - 就地修改 results
    """
    budget_remaining = max_tokens
    truncated_any = False
    for item in results:
        if budget_remaining <= 0:
            item["content"] = _OMITTED_MARKER
            truncated_any = True
            continue
        content = str(item.get("content") or "")
        token_cost = _estimate_tokens(content)
        if token_cost <= budget_remaining:
            budget_remaining -= token_cost
            continue
        # 当前 chunk 超出剩余预算：截断到剩余预算对应的字符数
        # 按 token/字符比率近似换算
        chars_to_keep = int(len(content) * budget_remaining / max(token_cost, 1))
        if chars_to_keep > 0:
            item["content"] = content[:chars_to_keep] + "\n..." + _OMITTED_MARKER
        else:
            item["content"] = _OMITTED_MARKER
        budget_remaining = 0
        truncated_any = True
    if truncated_any:
        logger.info(
            f"query_kb 结果因 token 预算({max_tokens})截断，共 {len(results)} 个 chunk，部分低 rank 内容已省略"
        )


def _kb_citation_source(
    kb_id: str,
    file_id: str,
    *,
    chunk_id: str = "",
    start_line: int = 0,
    end_line: int = 0,
) -> str:
    """构造稳定来源标识 kb://{kb_id}/{file_id}?chunk={chunk_id}

    统一使用 ?chunk= 格式，避免 LLM 看到矛盾 URL 格式导致"暂无引用摘要"。
    """
    source = f"kb://{quote(str(kb_id), safe='')}/{quote(str(file_id), safe='')}"
    if chunk_id:
        return f"{source}?chunk={quote(str(chunk_id), safe='')}"
    if start_line > 0 or end_line > 0:
        return f"{source}?chunk=lines_{start_line}_{max(start_line, end_line)}"
    return source


async def _resolve_manager(kb_manager=None):
    """Resolve the KB manager: explicit argument wins, else the global one."""
    if kb_manager is not None:
        return kb_manager
    from yusu_kb.api.deps import get_manager

    return await get_manager()


async def _resolve_kb(kb_manager, kb_id: str):
    """Resolve the KB instance owning ``kb_id``; None when it does not exist."""
    try:
        return await kb_manager._require_kb_for_database(kb_id)
    except KBNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Input schemas (exposed to the LLM via model_json_schema)
# ---------------------------------------------------------------------------


class ListKBsInput(BaseModel):
    """列出当前可访问的知识库列表。"""


class QueryKBInput(BaseModel):
    kb_id: str = Field(description="知识库资源 ID（kb_id）")
    query_text: str = Field(description="查询内容")
    file_name: str | None = Field(default=None, description="按文件名过滤（可选）")
    use_reranker: bool = Field(default=True, description="启用 rerank 重排序（默认启用）")
    use_graph_retrieval: bool = Field(default=True, description="启用知识图谱检索（默认启用）")


class OpenKBDocumentInput(BaseModel):
    kb_id: str = Field(description="知识库资源 ID（kb_id）")
    file_id: str = Field(description="知识库文件 ID")
    line: int | None = Field(default=None, ge=1, description="起始行号（1 起，与 offset 二选一）")
    offset: int | None = Field(default=None, ge=0, description="起始偏移（行，0 起）")
    window_size: int = Field(default=1800, ge=1, le=2000, description="窗口大小（行数）")


class FindKBDocumentInput(BaseModel):
    kb_id: str = Field(description="知识库资源 ID（kb_id）")
    file_id: str = Field(description="知识库文件 ID")
    patterns: list[str] = Field(description="查找模式列表")
    use_regex: bool = Field(default=False, description="是否使用正则表达式")
    case_sensitive: bool = Field(default=False, description="是否区分大小写")
    max_windows: int = Field(default=5, ge=1, le=20, description="最大窗口数量")
    window_size: int = Field(default=80, ge=1, le=200, description="窗口大小（行数）")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def list_kbs(kb_manager=None) -> list[dict]:
    """列出当前可访问的知识库列表

    返回每项含 kb_id / name / description。
    """
    try:
        manager = await _resolve_manager(kb_manager)
        data = await manager.get_databases(include_files=False)
        return [
            {
                "kb_id": db.get("kb_id"),
                "name": db.get("name", ""),
                "description": db.get("description") or "无描述",
            }
            for db in data.get("databases", [])
        ]
    except Exception as exc:  # noqa: BLE001 - 工具失败返回错误字符串而非抛出
        logger.error(f"获取知识库列表失败: {exc}")
        return [{"error": f"获取知识库列表失败: {exc}"}]


async def query_kb(
    kb_id: str,
    query_text: str,
    file_name: str | None = None,
    use_reranker: bool = True,
    use_graph_retrieval: bool = True,
    kb_manager=None,
) -> dict[str, Any] | str:
    """在指定知识库中检索内容

    当用户需要查询具体内容时使用此工具。kb_id 是知识库资源 ID，也就是 kb_id；返回结果中的
    file_id 可继续用于 find_kb_document 或 open_kb_document。

    可选参数：
    - use_reranker: 启用 rerank 重排序，提高检索精度（默认启用）
    - use_graph_retrieval: 启用知识图谱检索，支持多跳推理（默认启用）
    """
    if not kb_id:
        return "请提供 kb_id"
    if not query_text:
        return "请提供查询内容"

    try:
        manager = await _resolve_manager(kb_manager)
        kb = await _resolve_kb(manager, kb_id)
        if kb is None:
            return f"知识库资源 '{kb_id}' 不存在"

        query_kwargs: dict[str, Any] = {}
        if file_name:
            query_kwargs["file_name"] = file_name
        # 显式传参覆盖 KB 保存的 options：工具调用语义与 YUSU 一致（默认全开）
        query_kwargs["use_reranker"] = use_reranker
        query_kwargs["use_graph_retrieval"] = use_graph_retrieval

        result = await kb.aquery(query_text, kb_id, agent_call=True, **query_kwargs)
        output = kb.build_search_output(kb_id, result)
        if not isinstance(output, dict) or not isinstance(output.get("results"), list):
            return output
        for item in output["results"]:
            item["citation_source"] = _kb_citation_source(
                kb_id,
                str(item.get("file_id") or ""),
                chunk_id=str(item.get("id") or ""),
            )
        # token 预算截断：高 rank 保留完整，低 rank 截断尾部，避免总长超模型窗口
        _apply_token_budget(output["results"], _QUERY_KB_MAX_OUTPUT_TOKENS)
        return output
    except Exception as exc:  # noqa: BLE001 - 工具失败返回错误字符串而非抛出
        logger.error(f"检索失败: {exc}")
        return f"检索失败: {exc}"


async def open_kb_document(
    kb_id: str,
    file_id: str,
    line: int | None = None,
    offset: int | None = None,
    window_size: int = 1800,
    kb_manager=None,
) -> dict[str, Any] | str:
    """按行窗口打开知识库文档原文

    当 query_kb 返回的片段不足以回答问题，或需要查看某个文档的上下文时使用。
    kb_id 是知识库资源 ID，也就是 kb_id；file_id 是知识库文件 ID。
    """
    normalized_kb_id = str(kb_id or "").strip()
    normalized_file_id = str(file_id or "").strip()
    if not normalized_kb_id:
        return "请提供 kb_id"
    if not normalized_file_id:
        return "请提供 file_id"

    try:
        manager = await _resolve_manager(kb_manager)
        kb = await _resolve_kb(manager, normalized_kb_id)
        if kb is None:
            return f"知识库资源 '{normalized_kb_id}' 不存在"

        start_offset = int(line) - 1 if line is not None else int(offset or 0)
        window = await manager.open_file_content(
            normalized_kb_id, normalized_file_id, offset=start_offset, limit=window_size
        )
        citation_source = _kb_citation_source(
            normalized_kb_id,
            normalized_file_id,
            start_line=int(window.get("start_line") or 0),
            end_line=int(window.get("end_line") or 0),
        )
        return {
            "kb_id": normalized_kb_id,
            "file_id": normalized_file_id,
            "citation_source": citation_source,
            **window,
        }
    except Exception as exc:  # noqa: BLE001 - 工具失败返回错误字符串而非抛出
        logger.error(f"打开知识库文档失败: {exc}")
        return f"打开知识库文档失败: {exc}"


async def find_kb_document(
    kb_id: str,
    file_id: str,
    patterns: list[str],
    use_regex: bool = False,
    case_sensitive: bool = False,
    max_windows: int = 5,
    window_size: int = 80,
    kb_manager=None,
) -> dict[str, Any] | str:
    """在已知知识库文件内做关键词或正则定位。

    当 query_kb 已找到候选文件，但需要在该文件内定位术语、指标、章节或实体时使用。
    """
    normalized_kb_id = str(kb_id or "").strip()
    normalized_file_id = str(file_id or "").strip()
    if not normalized_kb_id:
        return "请提供 kb_id"
    if not normalized_file_id:
        return "请提供 file_id"
    if not patterns:
        return "请提供 patterns"

    try:
        manager = await _resolve_manager(kb_manager)
        kb = await _resolve_kb(manager, normalized_kb_id)
        if kb is None:
            return f"知识库资源 '{normalized_kb_id}' 不存在"

        result = await manager.find_file_content(
            normalized_kb_id,
            normalized_file_id,
            patterns,
            use_regex=use_regex,
            case_sensitive=case_sensitive,
            max_windows=max_windows,
            window_size=window_size,
        )
        windows = []
        for window in result.get("windows") or []:
            windows.append(
                {
                    **window,
                    "citation_source": _kb_citation_source(
                        normalized_kb_id,
                        normalized_file_id,
                        start_line=int(window.get("start_line") or 0),
                        end_line=int(window.get("end_line") or 0),
                    ),
                }
            )
        return {**result, "kb_id": normalized_kb_id, "file_id": normalized_file_id, "windows": windows}
    except Exception as exc:  # noqa: BLE001 - 工具失败返回错误字符串而非抛出
        logger.error(f"知识库文档内检索失败: {exc}")
        return f"知识库文档内检索失败: {exc}"


# ---------------------------------------------------------------------------
# Tool registry (name, description, args_schema, func)
# ---------------------------------------------------------------------------

_KB_TOOLS: list[tuple[str, str, type[BaseModel], Any]] = [
    (
        "list_kbs",
        "列出当前可访问的知识库列表（每项含 kb_id / name / description）。",
        ListKBsInput,
        list_kbs,
    ),
    (
        "query_kb",
        (
            "在指定知识库中检索内容。当用户需要查询具体内容时使用此工具。"
            "kb_id 是知识库资源 ID；返回结果中的 file_id 可继续用于 find_kb_document 或 open_kb_document。"
            "可选参数：use_reranker 启用 rerank 重排序（默认启用）；"
            "use_graph_retrieval 启用知识图谱检索（默认启用）。"
        ),
        QueryKBInput,
        query_kb,
    ),
    (
        "open_kb_document",
        (
            "按行窗口打开知识库文档原文。当 query_kb 返回的片段不足以回答问题，"
            "或需要查看某个文档的上下文时使用。"
        ),
        OpenKBDocumentInput,
        open_kb_document,
    ),
    (
        "find_kb_document",
        (
            "在已知知识库文件内做关键词或正则定位。当 query_kb 已找到候选文件，"
            "但需要在该文件内定位术语、指标、章节或实体时使用。"
        ),
        FindKBDocumentInput,
        find_kb_document,
    ),
]


def get_kb_tool_definitions() -> list[dict[str, Any]]:
    """OpenAI function-calling tool definitions (for the ``tools`` payload)."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": args_schema.model_json_schema(),
            },
        }
        for name, description, args_schema, _func in _KB_TOOLS
    ]


def get_kb_tool_names() -> list[str]:
    """Names of the registered KB tools."""
    return [name for name, _description, _args_schema, _func in _KB_TOOLS]


async def execute_kb_tool(name: str, arguments: dict[str, Any], *, kb_manager=None) -> str:
    """Dispatch a tool call by name; returns a JSON string (errors as strings)."""
    for tool_name, _description, _args_schema, func in _KB_TOOLS:
        if tool_name != name:
            continue
        try:
            result = await func(**arguments, kb_manager=kb_manager)
        except Exception as exc:  # noqa: BLE001 - 工具执行失败返回错误字符串
            logger.error(f"工具 {name} 执行失败: {exc}")
            return f"工具执行失败: {exc}"
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)
    return f"未知工具: {name}"


__all__ = [
    "FindKBDocumentInput",
    "ListKBsInput",
    "OpenKBDocumentInput",
    "QueryKBInput",
    "execute_kb_tool",
    "find_kb_document",
    "get_kb_tool_definitions",
    "get_kb_tool_names",
    "list_kbs",
    "open_kb_document",
    "query_kb",
]
