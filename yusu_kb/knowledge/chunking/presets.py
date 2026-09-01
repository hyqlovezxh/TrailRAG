"""Chunk preset registry and processing-param resolution.

Ported from YUSU ``yuxi.knowledge.chunking.ragflow_like.presets``.
The ``case_document`` preset (police case files) is ported and set as the
default for the event-driven edition: it auto-detects transcript /
chat_record / csv_table / spreadsheet and time-window-chunks chat records
with 【时间段 X ~ Y】 anchors that feed the event extractor.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from yusu_kb.utils.logger import logger

# 事件驱动版默认预设：case_document 自动检测案件文档类型并分发
# （笔录问答切分 / 聊天 30 分钟时间窗 / CSV/表格），为事件抽取提供时间锚。
DEFAULT_CHUNK_PRESET_ID = "case_document"

CHUNK_PRESETS: dict[str, dict[str, str]] = {
    "general": {
        "label": "General",
        "description": "通用分块：按分隔符和长度切分，适合大多数普通文档。",
    },
    "qa": {
        "label": "QA",
        "description": "问答分块：优先抽取问题-回答结构，适合 FAQ、题库、问答手册。",
    },
    "book": {
        "label": "Book",
        "description": "书籍分块：强化章节标题识别并做层级合并，适合教材、手册、长章节文档。",
    },
    "laws": {
        "label": "Laws",
        "description": "法规分块：按法条层级组织与合并，适合法律法规、制度规范类文本。",
    },
    "semantic": {
        "label": "Semantic",
        "description": "语义分块：利用嵌入和聚类算法进行语义切分，并自动增强标题上下文。",
    },
    "separator": {
        "label": "Separator",
        "description": "严格分隔：命中分隔符即切分，仅超长片段内部继续按长度切分。",
    },
    "case_document": {
        "label": "Case Document",
        "description": "案件文档分块：自动识别笔录/聊天/CSV/表格并分发专用分块（笔录问答切分、"
        "聊天 30 分钟时间窗 + 时间段锚定），事件驱动路径推荐。",
    },
}

CHUNK_PRESET_IDS = set(CHUNK_PRESETS)

CHUNK_ENGINE_VERSION = "ragflow_like_v1"
GENERAL_INTERNAL_PARSER_ID = "naive"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def normalize_chunk_preset_id(value: str | None) -> str:
    if not value:
        return DEFAULT_CHUNK_PRESET_ID

    normalized = str(value).strip().lower()
    # "naive" is the internal parser_id of the "general" preset, not a preset itself
    if normalized == GENERAL_INTERNAL_PARSER_ID:
        return "general"

    if normalized in CHUNK_PRESET_IDS:
        return normalized

    logger.warning(f"Unknown chunk preset id '{value}', fallback to {DEFAULT_CHUNK_PRESET_ID}")
    return DEFAULT_CHUNK_PRESET_ID


def map_to_internal_parser_id(preset_id: str) -> str:
    normalized = normalize_chunk_preset_id(preset_id)
    # Only the "general" preset maps to the "naive" internal parser.
    if normalized == "general":
        return GENERAL_INTERNAL_PARSER_ID
    return normalized


def get_default_chunk_parser_config(preset_id: str) -> dict[str, Any]:
    preset_id = normalize_chunk_preset_id(preset_id)
    return {}


def ensure_chunk_defaults_in_additional_params(additional_params: dict[str, Any] | None) -> dict[str, Any]:
    params = dict(additional_params or {})
    params["chunk_preset_id"] = normalize_chunk_preset_id(params.get("chunk_preset_id"))

    if "chunk_parser_config" in params and not isinstance(params.get("chunk_parser_config"), dict):
        logger.warning("Invalid chunk_parser_config in additional_params, fallback to empty dict")
        params["chunk_parser_config"] = {}

    return params


def resolve_chunk_processing_params(
    kb_additional_params: dict[str, Any] | None,
    file_processing_params: dict[str, Any] | None,
    request_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kb_additional = ensure_chunk_defaults_in_additional_params(kb_additional_params)
    file_params = dict(file_processing_params or {})
    request = dict(request_params or {})

    preset_id = normalize_chunk_preset_id(
        request.get("chunk_preset_id") or file_params.get("chunk_preset_id") or kb_additional.get("chunk_preset_id")
    )

    parser_config = get_default_chunk_parser_config(preset_id)

    kb_parser_config = kb_additional.get("chunk_parser_config")
    if isinstance(kb_parser_config, dict):
        parser_config = deep_merge(parser_config, kb_parser_config)

    file_parser_config = file_params.get("chunk_parser_config")
    if isinstance(file_parser_config, dict):
        parser_config = deep_merge(parser_config, file_parser_config)

    req_parser_config = request.get("chunk_parser_config")
    if isinstance(req_parser_config, dict):
        parser_config = deep_merge(parser_config, req_parser_config)

    return {
        "chunk_preset_id": preset_id,
        "chunk_parser_config": parser_config,
        "chunk_engine_version": CHUNK_ENGINE_VERSION,
    }


def get_chunk_preset_options() -> list[dict[str, str]]:
    return [
        {"value": preset_id, "label": preset["label"], "description": preset["description"]}
        for preset_id, preset in CHUNK_PRESETS.items()
    ]