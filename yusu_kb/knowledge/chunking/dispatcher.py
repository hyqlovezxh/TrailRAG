"""Chunk parser dispatch.

Ported from YUSU ``yuxi.knowledge.chunking.ragflow_like.dispatcher``,
including the ``case_document`` parser (案件文档：笔录/聊天/CSV/表格).
"""

from __future__ import annotations

from typing import Any

from yusu_kb.knowledge.chunking.parsers import (
    book,
    case_document,
    general,
    laws,
    qa,
    semantic,
    separator,
)
from yusu_kb.knowledge.chunking.presets import (
    map_to_internal_parser_id,
    normalize_chunk_preset_id,
)


def _build_chunk_records(
    text_chunks: list[str],
    file_id: str,
    filename: str,
    source_text: str | None = None,
    *,
    doc_type: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    search_from = 0

    for idx, chunk_content in enumerate(text_chunks):
        text = (chunk_content or "").strip()
        if not text:
            continue

        start_char_pos = None
        end_char_pos = None
        if source_text:
            found_at = source_text.find(text, search_from)
            if found_at >= 0:
                start_char_pos = found_at
                end_char_pos = found_at + len(text)
                search_from = end_char_pos

        records.append(
            {
                "id": f"{file_id}_chunk_{idx}",
                "content": text,
                "file_id": file_id,
                "filename": filename,
                "chunk_index": idx,
                "source": filename,
                "chunk_id": f"{file_id}_chunk_{idx}",
                "start_char_pos": start_char_pos,
                "end_char_pos": end_char_pos,
                "start_token_pos": None,
                "end_token_pos": None,
                "extraction_result": None,
                "doc_type": doc_type,
            }
        )

    return records


def _dispatch_markdown_parser(
    preset_id: str, filename: str, markdown_content: str, parser_config: dict[str, Any]
) -> list[str]:
    parser_id = map_to_internal_parser_id(preset_id)

    if parser_id == "naive":
        return general.chunk_markdown(markdown_content, parser_config)
    if parser_id == "qa":
        return qa.chunk_markdown(filename, markdown_content, parser_config)
    if parser_id == "book":
        return book.chunk_markdown(markdown_content, parser_config)
    if parser_id == "laws":
        return laws.chunk_markdown(filename, markdown_content, parser_config)
    if parser_id == "semantic":
        return semantic.chunk_markdown(markdown_content, parser_config)
    if parser_id == "separator":
        return separator.chunk_markdown(markdown_content, parser_config)
    if parser_id == "case_document":
        return case_document.chunk_markdown(filename, markdown_content, parser_config)

    return general.chunk_markdown(markdown_content, parser_config)


def chunk_markdown(
    markdown_content: str, file_id: str, filename: str, processing_params: dict[str, Any]
) -> list[dict[str, Any]]:
    params = dict(processing_params or {})
    preset_id = normalize_chunk_preset_id(params.get("chunk_preset_id"))
    parser_config = params.get("chunk_parser_config") if isinstance(params.get("chunk_parser_config"), dict) else {}

    text_chunks = _dispatch_markdown_parser(preset_id, filename, markdown_content, parser_config)
    # doc_type 随 chunk 记录落库（双路径路由判据，构建与查询读同一份）
    doc_type = _detect_chunk_doc_type(preset_id, filename, markdown_content, parser_config)
    return _build_chunk_records(text_chunks, file_id, filename, markdown_content, doc_type=doc_type)


def _detect_chunk_doc_type(
    preset_id: str, filename: str, markdown_content: str, parser_config: dict[str, Any]
) -> str | None:
    """按预设与内容判定 chunk 级 doc_type（缺失时返回 None，构建期回退探测）。

    仅 case_document 预设产出细粒度 doc_type；其他预设沿用自身语义。
    """
    if preset_id == "case_document":
        try:
            return case_document._detect_document_type(filename, markdown_content, parser_config)
        except Exception:  # noqa: BLE001 - 检测失败返回 None，构建期再兜底
            return None
    return preset_id


def chunk_file(
    file_content: str, file_id: str, filename: str, processing_params: dict[str, Any]
) -> list[dict[str, Any]]:
    # The current pipeline converts everything to markdown before chunking.
    return chunk_markdown(file_content, file_id, filename, processing_params)