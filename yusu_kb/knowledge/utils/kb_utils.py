"""Knowledge base utility helpers.

Ported from YUSU ``yuxi.knowledge.utils.kb_utils`` (demo edition):
OCR and MinIO branches removed.
"""

from __future__ import annotations

import hashlib
import os

from yusu_kb.knowledge.chunking.presets import (
    normalize_chunk_preset_id,
    resolve_chunk_processing_params,
)
from yusu_kb.utils.logger import logger

CHUNK_ENGINE_VERSION = "ragflow_like_v1"

CHUNK_PARAMS_DEFAULTS = {
    "chunk_engine_version": CHUNK_ENGINE_VERSION,
    "chunk_preset_id": "general",
    "chunk_parser_config": {},
    "preset_id": "general",
    "max_chunk_tokens": 400,
    "max_chunk_overlap_tokens": 0,
    "do_clean_markdown": True,
    "do_remove_tables": False,
    "do_remove_links": False,
    "do_remove_images": False,
    "do_remove_extra_whitespace": False,
    "embed_chunk_titles": True,
    "max_content_tokens": 1000,
    "extra_presets": {},
    "source_parse": {},
    "eval": {},
}

PARSE_PARAMS_DEFAULTS = {
    "multimodal_ocr": False,
    "multimodal_vlm": False,
    "table_to_markdown": True,
}

MAX_CONTENT_TOKENS = int(os.environ.get("YUSU_MAX_CONTENT_TOKENS") or 1000)


def sanitize_processing_params(params: dict | None) -> dict:
    """Sanitize processing params: keep only safe, serializable values."""
    if not isinstance(params, dict):
        return {}

    allowed_keys = set(CHUNK_PARAMS_DEFAULTS) | set(PARSE_PARAMS_DEFAULTS)
    sanitized: dict = {}
    for key, value in params.items():
        if key in allowed_keys:
            if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
    return sanitized


def merge_processing_params(
    base_params: dict | None,
    request_params: dict | None,
    allowed_keys: set[str] | None = None,
) -> dict:
    """Merge request params over base params, keeping only allowed keys."""
    allowed = allowed_keys if allowed_keys is not None else set(CHUNK_PARAMS_DEFAULTS) | set(PARSE_PARAMS_DEFAULTS)
    merged = dict(base_params or {})
    for key, value in (request_params or {}).items():
        if key in allowed:
            merged[key] = value
    return merged


def resolve_processing_params(
    kb_additional_params: dict | None = None,
    file_processing_params: dict | None = None,
    request_params: dict | None = None,
) -> dict:
    """Resolve effective processing params.

    Precedence: request params > file params > kb additional params > defaults.

    与 YUSU ``resolve_processing_params`` 契约一致：除扁平参数（preset_id /
    max_chunk_tokens 等，用于元数据展示与解析流程）外，还合并命名空间化 chunk
    参数（chunk_preset_id / chunk_parser_config），供 chunk_markdown 消费；
    扁平参数在无命名空间化配置时作为回退翻译（preset_id -> chunk_preset_id，
    max_chunk_tokens -> chunk_parser_config.chunk_token_num）。
    """
    effective: dict = dict(CHUNK_PARAMS_DEFAULTS)
    effective.update(PARSE_PARAMS_DEFAULTS)
    effective["chunk_engine_version"] = CHUNK_ENGINE_VERSION

    for params in (kb_additional_params, file_processing_params, request_params):
        if isinstance(params, dict):
            effective.update(sanitize_processing_params(params))

    chunk_params = resolve_chunk_processing_params(
        kb_additional_params=kb_additional_params,
        file_processing_params=file_processing_params,
        request_params=request_params,
    )
    if not chunk_params.get("chunk_parser_config"):
        flat_config: dict = {}
        flat_tokens = effective.get("max_chunk_tokens")
        if flat_tokens:
            flat_config["chunk_token_num"] = int(flat_tokens)
        if not chunk_params.get("chunk_preset_id") and effective.get("preset_id"):
            chunk_params["chunk_preset_id"] = normalize_chunk_preset_id(effective["preset_id"])
        chunk_params["chunk_parser_config"] = flat_config
    effective.update(chunk_params)
    return effective


def calculate_content_hash(content: str | bytes) -> str:
    """Calculate the SHA-256 content hash of a document's text content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _normalize_source_path(source_path: str) -> str:
    """Normalize a source path for storage key use."""
    return source_path.replace("\\", "/").lstrip("/")


def _resolve_relative_path(base_path: str, relative_path: str) -> str:
    """Resolve a relative path against a base directory, preventing escapes."""
    from pathlib import Path

    base = Path(base_path).resolve()
    target = (base / relative_path).resolve()
    if base != target and base not in target.parents:
        logger.warning(f"Blocked path escape attempt: {relative_path}")
        raise ValueError(f"非法路径: {relative_path}")
    return str(target)
