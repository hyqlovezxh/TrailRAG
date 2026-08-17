"""RAGFlow-like chunking engine, ported from YUSU ``yuxi.knowledge.chunking.ragflow_like``.

The ``case_document`` preset (police case files) is intentionally not ported.
``semantic_utils`` replaces NLTK/sklearn with a self-contained numpy
implementation so the demo needs no model downloads and no heavy native deps.
"""

from .dispatcher import chunk_file, chunk_markdown
from .presets import (
    CHUNK_ENGINE_VERSION,
    CHUNK_PRESETS,
    DEFAULT_CHUNK_PRESET_ID,
    GENERAL_INTERNAL_PARSER_ID,
    deep_merge,
    ensure_chunk_defaults_in_additional_params,
    get_chunk_preset_options,
    map_to_internal_parser_id,
    normalize_chunk_preset_id,
    resolve_chunk_processing_params,
)

__all__ = [
    "CHUNK_ENGINE_VERSION",
    "CHUNK_PRESETS",
    "DEFAULT_CHUNK_PRESET_ID",
    "GENERAL_INTERNAL_PARSER_ID",
    "chunk_file",
    "chunk_markdown",
    "deep_merge",
    "ensure_chunk_defaults_in_additional_params",
    "get_chunk_preset_options",
    "map_to_internal_parser_id",
    "normalize_chunk_preset_id",
    "resolve_chunk_processing_params",
]