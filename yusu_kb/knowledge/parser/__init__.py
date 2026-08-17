"""Document parser package (migrated from YUSU ``yuxi.knowledge.parser``).

Demo edition: local lightweight engines only — no OCR, no external parsing
services. Supported formats are declared in ``unified.SUPPORTED_FILE_EXTENSIONS``.
"""

from .unified import MarkdownParseResult, Parser, is_supported_file_extension

__all__ = ["MarkdownParseResult", "Parser", "is_supported_file_extension"]