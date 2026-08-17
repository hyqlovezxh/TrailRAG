"""Knowledge base core package (migrated from YUSU ``yuxi.knowledge``).

Demo edition: single knowledge base type (``local``, SQLite + NanoVectorDB),
no folders, no share config, no URL import, no mindmap.
"""

# Import implementations so the @register decorators take effect (LocalKB).
from . import implementations  # noqa: F401
from .base import (
    INDEXED_STATS_STATUSES,
    FileStatus,
    KBNotFoundError,
    KBOperationError,
    KnowledgeBase,
    KnowledgeBaseException,
)
from .factory import KnowledgeBaseFactory
from .manager import KnowledgeBaseManager
from .schemas import (
    FindInputSchema,
    FindOutputSchema,
    FindWindowSchema,
    OpenInputSchema,
    OpenOutputSchema,
    SearchInputSchema,
    SearchOutputSchema,
    SearchResultSchema,
)

__all__ = [
    "INDEXED_STATS_STATUSES",
    "FileStatus",
    "FindInputSchema",
    "FindOutputSchema",
    "FindWindowSchema",
    "KBNotFoundError",
    "KBOperationError",
    "KnowledgeBase",
    "KnowledgeBaseException",
    "KnowledgeBaseFactory",
    "KnowledgeBaseManager",
    "OpenInputSchema",
    "OpenOutputSchema",
    "SearchInputSchema",
    "SearchOutputSchema",
    "SearchResultSchema",
]