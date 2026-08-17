"""Knowledge base type registry.

Ported from YUSU ``yuxi.knowledge.factory`` (demo edition, single registered
type: ``local``).
"""

from __future__ import annotations

from typing import ClassVar

from yusu_kb.utils.logger import logger

from .base import KnowledgeBase


class KnowledgeBaseFactory:
    """Registry and factory for knowledge base types."""

    _implementations: ClassVar[dict[str, type[KnowledgeBase]]] = {}

    @classmethod
    def register(cls, kb_type: str):
        """Decorator to register a knowledge base type."""

        def decorator(kb_class: type[KnowledgeBase]) -> type[KnowledgeBase]:
            cls._implementations[kb_type] = kb_class
            return kb_class

        return decorator

    @classmethod
    def create(cls, kb_type: str, work_dir: str) -> KnowledgeBase:
        """Create a knowledge base instance of the given type."""
        kb_class = cls._implementations.get(kb_type)
        if kb_class is None:
            raise ValueError(f"未知的知识库类型: {kb_type}")
        return kb_class(work_dir)

    @classmethod
    def get_available_types(cls) -> dict[str, dict]:
        """Get metadata for all registered knowledge base types."""
        result = {}
        for kb_type, kb_class in cls._implementations.items():
            try:
                options = kb_class.get_create_params_config() or {}
                if not isinstance(options, dict):
                    options = {"options": []}
            except Exception as exc:  # noqa: BLE001 - registry introspection must not break other types
                logger.error(f"获取知识库类型 {kb_type} 的配置失败: {exc}")
                options = {"options": []}
            result[kb_type] = {
                "name": kb_class.name,
                "description": kb_class.description,
                "requires_embedding_model": kb_class.requires_embedding_model,
                "supports_documents": kb_class.supports_documents,
                "params": options,
            }
        return result

    @classmethod
    def get_kb_class(cls, kb_type: str) -> type[KnowledgeBase] | None:
        """Get the class for a knowledge base type, or None if unknown."""
        return cls._implementations.get(kb_type)

    @classmethod
    def is_type_supported(cls, kb_type: str) -> bool:
        """Check whether a knowledge base type is supported."""
        return kb_type in cls._implementations