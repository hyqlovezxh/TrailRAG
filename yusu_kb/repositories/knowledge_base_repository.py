"""Knowledge base repository (SQLite).

Ported from YUSU ``yuxi.repositories.knowledge_base_repository`` (demo
edition: advisory-lock helpers and additional-params merge dropped).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select

from yusu_kb.storage.sqlite.models_knowledge import KnowledgeBase

from . import get_session_factory

_ALLOWED_FIELDS = {
    "kb_id",
    "name",
    "description",
    "kb_type",
    "embedding_model_spec",
    "llm_model_spec",
    "query_params",
    "additional_params",
    "share_config",
    "created_by",
}


def _sanitize_data(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key in _ALLOWED_FIELDS and value is not None}


class KnowledgeBaseRepository:
    """SQLite-backed knowledge base repository."""

    async def get_all(self) -> list[KnowledgeBase]:
        async with get_session_factory()() as session:
            result = await session.execute(select(KnowledgeBase).order_by(KnowledgeBase.id))
            return list(result.scalars().all())

    async def get_by_kb_id(self, kb_id: str) -> KnowledgeBase | None:
        async with get_session_factory()() as session:
            result = await session.execute(select(KnowledgeBase).where(KnowledgeBase.kb_id == kb_id))
            return result.scalar_one_or_none()

    async def create(self, data: dict[str, Any]) -> KnowledgeBase:
        async with get_session_factory()() as session:
            record = KnowledgeBase(**_sanitize_data(data))
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def update(self, kb_id: str, data: dict[str, Any]) -> KnowledgeBase | None:
        async with get_session_factory()() as session:
            result = await session.execute(select(KnowledgeBase).where(KnowledgeBase.kb_id == kb_id))
            record = result.scalar_one_or_none()
            if record is None:
                return None
            for key, value in _sanitize_data(data).items():
                setattr(record, key, value)
            await session.commit()
            await session.refresh(record)
            return record

    async def delete(self, kb_id: str) -> None:
        async with get_session_factory()() as session:
            await session.execute(delete(KnowledgeBase).where(KnowledgeBase.kb_id == kb_id))
            await session.commit()