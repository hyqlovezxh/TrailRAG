"""Knowledge graph repository (SQLite).

Stores extracted graph data — entities, triples and their chunk mentions —
for the knowledge graph pipeline. Mention rows express which chunks (and
therefore which files) reference each entity / triple; triple rows carry no
``file_ids`` column in this schema, so file attribution is derived from the
mention tables only.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import case, delete, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from yusu_kb.storage.sqlite.models_knowledge import (
    KnowledgeGraphEntity,
    KnowledgeGraphEntityMention,
    KnowledgeGraphTriple,
    KnowledgeGraphTripleMention,
)
from yusu_kb.utils.datetime_utils import utc_now_naive

from . import get_session_factory

SQL_IN_BATCH_SIZE = 500


def _iter_batches(items: list[Any], batch_size: int = SQL_IN_BATCH_SIZE) -> Iterator[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _merged_description_expr(column, new_desc) -> Any:
    """SQL expression merging a new description into an existing one.

    Keeps the old value unless the new one is non-empty and not already
    contained in it, in which case the new one is appended with ``'; '``.
    """
    return case(
        (or_(new_desc.is_(None), new_desc == ""), column),
        (or_(column.is_(None), column == ""), new_desc),
        (func.instr(column, new_desc) > 0, column),
        else_=column + "; " + new_desc,
    )


class KnowledgeGraphRepository:
    """SQLite-backed knowledge graph repository."""

    async def upsert_entities(self, kb_id: str, entities: list[dict[str, Any]]) -> None:
        if not entities:
            return
        now = utc_now_naive()
        async with get_session_factory()() as session:
            for batch in _iter_batches(entities):
                values = [
                    {
                        "entity_id": item["entity_id"],
                        "kb_id": kb_id,
                        "normalized_name": item["normalized_name"],
                        "label": item["label"],
                        "name": item["name"],
                        "attributes": item.get("attributes"),
                        "description": item.get("description"),
                    }
                    for item in batch
                ]
                stmt = sqlite_insert(KnowledgeGraphEntity).values(values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[
                        KnowledgeGraphEntity.kb_id,
                        KnowledgeGraphEntity.normalized_name,
                        KnowledgeGraphEntity.label,
                    ],
                    set_={
                        "name": stmt.excluded.name,
                        # None keeps the stored attributes instead of wiping them;
                        # the JSON column binds None as the literal string "null"
                        # (none_as_null=False), so both shapes must be matched
                        "attributes": case(
                            (
                                or_(
                                    stmt.excluded.attributes.is_(None),
                                    stmt.excluded.attributes == "null",
                                ),
                                KnowledgeGraphEntity.attributes,
                            ),
                            else_=stmt.excluded.attributes,
                        ),
                        "description": _merged_description_expr(
                            KnowledgeGraphEntity.description, stmt.excluded.description
                        ),
                        "updated_at": now,
                    },
                )
                await session.execute(stmt)
            await session.commit()

    async def upsert_relations(self, kb_id: str, triples: list[dict[str, Any]]) -> None:
        if not triples:
            return
        now = utc_now_naive()
        async with get_session_factory()() as session:
            for batch in _iter_batches(triples):
                values = [
                    {
                        "triple_id": item["triple_id"],
                        "kb_id": kb_id,
                        "source_entity_id": item["source_entity_id"],
                        "target_entity_id": item["target_entity_id"],
                        "relation_type": item["relation_type"],
                        "content": item["content"],
                        "description": item.get("description"),
                    }
                    for item in batch
                ]
                stmt = sqlite_insert(KnowledgeGraphTriple).values(values)
                stmt = stmt.on_conflict_do_update(
                    index_elements=[KnowledgeGraphTriple.triple_id],
                    set_={
                        "content": stmt.excluded.content,
                        "description": _merged_description_expr(
                            KnowledgeGraphTriple.description, stmt.excluded.description
                        ),
                        "updated_at": now,
                    },
                )
                await session.execute(stmt)
            await session.commit()

    async def upsert_mentions(self, kb_id: str, entity_mentions: list[dict[str, Any]]) -> None:
        if not entity_mentions:
            return
        async with get_session_factory()() as session:
            for batch in _iter_batches(entity_mentions):
                values = [
                    {
                        "entity_id": item["entity_id"],
                        "kb_id": kb_id,
                        "file_id": item["file_id"],
                        "chunk_id": item["chunk_id"],
                    }
                    for item in batch
                ]
                stmt = sqlite_insert(KnowledgeGraphEntityMention).values(values)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=[
                        KnowledgeGraphEntityMention.entity_id,
                        KnowledgeGraphEntityMention.chunk_id,
                    ]
                )
                await session.execute(stmt)
            await session.commit()

    async def upsert_triple_mentions(self, kb_id: str, triple_mentions: list[dict[str, Any]]) -> None:
        if not triple_mentions:
            return
        async with get_session_factory()() as session:
            for batch in _iter_batches(triple_mentions):
                values = [
                    {
                        "triple_id": item["triple_id"],
                        "kb_id": kb_id,
                        "file_id": item["file_id"],
                        "chunk_id": item["chunk_id"],
                        "text": item.get("text"),
                        "extractor_type": item.get("extractor_type"),
                    }
                    for item in batch
                ]
                stmt = sqlite_insert(KnowledgeGraphTripleMention).values(values)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=[
                        KnowledgeGraphTripleMention.triple_id,
                        KnowledgeGraphTripleMention.chunk_id,
                    ]
                )
                await session.execute(stmt)
            await session.commit()

    async def count_by_kb_id(self, kb_id: str) -> tuple[int, int]:
        """Return ``(entity_count, triple_count)`` for a knowledge base."""
        async with get_session_factory()() as session:
            entity_count = await session.execute(
                select(func.count(KnowledgeGraphEntity.id)).where(KnowledgeGraphEntity.kb_id == kb_id)
            )
            triple_count = await session.execute(
                select(func.count(KnowledgeGraphTriple.id)).where(KnowledgeGraphTriple.kb_id == kb_id)
            )
            return int(entity_count.scalar_one()), int(triple_count.scalar_one())

    async def delete_file_references(self, kb_id: str, file_id: str) -> dict[str, list[str]]:
        """Drop every graph mention of a file and report now-unreferenced ids.

        Only mention rows are removed; entity / triple rows are left untouched
        so the caller (GraphService) can decide whether to purge rows whose
        mentions are all gone. Orphans are the involved ids with no residual
        mentions left after deletion (i.e. not shared with any other file).
        """
        async with get_session_factory()() as session:
            entity_rows = await session.execute(
                select(KnowledgeGraphEntityMention.entity_id).where(
                    KnowledgeGraphEntityMention.kb_id == kb_id,
                    KnowledgeGraphEntityMention.file_id == file_id,
                )
            )
            involved_entity_ids = set(entity_rows.scalars().all())
            triple_rows = await session.execute(
                select(KnowledgeGraphTripleMention.triple_id).where(
                    KnowledgeGraphTripleMention.kb_id == kb_id,
                    KnowledgeGraphTripleMention.file_id == file_id,
                )
            )
            involved_triple_ids = set(triple_rows.scalars().all())

            await session.execute(
                delete(KnowledgeGraphEntityMention).where(
                    KnowledgeGraphEntityMention.kb_id == kb_id,
                    KnowledgeGraphEntityMention.file_id == file_id,
                )
            )
            await session.execute(
                delete(KnowledgeGraphTripleMention).where(
                    KnowledgeGraphTripleMention.kb_id == kb_id,
                    KnowledgeGraphTripleMention.file_id == file_id,
                )
            )

            orphan_entity_ids: list[str] = []
            if involved_entity_ids:
                residual = await session.execute(
                    select(KnowledgeGraphEntityMention.entity_id)
                    .where(
                        KnowledgeGraphEntityMention.kb_id == kb_id,
                        KnowledgeGraphEntityMention.entity_id.in_(involved_entity_ids),
                    )
                    .distinct()
                )
                orphan_entity_ids = sorted(involved_entity_ids - set(residual.scalars().all()))

            orphan_triple_ids: list[str] = []
            if involved_triple_ids:
                residual = await session.execute(
                    select(KnowledgeGraphTripleMention.triple_id)
                    .where(
                        KnowledgeGraphTripleMention.kb_id == kb_id,
                        KnowledgeGraphTripleMention.triple_id.in_(involved_triple_ids),
                    )
                    .distinct()
                )
                orphan_triple_ids = sorted(involved_triple_ids - set(residual.scalars().all()))

            await session.commit()
        return {"orphan_entity_ids": orphan_entity_ids, "orphan_triple_ids": orphan_triple_ids}

    async def delete_by_kb_id(self, kb_id: str) -> None:
        """Delete all graph rows of a knowledge base (mentions before main rows).

        The schema declares ``ON DELETE CASCADE`` foreign keys, but explicit
        per-table deletes keep the behaviour independent of cascade config.
        """
        async with get_session_factory()() as session:
            await session.execute(
                delete(KnowledgeGraphTripleMention).where(KnowledgeGraphTripleMention.kb_id == kb_id)
            )
            await session.execute(
                delete(KnowledgeGraphEntityMention).where(KnowledgeGraphEntityMention.kb_id == kb_id)
            )
            await session.execute(delete(KnowledgeGraphTriple).where(KnowledgeGraphTriple.kb_id == kb_id))
            await session.execute(delete(KnowledgeGraphEntity).where(KnowledgeGraphEntity.kb_id == kb_id))
            await session.commit()

    async def update_entity_descriptions(self, kb_id: str, desc_map: dict[str, str]) -> None:
        if not desc_map:
            return
        async with get_session_factory()() as session:
            for entity_id, description in desc_map.items():
                await session.execute(
                    update(KnowledgeGraphEntity)
                    .where(KnowledgeGraphEntity.kb_id == kb_id, KnowledgeGraphEntity.entity_id == entity_id)
                    .values(description=description)
                )
            await session.commit()

    async def update_triple_descriptions(self, kb_id: str, desc_map: dict[str, str]) -> None:
        if not desc_map:
            return
        async with get_session_factory()() as session:
            for triple_id, description in desc_map.items():
                await session.execute(
                    update(KnowledgeGraphTriple)
                    .where(KnowledgeGraphTriple.kb_id == kb_id, KnowledgeGraphTriple.triple_id == triple_id)
                    .values(description=description)
                )
            await session.commit()

    async def list_entities_by_kb_id(
        self, kb_id: str, limit: int = 200, offset: int = 0
    ) -> list[KnowledgeGraphEntity]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeGraphEntity)
                .where(KnowledgeGraphEntity.kb_id == kb_id)
                .order_by(KnowledgeGraphEntity.id)
                .offset(max(offset, 0))
                .limit(max(min(limit, 2000), 1))
            )
            return list(result.scalars().all())

    async def list_triples_by_kb_id(
        self, kb_id: str, limit: int = 200, offset: int = 0
    ) -> list[KnowledgeGraphTriple]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeGraphTriple)
                .where(KnowledgeGraphTriple.kb_id == kb_id)
                .order_by(KnowledgeGraphTriple.id)
                .offset(max(offset, 0))
                .limit(max(min(limit, 2000), 1))
            )
            return list(result.scalars().all())


__all__ = ["KnowledgeGraphRepository"]