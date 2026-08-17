"""Knowledge chunk repository (SQLite).

Ported from YUSU ``yuxi.repositories.knowledge_chunk_repository`` (demo
edition: graph-pending helpers simplified).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from sqlalchemy import delete, func, or_, select

from yusu_kb.storage.sqlite.models_knowledge import KnowledgeChunk

from . import get_session_factory

SQL_IN_BATCH_SIZE = 500

_ALLOWED_FIELDS = {
    "chunk_id",
    "file_id",
    "kb_id",
    "chunk_index",
    "content",
    "start_char_pos",
    "end_char_pos",
    "start_token_pos",
    "end_token_pos",
    "graph_indexed",
    "ent_ids",
    "tags",
    "extraction_result",
}


def _iter_batches(items: list[str], batch_size: int = SQL_IN_BATCH_SIZE) -> Iterator[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _sanitize_data(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key in _ALLOWED_FIELDS and value is not None}


class KnowledgeChunkRepository:
    """SQLite-backed knowledge chunk repository."""

    async def get_by_chunk_id(self, chunk_id: str) -> KnowledgeChunk | None:
        async with get_session_factory()() as session:
            result = await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.chunk_id == chunk_id))
            return result.scalar_one_or_none()

    async def list_by_file_id(self, file_id: str) -> list[KnowledgeChunk]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeChunk)
                .where(KnowledgeChunk.file_id == file_id)
                .order_by(KnowledgeChunk.chunk_index)
            )
            return list(result.scalars().all())

    async def list_by_file_ids(self, file_ids: list[str]) -> list[KnowledgeChunk]:
        if not file_ids:
            return []
        records: list[KnowledgeChunk] = []
        async with get_session_factory()() as session:
            for batch in _iter_batches(file_ids):
                result = await session.execute(
                    select(KnowledgeChunk).where(KnowledgeChunk.file_id.in_(batch))
                )
                records.extend(result.scalars().all())
        return records

    async def count_by_file_ids(self, file_ids: list[str]) -> dict[str, int]:
        if not file_ids:
            return {}
        counts: dict[str, int] = {}
        async with get_session_factory()() as session:
            for batch in _iter_batches(file_ids):
                rows = await session.execute(
                    select(KnowledgeChunk.file_id, func.count(KnowledgeChunk.id))
                    .where(KnowledgeChunk.file_id.in_(batch))
                    .group_by(KnowledgeChunk.file_id)
                )
                for file_id, count in rows:
                    counts[file_id] = int(count)
        return counts

    async def list_by_kb_id(self, kb_id: str, limit: int = 500, offset: int = 0) -> list[KnowledgeChunk]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeChunk)
                .where(KnowledgeChunk.kb_id == kb_id)
                .order_by(KnowledgeChunk.id)
                .offset(max(offset, 0))
                .limit(max(min(limit, 2000), 1))
            )
            return list(result.scalars().all())

    async def list_by_chunk_ids(self, chunk_ids: list[str]) -> list[KnowledgeChunk]:
        if not chunk_ids:
            return []
        records: list[KnowledgeChunk] = []
        async with get_session_factory()() as session:
            for batch in _iter_batches(chunk_ids):
                result = await session.execute(
                    select(KnowledgeChunk).where(KnowledgeChunk.chunk_id.in_(batch))
                )
                records.extend(result.scalars().all())
        return records

    async def search_chunks_by_terms(self, *, kb_id: str, terms: list[str], limit: int = 2000) -> list[KnowledgeChunk]:
        """Keyword search: chunks whose content contains any of the given terms.

        Ranking happens at the caller (simple match-count); this only narrows
        the candidate set via SQL ``LIKE`` conditions.
        """
        terms = [term for term in terms if term]
        if not terms or limit <= 0:
            return []
        conditions = [KnowledgeChunk.content.contains(term) for term in terms]
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeChunk)
                .where(KnowledgeChunk.kb_id == kb_id, or_(*conditions))
                .order_by(KnowledgeChunk.id)
                .limit(limit)
            )
            return list(result.scalars().all())

    async def batch_upsert(self, chunks: list[dict[str, Any]]) -> list[KnowledgeChunk]:
        records: list[KnowledgeChunk] = []
        async with get_session_factory()() as session:
            for chunk in chunks:
                sanitized = _sanitize_data(chunk)
                chunk_id = sanitized.pop("chunk_id", None)
                if not chunk_id:
                    continue
                result = await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.chunk_id == chunk_id))
                record = result.scalar_one_or_none()
                if record is None:
                    record = KnowledgeChunk(chunk_id=chunk_id, **sanitized)
                    session.add(record)
                else:
                    for key, value in sanitized.items():
                        setattr(record, key, value)
                records.append(record)
            await session.commit()
            for record in records:
                await session.refresh(record)
        return records

    async def delete_by_file_id(self, file_id: str) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.file_id == file_id))
            await session.commit()
            return result.rowcount or 0

    async def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        total = 0
        async with get_session_factory()() as session:
            for batch in _iter_batches(chunk_ids):
                result = await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.chunk_id.in_(batch)))
                total += result.rowcount or 0
            await session.commit()
        return total

    async def delete_by_file_id_except(self, file_id: str, keep_chunk_ids: set[str]) -> list[str]:
        """Delete chunks of a file not in ``keep_chunk_ids``; returns deleted ids."""
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeChunk.chunk_id).where(
                    KnowledgeChunk.file_id == file_id,
                    KnowledgeChunk.chunk_id.not_in(keep_chunk_ids) if keep_chunk_ids else KnowledgeChunk.file_id == file_id,
                )
            )
            ids_to_delete = [row for row in result.scalars().all()]
            if ids_to_delete:
                for batch in _iter_batches(ids_to_delete):
                    await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.chunk_id.in_(batch)))
            await session.commit()
        return ids_to_delete

    async def delete_by_kb_id(self, kb_id: str) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.kb_id == kb_id))
            await session.commit()
            return result.rowcount or 0

    async def count_by_kb_id(self, kb_id: str) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(func.count(KnowledgeChunk.id)).where(KnowledgeChunk.kb_id == kb_id)
            )
            return int(result.scalar_one())

    async def count_graph_indexed_by_kb_id(self, kb_id: str) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(func.count(KnowledgeChunk.id)).where(
                    KnowledgeChunk.kb_id == kb_id, KnowledgeChunk.graph_indexed.is_(True)
                )
            )
            return int(result.scalar_one())

    async def count_graph_indexed_by_file_id(self, file_id: str) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(func.count(KnowledgeChunk.id)).where(
                    KnowledgeChunk.file_id == file_id, KnowledgeChunk.graph_indexed.is_(True)
                )
            )
            return int(result.scalar_one())

    async def count_graph_pending_by_kb_id(self, kb_id: str) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(func.count(KnowledgeChunk.id)).where(
                    KnowledgeChunk.kb_id == kb_id, KnowledgeChunk.graph_indexed.is_(False)
                )
            )
            return int(result.scalar_one())

    async def list_graph_pending_by_kb_id(
        self, kb_id: str, limit: int = 100, *, exclude_chunk_ids: set[str] | None = None
    ) -> list[KnowledgeChunk]:
        """List chunks whose graph is not built yet (``graph_indexed`` false).

        ``exclude_chunk_ids`` is pushed into the query as a SQL ``NOT IN``
        (empty/None sets are skipped). Without it, supply-style consumers that
        filter in Python would stall: ``ORDER BY id LIMIT N`` keeps returning
        the same first rows until the flusher marks them indexed.
        """
        async with get_session_factory()() as session:
            stmt = select(KnowledgeChunk).where(
                KnowledgeChunk.kb_id == kb_id, KnowledgeChunk.graph_indexed.is_(False)
            )
            if exclude_chunk_ids:
                stmt = stmt.where(~KnowledgeChunk.chunk_id.in_(exclude_chunk_ids))
            stmt = stmt.order_by(KnowledgeChunk.id).limit(max(min(limit, 1000), 1))
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update_extraction_result(self, chunk_id: str, extraction_result: dict[str, Any]) -> None:
        async with get_session_factory()() as session:
            result = await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.chunk_id == chunk_id))
            record = result.scalar_one_or_none()
            if record is None:
                return
            record.extraction_result = extraction_result
            await session.commit()

    async def mark_graph_indexed(
        self,
        chunk_id: str,
        extraction_result: dict[str, Any] | None = None,
        ent_ids: list[str] | None = None,
    ) -> None:
        async with get_session_factory()() as session:
            result = await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.chunk_id == chunk_id))
            record = result.scalar_one_or_none()
            if record is None:
                return
            record.graph_indexed = True
            if extraction_result is not None:
                record.extraction_result = extraction_result
            if ent_ids is not None:
                record.ent_ids = ent_ids
            await session.commit()

    async def reset_graph_state_by_kb_id(self, kb_id: str, clear_extraction_result: bool = False) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(select(KnowledgeChunk).where(KnowledgeChunk.kb_id == kb_id))
            records = list(result.scalars().all())
            for record in records:
                record.graph_indexed = False
                if clear_extraction_result:
                    record.extraction_result = None
            await session.commit()
        return len(records)


__all__ = ["KnowledgeChunkRepository"]