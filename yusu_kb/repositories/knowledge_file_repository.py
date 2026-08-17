"""Knowledge file repository (SQLite).

Ported from YUSU ``yuxi.repositories.knowledge_file_repository`` (demo
edition: folder-scoped helpers dropped).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from sqlalchemy import delete, func, select, update

from yusu_kb.storage.sqlite.models_knowledge import KnowledgeFile

from . import get_session_factory

SQL_IN_BATCH_SIZE = 500

_ALLOWED_FIELDS = {
    "kb_id",
    "parent_id",
    "filename",
    "original_filename",
    "file_type",
    "path",
    "local_url",
    "markdown_file",
    "status",
    "content_hash",
    "file_size",
    "chunk_count",
    "token_count",
    "content_type",
    "processing_params",
    "is_folder",
    "error_message",
    "created_by",
    "updated_by",
}


def _iter_batches(items: list[str], batch_size: int = SQL_IN_BATCH_SIZE) -> Iterator[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _sanitize_data(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key in _ALLOWED_FIELDS and value is not None}


class KnowledgeFileRepository:
    """SQLite-backed knowledge file repository."""

    async def get_all(self) -> list[KnowledgeFile]:
        async with get_session_factory()() as session:
            result = await session.execute(select(KnowledgeFile).order_by(KnowledgeFile.id))
            return list(result.scalars().all())

    async def get_by_file_id(self, file_id: str) -> KnowledgeFile | None:
        async with get_session_factory()() as session:
            result = await session.execute(select(KnowledgeFile).where(KnowledgeFile.file_id == file_id))
            return result.scalar_one_or_none()

    async def list_by_file_ids(self, file_ids: list[str]) -> list[KnowledgeFile]:
        if not file_ids:
            return []
        records: list[KnowledgeFile] = []
        async with get_session_factory()() as session:
            for batch in _iter_batches(file_ids):
                result = await session.execute(select(KnowledgeFile).where(KnowledgeFile.file_id.in_(batch)))
                records.extend(result.scalars().all())
        return records

    async def get_filenames_by_file_ids(self, *, kb_id: str, file_ids: list[str]) -> dict[str, str]:
        """Map file ids to filenames for chunk source hydration."""
        if not file_ids:
            return {}
        result_map: dict[str, str] = {}
        async with get_session_factory()() as session:
            for batch in _iter_batches(file_ids):
                result = await session.execute(
                    select(KnowledgeFile.file_id, KnowledgeFile.filename).where(
                        KnowledgeFile.kb_id == kb_id, KnowledgeFile.file_id.in_(batch)
                    )
                )
                result_map.update({row[0]: row[1] for row in result.all()})
        return result_map

    async def list_file_ids_by_filename_contains(self, *, kb_id: str, filename_pattern: str) -> list[str]:
        """List file ids whose filename contains the given pattern (case-insensitive)."""
        pattern = f"%{filename_pattern}%"
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeFile.file_id).where(
                    KnowledgeFile.kb_id == kb_id, KnowledgeFile.filename.like(pattern)
                )
            )
            return list(result.scalars().all())

    async def list_by_kb_id(self, kb_id: str) -> list[KnowledgeFile]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeFile).where(KnowledgeFile.kb_id == kb_id).order_by(KnowledgeFile.id)
            )
            return list(result.scalars().all())

    async def list_by_kb_id_after(self, kb_id: str, file_id: str) -> list[KnowledgeFile]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeFile)
                .where(KnowledgeFile.kb_id == kb_id, KnowledgeFile.file_id > file_id)
                .order_by(KnowledgeFile.file_id)
            )
            return list(result.scalars().all())

    async def list_same_name_files(self, *, kb_id: str, filename: str) -> list[KnowledgeFile]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeFile).where(KnowledgeFile.kb_id == kb_id, KnowledgeFile.filename == filename)
            )
            return list(result.scalars().all())

    async def exists_by_content_hash(self, *, kb_id: str, content_hash: str) -> bool:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeFile.id).where(
                    KnowledgeFile.kb_id == kb_id, KnowledgeFile.content_hash == content_hash
                )
            )
            return result.first() is not None

    async def exists_by_filename(self, *, kb_id: str, filename: str) -> bool:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeFile.id).where(
                    KnowledgeFile.kb_id == kb_id, KnowledgeFile.filename == filename
                )
            )
            return result.first() is not None

    async def count_all(self) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(select(func.count(KnowledgeFile.id)))
            return int(result.scalar_one())

    async def list_file_ids_by_exact_statuses(self, *, kb_id: str, statuses: Iterable[str]) -> list[str]:
        status_list = list(statuses)
        if not status_list:
            return []
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeFile.file_id).where(
                    KnowledgeFile.kb_id == kb_id, KnowledgeFile.status.in_(status_list)
                )
            )
            return list(result.scalars().all())

    @staticmethod
    def _status_condition(status: str | None):
        return KnowledgeFile.status == status if status else None

    def _document_filters(self, *, kb_id: str, status: str | None, filename: str | None):
        conditions = [KnowledgeFile.kb_id == kb_id, KnowledgeFile.is_folder.is_(False)]
        status_condition = self._status_condition(status)
        if status_condition is not None:
            conditions.append(status_condition)
        if filename:
            conditions.append(KnowledgeFile.filename.ilike(f"%{filename}%"))
        return conditions

    async def list_documents(
        self,
        *,
        kb_id: str,
        status: str | None = None,
        filename: str | None = None,
        offset: int = 0,
        limit: int = 200,
    ) -> list[KnowledgeFile]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeFile)
                .where(*self._document_filters(kb_id=kb_id, status=status, filename=filename))
                .order_by(KnowledgeFile.id)
                .offset(max(offset, 0))
                .limit(max(min(limit, 500), 1))
            )
            return list(result.scalars().all())

    async def get_kb_file_stats(self, kb_id: str) -> dict[str, int]:
        """Aggregate per-status file counts plus chunk/token totals."""
        status_counts: dict[str, int] = {}
        chunk_total = 0
        token_total = 0
        size_total = 0
        async with get_session_factory()() as session:
            rows = await session.execute(
                select(KnowledgeFile.status, func.count(KnowledgeFile.id), func.sum(KnowledgeFile.file_size))
                .where(KnowledgeFile.kb_id == kb_id, KnowledgeFile.is_folder.is_(False))
                .group_by(KnowledgeFile.status)
            )
            for status, count, size in rows:
                status_counts[status or "unknown"] = int(count)
                if size:
                    size_total += int(size)
            chunk_row = await session.execute(
                select(func.coalesce(func.sum(KnowledgeFile.chunk_count), 0)).where(KnowledgeFile.kb_id == kb_id)
            )
            chunk_total = int(chunk_row.scalar_one())
            token_row = await session.execute(
                select(func.coalesce(func.sum(KnowledgeFile.token_count), 0)).where(KnowledgeFile.kb_id == kb_id)
            )
            token_total = int(token_row.scalar_one())

        total = sum(status_counts.values())
        return {
            "file_count": total,
            "row_count": total,
            "folder_count": 0,
            "chunk_count": chunk_total,
            "token_count": token_total,
            "total_size": size_total,
            "pending_parse_count": int(status_counts.get("uploaded", 0)) + int(status_counts.get("error_parsing", 0)),
            "pending_index_count": int(status_counts.get("parsed", 0)),
            "processing_count": int(status_counts.get("parsing", 0)) + int(status_counts.get("indexing", 0)),
            "indexed_count": int(status_counts.get("indexed", 0)) + int(status_counts.get("done", 0)),
            "error_count": int(status_counts.get("error_indexing", 0)) + int(status_counts.get("error_parsing", 0)),
        }

    async def upsert(self, file_id: str, data: dict[str, Any]) -> KnowledgeFile:
        sanitized = _sanitize_data(data)
        async with get_session_factory()() as session:
            result = await session.execute(select(KnowledgeFile).where(KnowledgeFile.file_id == file_id))
            record = result.scalar_one_or_none()
            if record is None:
                record = KnowledgeFile(file_id=file_id, **sanitized)
                session.add(record)
            else:
                for key, value in sanitized.items():
                    setattr(record, key, value)
            await session.commit()
            await session.refresh(record)
            return record

    async def update_fields(self, file_id: str, kb_id: str, data: dict[str, Any]) -> KnowledgeFile | None:
        sanitized = _sanitize_data(data)
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeFile).where(KnowledgeFile.file_id == file_id, KnowledgeFile.kb_id == kb_id)
            )
            record = result.scalar_one_or_none()
            if record is None:
                return None
            for key, value in sanitized.items():
                setattr(record, key, value)
            await session.commit()
            await session.refresh(record)
            return record

    async def update_fields_if_status(
        self, kb_id: str, file_id: str, allowed_statuses: set[str], data: dict[str, Any]
    ) -> KnowledgeFile | None:
        """Update a file only if its status is in ``allowed_statuses``."""
        sanitized = _sanitize_data(data)
        async with get_session_factory()() as session:
            result = await session.execute(
                select(KnowledgeFile).where(
                    KnowledgeFile.file_id == file_id,
                    KnowledgeFile.kb_id == kb_id,
                    KnowledgeFile.status.in_(allowed_statuses),
                )
            )
            record = result.scalar_one_or_none()
            if record is None:
                return None
            for key, value in sanitized.items():
                setattr(record, key, value)
            await session.commit()
            await session.refresh(record)
            return record

    async def batch_get_file_statuses(self, kb_id: str, file_ids: list[str]) -> dict[str, str]:
        if not file_ids:
            return {}
        result: dict[str, str] = {}
        async with get_session_factory()() as session:
            for batch in _iter_batches(file_ids):
                rows = await session.execute(
                    select(KnowledgeFile.file_id, KnowledgeFile.status).where(
                        KnowledgeFile.kb_id == kb_id, KnowledgeFile.file_id.in_(batch)
                    )
                )
                for file_id, status in rows:
                    result[file_id] = status or "unknown"
        return result

    async def delete(self, file_id: str) -> None:
        async with get_session_factory()() as session:
            await session.execute(delete(KnowledgeFile).where(KnowledgeFile.file_id == file_id))
            await session.commit()

    async def delete_by_kb_id(self, kb_id: str) -> None:
        async with get_session_factory()() as session:
            await session.execute(delete(KnowledgeFile).where(KnowledgeFile.kb_id == kb_id))
            await session.commit()

    async def reset_stuck_statuses(self) -> dict[str, int]:
        """Reset PROCESSING-style statuses left behind by dead processes."""
        async with get_session_factory()() as session:
            result = await session.execute(
                update(KnowledgeFile)
                .where(KnowledgeFile.status.in_(["parsing", "indexing"]))
                .values(status="uploaded", error_message=None)
            )
            await session.commit()
            return {"reset_count": result.rowcount or 0}