"""Evaluation repository (SQLite): 4-table CRUD for RAG evaluation.

Covers evaluation datasets / dataset items / runs / run items. ``run_id``
format validation and error-only filtering are pushed into SQL where possible
(the ``is_error`` column carries the EVAL-2 marker written by the service).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, func, select

from yusu_kb.storage.sqlite.models_knowledge import (
    EvaluationDataset,
    EvaluationDatasetItem,
    EvaluationRun,
    EvaluationRunItem,
)

from . import get_session_factory

_DATASET_FIELDS = {
    "dataset_id",
    "kb_id",
    "name",
    "description",
    "item_count",
    "has_gold_chunks",
    "has_gold_answers",
    "build_metadata",
    "created_by",
}

_DATASET_ITEM_FIELDS = {
    "item_id",
    "dataset_id",
    "kb_id",
    "item_index",
    "query_text",
    "gold_chunk_ids",
    "gold_answer",
}

_RUN_FIELDS = {
    "run_id",
    "name",
    "kb_id",
    "dataset_id",
    "status",
    "retrieval_config",
    "metrics",
    "overall_score",
    "total_items",
    "completed_items",
    "started_at",
    "completed_at",
    "created_by",
    "error_message",
}

_RUN_ITEM_FIELDS = {
    "run_id",
    "dataset_item_id",
    "item_index",
    "query_text",
    "gold_chunk_ids",
    "gold_answer",
    "generated_answer",
    "retrieved_chunks",
    "metrics",
    "is_error",
}


def _sanitize(data: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key in allowed and value is not None}


class EvaluationRepository:
    """SQLite-backed repository for evaluation datasets and runs."""

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    async def create_dataset(self, data: dict[str, Any]) -> EvaluationDataset:
        async with get_session_factory()() as session:
            record = EvaluationDataset(**_sanitize(data, _DATASET_FIELDS))
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def create_dataset_with_items(
        self, data: dict[str, Any], items: list[dict[str, Any]]
    ) -> EvaluationDataset:
        async with get_session_factory()() as session:
            record = EvaluationDataset(**_sanitize(data, _DATASET_FIELDS))
            session.add(record)
            for item in items:
                session.add(EvaluationDatasetItem(**_sanitize(item, _DATASET_ITEM_FIELDS)))
            await session.commit()
            await session.refresh(record)
            return record

    async def update_dataset(self, dataset_id: str, data: dict[str, Any]) -> None:
        async with get_session_factory()() as session:
            result = await session.execute(select(EvaluationDataset).where(EvaluationDataset.dataset_id == dataset_id))
            record = result.scalar_one_or_none()
            if record is None:
                return
            for key, value in _sanitize(data, _DATASET_FIELDS).items():
                setattr(record, key, value)
            await session.commit()

    async def get_dataset(self, dataset_id: str) -> EvaluationDataset | None:
        async with get_session_factory()() as session:
            result = await session.execute(select(EvaluationDataset).where(EvaluationDataset.dataset_id == dataset_id))
            return result.scalar_one_or_none()

    async def list_datasets(self, kb_id: str) -> list[EvaluationDataset]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(EvaluationDataset)
                .where(EvaluationDataset.kb_id == kb_id)
                .order_by(EvaluationDataset.created_at.desc())
            )
            return list(result.scalars().all())

    async def count_dataset_items(self, dataset_id: str) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(func.count(EvaluationDatasetItem.id)).where(
                    EvaluationDatasetItem.dataset_id == dataset_id
                )
            )
            return int(result.scalar_one())

    async def list_dataset_items(self, dataset_id: str, offset: int, limit: int) -> list[EvaluationDatasetItem]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(EvaluationDatasetItem)
                .where(EvaluationDatasetItem.dataset_id == dataset_id)
                .order_by(EvaluationDatasetItem.item_index)
                .offset(max(offset, 0))
                .limit(max(min(limit, 2000), 1))
            )
            return list(result.scalars().all())

    async def list_all_dataset_items(self, dataset_id: str) -> list[EvaluationDatasetItem]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(EvaluationDatasetItem)
                .where(EvaluationDatasetItem.dataset_id == dataset_id)
                .order_by(EvaluationDatasetItem.item_index)
            )
            return list(result.scalars().all())

    async def add_dataset_items(self, items: list[dict[str, Any]]) -> None:
        async with get_session_factory()() as session:
            for item in items:
                session.add(EvaluationDatasetItem(**_sanitize(item, _DATASET_ITEM_FIELDS)))
            await session.commit()

    async def delete_dataset(self, dataset_id: str) -> None:
        async with get_session_factory()() as session:
            await session.execute(delete(EvaluationDatasetItem).where(EvaluationDatasetItem.dataset_id == dataset_id))
            await session.execute(delete(EvaluationDataset).where(EvaluationDataset.dataset_id == dataset_id))
            await session.commit()

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    async def create_run(self, data: dict[str, Any]) -> EvaluationRun:
        async with get_session_factory()() as session:
            record = EvaluationRun(**_sanitize(data, _RUN_FIELDS))
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def update_run(self, run_id: str, data: dict[str, Any]) -> None:
        async with get_session_factory()() as session:
            result = await session.execute(select(EvaluationRun).where(EvaluationRun.run_id == run_id))
            record = result.scalar_one_or_none()
            if record is None:
                return
            for key, value in _sanitize(data, _RUN_FIELDS).items():
                setattr(record, key, value)
            await session.commit()

    async def get_run(self, run_id: str) -> EvaluationRun | None:
        async with get_session_factory()() as session:
            result = await session.execute(select(EvaluationRun).where(EvaluationRun.run_id == run_id))
            return result.scalar_one_or_none()

    async def list_runs(self, kb_id: str) -> list[EvaluationRun]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(EvaluationRun).where(EvaluationRun.kb_id == kb_id).order_by(EvaluationRun.started_at.desc())
            )
            return list(result.scalars().all())

    async def delete_run(self, run_id: str) -> None:
        async with get_session_factory()() as session:
            await session.execute(delete(EvaluationRunItem).where(EvaluationRunItem.run_id == run_id))
            await session.execute(delete(EvaluationRun).where(EvaluationRun.run_id == run_id))
            await session.commit()

    # ------------------------------------------------------------------
    # Run items
    # ------------------------------------------------------------------

    async def upsert_run_item(self, run_id: str, item_index: int, data: dict[str, Any]) -> None:
        """Insert or update a run item by the ``(run_id, item_index)`` key."""
        payload = _sanitize(data, _RUN_ITEM_FIELDS)
        payload["run_id"] = run_id
        payload["item_index"] = item_index
        async with get_session_factory()() as session:
            result = await session.execute(
                select(EvaluationRunItem).where(
                    EvaluationRunItem.run_id == run_id, EvaluationRunItem.item_index == item_index
                )
            )
            record = result.scalar_one_or_none()
            if record is None:
                session.add(EvaluationRunItem(**payload))
            else:
                for key, value in payload.items():
                    setattr(record, key, value)
            await session.commit()

    async def count_run_items(self, run_id: str) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(func.count(EvaluationRunItem.id)).where(EvaluationRunItem.run_id == run_id)
            )
            return int(result.scalar_one())

    async def list_run_items(self, run_id: str, offset: int, limit: int) -> list[EvaluationRunItem]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(EvaluationRunItem)
                .where(EvaluationRunItem.run_id == run_id)
                .order_by(EvaluationRunItem.item_index)
                .offset(max(offset, 0))
                .limit(max(min(limit, 2000), 1))
            )
            return list(result.scalars().all())

    async def count_run_items_by_error(self, run_id: str, *, is_error: bool) -> int:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(func.count(EvaluationRunItem.id)).where(
                    EvaluationRunItem.run_id == run_id, EvaluationRunItem.is_error.is_(is_error)
                )
            )
            return int(result.scalar_one())

    async def list_run_items_by_error(
        self, run_id: str, *, is_error: bool, offset: int, limit: int
    ) -> list[EvaluationRunItem]:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(EvaluationRunItem)
                .where(EvaluationRunItem.run_id == run_id, EvaluationRunItem.is_error.is_(is_error))
                .order_by(EvaluationRunItem.item_index)
                .offset(max(offset, 0))
                .limit(max(min(limit, 2000), 1))
            )
            return list(result.scalars().all())


__all__ = ["EvaluationRepository"]
