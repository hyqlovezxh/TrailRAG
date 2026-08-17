"""Tests for the SQLite schema: idempotent init, CRUD, cascades."""

from __future__ import annotations

from sqlalchemy import func, select

from yusu_kb.storage.sqlite.engine import create_engine, init_db
from yusu_kb.storage.sqlite.models_knowledge import (
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeFile,
)


async def test_init_db_idempotent(tmp_workdir):
    engine = create_engine(tmp_workdir / "yusu.db")
    await init_db(engine)
    await init_db(engine)
    async with engine.begin() as conn:
        from sqlalchemy import inspect

        tables = await conn.run_sync(lambda sc: list(inspect(sc).get_table_names()))
    await engine.dispose()
    expected = {
        "ys_knowledge_bases",
        "ys_knowledge_files",
        "ys_knowledge_chunks",
        "ys_knowledge_graph_entities",
        "ys_knowledge_graph_entity_mentions",
        "ys_knowledge_graph_triples",
        "ys_knowledge_graph_triple_mentions",
        "ys_evaluation_datasets",
        "ys_evaluation_dataset_items",
        "ys_evaluation_runs",
        "ys_evaluation_run_items",
        "ys_llm_cache_entries",
        "ys_embedding_cache_entries",
    }
    assert expected.issubset(set(tables))


async def test_kb_crud(session_factory):
    async with session_factory() as session:
        kb = KnowledgeBase(kb_id="kb-1", name="测试库", kb_type="milvus", description="demo")
        session.add(kb)
        await session.commit()

    async with session_factory() as session:
        result = await session.get(KnowledgeBase, kb.id)
        assert result is not None
        assert result.kb_id == "kb-1"
        assert result.name == "测试库"

    async with session_factory() as session:
        await session.delete(kb)
        await session.commit()

    async with session_factory() as session:
        assert await session.get(KnowledgeBase, kb.id) is None


async def test_kb_id_unique(session_factory):
    async with session_factory() as session:
        session.add_all(
            [
                KnowledgeBase(kb_id="kb-x", name="A"),
                KnowledgeBase(kb_id="kb-x", name="B"),
            ]
        )
        import pytest
        from sqlalchemy.exc import IntegrityError

        with pytest.raises(IntegrityError):
            await session.commit()


async def test_file_and_chunk_cascade_delete(session_factory):
    async with session_factory() as session:
        kb = KnowledgeBase(kb_id="kb-2", name="级联测试", kb_type="local")
        session.add(kb)
        await session.flush()
        f = KnowledgeFile(
            file_id="file-1",
            kb_id=kb.kb_id,
            filename="a.md",
            original_filename="a.md",
            file_type="md",
            status="uploaded",
        )
        session.add(f)
        await session.flush()
        session.add(
            KnowledgeChunk(
                chunk_id="chunk-1",
                file_id=f.file_id,
                kb_id=kb.kb_id,
                chunk_index=0,
                content="内容",
            )
        )
        await session.commit()
        await session.delete(kb)
        await session.commit()

    async with session_factory() as session:
        # Cascade: deleting the KB removes files and chunks.
        assert (await session.execute(select(func.count()).select_from(KnowledgeFile))).scalar() == 0
        assert (await session.execute(select(func.count()).select_from(KnowledgeChunk))).scalar() == 0
