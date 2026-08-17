"""Tests for the SQLite repositories (kb / file / chunk)."""

from __future__ import annotations

import pytest

from yusu_kb.repositories import configure_repositories
from yusu_kb.repositories.knowledge_base_repository import KnowledgeBaseRepository
from yusu_kb.repositories.knowledge_chunk_repository import KnowledgeChunkRepository
from yusu_kb.repositories.knowledge_file_repository import KnowledgeFileRepository


@pytest.fixture()
def repos(engine):
    configure_repositories(engine)
    yield
    configure_repositories(None)


class TestKnowledgeBaseRepository:
    async def test_create_get_update_delete(self, repos):
        del repos
        repo = KnowledgeBaseRepository()
        created = await repo.create(
            {
                "kb_id": "kb_test01",
                "name": "测试库",
                "description": "desc",
                "kb_type": "local",
                "additional_params": {"preset_id": "general"},
            }
        )
        assert created.kb_id == "kb_test01"
        assert created.kb_type == "local"

        fetched = await repo.get_by_kb_id("kb_test01")
        assert fetched is not None
        assert fetched.name == "测试库"

        updated = await repo.update("kb_test01", {"name": "改名"})
        assert updated is not None
        assert updated.name == "改名"

        all_rows = await repo.get_all()
        assert [row.kb_id for row in all_rows] == ["kb_test01"]

        await repo.delete("kb_test01")
        assert await repo.get_by_kb_id("kb_test01") is None

    async def test_update_missing_returns_none(self, repos):
        del repos
        assert await KnowledgeBaseRepository().update("kb_missing", {"name": "x"}) is None


class TestKnowledgeFileRepository:
    async def _seed_kb(self, kb_id: str = "kb_test01"):
        await KnowledgeBaseRepository().create({"kb_id": kb_id, "name": kb_id, "kb_type": "local"})

    async def test_upsert_and_query(self, repos):
        del repos
        await self._seed_kb()
        repo = KnowledgeFileRepository()
        await repo.upsert(
            "file_1",
            {"kb_id": "kb_test01", "filename": "a.md", "status": "uploaded", "file_size": 10},
        )
        await repo.upsert(
            "file_2",
            {"kb_id": "kb_test01", "filename": "b.md", "status": "parsed", "chunk_count": 3},
        )
        await repo.upsert(
            "file_3",
            {"kb_id": "kb_test01", "filename": "c.md", "status": "parsed"},
        )

        record = await repo.get_by_file_id("file_1")
        assert record is not None and record.filename == "a.md"

        assert {f.filename for f in await repo.list_by_kb_id("kb_test01")} == {"a.md", "b.md", "c.md"}
        assert {f.file_id for f in await repo.list_by_file_ids(["file_1", "file_3"])} == {"file_1", "file_3"}
        assert await repo.exists_by_filename(kb_id="kb_test01", filename="b.md") is True
        assert await repo.exists_by_filename(kb_id="kb_test01", filename="zz.md") is False
        assert await repo.exists_by_content_hash(kb_id="kb_test01", content_hash="abc") is False

        assert await repo.list_file_ids_by_exact_statuses(kb_id="kb_test01", statuses=["parsed"]) == ["file_2", "file_3"]

        docs = await repo.list_documents(kb_id="kb_test01", status="parsed")
        assert [d.file_id for d in docs] == ["file_2", "file_3"]

        docs_by_name = await repo.list_documents(kb_id="kb_test01", filename="a")
        assert [d.file_id for d in docs_by_name] == ["file_1"]

    async def test_update_fields_and_status_guard(self, repos):
        del repos
        await self._seed_kb()
        repo = KnowledgeFileRepository()
        await repo.upsert("file_1", {"kb_id": "kb_test01", "filename": "a.md", "status": "uploaded"})

        updated = await repo.update_fields("file_1", "kb_test01", {"status": "parsing"})
        assert updated is not None and updated.status == "parsing"

        claimed = await repo.update_fields_if_status(
            kb_id="kb_test01",
            file_id="file_1",
            allowed_statuses={"uploaded", "error_parsing"},
            data={"status": "parsed"},
        )
        assert claimed is None  # status is "parsing", guard refuses

        claimed = await repo.update_fields_if_status(
            kb_id="kb_test01",
            file_id="file_1",
            allowed_statuses={"parsing"},
            data={"status": "parsed", "error_message": None},
        )
        assert claimed is not None and claimed.status == "parsed"

        assert await repo.batch_get_file_statuses("kb_test01", ["file_1", "nope"]) == {"file_1": "parsed"}

    async def test_delete_and_stats(self, repos):
        del repos
        await self._seed_kb("kb_a")
        repo = KnowledgeFileRepository()
        for file_id, status in (("f1", "uploaded"), ("f2", "parsed"), ("f3", "indexed")):
            await repo.upsert(file_id, {"kb_id": "kb_a", "filename": f"{file_id}.md", "status": status, "file_size": 100})

        stats = await repo.get_kb_file_stats("kb_a")
        assert stats["file_count"] == 3
        assert stats["pending_parse_count"] == 1
        assert stats["pending_index_count"] == 1
        assert stats["indexed_count"] == 1
        assert stats["total_size"] == 300

        await repo.delete("f1")
        assert await repo.get_by_file_id("f1") is None

        await repo.delete_by_kb_id("kb_a")
        assert await repo.list_by_kb_id("kb_a") == []
        assert await repo.count_all() == 0

    async def test_reset_stuck_statuses(self, repos):
        del repos
        await self._seed_kb("kb_a")
        repo = KnowledgeFileRepository()
        await repo.upsert("f1", {"kb_id": "kb_a", "filename": "a.md", "status": "parsing"})
        await repo.upsert("f2", {"kb_id": "kb_a", "filename": "b.md", "status": "indexed"})
        result = await repo.reset_stuck_statuses()
        assert result["reset_count"] == 1
        assert (await repo.get_by_file_id("f1")).status == "uploaded"


class TestKnowledgeChunkRepository:
    async def _seed_parents(self):
        """Chunk rows FK-reference ys_knowledge_bases and ys_knowledge_files."""
        await KnowledgeBaseRepository().create({"kb_id": "kb_a", "name": "A", "kb_type": "local"})
        file_repo = KnowledgeFileRepository()
        for file_id in ("file_1", "file_2", "f1", "f2"):
            await file_repo.upsert(file_id, {"kb_id": "kb_a", "filename": f"{file_id}.md", "status": "parsed"})

    async def test_batch_upsert_and_queries(self, repos):
        del repos
        await self._seed_parents()
        repo = KnowledgeChunkRepository()
        chunks = [
            {"chunk_id": "c1", "file_id": "file_1", "kb_id": "kb_a", "chunk_index": 0, "content": "第一块"},
            {"chunk_id": "c2", "file_id": "file_1", "kb_id": "kb_a", "chunk_index": 1, "content": "第二块"},
            {"chunk_id": "c3", "file_id": "file_2", "kb_id": "kb_a", "chunk_index": 0, "content": "第三块"},
        ]
        created = await repo.batch_upsert(chunks)
        assert len(created) == 3

        assert (await repo.get_by_chunk_id("c1")).content == "第一块"
        assert [c.chunk_id for c in await repo.list_by_file_id("file_1")] == ["c1", "c2"]
        assert await repo.count_by_file_ids(["file_1", "file_2"]) == {"file_1": 2, "file_2": 1}
        assert await repo.count_by_kb_id("kb_a") == 3

        # upsert again with changed content (idempotent by chunk_id)
        await repo.batch_upsert([{"chunk_id": "c1", "file_id": "file_1", "kb_id": "kb_a", "chunk_index": 0, "content": "改写"}])
        assert (await repo.get_by_chunk_id("c1")).content == "改写"
        assert await repo.count_by_kb_id("kb_a") == 3

    async def test_delete_and_graph_state(self, repos):
        del repos
        await self._seed_parents()
        repo = KnowledgeChunkRepository()
        await repo.batch_upsert(
            [
                {"chunk_id": "c1", "file_id": "f1", "kb_id": "kb_a", "chunk_index": 0, "content": "x"},
                {"chunk_id": "c2", "file_id": "f1", "kb_id": "kb_a", "chunk_index": 1, "content": "y"},
                {"chunk_id": "c3", "file_id": "f2", "kb_id": "kb_a", "chunk_index": 0, "content": "z"},
            ]
        )

        await repo.mark_graph_indexed("c1", extraction_result={"entities": []})
        assert await repo.count_graph_indexed_by_kb_id("kb_a") == 1
        assert await repo.count_graph_indexed_by_file_id("f1") == 1
        assert await repo.count_graph_pending_by_kb_id("kb_a") == 2
        assert [c.chunk_id for c in await repo.list_graph_pending_by_kb_id("kb_a")] == ["c2", "c3"]

        await repo.update_extraction_result("c2", {"entities": ["e1"]})
        assert (await repo.get_by_chunk_id("c2")).extraction_result == {"entities": ["e1"]}

        deleted = await repo.delete_by_file_id_except("f1", keep_chunk_ids={"c2"})
        assert sorted(deleted) == ["c1"]
        assert [c.chunk_id for c in await repo.list_by_file_id("f1")] == ["c2"]

        assert await repo.delete_by_chunk_ids(["c2", "c3"]) == 2
        assert await repo.count_by_kb_id("kb_a") == 0

    async def test_reset_graph_state(self, repos):
        del repos
        await self._seed_parents()
        repo = KnowledgeChunkRepository()
        await repo.batch_upsert(
            [
                {"chunk_id": "c1", "file_id": "f1", "kb_id": "kb_a", "chunk_index": 0, "content": "x"},
                {"chunk_id": "c2", "file_id": "f1", "kb_id": "kb_a", "chunk_index": 1, "content": "y"},
            ]
        )
        await repo.mark_graph_indexed("c1", extraction_result={"entities": ["e"]})
        reset = await repo.reset_graph_state_by_kb_id("kb_a", clear_extraction_result=True)
        assert reset == 2
        assert await repo.count_graph_pending_by_kb_id("kb_a") == 2
        assert (await repo.get_by_chunk_id("c1")).extraction_result is None