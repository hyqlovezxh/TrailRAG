"""Tests for the knowledge base lifecycle (create/add/parse/preview/find/delete)
exercised through a minimal concrete subclass of ``KnowledgeBase``."""

from __future__ import annotations

from pathlib import Path

import pytest

from yusu_kb.knowledge.base import FileStatus, KnowledgeBase
from yusu_kb.knowledge.factory import KnowledgeBaseFactory
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.repositories import configure_repositories
from yusu_kb.repositories.knowledge_base_repository import KnowledgeBaseRepository


@KnowledgeBaseFactory.register("fake")
class FakeKB(KnowledgeBase):
    """Minimal concrete knowledge base for testing shared base logic."""

    kb_type = "fake"
    name = "Fake KB"
    description = "Test knowledge base type"

    def get_query_params_config(self, kb_id: str, **kwargs) -> dict:
        return {
            "options": [
                {"key": "top_k", "label": "Top K", "type": "number", "default": 10},
                {"key": "mode", "label": "Mode", "type": "select", "default": "mix", "options": ["mix", "local", "global"]},
            ]
        }

    async def _create_kb_instance(self, kb_id: str, config: dict):
        return {"id": kb_id}

    async def _initialize_kb_instance(self, instance) -> None:
        return None

    async def index_file(self, kb_id: str, file_id: str, operator_id: str | None = None) -> dict:
        return {"status": FileStatus.INDEXED}

    async def update_content(self, kb_id: str, file_ids: list[str], params: dict | None = None) -> list[dict]:
        return []

    async def aquery(self, query_text: str, kb_id: str, **kwargs) -> list[dict]:
        return []

    async def delete_file(self, kb_id: str, file_id: str) -> None:
        return None

    async def get_file_basic_info(self, kb_id: str, file_id: str) -> dict:
        meta = await self._get_file_meta(kb_id, file_id)
        return self._file_record_to_meta(meta) if not isinstance(meta, dict) else meta

    async def get_file_content(self, kb_id: str, file_id: str) -> dict:
        return {"chunks": []}

    async def get_file_info(self, kb_id: str, file_id: str) -> dict:
        basic = await self.get_file_basic_info(kb_id, file_id)
        content = await self.get_file_content(kb_id, file_id)
        return {**basic, **content}


@pytest.fixture()
async def manager(tmp_path, engine):
    configure_repositories(engine)
    kb_manager = KnowledgeBaseManager(str(tmp_path / "kb_work"))
    yield kb_manager
    await kb_manager.close()
    configure_repositories(None)


@pytest.fixture()
def sample_md(tmp_path: Path) -> str:
    path = tmp_path / "intro.md"
    path.write_text("# 简介\n\n这是知识库的第一份测试文档。\n\n## 章节\n\n包含若干段落内容。", encoding="utf-8")
    return str(path)


class TestManagerLifecycle:
    async def test_available_types(self, manager):
        types = manager.get_available_types()
        assert "fake" in types
        assert types["fake"]["name"] == "Fake KB"
        assert manager.is_type_supported("fake")
        assert not manager.is_type_supported("nope")

    async def test_create_get_update_delete_database(self, manager):
        created = await manager.create_database(name="测试知识库", description="演示", kb_type="fake")
        kb_id = created["kb_id"]
        assert kb_id.startswith("kb_")
        assert created["metadata"]["stats"]["file_count"] == 0

        info = await manager.get_database_info(kb_id)
        assert info is not None
        assert info["name"] == "测试知识库"

        databases = await manager.get_databases()
        assert [db["kb_id"] for db in databases["databases"]] == [kb_id]

        updated = await manager.update_database(kb_id, name="改名库", description="新描述")
        assert updated["name"] == "改名库"

        # persistence round-trip via metadata reload
        await manager.get_instance("fake")._load_metadata()
        assert kb_id in manager.get_instance("fake").databases_meta

        result = await manager.delete_database(kb_id)
        assert result["message"] == "删除成功"
        assert await manager.get_database_info(kb_id) is None

    async def test_create_database_validations(self, manager):
        from yusu_kb.knowledge.base import KBOperationError

        with pytest.raises(KBOperationError):
            await manager.create_database(name="x", kb_type="unknown_type")
        with pytest.raises(KBOperationError):
            await manager.create_database(name="  ", kb_type="fake")

    async def test_query_params_defaults_and_overrides(self, manager):
        kb_id = (await manager.create_database(name="参数库", kb_type="fake"))["kb_id"]

        defaults = await manager.get_query_params(kb_id)
        assert defaults["options"]["top_k"] == 10
        assert defaults["options"]["mode"] == "mix"

        merged = await manager.update_query_params(kb_id, {"mode": "local", "bogus_key": 1})
        assert merged["options"]["mode"] == "local"
        assert "bogus_key" not in merged["options"]

    async def test_add_record_parse_preview_find_open(self, manager, sample_md):
        kb_id = (await manager.create_database(name="文件库", kb_type="fake"))["kb_id"]

        record = await manager.add_file_record(kb_id, sample_md)
        file_id = record["file_id"]
        assert record["status"] == FileStatus.UPLOADED
        assert record["filename"] == "intro.md"

        statuses = await manager.get_instance("fake").batch_get_file_statuses(kb_id, [file_id])
        assert statuses[file_id] == FileStatus.UPLOADED

        # adding the same file again yields a distinct file id
        duplicate = await manager.add_file_record(kb_id, sample_md)
        assert duplicate["file_id"] != file_id

        parsed = await manager.parse_file(kb_id, file_id)
        assert parsed["status"] == FileStatus.PARSED
        assert parsed["markdown_file"]

        stats = await manager.refresh_database_stats(kb_id)
        assert stats["file_count"] == 2

        preview = await manager.read_file_preview(kb_id, file_id)
        assert preview["preview_type"] == "markdown"
        assert "简介" in preview["content"]

        opened = await manager.open_file_content(kb_id, file_id)
        assert opened["total_lines"] >= 3
        assert opened["start_line"] == 1

        found = await manager.find_file_content(kb_id, file_id, ["知识库"])
        assert found["total_matches"] >= 1
        assert found["match_mode"] == "keyword"
        assert found["windows"][0]["content"]

        download = await manager.get_file_download(kb_id, file_id, variant="parsed")
        assert "简介" in download["content"].decode("utf-8")

    async def test_parse_file_requires_supported_extension(self, manager, tmp_path):
        kb_id = (await manager.create_database(name="文件库", kb_type="fake"))["kb_id"]
        bad_path = tmp_path / "script.exe"
        bad_path.write_bytes(b"MZ")
        with pytest.raises(ValueError):
            await manager.add_file_record(kb_id, str(bad_path))

    async def test_parse_error_sets_status(self, manager, tmp_path):
        kb_id = (await manager.create_database(name="文件库", kb_type="fake"))["kb_id"]
        bad_pdf = tmp_path / "broken.pdf"
        bad_pdf.write_bytes(b"not a pdf at all")
        record = await manager.add_file_record(kb_id, str(bad_pdf))
        with pytest.raises(ValueError):
            await manager.parse_file(kb_id, record["file_id"])
        statuses = await manager.get_instance("fake").batch_get_file_statuses(kb_id, [record["file_id"]])
        assert statuses[record["file_id"]] == FileStatus.ERROR_PARSING

    async def test_update_file_params(self, manager, sample_md):
        kb_id = (await manager.create_database(name="文件库", kb_type="fake"))["kb_id"]
        file_id = (await manager.add_file_record(kb_id, sample_md))["file_id"]

        await manager.update_file_params(kb_id, file_id, {"max_chunk_tokens": 200})
        meta = await manager.get_file_basic_info(kb_id, file_id)
        assert meta["processing_params"]["max_chunk_tokens"] == 200
        assert meta["processing_params"]["preset_id"] == "general"


class TestBasePersistence:
    async def test_metadata_round_trip_across_instances(self, manager, tmp_path, engine):
        del manager
        kb_id = None
        work_dir = str(tmp_path / "kb_work")

        m1 = KnowledgeBaseManager(work_dir)
        created = await m1.create_database(name="持久化库", description="d", kb_type="fake")
        kb_id = created["kb_id"]
        await m1.close()

        # a fresh manager must rediscover the database from SQLite
        m2 = KnowledgeBaseManager(work_dir)
        await m2.load_all_metadata()
        info = await m2.get_database_info(kb_id)
        assert info is not None and info["name"] == "持久化库"
        await m2.close()

        # row also visible via the repository directly
        configure_repositories(engine)
        row = await KnowledgeBaseRepository().get_by_kb_id(kb_id)
        assert row is not None and row.kb_type == "fake"
        configure_repositories(None)