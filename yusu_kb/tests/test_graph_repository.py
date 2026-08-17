"""Tests for the SQLite knowledge graph repository."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from yusu_kb.repositories import configure_repositories
from yusu_kb.repositories.knowledge_base_repository import KnowledgeBaseRepository
from yusu_kb.repositories.knowledge_chunk_repository import KnowledgeChunkRepository
from yusu_kb.repositories.knowledge_file_repository import KnowledgeFileRepository
from yusu_kb.repositories.knowledge_graph_repository import KnowledgeGraphRepository
from yusu_kb.storage.sqlite.models_knowledge import (
    KnowledgeGraphEntity,
    KnowledgeGraphEntityMention,
    KnowledgeGraphTriple,
    KnowledgeGraphTripleMention,
)


@pytest.fixture()
def repos(engine):
    configure_repositories(engine)
    yield
    configure_repositories(None)


class TestKnowledgeGraphRepository:
    KB_ID = "kb_graph"

    async def _seed_parents(self) -> None:
        """KB + files + chunks rows the graph tables FK-reference."""
        await KnowledgeBaseRepository().create({"kb_id": self.KB_ID, "name": "Graph", "kb_type": "local"})
        file_repo = KnowledgeFileRepository()
        for file_id in ("fA", "fB"):
            await file_repo.upsert(file_id, {"kb_id": self.KB_ID, "filename": f"{file_id}.md", "status": "parsed"})
        await KnowledgeChunkRepository().batch_upsert(
            [
                {"chunk_id": "cA1", "file_id": "fA", "kb_id": self.KB_ID, "chunk_index": 0, "content": "块A1"},
                {"chunk_id": "cA2", "file_id": "fA", "kb_id": self.KB_ID, "chunk_index": 1, "content": "块A2"},
                {"chunk_id": "cB1", "file_id": "fB", "kb_id": self.KB_ID, "chunk_index": 0, "content": "块B1"},
            ]
        )

    async def _seed_entities(self, repo: KnowledgeGraphRepository) -> None:
        await repo.upsert_entities(
            self.KB_ID,
            [
                {
                    "entity_id": "e1",
                    "normalized_name": "alpha",
                    "label": "concept",
                    "name": "Alpha",
                    "attributes": [{"key": "v1"}],
                    "description": "Alpha 描述",
                },
                {
                    "entity_id": "e2",
                    "normalized_name": "beta",
                    "label": "concept",
                    "name": "Beta",
                    "attributes": None,
                    "description": "Beta 描述",
                },
                {
                    "entity_id": "e3",
                    "normalized_name": "gamma",
                    "label": "org",
                    "name": "Gamma",
                    "attributes": None,
                    "description": "Gamma 描述",
                },
            ],
        )

    async def _seed_triples(self, repo: KnowledgeGraphRepository) -> None:
        await repo.upsert_relations(
            self.KB_ID,
            [
                {
                    "triple_id": "t1",
                    "source_entity_id": "e1",
                    "target_entity_id": "e2",
                    "relation_type": "related_to",
                    "content": "alpha 关联 beta",
                    "description": "关系一",
                },
                {
                    "triple_id": "t2",
                    "source_entity_id": "e2",
                    "target_entity_id": "e3",
                    "relation_type": "part_of",
                    "content": "beta 属于 gamma",
                    "description": "关系二",
                },
            ],
        )

    async def _seed_mentions(self, repo: KnowledgeGraphRepository) -> None:
        await repo.upsert_mentions(
            self.KB_ID,
            [
                {"entity_id": "e1", "file_id": "fA", "chunk_id": "cA1"},
                {"entity_id": "e2", "file_id": "fA", "chunk_id": "cA1"},
                {"entity_id": "e2", "file_id": "fB", "chunk_id": "cB1"},
                {"entity_id": "e3", "file_id": "fB", "chunk_id": "cB1"},
            ],
        )
        await repo.upsert_triple_mentions(
            self.KB_ID,
            [
                {"triple_id": "t1", "file_id": "fA", "chunk_id": "cA1", "text": "文中句一", "extractor_type": "llm"},
                {"triple_id": "t2", "file_id": "fA", "chunk_id": "cA1", "text": "文中句二", "extractor_type": "llm"},
                {"triple_id": "t2", "file_id": "fB", "chunk_id": "cB1", "text": "文中句三", "extractor_type": "llm"},
            ],
        )

    async def _full_seed(self) -> None:
        repo = KnowledgeGraphRepository()
        await self._seed_parents()
        await self._seed_entities(repo)
        await self._seed_triples(repo)
        await self._seed_mentions(repo)

    async def _table_count(self, session_factory, model) -> int:
        async with session_factory() as session:
            result = await session.execute(select(func.count()).select_from(model))
            return int(result.scalar_one())

    async def test_upsert_entities_idempotent_and_merge(self, repos, session_factory):
        del repos
        await self._seed_parents()
        repo = KnowledgeGraphRepository()
        await self._seed_entities(repo)

        assert await repo.count_by_kb_id(self.KB_ID) == (3, 0)
        assert await self._table_count(session_factory, KnowledgeGraphEntity) == 3

        # same (normalized_name, label) again → no new row, description appended, attributes replaced
        await repo.upsert_entities(
            self.KB_ID,
            [
                {
                    "entity_id": "e1",
                    "normalized_name": "alpha",
                    "label": "concept",
                    "name": "Alpha",
                    "attributes": [{"key": "v2"}],
                    "description": "第二描述",
                },
            ],
        )
        assert await repo.count_by_kb_id(self.KB_ID) == (3, 0)
        assert await self._table_count(session_factory, KnowledgeGraphEntity) == 3
        alpha = next(e for e in await repo.list_entities_by_kb_id(self.KB_ID) if e.entity_id == "e1")
        assert alpha.description == "Alpha 描述; 第二描述"
        assert alpha.attributes == [{"key": "v2"}]

        # duplicate description → unchanged; None attributes keep the old ones
        await repo.upsert_entities(
            self.KB_ID,
            [
                {
                    "entity_id": "e1",
                    "normalized_name": "alpha",
                    "label": "concept",
                    "name": "Alpha",
                    "attributes": None,
                    "description": "第二描述",
                },
            ],
        )
        alpha = next(e for e in await repo.list_entities_by_kb_id(self.KB_ID) if e.entity_id == "e1")
        assert alpha.description == "Alpha 描述; 第二描述"
        assert alpha.attributes == [{"key": "v2"}]

    async def test_upsert_relations_idempotent_and_merge(self, repos, session_factory):
        del repos
        await self._seed_parents()
        await self._seed_entities(KnowledgeGraphRepository())
        repo = KnowledgeGraphRepository()
        await self._seed_triples(repo)

        assert await repo.count_by_kb_id(self.KB_ID) == (3, 2)

        # same triple_id again → one row, content replaced, description appended
        await repo.upsert_relations(
            self.KB_ID,
            [
                {
                    "triple_id": "t1",
                    "source_entity_id": "e1",
                    "target_entity_id": "e2",
                    "relation_type": "related_to",
                    "content": "内容改写",
                    "description": "新描述",
                },
            ],
        )
        assert await repo.count_by_kb_id(self.KB_ID) == (3, 2)
        assert await self._table_count(session_factory, KnowledgeGraphTriple) == 2
        t1 = next(t for t in await repo.list_triples_by_kb_id(self.KB_ID) if t.triple_id == "t1")
        assert t1.content == "内容改写"
        assert t1.description == "关系一; 新描述"

        # duplicate description → unchanged
        await repo.upsert_relations(
            self.KB_ID,
            [
                {
                    "triple_id": "t1",
                    "source_entity_id": "e1",
                    "target_entity_id": "e2",
                    "relation_type": "related_to",
                    "content": "内容改写",
                    "description": "新描述",
                },
            ],
        )
        t1 = next(t for t in await repo.list_triples_by_kb_id(self.KB_ID) if t.triple_id == "t1")
        assert t1.description == "关系一; 新描述"

    async def test_upsert_mentions_idempotent(self, repos, session_factory):
        del repos
        await self._full_seed()
        repo = KnowledgeGraphRepository()

        # re-upserting the same (entity_id, chunk_id) / (triple_id, chunk_id) adds nothing
        await repo.upsert_mentions(self.KB_ID, [{"entity_id": "e1", "file_id": "fA", "chunk_id": "cA1"}])
        await repo.upsert_triple_mentions(
            self.KB_ID,
            [{"triple_id": "t1", "file_id": "fA", "chunk_id": "cA1", "text": "另一句", "extractor_type": "rule"}],
        )
        assert await self._table_count(session_factory, KnowledgeGraphEntityMention) == 4
        assert await self._table_count(session_factory, KnowledgeGraphTripleMention) == 3

        # first-row text/extractor_type survive the do-nothing conflict
        async with session_factory() as session:
            result = await session.execute(
                select(KnowledgeGraphTripleMention.text, KnowledgeGraphTripleMention.extractor_type).where(
                    KnowledgeGraphTripleMention.triple_id == "t1",
                    KnowledgeGraphTripleMention.chunk_id == "cA1",
                )
            )
            text, extractor = result.one()
        assert (text, extractor) == ("文中句一", "llm")

    async def test_count_by_kb_id(self, repos):
        del repos
        await self._full_seed()
        repo = KnowledgeGraphRepository()
        assert await repo.count_by_kb_id(self.KB_ID) == (3, 2)
        assert await repo.count_by_kb_id("kb_other") == (0, 0)

    async def test_delete_file_references_orphan_detection(self, repos, session_factory):
        del repos
        await self._full_seed()
        repo = KnowledgeGraphRepository()

        # e1 / t1 are mentioned only in fA → orphans after removal
        result = await repo.delete_file_references(self.KB_ID, "fA")
        assert result == {"orphan_entity_ids": ["e1"], "orphan_triple_ids": ["t1"]}

        # entity / triple rows themselves are left untouched (caller decides purge)
        assert await repo.count_by_kb_id(self.KB_ID) == (3, 2)

        # e2 / t2 are shared with fB → not orphans; deleting fB orphans them
        result = await repo.delete_file_references(self.KB_ID, "fB")
        assert sorted(result["orphan_entity_ids"]) == ["e2", "e3"]
        assert result["orphan_triple_ids"] == ["t2"]

        assert await self._table_count(session_factory, KnowledgeGraphEntityMention) == 0
        assert await self._table_count(session_factory, KnowledgeGraphTripleMention) == 0

    async def test_delete_by_kb_id(self, repos, session_factory):
        del repos
        await self._full_seed()
        repo = KnowledgeGraphRepository()
        await repo.delete_by_kb_id(self.KB_ID)
        assert await repo.count_by_kb_id(self.KB_ID) == (0, 0)
        for model in (
            KnowledgeGraphEntity,
            KnowledgeGraphEntityMention,
            KnowledgeGraphTriple,
            KnowledgeGraphTripleMention,
        ):
            assert await self._table_count(session_factory, model) == 0

    async def test_update_descriptions(self, repos):
        del repos
        await self._full_seed()
        repo = KnowledgeGraphRepository()
        await repo.update_entity_descriptions(self.KB_ID, {"e1": "实体新描述", "e2": "Beta 新描述"})
        await repo.update_triple_descriptions(self.KB_ID, {"t1": "三元组新描述"})
        entities = {e.entity_id: e for e in await repo.list_entities_by_kb_id(self.KB_ID)}
        triples = {t.triple_id: t for t in await repo.list_triples_by_kb_id(self.KB_ID)}
        assert entities["e1"].description == "实体新描述"
        assert entities["e2"].description == "Beta 新描述"
        assert entities["e3"].description == "Gamma 描述"
        assert triples["t1"].description == "三元组新描述"
        assert triples["t2"].description == "关系二"

    async def test_list_pagination(self, repos):
        del repos
        await self._full_seed()
        repo = KnowledgeGraphRepository()

        page1 = await repo.list_entities_by_kb_id(self.KB_ID, limit=2, offset=0)
        page2 = await repo.list_entities_by_kb_id(self.KB_ID, limit=2, offset=2)
        assert [e.entity_id for e in page1] == ["e1", "e2"]
        assert [e.entity_id for e in page2] == ["e3"]

        triples_page = await repo.list_triples_by_kb_id(self.KB_ID, limit=1, offset=1)
        assert [t.triple_id for t in triples_page] == ["t2"]