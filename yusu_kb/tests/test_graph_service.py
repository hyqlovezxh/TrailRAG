"""Tests for GraphService (build orchestration, status, cascading delete).

Uses a real SQLite database (conftest ``engine`` fixture) plus injected
FakeChat / fake-embedding stubs, so the whole supply → worker → flusher →
post-process pipeline runs against real repository rows and real JSON files
under ``tmp_workdir``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import func, select

from yusu_kb.knowledge.graphs.description_merger import DescriptionMerger
from yusu_kb.knowledge.graphs.graph_service import GraphService
from yusu_kb.knowledge.graphs.graph_utils import (
    DESC_SEPARATOR,
    compute_entity_id,
    compute_triple_id,
    normalize_entity_name,
)
from yusu_kb.models.chat import GeneralResponse
from yusu_kb.models.embed import EmbeddingFunc, wrap_embedding_func_with_attrs
from yusu_kb.repositories import configure_repositories
from yusu_kb.repositories.knowledge_base_repository import KnowledgeBaseRepository
from yusu_kb.repositories.knowledge_chunk_repository import KnowledgeChunkRepository
from yusu_kb.repositories.knowledge_file_repository import KnowledgeFileRepository
from yusu_kb.repositories.knowledge_graph_repository import KnowledgeGraphRepository
from yusu_kb.storage.sqlite.models_knowledge import KnowledgeGraphEntityMention

KB_ID = "kb_svc"
FILE_A = "fA"
FILE_B = "fB"

PAYLOAD_A = {
    "entities": [
        {"text": "甲公司", "label": "机构", "description": "甲公司，主营支付系统开发。"},
        {"text": "王五", "label": "人物", "description": "王五，甲公司员工。"},
    ],
    "relations": [
        {"source": "王五", "target": "甲公司", "text": "任职于", "label": "任职于", "description": "王五任职于甲公司。"}
    ],
}

PAYLOAD_B = {
    "entities": [
        {"text": "乙公司", "label": "机构", "description": "乙公司，主营风控系统开发。"},
        {"text": "王五", "label": "人物", "description": "王五，乙公司顾问。"},
    ],
    "relations": [
        {"source": "王五", "target": "乙公司", "text": "任职于", "label": "任职于", "description": "王五任职于乙公司。"}
    ],
}


class FakeChat:
    """Injected chat_model_fn stand-in: async (messages) -> GeneralResponse.

    Serves payloads in order (last one repeats); records every call.
    """

    def __init__(self, payloads: list[str]):
        self.payloads = payloads
        self.calls: list[list[dict]] = []

    async def __call__(self, messages: list[dict]) -> GeneralResponse:
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self.payloads) - 1)
        return GeneralResponse(self.payloads[idx])


class SlowChat(FakeChat):
    """FakeChat with an artificial delay so builds stay observable."""

    async def __call__(self, messages: list[dict]) -> GeneralResponse:
        await asyncio.sleep(0.25)
        return await super().__call__(messages)


class IntermittentChat:
    """Chat stub raising for chunks whose content matches ``fail_on`` markers.

    Lets a test fail extraction for a subset of chunks (e.g. one file) while
    the rest succeed, to exercise per-worker failure isolation.
    """

    def __init__(self, payload: str, fail_on: set[str]):
        self.payload = payload
        self.fail_on = fail_on
        self.calls: list[list[dict]] = []

    async def __call__(self, messages: list[dict]) -> GeneralResponse:
        self.calls.append(messages)
        content = messages[-1]["content"] if messages else ""
        if any(marker in content for marker in self.fail_on):
            raise RuntimeError("simulated extraction failure")
        return GeneralResponse(self.payload)


class ContentFakeChat:
    """Chat stub picking a payload by the marker text inside the chunk."""

    def __init__(self, payload_a: dict, payload_b: dict):
        self.payload_a = payload_a
        self.payload_b = payload_b
        self.calls: list[list[dict]] = []

    async def __call__(self, messages: list[dict]) -> GeneralResponse:
        self.calls.append(messages)
        content = messages[-1]["content"] if messages else ""
        payload = self.payload_a if "甲公司" in content else self.payload_b
        return GeneralResponse(json.dumps(payload, ensure_ascii=False))


FAKE_DIM = 8


def _fake_embed_one(text: str, dim: int = FAKE_DIM) -> np.ndarray:
    vec = np.zeros(dim)
    for ch in text:
        vec[ord(ch) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class _FakeEmbed:
    def __init__(self, dim: int | None = FAKE_DIM):
        self.dim = dim
        self.calls: list[int] = []

    async def __call__(self, texts, **kwargs) -> np.ndarray:
        del kwargs
        self.calls.append(len(texts))
        dim = self.dim or FAKE_DIM
        return np.array([_fake_embed_one(t, dim) for t in texts])


def _embedding_func(embed: _FakeEmbed) -> EmbeddingFunc:
    return wrap_embedding_func_with_attrs(embedding_dim=embed.dim)(embed)


@pytest.fixture()
def repos(engine):
    configure_repositories(engine)
    yield
    configure_repositories(None)


def _entity_id(kb_id: str, text: str, label: str) -> str:
    return compute_entity_id(kb_id, normalize_entity_name(text), label)


def _triple_id(kb_id: str, source: str, source_label: str, rtype: str, target: str, target_label: str) -> str:
    return compute_triple_id(
        kb_id,
        normalize_entity_name(source),
        source_label,
        rtype,
        normalize_entity_name(target),
        target_label,
    )


async def _seed_kb() -> None:
    """KB + two files + six chunks (three per file, distinct markers)."""
    await KnowledgeBaseRepository().create({"kb_id": KB_ID, "name": "Svc", "kb_type": "local"})
    file_repo = KnowledgeFileRepository()
    for file_id in (FILE_A, FILE_B):
        await file_repo.upsert(file_id, {"kb_id": KB_ID, "filename": f"{file_id}.md", "status": "parsed"})
    await KnowledgeChunkRepository().batch_upsert(
        [
            {"chunk_id": "cA1", "file_id": FILE_A, "kb_id": KB_ID, "chunk_index": 0, "content": "甲公司开发支付系统，王五负责研发。"},
            {"chunk_id": "cA2", "file_id": FILE_A, "kb_id": KB_ID, "chunk_index": 1, "content": "甲公司拓展海外市场，王五主导产品设计。"},
            {"chunk_id": "cA3", "file_id": FILE_A, "kb_id": KB_ID, "chunk_index": 2, "content": "甲公司总部位于深圳，王五常驻杭州。"},
            {"chunk_id": "cB1", "file_id": FILE_B, "kb_id": KB_ID, "chunk_index": 0, "content": "乙公司开发风控系统，王五提供咨询。"},
            {"chunk_id": "cB2", "file_id": FILE_B, "kb_id": KB_ID, "chunk_index": 1, "content": "乙公司服务多家银行，王五负责对接。"},
            {"chunk_id": "cB3", "file_id": FILE_B, "kb_id": KB_ID, "chunk_index": 2, "content": "乙公司位于上海，王五参与算法评审。"},
        ]
    )


def _make_service(tmp_workdir: Path, chat=None, embed=None, **kwargs) -> GraphService:
    return GraphService(
        kb_id=KB_ID,
        work_dir=tmp_workdir,
        chunk_repo=KnowledgeChunkRepository(),
        graph_repo=KnowledgeGraphRepository(),
        kb_repo=KnowledgeBaseRepository(),
        chat_model_fn=chat,
        embed_func=_embedding_func(embed) if embed is not None else None,
        **kwargs,
    )


async def _wait_build_done(svc: GraphService, kb_id: str, timeout: float = 15.0) -> dict:
    """Poll get_status until the background build reaches a terminal state."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        status = await svc.get_status(kb_id)
        if status["build_task_status"] in ("completed", "failed", "cancelled"):
            return status
        if loop.time() > deadline:
            raise AssertionError(f"build did not finish within {timeout}s: {status}")
        await asyncio.sleep(0.02)


class TestGraphService:
    async def test_get_status_before_configure(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        svc = _make_service(tmp_workdir, FakeChat(['{"entities": [], "relations": []}']), _FakeEmbed())
        status = await svc.get_status(KB_ID)
        assert status["configured"] is False
        assert status["locked"] is False
        assert status["config"] is None
        assert status["total_chunks"] == 6
        assert status["pending_chunks"] == 6
        assert status["indexed_chunks"] == 0
        assert status["entity_count"] == 0
        assert status["build_task_status"] is None
        assert status["build_task_progress"] == 0.0
        with pytest.raises(ValueError, match="不存在"):
            await svc.get_status("kb_missing")

    async def test_configure_validation_and_persistence(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        svc = _make_service(tmp_workdir, FakeChat(['{"entities": [], "relations": []}']), _FakeEmbed())
        with pytest.raises(ValueError, match="model_spec"):
            await svc.configure(KB_ID, extractor_type="llm", extractor_options={})
        with pytest.raises(ValueError, match="gleaning_count"):
            await svc.configure(KB_ID, extractor_type="llm", extractor_options={"model_spec": "m", "gleaning_count": 5})
        with pytest.raises(ValueError, match="未知的图谱抽取器类型"):
            await svc.configure(KB_ID, extractor_type="keyword", extractor_options={"model_spec": "m"})

        result = await svc.configure(KB_ID, extractor_type="llm", extractor_options={"model_spec": "fake-model"})
        assert result["locked"] is True
        assert result["extractor_type"] == "llm"
        assert result["created_by"] == "system"
        assert result["created_at"]

        status = await svc.get_status(KB_ID)
        assert status["configured"] is True
        assert status["locked"] is True
        assert status["config"]["extractor_type"] == "llm"
        assert status["config"]["extractor_options"]["model_spec"] == "fake-model"

    async def test_build_pipeline_end_to_end(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        chat = FakeChat([json.dumps(PAYLOAD_A, ensure_ascii=False)])
        svc = _make_service(tmp_workdir, chat, _FakeEmbed())
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model"})

        start = await svc.build_pending_chunks(KB_ID, batch_size=4)
        assert start["message"] == "图谱构建已在后台启动"
        assert start["remaining"] == 6

        status = await _wait_build_done(svc, KB_ID)
        assert status["build_task_status"] == "completed"
        assert status["build_task_progress"] == 100.0
        assert status["entity_count"] == 2
        assert status["relation_count"] == 1
        assert status["indexed_chunks"] == 6
        assert status["pending_chunks"] == 0

        storage = svc.get_storage(KB_ID)
        stats = storage.get_stats()
        assert stats["chunks"] == 6
        assert stats["entities"] == 2
        assert stats["relations"] == 1
        assert stats["mentions"] == 12

        vstore = await svc.get_vector_store(KB_ID, "entity")
        hits = await vstore.search("王五", top_k=5)
        assert hits
        assert hits[0]["id"] == _entity_id(KB_ID, "王五", "人物")
        tstore = await svc.get_vector_store(KB_ID, "triple")
        thits = await tstore.search("王五 甲公司", top_k=5)
        assert thits and "任职于" in thits[0]["content"]

        chunks = await KnowledgeChunkRepository().list_by_kb_id(KB_ID)
        assert all(c.graph_indexed for c in chunks)
        assert all(c.extraction_result for c in chunks)
        assert len(chat.calls) == 6

    async def test_build_reuses_cached_extraction(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        chat = FakeChat([json.dumps(PAYLOAD_A, ensure_ascii=False)])
        svc = _make_service(tmp_workdir, chat, _FakeEmbed())
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model"})
        await svc.build_pending_chunks(KB_ID, batch_size=4)
        status = await _wait_build_done(svc, KB_ID)
        assert status["build_task_status"] == "completed"
        calls_after_first = len(chat.calls)
        assert calls_after_first == 6

        # graph state reset, extraction cache kept → rebuild without LLM calls
        await KnowledgeChunkRepository().reset_graph_state_by_kb_id(KB_ID, clear_extraction_result=False)
        await svc.build_pending_chunks(KB_ID, batch_size=4)
        status = await _wait_build_done(svc, KB_ID)
        assert status["build_task_status"] == "completed"
        assert status["indexed_chunks"] == 6
        assert status["entity_count"] == 2
        assert len(chat.calls) == calls_after_first

    async def test_configure_change_clears_extraction_cache(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        chat = FakeChat([json.dumps(PAYLOAD_A, ensure_ascii=False)])
        svc = _make_service(tmp_workdir, chat, _FakeEmbed())
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model"})
        await svc.build_pending_chunks(KB_ID, batch_size=4)
        status = await _wait_build_done(svc, KB_ID)
        assert status["build_task_status"] == "completed"

        # same options again → no clear
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model"})
        chunks = await KnowledgeChunkRepository().list_by_kb_id(KB_ID)
        assert all(c.extraction_result for c in chunks)
        assert all(c.graph_indexed for c in chunks)

        # model_spec change → extraction cache cleared + graph_indexed reset
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model-v2"})
        chunks = await KnowledgeChunkRepository().list_by_kb_id(KB_ID)
        assert all(c.extraction_result is None for c in chunks)
        assert all(not c.graph_indexed for c in chunks)
        status = await svc.get_status(KB_ID)
        assert status["pending_chunks"] == 6
        assert status["config"]["extractor_options"]["model_spec"] == "fake-model-v2"

    async def test_build_dedupes_concurrent_runs(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        svc = _make_service(tmp_workdir, SlowChat([json.dumps(PAYLOAD_A, ensure_ascii=False)]), _FakeEmbed())
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model"})
        await svc.build_pending_chunks(KB_ID, batch_size=4)
        second = await svc.build_pending_chunks(KB_ID, batch_size=4)
        assert second["message"] == "图谱构建已在后台运行，请稍后查询状态"
        status = await _wait_build_done(svc, KB_ID)
        assert status["build_task_status"] == "completed"
        assert status["indexed_chunks"] == 6

    async def test_worker_exception_isolation_and_retry(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        # fB chunks (marker 乙公司) fail extraction on the first build; the
        # failures must not kill the workers, and the failed chunks must be
        # retried by the next build once the failure stops.
        chat = IntermittentChat(json.dumps(PAYLOAD_A, ensure_ascii=False), fail_on={"乙公司"})
        svc = _make_service(tmp_workdir, chat, _FakeEmbed())
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model"})
        await svc.build_pending_chunks(KB_ID, batch_size=2)
        status = await _wait_build_done(svc, KB_ID)
        assert status["build_task_status"] == "failed"
        assert status["indexed_chunks"] == 3
        assert status["pending_chunks"] == 3
        chunks = await KnowledgeChunkRepository().list_by_kb_id(KB_ID)
        assert all(c.graph_indexed for c in chunks if c.file_id == FILE_A)
        assert all(not c.graph_indexed for c in chunks if c.file_id == FILE_B)
        stats = svc.get_storage(KB_ID).get_stats()
        assert stats["chunks"] == 3
        assert stats["mentions"] == 6

        # failure stops → the next build retries only the failed chunks
        chat.fail_on = set()
        await svc.build_pending_chunks(KB_ID, batch_size=2)
        status = await _wait_build_done(svc, KB_ID)
        assert status["build_task_status"] == "completed"
        assert status["indexed_chunks"] == 6
        assert status["pending_chunks"] == 0
        chunks = await KnowledgeChunkRepository().list_by_kb_id(KB_ID)
        assert all(c.graph_indexed for c in chunks)
        stats = svc.get_storage(KB_ID).get_stats()
        assert stats["chunks"] == 6
        assert stats["mentions"] == 12

    async def test_delete_file_graph_cascades(self, repos, tmp_workdir, session_factory):
        del repos
        await _seed_kb()
        svc = _make_service(tmp_workdir, ContentFakeChat(PAYLOAD_A, PAYLOAD_B), _FakeEmbed())
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model"})
        await svc.build_pending_chunks(KB_ID, batch_size=4)
        status = await _wait_build_done(svc, KB_ID)
        assert status["build_task_status"] == "completed"
        assert status["entity_count"] == 3
        assert status["relation_count"] == 2

        jia_id = _entity_id(KB_ID, "甲公司", "机构")
        wang_id = _entity_id(KB_ID, "王五", "人物")
        jia_wang_triple = _triple_id(KB_ID, "王五", "人物", "任职于", "甲公司", "机构")

        await svc.delete_file_graph(KB_ID, FILE_A)

        # fA mentions gone; shared entity keeps its fB mentions
        async with session_factory() as session:
            fA_mentions = await session.execute(
                select(func.count()).select_from(KnowledgeGraphEntityMention).where(
                    KnowledgeGraphEntityMention.file_id == FILE_A
                )
            )
            assert int(fA_mentions.scalar_one()) == 0
            wang_mentions = await session.execute(
                select(func.count()).select_from(KnowledgeGraphEntityMention).where(
                    KnowledgeGraphEntityMention.entity_id == wang_id
                )
            )
            assert int(wang_mentions.scalar_one()) == 3

        # fA-only entity/triple removed from the vectors; shared ones kept
        vstore = await svc.get_vector_store(KB_ID, "entity")
        hits = await vstore.search("甲公司", top_k=5)
        assert jia_id not in {h["id"] for h in hits}
        hits = await vstore.search("王五", top_k=5)
        assert wang_id in {h["id"] for h in hits}
        tstore = await svc.get_vector_store(KB_ID, "triple")
        thits = await tstore.search("王五 甲公司", top_k=5)
        assert jia_wang_triple not in {h["id"] for h in thits}

        # storage: fA chunks gone, mentions halved, shared relation kept
        stats = svc.get_storage(KB_ID).get_stats()
        assert stats["chunks"] == 3
        assert stats["relations"] == 1
        assert stats["mentions"] == 6
        assert stats["entities"] == 3

        # repository rows are left untouched (caller decides row purge)
        status = await svc.get_status(KB_ID)
        assert status["entity_count"] == 3
        assert status["relation_count"] == 2

    async def test_reset_clears_everything(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        svc = _make_service(tmp_workdir, FakeChat([json.dumps(PAYLOAD_A, ensure_ascii=False)]), _FakeEmbed())
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model"})
        await svc.build_pending_chunks(KB_ID, batch_size=4)
        status = await _wait_build_done(svc, KB_ID)
        assert status["build_task_status"] == "completed"

        result = await svc.reset(KB_ID, clear_extraction_result=True, clear_config=True)
        assert result["reset_chunks"] == 6
        status = await svc.get_status(KB_ID)
        assert status["configured"] is False
        assert status["entity_count"] == 0
        assert status["relation_count"] == 0
        assert status["pending_chunks"] == 6
        assert status["build_task_status"] is None
        assert not (tmp_workdir / KB_ID / "graph_storage.json").exists()
        assert not (tmp_workdir / KB_ID / "graph_vdb_entity.json").exists()
        assert not (tmp_workdir / KB_ID / "graph_vdb_triple.json").exists()
        chunks = await KnowledgeChunkRepository().list_by_kb_id(KB_ID)
        assert all(c.extraction_result is None for c in chunks)

        # reset without clearing keeps config + extraction cache
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model"})
        await svc.build_pending_chunks(KB_ID, batch_size=4)
        await _wait_build_done(svc, KB_ID)
        result = await svc.reset(KB_ID, clear_extraction_result=False, clear_config=False)
        assert result["reset_chunks"] == 6
        status = await svc.get_status(KB_ID)
        assert status["configured"] is True
        chunks = await KnowledgeChunkRepository().list_by_kb_id(KB_ID)
        assert all(not c.graph_indexed for c in chunks)
        assert all(c.extraction_result for c in chunks)

    async def test_reset_cancels_running_build(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        svc = _make_service(tmp_workdir, SlowChat([json.dumps(PAYLOAD_A, ensure_ascii=False)]), _FakeEmbed())
        await svc.configure(KB_ID, extractor_options={"model_spec": "fake-model"})
        await svc.build_pending_chunks(KB_ID, batch_size=4)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10.0
        while loop.time() < deadline:
            status = await svc.get_status(KB_ID)
            if status["build_task_status"] == "running" and status["build_task_progress"] < 80.0:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("build did not reach an in-progress state")

        result = await svc.reset(KB_ID, clear_extraction_result=True, clear_config=True)
        assert result["reset_chunks"] == 6
        status = await svc.get_status(KB_ID)
        assert status["build_task_status"] is None
        assert status["pending_chunks"] == 6
        assert status["indexed_chunks"] == 0
        assert not (tmp_workdir / KB_ID / "graph_storage.json").exists()
        assert not (tmp_workdir / KB_ID / "graph_vdb_entity.json").exists()

    async def test_get_instance_singleton_and_evict(self, tmp_workdir):
        first = GraphService.get_instance(kb_id="kb_inst", work_dir=tmp_workdir)
        second = GraphService.get_instance(kb_id="kb_inst", work_dir=tmp_workdir)
        assert first is second
        await GraphService.evict("kb_inst")
        third = GraphService.get_instance(kb_id="kb_inst", work_dir=tmp_workdir)
        assert third is not first
        await GraphService.evict("kb_inst")

    async def test_cleanup_orphan_entities(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        svc = _make_service(tmp_workdir, FakeChat(['{"entities": [], "relations": []}']), _FakeEmbed())
        storage = svc.get_storage(KB_ID)
        storage.upsert_entity(entity_id="e_orphan", normalized_name="孤儿实体", label="Entity", name="孤儿实体")
        storage.upsert_entity(entity_id="e_kept", normalized_name="王五", label="人物", name="王五")
        storage.add_mention(entity_id="e_kept", chunk_id="cA1", file_id=FILE_A)
        vstore = await svc.get_vector_store(KB_ID, "entity")
        await vstore.upsert(
            [
                {"id": "e_orphan", "content": "孤儿实体", "label": "Entity"},
                {"id": "e_kept", "content": "王五", "label": "人物"},
            ]
        )

        removed = await svc._cleanup_orphan_entities(KB_ID, storage)
        assert removed == 1
        hits = await vstore.search("孤儿实体", top_k=5)
        assert "e_orphan" not in {h["id"] for h in hits}
        hits = await vstore.search("王五", top_k=5)
        assert "e_kept" in {h["id"] for h in hits}

    async def test_merge_cross_chunk_descriptions(self, repos, tmp_workdir):
        del repos
        await _seed_kb()
        summary = "王五，综合了所有片段的人物概述，覆盖全部关键信息。"
        svc = _make_service(tmp_workdir, FakeChat([summary]), _FakeEmbed())
        storage = svc.get_storage(KB_ID)
        fragments = [f"王五片段{i}，说明人物背景。" for i in range(8)]
        storage.upsert_entity(
            entity_id="e_wang",
            normalized_name="王五",
            label="人物",
            name="王五",
            description=DESC_SEPARATOR.join(fragments),
        )

        merged = await svc._merge_cross_chunk_descriptions(
            KB_ID, storage, DescriptionMerger(model_spec="fake-model", chat_model_fn=svc.chat_model_fn)
        )
        assert merged == 1
        assert storage.get_entity_node("e_wang")["description"] == summary
        assert len(svc.chat_model_fn.calls) == 1

        vstore = await svc.get_vector_store(KB_ID, "entity")
        hits = await vstore.search("王五", top_k=5)
        assert hits and hits[0]["id"] == "e_wang"
        assert summary in hits[0]["content"]
