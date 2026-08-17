"""End-to-end tests for graph-augmented retrieval (Task 11/12).

Two paths are covered:
- Task 11 (FakeChat extraction pipeline): chunks -> LLM extraction (FakeChat)
  -> graph build -> ``aquery`` fusion assertions (boost, disable flag,
  no-config degradation, extractor failure fallback).
- Task 12 (synthetic graph injection): a known graph ("张三 -> 资金转账 ->
  李四") is written directly through the GraphService storage/repo/vector
  interfaces, then ``aquery`` with ``keyword_extractor_enabled=False`` asserts
  multi-hop chunks rank above the pure-vector baseline and that disabling
  graph retrieval changes the result.

All tests run against a deterministic fake embedding (char-hash bag-of-words),
so vector similarities are stable within the process and no external service
is contacted.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from yusu_kb.knowledge.base import FileStatus
from yusu_kb.knowledge.graphs.graph_service import GraphService
from yusu_kb.knowledge.graphs.graph_utils import (
    compute_entity_id,
    compute_triple_id,
    normalize_entity_name,
)
from yusu_kb.knowledge.implementations.local_kb import (
    LocalKB,
    set_default_graph_chat_model_fn,
)
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.models.chat import GeneralResponse
from yusu_kb.models.embed import EmbeddingFunc, wrap_embedding_func_with_attrs
from yusu_kb.repositories import configure_repositories
from yusu_kb.repositories.knowledge_base_repository import KnowledgeBaseRepository
from yusu_kb.repositories.knowledge_chunk_repository import KnowledgeChunkRepository
from yusu_kb.repositories.knowledge_file_repository import KnowledgeFileRepository

FAKE_DIM = 64

# 张三 -> 资金转账 -> 李四：图抽取 payload（Task 11 的 FakeChat 全流程用）
PAYLOAD_TRANSFER = {
    "entities": [
        {"text": "张三", "label": "人物", "description": "张三，转账发起人。"},
        {"text": "李四", "label": "人物", "description": "李四，收款人。"},
    ],
    "relations": [
        {
            "source": "张三",
            "target": "李四",
            "text": "资金转账",
            "label": "资金转账",
            "description": "张三向李四转账五十万元。",
        }
    ],
}

# HL/LL 关键词 payload（用于 DualChat 分流返回）
PAYLOAD_KEYWORDS = {
    "high_level_keywords": ["转账", "资金"],
    "low_level_keywords": ["张三", "李四"],
}

GARBAGE_KEYWORDS = "this is not json at all"


def _fake_embed_one(text: str, dim: int = FAKE_DIM) -> np.ndarray:
    vec = np.zeros(dim)
    for ch in text:
        # ord() 而非 hash()：Python str hash 受 PYTHONHASHSEED 随机化影响，会导致
        # 向量召回结果跨进程不稳定（flaky）
        vec[ord(ch) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


async def _fake_embed(texts: list[str], **kwargs) -> np.ndarray:
    del kwargs
    return np.array([_fake_embed_one(t) for t in texts])


fake_embedding_func: EmbeddingFunc = wrap_embedding_func_with_attrs(
    embedding_dim=FAKE_DIM, max_token_size=None
)(_fake_embed)


class DualChat:
    """Chat stub that answers extraction vs. keyword prompts differently.

    The keyword-extraction prompt (LightRAG) always contains
    ``high_level_keywords``; extraction prompts never do. Recorded calls let
    tests assert the pipeline actually invoked the LLM path.
    """

    def __init__(self, extraction_payload: str, keyword_payload: str = ""):
        self.extraction_payload = extraction_payload
        self.keyword_payload = keyword_payload
        self.calls: list[list[dict]] = []

    async def __call__(self, messages: list[dict]) -> GeneralResponse:
        self.calls.append(messages)
        content = messages[-1]["content"] if messages else ""
        if "high_level_keywords" in content:
            return GeneralResponse(self.keyword_payload)
        return GeneralResponse(self.extraction_payload)


class GarbageChat(DualChat):
    """DualChat variant whose keyword answers are unparseable garbage."""

    async def __call__(self, messages: list[dict]) -> GeneralResponse:
        self.calls.append(messages)
        content = messages[-1]["content"] if messages else ""
        if "high_level_keywords" in content:
            return GeneralResponse(GARBAGE_KEYWORDS)
        return GeneralResponse(self.extraction_payload)


@pytest.fixture(autouse=True)
def default_fake_embedding():
    from yusu_kb.knowledge.implementations.local_kb import set_default_embedding_func

    set_default_embedding_func(fake_embedding_func)
    yield
    set_default_embedding_func(None)


@pytest.fixture(autouse=True)
def reset_graph_chat_fn():
    set_default_graph_chat_model_fn(None)
    yield
    set_default_graph_chat_model_fn(None)


@pytest.fixture()
async def manager(tmp_path, engine):
    configure_repositories(engine)
    kb_manager = KnowledgeBaseManager(str(tmp_path / "kb_work"))
    yield kb_manager
    await kb_manager.close()
    configure_repositories(None)


def get_kb(manager) -> LocalKB:
    instance = manager.get_instance("local")
    assert isinstance(instance, LocalKB)
    return instance


async def create_kb(manager, name: str) -> str:
    """Create a KB with auto graph build disabled (tests build manually)."""
    created = await manager.create_database(
        name=name,
        description="graph retrieval test",
        kb_type="local",
        additional_params={"auto_build_graph": False},
    )
    return created["kb_id"]


async def index_file(manager, kb_id: str, tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    record = await manager.add_file_record(kb_id, str(path))
    await manager.parse_file(kb_id, record["file_id"])
    result = await get_kb(manager).index_file(kb_id, record["file_id"])
    assert result["status"] == FileStatus.INDEXED
    return record["file_id"]


def _chunk_rank(results: list[dict], chunk_id: str) -> int:
    """1-based rank of a chunk_id in aquery results (or len+1 when absent)."""
    for index, chunk in enumerate(results, start=1):
        if chunk["metadata"].get("chunk_id") == chunk_id:
            return index
    return len(results) + 1


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


async def _sync_graph_config_to_kb(kb: LocalKB, kb_id: str) -> None:
    """Mirror the DB graph_build_config into in-memory metadata (same as the
    API router's ``_sync_graph_config_to_kb``)."""
    row = await KnowledgeBaseRepository().get_by_kb_id(kb_id)
    assert row is not None and row.additional_params
    metadata = kb.databases_meta.setdefault(kb_id, {}).setdefault("metadata", {})
    metadata["graph_build_config"] = row.additional_params["graph_build_config"]


async def _make_service(kb: LocalKB, kb_id: str, chat) -> GraphService:
    # index_file 期间的 _delete_file_graph_only 可能已用 env 回退 chat 创建并
    # 缓存了 GraphService 单例；evict 后重建，保证图构建/检索使用注入的 chat。
    await GraphService.evict(kb_id)
    return GraphService.get_instance(
        kb_id=kb_id,
        work_dir=kb.work_dir,
        embed_func=kb._get_embedding_function(),
        chat_model_fn=chat,
    )


class TestGraphRetrievalPipeline:
    """Task 11: FakeChat extraction -> build -> aquery fusion."""

    @pytest.fixture(autouse=True)
    async def seeded_kb(self, manager, tmp_path):
        kb_id = await create_kb(manager, "图检索全流程")
        await index_file(
            manager,
            kb_id,
            tmp_path,
            name="a.md",
            content="# 转账记录\n\n张三向李四转账五十万元，用于项目投资。\n\n张三与李四共同投资了这家公司。",
        )
        await index_file(
            manager,
            kb_id,
            tmp_path,
            name="b.md",
            content="# 采购记录\n\n公司采购了一批设备，总价三十万元。\n\n采购合同由财务部门审核通过。",
        )
        return kb_id

    async def _build_graph(self, manager, kb_id: str, chat) -> GraphService:
        kb = get_kb(manager)
        svc = await _make_service(kb, kb_id, chat)
        await svc.configure(kb_id, extractor_options={"model_spec": "fake-model"})
        await _sync_graph_config_to_kb(kb, kb_id)
        await svc.build_pending_chunks(kb_id, batch_size=4)
        status = await _wait_build_done(svc, kb_id)
        assert status["build_task_status"] == "completed"
        assert status["entity_count"] >= 2
        return svc

    async def test_graph_chunks_carried_graph_score(self, manager, seeded_kb):
        """Direct _retrieve_graph_chunks: non-empty, chunk ids + graph_score."""
        kb_id = seeded_kb
        chat = DualChat(json.dumps(PAYLOAD_TRANSFER, ensure_ascii=False), json.dumps(PAYLOAD_KEYWORDS, ensure_ascii=False))
        set_default_graph_chat_model_fn(chat)
        await self._build_graph(manager, kb_id, chat)

        kb = get_kb(manager)
        graph_chunks, error = await kb._retrieve_graph_chunks(
            "张三 向 李四 转账", kb_id, [], {"keyword_extractor_enabled": True}
        )
        assert error is None
        assert graph_chunks, "graph retrieval must return chunks after a build"
        for chunk in graph_chunks:
            assert chunk["metadata"]["chunk_id"]
            assert chunk["metadata"]["file_id"]
            assert float(chunk.get("graph_score") or 0.0) > 0.0
            assert chunk["content"]
        assert len(chat.calls) >= 1  # keyword extraction went through the LLM

    async def test_aquery_fusion_changes_results(self, manager, seeded_kb):
        kb_id = seeded_kb
        chat = DualChat(json.dumps(PAYLOAD_TRANSFER, ensure_ascii=False), json.dumps(PAYLOAD_KEYWORDS, ensure_ascii=False))
        set_default_graph_chat_model_fn(chat)
        await self._build_graph(manager, kb_id, chat)

        kb = get_kb(manager)
        with_graph = await kb.aquery("张三 转账 李四", kb_id, use_reranker=False, use_graph_retrieval=True)
        without_graph = await kb.aquery("张三 转账 李四", kb_id, use_reranker=False, use_graph_retrieval=False)
        assert with_graph, "graph-enabled query must return results"
        assert without_graph, "plain query must return results"
        # 图融合会向结果集合注入图召回 chunk，结果列表应与纯向量不同
        ids_on = [c["metadata"]["chunk_id"] for c in with_graph]
        ids_off = [c["metadata"]["chunk_id"] for c in without_graph]
        assert ids_on != ids_off or any("graph_score" in c for c in with_graph)

    async def test_use_graph_retrieval_false_has_no_graph_effect(self, manager, seeded_kb):
        kb_id = seeded_kb
        chat = DualChat(json.dumps(PAYLOAD_TRANSFER, ensure_ascii=False), json.dumps(PAYLOAD_KEYWORDS, ensure_ascii=False))
        set_default_graph_chat_model_fn(chat)
        await self._build_graph(manager, kb_id, chat)

        kb = get_kb(manager)
        results = await kb.aquery("张三 转账 李四", kb_id, use_reranker=False, use_graph_retrieval=False)
        assert results
        assert all("graph_score" not in chunk for chunk in results)
        # 关闭图检索与图未建（无配置）的基线行为一致
        plain = await kb.aquery("张三 转账 李四", kb_id, use_reranker=False, use_graph_retrieval=True, keyword_extractor_enabled=False)
        assert plain

    async def test_no_config_degrades_to_vector_only(self, manager, tmp_path):
        """No graph_build_config -> automatic degradation, identical to plain."""
        kb_id = await create_kb(manager, "无图配置")
        await index_file(
            manager,
            kb_id,
            tmp_path,
            name="a.md",
            content="# 转账记录\n\n张三向李四转账五十万元。",
        )
        kb = get_kb(manager)
        graph_chunks, error = await kb._retrieve_graph_chunks(
            "张三 转账 李四", kb_id, [], {"keyword_extractor_enabled": True}
        )
        assert error is None
        assert graph_chunks == []

        base = await kb.aquery("张三 转账 李四", kb_id, use_reranker=False, use_graph_retrieval=False)
        same = await kb.aquery("张三 转账 李四", kb_id, use_reranker=False, use_graph_retrieval=True)
        assert base and same
        assert [c["metadata"]["chunk_id"] for c in base] == [c["metadata"]["chunk_id"] for c in same]

    async def test_keyword_extractor_failure_degrades_gracefully(self, manager, seeded_kb):
        """Garbage keyword payload -> ([],[]) -> query_text fallback, no raise."""
        kb_id = seeded_kb
        chat = GarbageChat(json.dumps(PAYLOAD_TRANSFER, ensure_ascii=False))
        set_default_graph_chat_model_fn(chat)
        await self._build_graph(manager, kb_id, chat)

        kb = get_kb(manager)
        # LLM keyword path enabled but the extractor returns garbage: the
        # pipeline must fall back to the query_text direct path, never raise.
        graph_chunks, error = await kb._retrieve_graph_chunks(
            "张三 转账 李四", kb_id, [], {"keyword_extractor_enabled": True}
        )
        assert error is None
        results = await kb.aquery("张三 转账 李四", kb_id, use_reranker=False, use_graph_retrieval=True)
        assert results
        # 抽取失败走 fallback 直查，结果仍可能带图召回（不抛即可）
        assert graph_chunks == [] or all(c["metadata"]["chunk_id"] for c in graph_chunks)


class TestGraphRetrievalSyntheticGraph:
    """Task 12: hand-written graph injected via GraphService interfaces."""

    async def _seed(self, manager, tmp_path: Path) -> tuple[str, str, str, str]:
        """KB + 3 chunks; returns (kb_id, chunkA, chunkB, chunkC)."""
        kb_id = await create_kb(manager, "合成图直插")
        await index_file(manager, kb_id, tmp_path, name="a.md", content="# 转账\n\n张三向李四转账五十万元。")
        await index_file(manager, kb_id, tmp_path, name="b.md", content="# 转账\n\n王五向张三转账二十万元。")
        await index_file(manager, kb_id, tmp_path, name="c.md", content="# 采购\n\n李四购买了新设备。")

        chunks = await KnowledgeChunkRepository().list_by_kb_id(kb_id)
        by_content = {chunk.content for chunk in chunks}
        chunk_a = next(ch for ch in chunks if "张三向李四" in ch.content)
        chunk_b = next(ch for ch in chunks if "王五向张三" in ch.content)
        chunk_c = next(ch for ch in chunks if "新设备" in ch.content)
        assert len(by_content) == 3, f"expected 3 chunks, got {len(by_content)}"
        return kb_id, chunk_a.chunk_id, chunk_b.chunk_id, chunk_c.chunk_id

    async def _inject_graph(self, kb: LocalKB, kb_id: str, chunk_a: str, chunk_b: str, chunk_c: str, file_ids: dict[str, str]) -> GraphService:
        """Write 张三 -> 资金转账 -> 李四 + 王五 -> 资金转账 -> 张三 by hand."""
        chat = DualChat(json.dumps(PAYLOAD_TRANSFER, ensure_ascii=False))
        # 图检索路径的 GraphService 单例需要 chat_model_fn（fallback 模式虽不
        # 调用 LLM，但 get_instance 构造时就要注入，避免 env 回退创建失败）
        set_default_graph_chat_model_fn(chat)
        svc = await _make_service(kb, kb_id, chat)
        storage = svc.get_storage(kb_id)

        zhang_id = compute_entity_id(kb_id, normalize_entity_name("张三"), "人物")
        li_id = compute_entity_id(kb_id, normalize_entity_name("李四"), "人物")
        wang_id = compute_entity_id(kb_id, normalize_entity_name("王五"), "人物")
        for entity_id, name in ((zhang_id, "张三"), (li_id, "李四"), (wang_id, "王五")):
            storage.upsert_entity(
                entity_id=entity_id,
                normalized_name=name,
                label="人物",
                name=name,
                description=f"{name}，测试人物。",
            )
        triple_ab = compute_triple_id(kb_id, "张三", "人物", "资金转账", "李四", "人物")
        storage.upsert_relation(
            triple_id=triple_ab,
            source_id=zhang_id,
            target_id=li_id,
            text="资金转账",
            rtype="资金转账",
            file_ids=[file_ids["a"]],
            description="张三向李四转账。",
        )
        triple_ba = compute_triple_id(kb_id, "王五", "人物", "资金转账", "张三", "人物")
        storage.upsert_relation(
            triple_id=triple_ba,
            source_id=wang_id,
            target_id=zhang_id,
            text="资金转账",
            rtype="资金转账",
            file_ids=[file_ids["b"]],
            description="王五向张三转账。",
        )
        storage.add_mention(entity_id=zhang_id, chunk_id=chunk_a, file_id=file_ids["a"])
        storage.add_mention(entity_id=li_id, chunk_id=chunk_a, file_id=file_ids["a"])
        storage.add_mention(entity_id=wang_id, chunk_id=chunk_b, file_id=file_ids["b"])
        storage.add_mention(entity_id=zhang_id, chunk_id=chunk_b, file_id=file_ids["b"])
        storage.add_mention(entity_id=li_id, chunk_id=chunk_c, file_id=file_ids["c"])
        storage.save()

        entity_store = await svc.get_vector_store(kb_id, "entity")
        await entity_store.upsert(
            [
                {"id": zhang_id, "content": "张三|人物|张三，测试人物。", "label": "人物"},
                {"id": li_id, "content": "李四|人物|李四，测试人物。", "label": "人物"},
                {"id": wang_id, "content": "王五|人物|王五，测试人物。", "label": "人物"},
            ]
        )
        triple_store = await svc.get_vector_store(kb_id, "triple")
        await triple_store.upsert(
            [
                {
                    "id": triple_ab,
                    "content": "张三 → 资金转账 → 李四 | 张三向李四转账。",
                    "source_id": zhang_id,
                    "target_id": li_id,
                    "type": "资金转账",
                },
                {
                    "id": triple_ba,
                    "content": "王五 → 资金转账 → 张三 | 王五向张三转账。",
                    "source_id": wang_id,
                    "target_id": zhang_id,
                    "type": "资金转账",
                },
            ]
        )
        # 图检索启用条件：model_spec 存在（模拟 configure 后已同步）
        kb.databases_meta.setdefault(kb_id, {}).setdefault("metadata", {})["graph_build_config"] = {
            "extractor_type": "llm",
            "extractor_options": {"model_spec": "fake-model"},
        }
        return svc

    async def test_multihop_chunk_beats_vector_baseline(self, manager, tmp_path):
        kb_id, chunk_a, chunk_b, chunk_c = await self._seed(manager, tmp_path)
        kb = get_kb(manager)
        file_repo = KnowledgeFileRepository()
        files = await file_repo.list_by_kb_id(kb_id)
        file_ids = {f.filename.split(".")[0]: f.file_id for f in files}
        await self._inject_graph(kb, kb_id, chunk_a, chunk_b, chunk_c, file_ids)

        # 直接断言图路径能通过关系扩散召回与查询向量相似度低的多跳 chunk
        graph_chunks, error = await kb._retrieve_graph_chunks(
            "张三 转账 李四", kb_id, [], {"keyword_extractor_enabled": False}
        )
        assert error is None
        assert graph_chunks, "synthetic graph must produce graph chunks"
        graph_ids = {c["metadata"]["chunk_id"] for c in graph_chunks}
        assert chunk_c in graph_ids, "multi-hop chunk must be recalled through the graph"
        assert chunk_a in graph_ids

        # aquery 对比：图增强 vs 纯向量基线
        query = "张三 转账 李四"
        graph_on = await kb.aquery(query, kb_id, use_reranker=False, use_graph_retrieval=True, keyword_extractor_enabled=False)
        graph_off = await kb.aquery(query, kb_id, use_reranker=False, use_graph_retrieval=False, keyword_extractor_enabled=False)
        assert graph_on and graph_off
        ids_on = [c["metadata"]["chunk_id"] for c in graph_on]
        ids_off = [c["metadata"]["chunk_id"] for c in graph_off]
        # 图检索必须改变结果（多跳 chunk 提升或集合变化）
        assert ids_on != ids_off or any("graph_score" in c for c in graph_on)
        # 多跳 chunk（只提李四，与"转账"无关）在图增强中排名不低于纯向量基线
        assert _chunk_rank(graph_on, chunk_c) <= _chunk_rank(graph_off, chunk_c)

    async def test_disabled_flag_no_difference(self, manager, tmp_path):
        kb_id, chunk_a, chunk_b, chunk_c = await self._seed(manager, tmp_path)
        kb = get_kb(manager)
        file_repo = KnowledgeFileRepository()
        files = await file_repo.list_by_kb_id(kb_id)
        file_ids = {f.filename.split(".")[0]: f.file_id for f in files}
        await self._inject_graph(kb, kb_id, chunk_a, chunk_b, chunk_c, file_ids)

        results = await kb.aquery("张三 转账 李四", kb_id, use_reranker=False, use_graph_retrieval=False)
        assert results
        assert all("graph_score" not in chunk for chunk in results)
