"""Integration tests for the implicit-graph fallback wiring (plan Phase 2).

Regression guards required by the plan:
- the switch is off by default and adds zero behaviour change (the untouched
  ``test_no_config_degrades_to_vector_only`` in test_graph_retrieval.py stays
  green; here we additionally assert the snapshot / implicit entry points are
  never touched when disabled);
- coverage == 1 keeps the explicit graph path and produces chunk-id-identical
  results with the switch on vs off;
- a configured-but-unbuilt graph routes through the implicit channel and fuses
  with ``implicit_weight`` (0.4, weaker than ``graph_weight`` 0.5);
- deleting chunks / databases invalidates the implicit corpus-graph cache;
- the adjacent/lexical disable flags map onto zero edge weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import yusu_kb.knowledge.implementations.local_kb as local_kb_module
from yusu_kb.knowledge.base import FileStatus
from yusu_kb.knowledge.graphs.graph_service import GraphService
from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage
from yusu_kb.knowledge.graphs.implicit_graph import (
    _GRAPH_CACHE,
    ImplicitGraphConfig,
)
from yusu_kb.knowledge.implementations.local_kb import (
    LocalKB,
    _retrieval_config_options,
    set_default_embedding_func,
    set_default_graph_chat_model_fn,
)
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.models.chat import GeneralResponse
from yusu_kb.models.embed import EmbeddingFunc, wrap_embedding_func_with_attrs
from yusu_kb.repositories import configure_repositories
from yusu_kb.repositories.knowledge_base_repository import KnowledgeBaseRepository
from yusu_kb.storage.vector_store import VectorStore

FAKE_DIM = 64

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

PAYLOAD_KEYWORDS = {
    "high_level_keywords": ["转账", "资金"],
    "low_level_keywords": ["张三", "李四"],
}


def _fake_embed_one(text: str, dim: int = FAKE_DIM) -> np.ndarray:
    vec = np.zeros(dim)
    for ch in text:
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
    """Extraction vs keyword prompt stub (same contract as test_graph_retrieval)."""

    def __init__(self, extraction_payload: str, keyword_payload: str = ""):
        self.extraction_payload = extraction_payload
        self.keyword_payload = keyword_payload

    async def __call__(self, messages: list[dict]) -> GeneralResponse:
        content = messages[-1]["content"] if messages else ""
        if "high_level_keywords" in content:
            return GeneralResponse(self.keyword_payload)
        return GeneralResponse(self.extraction_payload)


@pytest.fixture(autouse=True)
def _default_embedding():
    set_default_embedding_func(fake_embedding_func)
    yield
    set_default_embedding_func(None)


@pytest.fixture(autouse=True)
def _reset_graph_chat_fn():
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
    created = await manager.create_database(
        name=name,
        description="implicit graph integration test",
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


async def configure_model_spec(kb: LocalKB, kb_id: str) -> None:
    """Mark the KB as graph-configured (model_spec) without building the graph."""
    chat = DualChat(json.dumps(PAYLOAD_TRANSFER, ensure_ascii=False), json.dumps(PAYLOAD_KEYWORDS, ensure_ascii=False))
    svc = GraphService.get_instance(
        kb_id=kb_id,
        work_dir=kb.work_dir,
        embed_func=kb._get_embedding_function(),
        chat_model_fn=chat,
    )
    await svc.configure(kb_id, extractor_options={"model_spec": "fake-model"})
    row = await KnowledgeBaseRepository().get_by_kb_id(kb_id)
    assert row is not None and row.additional_params
    metadata = kb.databases_meta.setdefault(kb_id, {}).setdefault("metadata", {})
    metadata["graph_build_config"] = row.additional_params["graph_build_config"]


async def _chunk_ids(results: list[dict]) -> list[str]:
    return [chunk["metadata"]["chunk_id"] for chunk in results]


async def _wait_build_done(svc: GraphService, kb_id: str, timeout: float = 15.0) -> dict:
    """Poll get_status until the background build reaches a terminal state."""
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        status = await svc.get_status(kb_id)
        if status["build_task_status"] in ("completed", "failed", "cancelled"):
            return status
        if loop.time() > deadline:
            raise AssertionError(f"build did not finish within {timeout}s: {status}")
        await asyncio.sleep(0.02)


class TestImplicitFallbackWiring:
    async def test_switch_off_never_touches_implicit_path(self, manager, tmp_path, monkeypatch):
        """Default off: snapshot and the implicit entry point are never reached."""
        kb_id = await create_kb(manager, "开关关闭零开销")
        await index_file(manager, kb_id, tmp_path, "a.md", "# 转账记录\n\n张三向李四转账五十万元。")
        kb = get_kb(manager)

        def _no_snapshot():
            raise AssertionError("snapshot must not be accessed while the switch is off")

        async def _no_implicit(**kwargs):
            raise AssertionError("retrieve_implicit_chunks must not be called while off")

        monkeypatch.setattr(VectorStore, "snapshot", _no_snapshot)
        monkeypatch.setattr(local_kb_module, "retrieve_implicit_chunks", _no_implicit)

        baseline = await kb.aquery("张三 转账 李四", kb_id, use_reranker=False)
        assert baseline
        # 显式开启但从未配置图谱（无 model_spec）→ model_spec 门禁同样拦下
        enabled = await kb.aquery(
            "张三 转账 李四", kb_id, use_reranker=False, implicit_graph_enabled=True
        )
        assert await _chunk_ids(enabled) == await _chunk_ids(baseline)

    async def test_implicit_fallback_runs_and_fuses_with_implicit_weight(
        self, manager, tmp_path, monkeypatch
    ):
        """Graph configured but coverage 0 → implicit channel runs, weight 0.4."""
        kb_id = await create_kb(manager, "隐式回退全链路")
        await index_file(manager, kb_id, tmp_path, "a.md", "# 转账记录\n\n张三向李四转账五十万元，用于项目投资。")
        await index_file(manager, kb_id, tmp_path, "b.md", "# 采购记录\n\n公司采购了一批设备，总价三十万元。")
        kb = get_kb(manager)
        await configure_model_spec(kb, kb_id)
        assert await kb._graph_coverage(kb_id) == 0.0

        fused_weights: list[float] = []
        original_fuse = kb._fuse_chunk_rankings

        def _spy(base_chunks, graph_chunks, graph_weight, rrf_k=60.0):
            fused_weights.append(float(graph_weight))
            return original_fuse(base_chunks, graph_chunks, graph_weight, rrf_k)

        monkeypatch.setattr(kb, "_fuse_chunk_rankings", _spy)

        results = await kb.aquery(
            "张三 转账 李四",
            kb_id,
            use_reranker=False,
            use_graph_retrieval=True,
            implicit_graph_enabled=True,
            implicit_timeout_ms=4000,
        )
        assert results, "implicit fallback must still deliver results"
        assert fused_weights, "graph fusion must have run"
        assert fused_weights[-1] == pytest.approx(0.4), "implicit channel must fuse with implicit_weight"

        # 直接调用 wrapper：chunk 组装形与图通道一致（graph_score + implicit_score）
        vector_hits = await kb.aquery("张三 转账 李四", kb_id, use_reranker=False, use_graph_retrieval=False)
        chunks, error = await kb._retrieve_implicit_graph_chunks(
            "张三 转账 李四",
            kb_id,
            vector_hits,
            {"implicit_graph_enabled": True, "implicit_timeout_ms": 4000},
            0.0,
        )
        assert error is None
        assert chunks
        for chunk in chunks:
            assert chunk["metadata"]["chunk_id"]
            assert float(chunk.get("graph_score") or 0.0) > 0.0
            assert float(chunk.get("implicit_score") or 0.0) > 0.0

    async def test_coverage_one_keeps_explicit_graph_path(self, manager, tmp_path):
        """Graph fully built → implicit branch skipped, results identical on/off."""
        kb_id = await create_kb(manager, "建满后行为不变")
        await index_file(manager, kb_id, tmp_path, "a.md", "# 转账记录\n\n张三向李四转账五十万元，用于项目投资。")
        kb = get_kb(manager)
        chat = DualChat(json.dumps(PAYLOAD_TRANSFER, ensure_ascii=False), json.dumps(PAYLOAD_KEYWORDS, ensure_ascii=False))
        set_default_graph_chat_model_fn(chat)
        await GraphService.evict(kb_id)
        svc = GraphService.get_instance(
            kb_id=kb_id,
            work_dir=kb.work_dir,
            embed_func=kb._get_embedding_function(),
            chat_model_fn=chat,
        )
        await svc.configure(kb_id, extractor_options={"model_spec": "fake-model"})
        row = await KnowledgeBaseRepository().get_by_kb_id(kb_id)
        metadata = kb.databases_meta.setdefault(kb_id, {}).setdefault("metadata", {})
        metadata["graph_build_config"] = row.additional_params["graph_build_config"]
        await svc.build_pending_chunks(kb_id, batch_size=4)
        status = await _wait_build_done(svc, kb_id)
        assert status["build_task_status"] == "completed"

        coverage = await kb._graph_coverage(kb_id)
        assert coverage == pytest.approx(1.0), "full build must cover every chunk"

        with_switch = await kb.aquery(
            "张三 转账 李四", kb_id, use_reranker=False, implicit_graph_enabled=True
        )
        without_switch = await kb.aquery(
            "张三 转账 李四", kb_id, use_reranker=False, implicit_graph_enabled=False
        )
        assert await _chunk_ids(with_switch) == await _chunk_ids(without_switch)
        # 显式图通道产物不带 implicit_score
        graph_chunks, error = await kb._retrieve_graph_chunks(
            "张三 转账 李四", kb_id, [], {"keyword_extractor_enabled": False}
        )
        assert error is None
        assert all("implicit_score" not in chunk for chunk in graph_chunks)

    async def test_disable_flags_zero_edge_weights(self, manager, tmp_path, monkeypatch):
        """implicit_adjacent_enabled/implicit_lexical_enabled map to zero weights."""
        kb_id = await create_kb(manager, "边族开关映射")
        await index_file(manager, kb_id, tmp_path, "a.md", "# 转账记录\n\n张三向李四转账五十万元。")
        kb = get_kb(manager)

        captured: dict = {}

        async def _capture(**kwargs):
            captured.update(kwargs)
            return [], None, None

        monkeypatch.setattr(local_kb_module, "retrieve_implicit_chunks", _capture)

        base = [{"metadata": {"chunk_id": "ck000", "file_id": "f1", "chunk_index": 0}, "score": 0.9}]
        await kb._retrieve_implicit_graph_chunks(
            "张三 转账",
            kb_id,
            base,
            {"implicit_graph_enabled": True, "implicit_adjacent_enabled": False},
            0.0,
        )
        cfg: ImplicitGraphConfig = captured["cfg"]
        assert cfg.weight_adjacent == 0.0
        assert cfg.weight_lexical == pytest.approx(0.4)  # 未关闭词法边

        await kb._retrieve_implicit_graph_chunks(
            "张三 转账",
            kb_id,
            base,
            {"implicit_graph_enabled": True, "implicit_lexical_enabled": False},
            0.0,
        )
        cfg2: ImplicitGraphConfig = captured["cfg"]
        assert cfg2.weight_lexical == 0.0
        assert cfg2.weight_exact_id == 0.0
        assert cfg2.weight_adjacent == pytest.approx(0.5)  # 未关闭相邻边

    async def test_delete_chunks_invalidates_implicit_cache(self, manager, tmp_path):
        kb_id = await create_kb(manager, "删除失效缓存")
        file_id = await index_file(manager, kb_id, tmp_path, "a.md", "# 转账记录\n\n张三向李四转账五十万元。")
        _GRAPH_CACHE["integration-test-key"] = (kb_id, SimpleNamespace(kb_id=kb_id))
        await get_kb(manager).delete_file_chunks_only(kb_id, file_id)
        assert "integration-test-key" not in _GRAPH_CACHE

    async def test_config_schema_exposes_implicit_fields(self):
        """All 13 implicit_* knobs appear in the front-end schema with defaults."""
        options = {opt["key"]: opt for opt in _retrieval_config_options()}
        expected = {
            "implicit_graph_enabled": True,
            "implicit_graph_coverage_threshold": 0.999,
            "implicit_knn_k": 12,
            "implicit_sim_threshold": 0.55,
            "implicit_damping": 0.80,
            "implicit_query_mix": 0.6,
            "implicit_top_k": 20,
            "implicit_weight": 0.4,
            "implicit_max_corpus_nodes": 3000,
            "implicit_timeout_ms": 400,
            "implicit_adjacent_enabled": True,
            "implicit_lexical_enabled": True,
            "implicit_query_pool_size": 64,
        }
        for key, default in expected.items():
            assert key in options, f"missing schema option: {key}"
            assert options[key]["default"] == default
            assert options[key]["depend_on"] == ("use_graph_retrieval", True)


class TestRealChunkEdges:
    """Unit tests for GraphStorage.real_chunk_edges (hybrid REAL edges)."""

    def _seeded_storage(self, tmp_path: Path) -> NetworkXGraphStorage:
        storage = NetworkXGraphStorage("kb-real-edges", tmp_path)
        storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="张三转账")
        storage.add_chunk(chunk_id="c2", file_id="f1", chunk_index=1, content_preview="李四收款")
        storage.add_chunk(chunk_id="c3", file_id="f2", chunk_index=0, content_preview="王五在场")
        storage.upsert_entity(
            entity_id="e1", normalized_name="张三", label="人物", name="张三"
        )
        storage.upsert_entity(
            entity_id="e2", normalized_name="王五", label="人物", name="王五"
        )
        storage.add_mention(entity_id="e1", chunk_id="c1", file_id="f1")
        storage.add_mention(entity_id="e1", chunk_id="c2", file_id="f1")
        storage.add_mention(entity_id="e2", chunk_id="c1", file_id="f1")
        storage.add_mention(entity_id="e2", chunk_id="c3", file_id="f2")
        return storage

    def test_shared_entities_produce_symmetric_edges(self, tmp_path):
        storage = self._seeded_storage(tmp_path)
        edges = storage.real_chunk_edges(["c1"], weight_scale=0.6)
        by_pair = {(u, v): weight for u, v, weight in edges}
        # c1 与 c2 共享 e1，c1 与 c3 共享 e2；c2-c3 无共享实体
        assert ("c1", "c2") in by_pair and ("c1", "c3") in by_pair
        assert ("c2", "c3") not in by_pair
        assert by_pair[("c1", "c2")] == pytest.approx(0.6)
        assert by_pair[("c1", "c3")] == pytest.approx(0.6)

    def test_shared_count_normalises_to_strongest_pair(self, tmp_path):
        storage = self._seeded_storage(tmp_path)
        storage.upsert_entity(entity_id="e3", normalized_name="资金", label="事件", name="资金")
        storage.add_mention(entity_id="e3", chunk_id="c1", file_id="f1")
        storage.add_mention(entity_id="e3", chunk_id="c2", file_id="f1")
        edges = storage.real_chunk_edges(["c1", "c2"], weight_scale=0.6)
        by_pair = {(u, v): weight for u, v, weight in edges}
        # c1-c2 共享 2 个实体（e1/e3），峰值归一后为 1.0 * 0.6；c1-c3 只共享 1 个 → 0.3
        assert by_pair[("c1", "c2")] == pytest.approx(0.6)
        assert by_pair[("c1", "c3")] == pytest.approx(0.3)

    def test_empty_and_disabled_cases(self, tmp_path):
        storage = self._seeded_storage(tmp_path)
        assert storage.real_chunk_edges(["missing-chunk"]) == []
        assert storage.real_chunk_edges(["c1"], weight_scale=0.0) == []
        assert storage.real_chunk_edges([]) == []
