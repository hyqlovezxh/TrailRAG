"""Tests for GraphVectorStore (nano-vectordb wrapper for graph records).

Uses a deterministic fake embedding function; persistence is exercised
through the real JSON data file under a tmp dir.
"""

from __future__ import annotations

import numpy as np
import pytest

from yusu_kb.knowledge.graphs.graph_vector_store import GraphVectorStore
from yusu_kb.models.embed import EmbeddingFunc, wrap_embedding_func_with_attrs

FAKE_DIM = 8


def _fake_embed_one(text: str, dim: int = FAKE_DIM) -> np.ndarray:
    vec = np.zeros(dim)
    for ch in text:
        vec[ord(ch) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class _FakeEmbed:
    def __init__(self, dim: int | None = FAKE_DIM, fail: bool = False):
        self.dim = dim
        self.fail = fail
        self.calls: list[int] = []

    async def __call__(self, texts, **kwargs) -> np.ndarray:
        del kwargs
        self.calls.append(len(texts))
        if self.fail:
            raise RuntimeError("embedding failure injected")
        dim = self.dim or FAKE_DIM
        return np.array([_fake_embed_one(t, dim) for t in texts])


def _embedding_func(embed: _FakeEmbed) -> EmbeddingFunc:
    return wrap_embedding_func_with_attrs(embedding_dim=embed.dim)(embed)


def _make_store(
    tmp_path,
    *,
    kind: str = "entity",
    kb_id: str = "kb1",
    embed: _FakeEmbed | None = None,
    **kwargs,
) -> GraphVectorStore:
    return GraphVectorStore(
        kind=kind,
        kb_id=kb_id,
        work_dir=tmp_path,
        embedding_func=_embedding_func(embed or _FakeEmbed()),
        **kwargs,
    )


def _entities() -> list[dict]:
    return [
        {"id": "e1", "content": "苹果是红色的水果", "label": "fruit"},
        {"id": "e2", "content": "香蕉是黄色的水果", "label": "fruit"},
        {"id": "e3", "content": "汽车有四个轮子", "label": "vehicle"},
    ]


def _triples() -> list[dict]:
    return [
        {
            "id": "t1",
            "content": "苹果 → 是 → 红色 | 果皮颜色",
            "source_id": "e1",
            "target_id": "e_red",
            "type": "是",
        },
        {
            "id": "t2",
            "content": "香蕉 → 是 → 黄色 | 果皮颜色",
            "source_id": "e2",
            "target_id": "e_yellow",
            "type": "是",
        },
    ]


async def test_initialize_probes_dim_and_marks_initialized(tmp_path):
    embed = _FakeEmbed(dim=None)
    store = _make_store(tmp_path, embed=embed)
    assert not store.is_initialized()
    await store.initialize()
    assert store.is_initialized()
    assert store._client.embedding_dim == FAKE_DIM
    assert embed.calls == [1]


async def test_upsert_search_roundtrip(tmp_path):
    store = _make_store(tmp_path, cosine_better_than_threshold=0.0)
    await store.initialize()
    await store.upsert(_entities())
    results = await store.search("苹果红", top_k=3)
    assert len(results) == 3
    by_id = {hit["id"]: hit for hit in results}
    assert set(by_id) == {"e1", "e2", "e3"}
    assert results[0]["id"] == "e1"
    assert results[0]["score"] > 0.5
    assert by_id["e1"]["content"] == "苹果是红色的水果"
    assert by_id["e1"]["label"] == "fruit"


async def test_search_field_contract_and_score_desc(tmp_path):
    store = _make_store(tmp_path, cosine_better_than_threshold=0.0)
    await store.initialize()
    await store.upsert(_entities())
    results = await store.search("苹果", top_k=3)
    for hit in results:
        assert set(hit) == {"id", "content", "score", "label"}
        assert isinstance(hit["score"], float)
        assert "distance" not in hit
        assert "vector" not in hit
    scores = [hit["score"] for hit in results]
    assert scores == sorted(scores, reverse=True)


async def test_same_id_upsert_replaces(tmp_path):
    store = _make_store(tmp_path, cosine_better_than_threshold=0.0)
    await store.initialize()
    await store.upsert([{"id": "e1", "content": "旧内容", "label": "fruit"}])
    await store.upsert([{"id": "e1", "content": "新内容", "label": "fruit"}])
    results = await store.search("新内容", top_k=10)
    hits = [hit for hit in results if hit["id"] == "e1"]
    assert len(hits) == 1
    assert hits[0]["content"] == "新内容"


async def test_delete_ids_removes_rows(tmp_path):
    store = _make_store(tmp_path, cosine_better_than_threshold=0.0)
    await store.initialize()
    await store.upsert(_entities())
    await store.delete_ids(["e1"])
    results = await store.search("苹果", top_k=10)
    ids = {hit["id"] for hit in results}
    assert "e1" not in ids
    assert ids == {"e2", "e3"}


async def test_default_threshold_filters_weak_hits(tmp_path):
    store = _make_store(tmp_path)
    await store.initialize()
    await store.upsert(
        [
            {"id": "e1", "content": "苹果是红色的水果", "label": "fruit"},
            {"id": "e2", "content": "汽车四轮子", "label": "vehicle"},
        ]
    )
    results = await store.search("苹果", top_k=2)
    ids = {hit["id"] for hit in results}
    assert "e1" in ids
    assert "e2" not in ids


async def test_persistence_across_instances(tmp_path):
    first = _make_store(tmp_path, cosine_better_than_threshold=0.0)
    await first.initialize()
    await first.upsert(_entities())

    second = _make_store(tmp_path, cosine_better_than_threshold=0.0)
    await second.initialize()
    results = await second.search("香蕉", top_k=2)
    assert results[0]["id"] == "e2"
    assert results[0]["label"] == "fruit"
    assert results[0]["content"] == "香蕉是黄色的水果"


async def test_triple_kind_meta_roundtrip(tmp_path):
    store = _make_store(tmp_path, kind="triple", cosine_better_than_threshold=0.0)
    await store.initialize()
    await store.upsert(_triples())
    results = await store.search("苹果是红色", top_k=1)
    hit = results[0]
    assert hit["id"] == "t1"
    assert hit["source_id"] == "e1"
    assert hit["target_id"] == "e_red"
    assert hit["type"] == "是"
    data_file = tmp_path / "kb1" / "graph_vdb_triple.json"
    assert data_file.exists()


async def test_drop_removes_file_and_resets(tmp_path):
    store = _make_store(tmp_path, cosine_better_than_threshold=0.0)
    await store.initialize()
    await store.upsert(_entities())
    data_file = tmp_path / "kb1" / "graph_vdb_entity.json"
    assert data_file.exists()
    await store.drop()
    assert not data_file.exists()
    assert not store.is_initialized()
    with pytest.raises(RuntimeError, match="not initialized"):
        await store.search("苹果", top_k=1)


async def test_empty_upsert_and_top_k_zero(tmp_path):
    store = _make_store(tmp_path, cosine_better_than_threshold=0.0)
    await store.initialize()
    await store.upsert([])
    assert await store.search("苹果", top_k=0) == []
    assert await store.search("苹果", top_k=1) == []


async def test_requires_initialize(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(RuntimeError, match="not initialized"):
        await store.search("x", top_k=1)
    with pytest.raises(RuntimeError, match="not initialized"):
        await store.upsert(_entities())


async def test_embed_failure_persists_nothing(tmp_path):
    embed = _FakeEmbed(fail=True)
    store = _make_store(tmp_path, embed=embed)
    await store.initialize()
    with pytest.raises(RuntimeError, match="embedding failure"):
        await store.upsert(_entities())
    embed.fail = False
    assert await store.search("苹果", top_k=1) == []


async def test_duplicate_ids_in_one_call_last_wins(tmp_path):
    store = _make_store(tmp_path, cosine_better_than_threshold=0.0)
    await store.initialize()
    await store.upsert(
        [
            {"id": "e1", "content": "旧内容", "label": "fruit"},
            {"id": "e1", "content": "新内容", "label": "fruit"},
        ]
    )
    results = await store.search("新内容", top_k=10)
    hits = [hit for hit in results if hit["id"] == "e1"]
    assert len(hits) == 1
    assert hits[0]["content"] == "新内容"


async def test_batch_embedding_splits(tmp_path):
    embed = _FakeEmbed()
    store = _make_store(
        tmp_path,
        embed=embed,
        embedding_batch_num=2,
        cosine_better_than_threshold=0.0,
    )
    await store.initialize()
    docs = [
        {"id": f"e{i}", "content": f"实体内容{i}", "label": "fruit"} for i in range(5)
    ]
    await store.upsert(docs)
    assert embed.calls == [2, 2, 1]
    results = await store.search("实体内容", top_k=5)
    assert len(results) == 5