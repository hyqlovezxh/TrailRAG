"""Tests for VectorStore (self-contained nano-vectordb wrapper).

Uses a deterministic fake embedding function; persistence is exercised
through the real JSON data file under a tmp dir.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from yusu_kb.models.embed import EmbeddingFunc, wrap_embedding_func_with_attrs
from yusu_kb.storage.vector_store import VectorStore

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


@pytest.fixture()
async def store(tmp_path):
    instance = VectorStore(
        namespace="kb1",
        working_dir=str(tmp_path),
        workspace="kb1",
        embedding_func=_embedding_func(_FakeEmbed()),
        embedding_batch_num=200,
        meta_fields={"content", "file_id", "chunk_index"},
        cosine_better_than_threshold=0.0,
    )
    await instance.initialize()
    yield instance


def _documents() -> dict[str, dict]:
    return {
        "c0": {"content": "苹果是红色的水果", "file_id": "f1", "chunk_index": 0},
        "c1": {"content": "香蕉是黄色的水果", "file_id": "f1", "chunk_index": 1},
        "c2": {"content": "汽车有四个轮子", "file_id": "f2", "chunk_index": 0},
    }


async def test_upsert_flush_query_roundtrip(store):
    await store.upsert(_documents())
    await store.index_done_callback()
    results = await store.query("苹果", top_k=3)
    assert len(results) == 3
    by_id = {hit["id"]: hit for hit in results}
    assert set(by_id) == {"c0", "c1", "c2"}
    assert by_id["c0"]["content"] == "苹果是红色的水果"
    assert by_id["c0"]["file_id"] == "f1"
    assert by_id["c0"]["chunk_index"] == 0
    assert "distance" in by_id["c0"]
    assert "vector" not in by_id["c0"]
    scores = [hit["distance"] for hit in results]
    assert scores == sorted(scores, reverse=True)


async def test_buffered_upserts_are_invisible_until_flush(store):
    await store.upsert(_documents())
    assert await store.query("苹果", top_k=3) == []


async def test_persistence_across_instances(tmp_path):
    embed = _FakeEmbed()
    first = VectorStore(
        namespace="kb1",
        working_dir=str(tmp_path),
        workspace="kb1",
        embedding_func=_embedding_func(embed),
    )
    await first.initialize()
    await first.upsert(_documents())
    await first.index_done_callback()

    second = VectorStore(
        namespace="kb1",
        working_dir=str(tmp_path),
        workspace="kb1",
        embedding_func=_embedding_func(_FakeEmbed()),
    )
    await second.initialize()
    results = await second.query("香蕉", top_k=2)
    by_id = {hit["id"]: hit for hit in results}
    assert by_id["c1"]["content"] == "香蕉是黄色的水果"


async def test_delete_removes_rows(store):
    await store.upsert(_documents())
    await store.index_done_callback()
    await store.delete(["c0"])
    results = await store.query("苹果", top_k=3)
    assert "c0" not in {hit["id"] for hit in results}
    assert {hit["id"] for hit in results} == {"c1", "c2"}


async def test_upsert_overwrites_same_id(store):
    await store.upsert({"c0": {"content": "旧内容", "file_id": "f1", "chunk_index": 0}})
    await store.index_done_callback()
    await store.upsert({"c0": {"content": "新内容", "file_id": "f1", "chunk_index": 0}})
    await store.index_done_callback()
    results = await store.query("新内容", top_k=1)
    assert results[0]["id"] == "c0"
    assert results[0]["content"] == "新内容"


async def test_embedding_failure_keeps_pending(tmp_path):
    embed = _FakeEmbed(fail=True)
    store = VectorStore(
        namespace="kb1",
        working_dir=str(tmp_path),
        workspace="kb1",
        embedding_func=_embedding_func(embed),
    )
    await store.initialize()
    await store.upsert(_documents())
    with pytest.raises(RuntimeError, match="embedding failure"):
        await store.index_done_callback()
    assert store._pending  # pending 保留，下次 flush 重试

    embed.fail = False
    await store.index_done_callback()
    results = await store.query("苹果红色", top_k=1)
    assert results[0]["id"] == "c0"


async def test_dim_probed_when_unset(tmp_path):
    embed = _FakeEmbed(dim=None)

    async def probe_aware(texts, **kwargs):
        # 无配置维度时返回固定 8 维向量
        del kwargs
        return np.array([_fake_embed_one(t) for t in texts])

    func = wrap_embedding_func_with_attrs(embedding_dim=None)(probe_aware)
    store = VectorStore(
        namespace="kb1",
        working_dir=str(tmp_path),
        workspace="kb1",
        embedding_func=func,
    )
    await store.initialize()
    await store.upsert(_documents())
    await store.index_done_callback()
    results = await store.query("苹果红色", top_k=1)
    assert results[0]["id"] == "c0"
    del embed


async def test_batch_embedding_chunks(store):
    embed = _FakeEmbed()
    store.embedding_func = _embedding_func(embed)
    docs = {f"c{i}": {"content": f"文档内容{i}", "file_id": "f1", "chunk_index": i} for i in range(5)}
    await store.upsert(docs)
    store.embedding_batch_num = 2
    await store.index_done_callback()
    assert embed.calls == [2, 2, 1]


async def test_drop_removes_file(tmp_path):
    store = VectorStore(
        namespace="kb1",
        working_dir=str(tmp_path),
        workspace="kb1",
        embedding_func=_embedding_func(_FakeEmbed()),
    )
    await store.initialize()
    await store.upsert(_documents())
    await store.index_done_callback()
    data_file = tmp_path / "kb1" / "vdb_kb1.json"
    assert data_file.exists()
    await store.drop()
    assert not data_file.exists()
    assert store._pending == {}


async def test_requires_initialize():
    store = VectorStore(
        namespace="kb1",
        working_dir=".",
        workspace="kb1",
        embedding_func=_embedding_func(_FakeEmbed()),
    )
    with pytest.raises(RuntimeError, match="not initialized"):
        await store.query("x", top_k=1)


async def test_snapshot_matches_query_cosine(store):
    """Snapshot rows must reproduce the exact cosine scores that query() returns."""
    await store.upsert(_documents())
    await store.index_done_callback()

    query = "苹果"
    results = await store.query(query, top_k=3)
    snap = store.snapshot()
    assert snap is not None
    assert len(snap.ids) == 3

    row_of = {chunk_id: i for i, chunk_id in enumerate(snap.ids)}
    query_vec = _fake_embed_one(query)
    for hit in results:
        row = snap.matrix[row_of[hit["id"]]]
        # 行已 L2 归一化（nano-vectordb upsert 时 normalize），点积即余弦
        assert float(np.dot(row, query_vec)) == pytest.approx(
            float(hit["distance"]), abs=1e-5
        )


async def test_snapshot_is_row_aligned_with_metadata(store):
    await store.upsert(_documents())
    await store.index_done_callback()

    snap = store.snapshot()
    assert snap is not None
    assert len(snap.ids) == len(snap.metas) == snap.matrix.shape[0]
    # 按 id 对齐：每个 id 的 meta 必须仍属于它自己，且不含重载体/内部字段
    aligned = dict(zip(snap.ids, snap.metas))
    assert aligned["c0"]["content"] == "苹果是红色的水果"
    assert aligned["c2"]["file_id"] == "f2"
    assert "vector" not in aligned["c0"]
    assert "__vector__" not in aligned["c0"]


async def test_snapshot_is_none_before_flush(store):
    await store.upsert(_documents())
    # 未 flush 时分块对检索不可见，快照同样应不可用（与 query 语义一致）
    assert store.snapshot() is None


async def test_snapshot_replaced_not_mutated_on_new_flush(store):
    """A handed-out snapshot must stay valid when the corpus is later extended."""
    await store.upsert({"c0": {"content": "苹果是红色的水果", "file_id": "f1", "chunk_index": 0}})
    await store.index_done_callback()
    first = store.snapshot()
    assert first is not None
    frozen = np.array(first.matrix)  # 复制一份作为比对基准

    await store.upsert(_documents())
    await store.index_done_callback()
    second = store.snapshot()
    assert second is not None

    # 新快照是全新对象，旧对象内容毫发无损 → 缓存是"替换"而非"原地改写"
    assert second.matrix is not first.matrix
    assert np.array_equal(first.matrix, frozen)
    assert len(second.ids) == 3
    assert len(first.ids) == 1


async def test_concurrent_flush_serialized(tmp_path):
    embed = _FakeEmbed()
    store = VectorStore(
        namespace="kb1",
        working_dir=str(tmp_path),
        workspace="kb1",
        embedding_func=_embedding_func(embed),
    )
    await store.initialize()
    docs = {f"c{i}": {"content": f"内容{i}", "file_id": "f1", "chunk_index": i} for i in range(6)}
    await store.upsert(docs)
    store.embedding_batch_num = 3
    await asyncio.gather(store.index_done_callback(), store.index_done_callback())
    results = await store.query("内容", top_k=6)
    assert len(results) == 6
    assert len(store._pending) == 0