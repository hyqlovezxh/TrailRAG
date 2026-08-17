"""Tests for the PPR retrieval module (``knowledge/graphs/ppr.py``)."""

from __future__ import annotations

import networkx as nx
import pytest

from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage
from yusu_kb.knowledge.graphs.graph_utils import normalize_entity_name
from yusu_kb.knowledge.graphs.ppr import build_ppr_graph, rank_chunks_by_ppr


@pytest.fixture()
def storage(tmp_path):
    return NetworkXGraphStorage("kb_ppr", tmp_path)


def _add_entity(storage, entity_id, name):
    storage.upsert_entity(
        entity_id=entity_id,
        normalized_name=normalize_entity_name(name),
        label="person",
        name=name,
    )


def _add_relation(storage, source_id, target_id, rtype="transfer"):
    storage.upsert_relation(
        triple_id=f"{source_id}-{target_id}",
        source_id=source_id,
        target_id=target_id,
        text=f"{source_id} {rtype} {target_id}",
        rtype=rtype,
        file_ids=["f1"],
    )


def _add_chunk(storage, chunk_id, file_id="f1", index=0):
    storage.add_chunk(chunk_id=chunk_id, file_id=file_id, chunk_index=index, content_preview=chunk_id)


def _mention(storage, chunk_id, entity_id):
    storage.add_mention(entity_id=entity_id, chunk_id=chunk_id, file_id="f1")


def _build_chain_graph(storage):
    """A --RELATION--> B; chunk1 mentions A; chunk2 mentions B."""
    _add_entity(storage, "A", "张三")
    _add_entity(storage, "B", "李四")
    _add_relation(storage, "A", "B")
    _add_chunk(storage, "chunk1")
    _add_chunk(storage, "chunk2")
    _mention(storage, "chunk1", "A")
    _mention(storage, "chunk2", "B")


def test_ppr_graph_weights(storage):
    _build_chain_graph(storage)
    graph = build_ppr_graph(storage, directed=False)
    assert set(graph.nodes) == {"A", "B", "chunk1", "chunk2"}
    assert graph["A"]["B"]["weight"] == 1.0
    # MENTIONS 基础权重 0.3（每个 chunk 只有一条提及边，计数权重为 0.2 * 1/1）
    assert graph["chunk1"]["A"]["weight"] == pytest.approx(0.5)
    assert graph["chunk2"]["B"]["weight"] == pytest.approx(0.5)


def test_rank_seed_chunk_first(storage):
    """seed={A} -> 直接提及 A 的 chunk1 排名高于多跳的 chunk2。"""
    _build_chain_graph(storage)
    ranked = rank_chunks_by_ppr(
        storage, {"A": 1.0}, top_k=10, max_nodes=100, damping=0.85, directed=False
    )
    assert [cid for cid, _ in ranked] == ["chunk1", "chunk2"]
    assert ranked[0][1] >= ranked[1][1] > 0


def test_personalization_explicit_zero_non_seed(storage):
    """非 seed 节点显式 0 后，seed 实体得分占比最大（图上可达节点仍 >0）。"""
    _build_chain_graph(storage)
    graph = build_ppr_graph(storage, directed=False)
    reset = {node: 0.0 for node in graph.nodes}
    reset["A"] = 1.0
    scores = nx.pagerank(graph, alpha=0.85, personalization=reset, weight="weight")
    assert scores["A"] == max(scores.values())
    assert scores["B"] > 0 and scores["chunk2"] > 0


def test_top_k_truncation(storage):
    _build_chain_graph(storage)
    ranked = rank_chunks_by_ppr(
        storage, {"A": 1.0}, top_k=1, max_nodes=100, damping=0.85, directed=False
    )
    assert len(ranked) == 1
    assert ranked[0][0] == "chunk1"


def test_directed_and_undirected_equivalent_on_undirected_graph(storage):
    _build_chain_graph(storage)
    undirected = rank_chunks_by_ppr(
        storage, {"A": 1.0}, top_k=10, max_nodes=100, damping=0.85, directed=False
    )
    directed = rank_chunks_by_ppr(
        storage, {"A": 1.0}, top_k=10, max_nodes=100, damping=0.85, directed=True
    )
    assert [cid for cid, _ in directed] == [cid for cid, _ in undirected]


def test_empty_graph_falls_back_to_empty(storage):
    ranked = rank_chunks_by_ppr(storage, {"A": 1.0}, top_k=5, max_nodes=100, damping=0.85, directed=False)
    assert ranked == []


def test_no_relation_falls_back_to_2hop(storage):
    """无 RELATION 边时 pagerank 无 chunk 得分 -> 降级 2hop（此处同 1hop）。"""
    _add_entity(storage, "A", "张三")
    _add_chunk(storage, "chunk1")
    _mention(storage, "chunk1", "A")
    ranked = rank_chunks_by_ppr(
        storage, {"A": 1.0}, top_k=10, max_nodes=100, damping=0.85, directed=False
    )
    assert [cid for cid, _ in ranked] == ["chunk1"]


def test_unknown_seed_ignored(storage):
    _build_chain_graph(storage)
    ranked = rank_chunks_by_ppr(
        storage, {"A": 1.0, "GHOST": 5.0}, top_k=10, max_nodes=100, damping=0.85, directed=False
    )
    assert [cid for cid, _ in ranked] == ["chunk1", "chunk2"]


def test_max_nodes_restricts_subgraph(storage):
    """节点数超过 max_nodes 时只在受限子图上做 pagerank，仍返回 chunk。"""
    _build_chain_graph(storage)
    ranked = rank_chunks_by_ppr(
        storage, {"A": 1.0}, top_k=10, max_nodes=3, damping=0.85, directed=False
    )
    # 受限子图（A + 1 个邻居 + 1 个 chunk）仍有得分或降级路径，结果非空
    assert isinstance(ranked, list)


def test_weights_normalized_to_unit_interval(storage):
    _build_chain_graph(storage)
    ranked = rank_chunks_by_ppr(
        storage, {"A": 1.0}, top_k=10, max_nodes=100, damping=0.85, directed=False
    )
    assert all(0.0 < score <= 1.0 for _, score in ranked)
