"""Tests for the query-time implicit chunk graph (implicit_graph.py).

Covers: mutual-KNN symmetry / degree cap / hubness, threshold truncation and
weight shaping, ADJACENT / LEXICAL / EXACT_ID edge construction, PPR
determinism, personalization mixing, L5 zero-PPR fallback, tier split +
caching + invalidation, the full degradation ladder (skipped / real / no
signal / timeout / exception), hybrid real edges, file filtering and the
explanation beam.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from yusu_kb.knowledge.graphs import implicit_graph
from yusu_kb.knowledge.graphs.implicit_graph import (
    ImplicitChunkGraph,
    ImplicitGraphConfig,
    _get_or_build_full_graph,
    build_personalization,
    invalidate,
    retrieve_implicit_chunks,
)
from yusu_kb.knowledge.retrieval.query_analysis import QueryAnalysis
from yusu_kb.storage.vector_store import VectorSnapshot

_DIM = 16


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / max(float(np.linalg.norm(arr)), 1e-9)


def _snapshot(
    contents: list[str],
    vectors: list[np.ndarray] | None = None,
    file_ids: list[str] | None = None,
    chunk_indexes: list[int] | None = None,
) -> VectorSnapshot:
    count = len(contents)
    if vectors is None:
        rng = np.random.default_rng(hash(tuple(contents)) % (2**32))
        vectors = [rng.normal(size=_DIM).astype(np.float32) for _ in range(count)]
    matrix = np.stack([_unit(v.tolist()) for v in vectors])
    metas = [
        {
            "content": content,
            "file_id": (file_ids or [f"f{i // 4}" for i in range(count)])[pos],
            "chunk_index": (chunk_indexes or list(range(count)))[pos],
        }
        for pos, content in enumerate(contents)
    ]
    return VectorSnapshot(
        ids=[f"ck{i:03d}" for i in range(count)], matrix=matrix, metas=metas
    )


def _analysis(terms: tuple[str, ...] = (), tokens: tuple[str, ...] = ()) -> QueryAnalysis:
    return QueryAnalysis(phrase=" ".join(terms), exact_tokens=tokens, scoring_terms=terms, segmentation_used=False)


def _families(graph: ImplicitChunkGraph, **kwargs) -> dict[tuple[str, str], str]:
    nx_graph = graph.to_networkx(**kwargs)
    return {
        (min(u, v), max(u, v)): str(data.get("family"))
        for u, v, data in nx_graph.edges(data=True)
    }


def _cluster_snapshot(n_per_cluster: int = 55, clusters: int = 4) -> VectorSnapshot:
    rng = np.random.default_rng(42)
    vectors, contents = [], []
    for cluster in range(clusters):
        center = rng.normal(size=_DIM)
        for _ in range(n_per_cluster):
            vec = center + rng.normal(scale=0.05, size=_DIM)
            vectors.append(vec.astype(np.float32))
            contents.append(f"cluster{cluster} evidence text")
    return _snapshot(contents, vectors)


# --- SIM edges ---------------------------------------------------------------


def test_from_query_params_honours_explicit_zero_and_empty_fallback():
    """Explicit ``0`` must survive (edge-family disable flags rely on it);
    missing/empty values fall back to defaults -- a plain ``or`` would
    silently rewrite ``0.0`` to the default (regression guard)."""
    cfg = ImplicitGraphConfig.from_query_params(
        {
            "implicit_weight_adjacent": 0,
            "implicit_weight_lexical": 0.0,
            "implicit_weight_exact_id": 0,
            "implicit_query_mix": 0,
            "implicit_sim_threshold": 0,
            "implicit_hybrid_real_weight": 0.0,
        }
    )
    assert cfg.weight_adjacent == 0.0
    assert cfg.weight_lexical == 0.0
    assert cfg.weight_exact_id == 0.0
    assert cfg.query_mix == 0.0
    assert cfg.sim_threshold == 0.0
    assert cfg.hybrid_real_weight == 0.0

    fallback = ImplicitGraphConfig.from_query_params(
        {"implicit_weight_adjacent": "", "implicit_knn_k": None}
    )
    assert fallback.weight_adjacent == pytest.approx(0.5)
    assert fallback.knn_k == 12

    garbage = ImplicitGraphConfig.from_query_params({"implicit_damping": "not-a-number"})
    assert garbage.damping == pytest.approx(0.80)


def test_mutual_knn_symmetry_degree_cap_and_threshold():
    snap = _cluster_snapshot()
    cfg = ImplicitGraphConfig(knn_k=12, sim_threshold=0.55)
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=cfg)

    assert all(u < v for u, v, _ in graph.sim_edges)  # undirected pair stored once
    degree: dict[int, int] = {}
    for u, v, _ in graph.sim_edges:
        degree[u] = degree.get(u, 0) + 1
        degree[v] = degree.get(v, 0) + 1
    assert max(degree.values()) <= cfg.knn_k  # mutual adjacency caps the degree
    assert all(w > 0.0 for _, _, w in graph.sim_edges)


def test_sim_threshold_truncates_weak_edges():
    same = _unit([1.0] + [0.0] * (_DIM - 1))
    near = _unit([0.8, 0.6] + [0.0] * (_DIM - 2))  # cos(same, near) = 0.8
    snap = _snapshot(["a", "b", "c"], vectors=[same, same.copy(), near])
    tight = ImplicitGraphConfig(knn_k=2, sim_threshold=0.9)
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=tight)
    assert ("ck000", "ck001") in {(ids_to(u), ids_to(v)) for u, v, _ in graph.sim_edges}
    assert all("ck002" not in (ids_to(u), ids_to(v)) for u, v, _ in graph.sim_edges)

    loose = ImplicitGraphConfig(knn_k=2, sim_threshold=0.75)
    loose_graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=loose)
    assert any("ck002" in (ids_to(u), ids_to(v)) for u, v, _ in loose_graph.sim_edges)


def ids_to(row: int) -> str:
    return f"ck{row:03d}"


def test_sim_weight_formula_peaks_at_one():
    same = _unit([1.0] + [0.0] * (_DIM - 1))
    snap = _snapshot(["a", "b"], vectors=[same, same.copy()])
    cfg = ImplicitGraphConfig(knn_k=1, sim_threshold=0.55, sim_exponent=1.5)
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=cfg)
    weight = graph.sim_edges[0][2]
    assert weight == pytest.approx(1.0)  # ((1 - tau) / (1 - tau)) ** 1.5


def test_hubness_index_below_target_on_clustered_corpus():
    snap = _cluster_snapshot(n_per_cluster=55, clusters=4)
    cfg = ImplicitGraphConfig(knn_k=12, sim_threshold=0.55)
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=cfg)
    assert graph.hub_index < 0.05


# --- ADJACENT / LEXICAL / EXACT_ID edges -------------------------------------


def test_adjacent_edges_window_and_weights():
    snap = _snapshot(
        [f"doc text {i}" for i in range(6)],
        file_ids=["f0"] * 4 + ["f1"] * 2,
        chunk_indexes=[0, 1, 2, 4, 0, 1],
    )
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=ImplicitGraphConfig())
    families = _families(graph)
    adjacent = {pair: fam for pair, fam in families.items() if fam == "ADJACENT"}
    # ck000-ck001 (Δ1 -> 0.5), ck001-ck002 (Δ1), ck002-ck003 (Δ2 -> 0.25);
    # Δ3 (ck000-ck003) absent because indexes 0->4 skip; f1 pair present (Δ1).
    assert ("ck000", "ck001") in adjacent
    assert ("ck001", "ck002") in adjacent
    assert ("ck002", "ck003") in adjacent
    assert ("ck000", "ck003") not in adjacent  # chunk_index gap of 4 > window
    assert ("ck004", "ck005") in adjacent  # same second file
    nx_graph = graph.to_networkx()
    assert nx_graph["ck000"]["ck001"]["weight"] == pytest.approx(0.5)
    assert nx_graph["ck002"]["ck003"]["weight"] == pytest.approx(0.25)


def test_exact_id_edges_hard_link():
    snap = _snapshot(
        ["call from 13800138000 about case", "victim number 13800138000 reported", "unrelated narrative"],
    )
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=ImplicitGraphConfig())
    families = _families(graph, exact_tokens=("13800138000",))
    assert families.get(("ck000", "ck001")) == "EXACT_ID"
    nx_graph = graph.to_networkx(exact_tokens=("13800138000",))
    assert nx_graph["ck000"]["ck001"]["weight"] == pytest.approx(0.9)


def test_lexical_edges_and_high_df_drop():
    # 'funds' appears in 2/8 chunks (df ratio 0.25 > default 0.2 -> dropped),
    # 'transfer' appears in 2/8 too, so use a larger corpus for a kept term.
    contents = ["funds transfer record"] * 2 + [f"filler narrative {i}" for i in range(6)]
    snap = _snapshot(contents)
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=ImplicitGraphConfig())
    families = _families(graph, query_terms=("funds",))
    assert ("ck000", "ck001") not in {pair for pair, fam in families.items() if fam == "LEXICAL"}

    sparse = _snapshot(
        ["funds record one", "funds record two"] + [f"filler {i}" for i in range(8)],
        file_ids=["fa", "fb"] + [f"f{i}" for i in range(8)],
        chunk_indexes=[0, 7] + [i for i in range(8)],
    )
    sparse_graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=sparse, cfg=ImplicitGraphConfig())
    sparse_families = _families(sparse_graph, query_terms=("funds",))
    assert sparse_families.get(("ck000", "ck001")) == "LEXICAL"
    expected = 0.4 * (np.log1p(10 / 2) / np.log1p(10)) * min(1.0, 8 / 2)
    nx_graph = sparse_graph.to_networkx(query_terms=("funds",))
    assert nx_graph["ck000"]["ck001"]["weight"] == pytest.approx(float(expected))


# --- PPR / personalization / L5 ------------------------------------------------


def test_ppr_ranking_deterministic():
    snap = _cluster_snapshot(n_per_cluster=20, clusters=3)
    cfg = ImplicitGraphConfig()
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=cfg)
    nx_graph = graph.to_networkx(seed_ids=["ck000"])
    seeds = {"ck000": 1.0}
    first = graph.rank(nx_graph, query_cosine=None, seed_weights=seeds, top_k=10)
    second = graph.rank(nx_graph, query_cosine=None, seed_weights=seeds, top_k=10)
    assert first == second
    assert 0.0 < first[0][1] <= 1.0  # max-normalised


def test_personalization_mixes_query_and_seeds():
    nodes = ["a", "b", "c", "d"]
    pure_query = build_personalization(
        nodes,
        query_cosine={"a": 0.9, "b": 0.8, "c": 0.1, "d": 0.05},
        seed_weights={},
        query_mix=1.0,
        query_pool_size=2,
    )
    assert pure_query["a"] > 0 and pure_query["b"] > 0
    assert pure_query["c"] == 0.0 and pure_query["d"] == 0.0  # outside top-M
    assert sum(pure_query.values()) == pytest.approx(1.0)

    pure_seed = build_personalization(
        nodes, query_cosine=None, seed_weights={"a": 3.0, "b": 1.0}, query_mix=0.6, query_pool_size=2
    )
    assert pure_seed["a"] == pytest.approx(0.75)
    assert pure_seed["b"] == pytest.approx(0.25)
    assert pure_seed["c"] == 0.0

    with pytest.raises(ValueError):
        build_personalization(nodes, query_cosine=None, seed_weights={}, query_mix=0.6, query_pool_size=2)


def test_l5_zero_ppr_falls_back_to_seed_knn(monkeypatch):
    snap = _cluster_snapshot(n_per_cluster=20, clusters=3)
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=ImplicitGraphConfig())
    monkeypatch.setattr(implicit_graph, "pagerank_scores", lambda *a, **k: {})
    nx_graph = graph.to_networkx(seed_ids=["ck000"])
    ranked = graph.rank(nx_graph, query_cosine=None, seed_weights={"ck000": 1.0}, top_k=10)
    assert ranked  # L5 guarantees contribution
    assert ranked[0][1] == pytest.approx(1.0)  # max-normalised fallback
    assert all(chunk_id in graph.index for chunk_id, _ in ranked)
    assert "ck000" not in {chunk_id for chunk_id, _ in ranked[:3]}  # neighbours, not the seed itself


# --- tiers / cache / ladder ----------------------------------------------------


async def test_tier_split_full_pool_and_cache_invalidate():
    snap = _cluster_snapshot(n_per_cluster=20, clusters=3)  # 60 nodes
    cfg = ImplicitGraphConfig(max_corpus_nodes=3000)
    first = await _get_or_build_full_graph(kb_id="kbA", snapshot=snap, cfg=cfg)
    second = await _get_or_build_full_graph(kb_id="kbA", snapshot=snap, cfg=cfg)
    assert first is second  # cached by md5 of sorted ids
    assert first.tier == "full"

    invalidate("kbA")
    third = await _get_or_build_full_graph(kb_id="kbA", snapshot=snap, cfg=cfg)
    assert third is not first

    pool_cfg = ImplicitGraphConfig(max_corpus_nodes=10, pool_node_cap=32)
    pool = ImplicitChunkGraph.build_pool(
        kb_id="kbA", snapshot=snap, cfg=pool_cfg, seed_ids=["ck000"], query_vector=None
    )
    assert pool.tier == "pool"
    assert len(pool.ids) <= pool_cfg.pool_node_cap
    assert "ck000" in pool.index  # seeds always retained


async def test_retrieve_ladder_skipped_real_and_no_signal():
    empty = await retrieve_implicit_chunks(
        snapshot=None, seed_weights={}, analysis=_analysis(), cfg=ImplicitGraphConfig()
    )
    assert empty[0] == [] and empty[2].mode == "skipped"

    snap = _cluster_snapshot(n_per_cluster=10, clusters=2)
    real = await retrieve_implicit_chunks(
        snapshot=snap, seed_weights={}, analysis=_analysis(), cfg=ImplicitGraphConfig(), coverage=1.0
    )
    assert real[0] == [] and real[2].mode == "real"

    no_signal = await retrieve_implicit_chunks(
        snapshot=snap,
        seed_weights={"missing": 1.0},
        analysis=_analysis(),
        cfg=ImplicitGraphConfig(query_mix=1.0),  # needs query signal, none given
    )
    assert no_signal[0] == [] and no_signal[1] and "no seeds" in no_signal[1]

    produced = await retrieve_implicit_chunks(
        snapshot=snap,
        seed_weights={"ck000": 2.0},
        analysis=_analysis(),
        cfg=ImplicitGraphConfig(),
    )
    assert produced[1] is None and produced[0], "seeded retrieval must produce chunks"
    assert produced[2].mode == "implicit"
    top = produced[0][0]
    assert top["metadata"]["chunk_id"] in snap.ids
    assert top["implicit_score"] == pytest.approx(top["graph_score"])


async def test_timeout_and_exception_degrade_without_raising(monkeypatch):
    snap = _cluster_snapshot(n_per_cluster=10, clusters=2)

    def _slow(*args, **kwargs):
        time.sleep(0.05)
        return {"ck000": 1.0}

    monkeypatch.setattr(implicit_graph, "pagerank_scores", _slow)
    timed_out = await retrieve_implicit_chunks(
        snapshot=snap,
        seed_weights={"ck000": 1.0},
        analysis=_analysis(),
        cfg=ImplicitGraphConfig(timeout_ms=1),
    )
    assert timed_out[0] == [] and "timeout" in timed_out[1]

    def _boom(*args, **kwargs):
        raise RuntimeError("ppr exploded")

    monkeypatch.setattr(implicit_graph, "pagerank_scores", _boom)
    failed = await retrieve_implicit_chunks(
        snapshot=snap, seed_weights={"ck000": 1.0}, analysis=_analysis(), cfg=ImplicitGraphConfig()
    )
    assert failed[0] == [] and failed[1] and "implicit graph: " in failed[1]


async def test_hybrid_real_edges_mode_and_family():
    snap = _cluster_snapshot(n_per_cluster=10, clusters=2)
    real_edges = [("ck000", "ck015", 0.6), ("ck000", "missing", 0.6)]
    result = await retrieve_implicit_chunks(
        snapshot=snap,
        seed_weights={"ck000": 1.0},
        analysis=_analysis(),
        cfg=ImplicitGraphConfig(),
        real_edges=real_edges,
        coverage=0.5,
    )
    assert result[2].mode == "hybrid"
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=ImplicitGraphConfig())
    families = _families(graph, extra_edges=[("ck000", "ck015", 0.6)])
    assert families.get(("ck000", "ck015")) == "REAL"


async def test_file_filter_and_candidate_widening():
    snap = _snapshot(
        [f"target file content {i}" for i in range(10)] + [f"other file content {i}" for i in range(10)],
        file_ids=["f0"] * 10 + ["f1"] * 10,
    )
    result = await retrieve_implicit_chunks(
        snapshot=snap,
        seed_weights={"ck000": 1.0, "ck010": 1.0},
        analysis=_analysis(),
        cfg=ImplicitGraphConfig(),
        allowed_file_ids={"f0"},
        top_k=4,
    )
    assert result[0], "filter must not empty the channel when f0 chunks exist"
    assert all(chunk["metadata"]["file_id"] == "f0" for chunk in result[0])
    assert len(result[0]) <= 4


async def test_explanation_beam_produces_evidence_paths():
    snap = _cluster_snapshot(n_per_cluster=15, clusters=3)
    result = await retrieve_implicit_chunks(
        snapshot=snap,
        seed_weights={"ck000": 1.0},
        analysis=_analysis(),
        cfg=ImplicitGraphConfig(include_explanations=True),
        top_k=10,
    )
    debug = result[2]
    assert debug is not None and debug.paths, "explanations requested -> paths returned"
    path = debug.paths[0]
    assert path.hops[0].event_id == "ck000"
    assert path.hops[0].relation_to_prev == "向量命中种子"
    for hop in path.hops[1:]:
        assert hop.edge_type in {"SIMILAR", "ADJACENT", "LEXICAL", "EXACT_ID", "REAL"}
        assert hop.evidence_chunk_id == hop.event_id
        assert hop.evidence_quote  # points back to the source text


def test_explain_never_visits_a_node_twice():
    snap = _cluster_snapshot(n_per_cluster=15, clusters=3)
    graph = ImplicitChunkGraph.build_full(kb_id="kb", snapshot=snap, cfg=ImplicitGraphConfig())
    nx_graph = graph.to_networkx(seed_ids=["ck000"])
    paths = graph.explain(
        nx_graph,
        seed_weights={"ck000": 1.0},
        ppr_scores={chunk_id: 0.01 for chunk_id in graph.ids},
        query_cosine=None,
    )
    for path in paths:
        chunk_ids = [hop.event_id for hop in path.hops]
        assert len(chunk_ids) == len(set(chunk_ids))  # acyclic evidence path
