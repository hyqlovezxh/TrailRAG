"""Query-time implicit chunk graph: dynamic graph construction + PPR multi-hop.

Contract source: implementation plan ``stellar-vortex-darwin-pdXOb18c`` (S2).
When the entity/event knowledge graph has not finished building (graph coverage
below threshold), the graph retrieval channel does not simply return empty:
it falls back to an *implicit* graph over chunks built from structural
evidence at query time, then runs personalised PageRank over it:

- SIM: mutual-KNN over the flushed vector snapshot (hubness-proof by
  construction -- mutual adjacency caps every node's degree at ``knn_k``);
- ADJACENT: same-file neighbours within ``adjacent_window`` chunk indices;
- LEXICAL: co-occurrence of query terms (matched at query time);
- EXACT_ID: chunks sharing an exact identifier token (phone / account / ...).

PPR provides candidate ranking (global structure, robust to seed noise); a
deterministic beam pass over the same graph produces ``HopExplanation``
evidence paths on demand, so the fallback keeps the evidence-traceable
property of the explicit graph channel.

Tiering: corpora with ``N <= max_corpus_nodes`` build the full mutual-KNN
graph once and cache it by the md5 of sorted chunk ids (never ``hash()`` --
PYTHONHASHSEED would change it across processes). Larger corpora build a
query-scoped anchor pool (seeds + their kNN + query neighbours), uncached.

Degradation ladder (never raises to the caller):
L0 skipped (no snapshot / toggle off upstream) -> L1 cached graph -> L2
uncached full build -> L3 anchor pool -> L4 timeout/exception -> error string
-> L5 zero PPR falls back to the seeds' 1-hop kNN neighbours.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from yusu_kb.knowledge.graphs.event_schemas import HopExplanation, MultiHopPath
from yusu_kb.knowledge.graphs.ppr import pagerank_scores
from yusu_kb.knowledge.retrieval.query_analysis import QueryAnalysis
from yusu_kb.storage.vector_store import VectorSnapshot
from yusu_kb.utils.logger import logger

# Hard cap for the full-corpus graph regardless of configuration: the mutual
# KNN build allocates an (N, N) similarity matrix, and beyond this size the
# memory cost outweighs the latency win -- such corpora always take the
# query-scoped anchor pool path (Tier B).
_ABSOLUTE_FULL_CAP = 6000
# Cache bound: at most this many implicit corpus graphs are kept in memory.
_CACHE_MAX_ENTRIES = 8
# Self-similarity sentinel on the diagonal (float32-safe, below any cosine).
_SELF_SIM = np.float32(-1e9)
# Edge-family priority for explanations: a pair connected by several evidence
# types is explained by the strongest (lowest value = highest priority).
_FAMILY_PRIORITY = {"EXACT_ID": 0, "LEXICAL": 1, "ADJACENT": 2, "SIMILAR": 3, "REAL": 4}

# Beam explanation constants (mirrors multi_hop.py scoring shape, with the
# value_weight term replaced by the PPR score -- the implicit graph's analogue).
_EXPLAIN_ALPHA = 0.5  # query cosine of the candidate chunk
_EXPLAIN_BETA = 0.3  # edge weight (transition strength)
_EXPLAIN_GAMMA = 0.2  # PPR score of the candidate chunk
_PPR_ZERO_EPS = 1e-10  # L5 trigger: PPR mass below this counts as unusable


@dataclass(frozen=True)
class ImplicitGraphConfig:
    """All tunables of the implicit graph fallback (query params -> frozen)."""

    knn_k: int = 12
    sim_threshold: float = 0.55
    sim_exponent: float = 1.5
    mutual_knn: bool = True
    weight_adjacent: float = 0.5
    adjacent_window: int = 3
    weight_lexical: float = 0.4
    weight_exact_id: float = 0.9
    lexical_max_df_ratio: float = 0.2
    lexical_clique_cap: int = 32
    damping: float = 0.80
    max_iter: int = 100
    tol: float = 1e-8
    query_mix: float = 0.6
    query_pool_size: int = 64
    max_corpus_nodes: int = 3000
    pool_node_cap: int = 512
    timeout_ms: int = 400
    hybrid_real_weight: float = 0.6
    include_explanations: bool = False
    explain_max_hops: int = 3
    explain_beam_width: int = 4
    explain_top_paths: int = 3

    @classmethod
    def from_query_params(cls, params: Mapping[str, Any]) -> ImplicitGraphConfig:
        """Map the ``implicit_*`` query params onto the frozen config, clamped.

        Missing/empty values fall back to defaults, but an explicit ``0`` is
        honoured -- the edge-family disable flags rely on zero weights, and a
        plain ``or`` fallback would silently rewrite ``0.0`` to the default.
        """

        def _num(key: str, default: float) -> float:
            raw = params.get(key)
            if raw is None or raw == "":
                return default
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default

        def _int(key: str, default: int) -> int:
            return int(_num(key, float(default)))

        return cls(
            knn_k=max(_int("implicit_knn_k", 12), 1),
            sim_threshold=min(max(_num("implicit_sim_threshold", 0.55), 0.0), 0.99),
            sim_exponent=max(_num("implicit_sim_exponent", 1.5), 0.0),
            mutual_knn=bool(params.get("implicit_mutual_knn", True)),
            weight_adjacent=max(_num("implicit_weight_adjacent", 0.5), 0.0),
            adjacent_window=max(_int("implicit_adjacent_window", 3), 1),
            weight_lexical=max(_num("implicit_weight_lexical", 0.4), 0.0),
            weight_exact_id=max(_num("implicit_weight_exact_id", 0.9), 0.0),
            lexical_max_df_ratio=min(max(_num("implicit_lexical_max_df_ratio", 0.2), 0.0), 1.0),
            damping=min(max(_num("implicit_damping", 0.80), 0.1), 0.99),
            max_iter=max(_int("implicit_max_iter", 100), 1),
            tol=max(_num("implicit_tol", 1e-8), 1e-12),
            query_mix=min(max(_num("implicit_query_mix", 0.6), 0.0), 1.0),
            query_pool_size=max(_int("implicit_query_pool_size", 64), 1),
            max_corpus_nodes=max(_int("implicit_max_corpus_nodes", 3000), 10),
            timeout_ms=max(_int("implicit_timeout_ms", 400), 1),
            hybrid_real_weight=max(_num("implicit_hybrid_real_weight", 0.6), 0.0),
            include_explanations=bool(params.get("implicit_include_explanations", False)),
        )

    def build_signature(self) -> str:
        """Stable fingerprint of every field that shapes the static graph.

        Part of the cache key: reusing a cached corpus graph built with
        different KNN/threshold parameters would silently serve stale
        structure, so parameter changes must produce a new key.
        """
        shaping = (
            self.knn_k,
            self.sim_threshold,
            self.sim_exponent,
            self.mutual_knn,
            self.weight_adjacent,
            self.adjacent_window,
        )
        return "\x1f".join(str(value) for value in shaping)


@dataclass(frozen=True)
class ImplicitGraphDebug:
    """Observability payload of one implicit-graph retrieval."""

    mode: str  # "implicit" | "hybrid" | "real" | "skipped"
    nodes: int = 0
    edges: int = 0
    hub_index: float = 0.0
    build_ms: float = 0.0
    ppr_ms: float = 0.0
    tier: str = ""  # "full" | "pool" | ""
    paths: tuple[MultiHopPath, ...] = field(default_factory=tuple)


def _sim_edge_weight(similarity: float, threshold: float, exponent: float) -> float:
    """Rescale a cosine above the threshold into a positive transition weight.

    ``(s - tau) / (1 - tau)`` maps [tau, 1] -> [0, 1]; the exponent pushes the
    long low-similarity tail further down so weak edges cannot dominate the
    per-node transition mass once networkx row-normalises the graph.
    """
    if similarity <= threshold:
        return 0.0
    return ((similarity - threshold) / (1.0 - threshold)) ** exponent


def build_personalization(
    node_ids: Sequence[str],
    *,
    query_cosine: Mapping[str, float] | None,
    seed_weights: Mapping[str, float],
    query_mix: float,
    query_pool_size: int,
) -> dict[str, float]:
    """Personalisation vector: query direct-check mixed with seed evidence.

    ``p[v] \\propto query_mix * softmax_topM(cos(q, v)) + (1 - query_mix) * seed_weights[v]``

    Every node receives an explicit entry (networkx silently defaults missing
    keys to 1.0, which would corrupt the reset distribution). Raises when the
    mixture is degenerate (no query signal AND no seeds) -- callers degrade.
    """
    query_mix = min(max(query_mix, 0.0), 1.0)
    seed_component = (1.0 - query_mix) if query_cosine else 1.0
    query_component = query_mix if query_cosine else 0.0

    seeds_total = sum(w for w in seed_weights.values() if w > 0.0)
    if seed_component > 0.0 and seeds_total <= 0.0:
        # Seed mass is zero: fold everything into the query component.
        query_component += seed_component
        seed_component = 0.0
    if query_component > 0.0 and not query_cosine:
        query_component = 0.0
    if seed_component <= 0.0 and query_component <= 0.0:
        raise ValueError("degenerate personalization: no seeds and no query signal")

    personalization = dict.fromkeys(node_ids, 0.0)

    if seed_component > 0.0 and seeds_total > 0.0:
        for node_id, weight in seed_weights.items():
            if weight > 0.0 and node_id in personalization:
                personalization[node_id] += seed_component * (weight / seeds_total)

    if query_component > 0.0 and query_cosine:
        top = sorted(
            ((node_id, float(sim)) for node_id, sim in query_cosine.items() if node_id in personalization),
            key=lambda item: (-item[1], item[0]),
        )[: max(query_pool_size, 1)]
        if top:
            peak = max(sim for _, sim in top)
            exps = {node_id: math.exp(max(sim - peak, -60.0)) for node_id, sim in top}
            total = sum(exps.values())
            if total > 0.0:
                for node_id, exp_value in exps.items():
                    personalization[node_id] += query_component * (exp_value / total)
    return personalization


class ImplicitChunkGraph:
    """Precomputed static structure (SIM + ADJACENT) over a chunk corpus.

    Instances are immutable after construction and, for Tier A corpora, cached
    across queries. Per-query evidence (LEXICAL / EXACT_ID / hybrid REAL
    edges) is materialised into a ``networkx.Graph`` by :meth:`to_networkx`.
    """

    def __init__(
        self,
        *,
        kb_id: str,
        cfg: ImplicitGraphConfig,
        ids: list[str],
        matrix: np.ndarray,
        metas: list[dict[str, Any]],
        sim_edges: list[tuple[int, int, float]],
        adjacent_edges: list[tuple[int, int, float]],
        hub_index: float,
        tier: str,
    ) -> None:
        self.kb_id = kb_id
        self.cfg = cfg
        self.ids = ids
        self.index = {chunk_id: row for row, chunk_id in enumerate(ids)}
        self.matrix = matrix
        self.metas = metas
        self.contents_lower = [str(meta.get("content") or "").lower() for meta in metas]
        self.sim_edges = sim_edges
        self.adjacent_edges = adjacent_edges
        self.hub_index = hub_index
        self.tier = tier
        self.static_edge_count = len(sim_edges) + len(adjacent_edges)

    # -- construction ------------------------------------------------------

    @classmethod
    def build_full(
        cls, *, kb_id: str, snapshot: VectorSnapshot, cfg: ImplicitGraphConfig
    ) -> ImplicitChunkGraph:
        """Tier A: mutual-KNN over the whole flushed corpus."""
        ids = list(snapshot.ids)
        matrix = np.ascontiguousarray(snapshot.matrix, dtype=np.float32)
        metas = list(snapshot.metas)
        sim_edges = cls._mutual_knn_edges(matrix, cfg)
        adjacent_edges = cls._adjacent_edges(metas, cfg)
        return cls(
            kb_id=kb_id,
            cfg=cfg,
            ids=ids,
            matrix=matrix,
            metas=metas,
            sim_edges=sim_edges,
            adjacent_edges=adjacent_edges,
            hub_index=cls._hub_index(sim_edges),
            tier="full",
        )

    @classmethod
    def build_pool(
        cls,
        *,
        kb_id: str,
        snapshot: VectorSnapshot,
        cfg: ImplicitGraphConfig,
        seed_ids: Sequence[str],
        query_vector: Sequence[float] | None,
    ) -> ImplicitChunkGraph:
        """Tier B: query-scoped anchor pool (seeds + kNN + query neighbours)."""
        ids = list(snapshot.ids)
        full_matrix = np.ascontiguousarray(snapshot.matrix, dtype=np.float32)
        metas = list(snapshot.metas)
        index = {chunk_id: row for row, chunk_id in enumerate(ids)}

        anchor_rows = sorted({index[seed_id] for seed_id in seed_ids if seed_id in index})
        pool: set[int] = set(anchor_rows)

        # One matvec per anchor (O(S*N)) instead of the full O(N^2) KNN.
        if anchor_rows:
            anchor_sims = full_matrix[anchor_rows] @ full_matrix.T
            k = min(cfg.knn_k, len(ids) - 1)
            if k > 0:
                for local_row in range(anchor_sims.shape[0]):
                    top = np.argpartition(-anchor_sims[local_row], k - 1)[: k + 1]
                    pool.update(int(neighbour) for neighbour in top.tolist())
        if query_vector is not None:
            q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(q))
            if norm > 0.0 and full_matrix.ndim == 2 and full_matrix.shape[1] == q.shape[0]:
                q_sims = full_matrix @ (q / norm)
                top_m = min(cfg.query_pool_size, len(ids))
                pool.update(int(neighbour) for neighbour in np.argpartition(-q_sims, top_m - 1)[:top_m].tolist())

        pool_rows = cls._bounded_pool_rows(pool, anchor_rows, cfg.pool_node_cap)
        if not pool_rows:
            pool_rows = list(range(min(len(ids), max(cfg.pool_node_cap, 1))))

        sub_matrix = np.ascontiguousarray(full_matrix[pool_rows])
        sub_ids = [ids[row] for row in pool_rows]
        sub_metas = [metas[row] for row in pool_rows]
        sim_edges = cls._mutual_knn_edges(sub_matrix, cfg)
        adjacent_edges = cls._adjacent_edges(sub_metas, cfg)
        return cls(
            kb_id=kb_id,
            cfg=cfg,
            ids=sub_ids,
            matrix=sub_matrix,
            metas=sub_metas,
            sim_edges=sim_edges,
            adjacent_edges=adjacent_edges,
            hub_index=cls._hub_index(sim_edges),
            tier="pool",
        )

    @staticmethod
    def _bounded_pool_rows(
        pool: set[int], anchor_rows: Sequence[int], pool_node_cap: int
    ) -> list[int]:
        """Truncate the pool to ``pool_node_cap`` while keeping every seed."""
        cap = max(pool_node_cap, 1)
        if len(pool) <= cap:
            return sorted(pool)
        anchor_set = set(anchor_rows)
        fill = [row for row in sorted(pool) if row not in anchor_set][: max(cap - len(anchor_set), 0)]
        return sorted(anchor_set | set(fill))

    @staticmethod
    def _mutual_knn_edges(matrix: np.ndarray, cfg: ImplicitGraphConfig) -> list[tuple[int, int, float]]:
        """Symmetric mutual-KNN SIM edges with thresholded, reshaped weights."""
        node_count = matrix.shape[0]
        if node_count < 2:
            return []
        sims = matrix @ matrix.T
        np.fill_diagonal(sims, _SELF_SIM)
        k = min(cfg.knn_k, node_count - 1)
        neighbours = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        neighbour_sets = [set(row.tolist()) for row in neighbours]
        edges: list[tuple[int, int, float]] = []
        for row in range(node_count):
            for col in neighbours[row].tolist():
                if col <= row:
                    continue  # emit each undirected pair once, from its lower index
                if cfg.mutual_knn and row not in neighbour_sets[col]:
                    continue
                weight = _sim_edge_weight(float(sims[row, col]), cfg.sim_threshold, cfg.sim_exponent)
                if weight > 0.0:
                    edges.append((row, int(col), weight))
        return edges

    @staticmethod
    def _adjacent_edges(metas: Sequence[Mapping[str, Any]], cfg: ImplicitGraphConfig) -> list[tuple[int, int, float]]:
        """Same-file edges between chunks ``1..adjacent_window`` indices apart."""
        by_file: dict[str, list[tuple[int, int]]] = {}
        for row, meta in enumerate(metas):
            file_id = str(meta.get("file_id") or "")
            chunk_index = meta.get("chunk_index")
            if file_id and chunk_index is not None:
                by_file.setdefault(file_id, []).append((int(chunk_index), row))
        edges: list[tuple[int, int, float]] = []
        for file_id in sorted(by_file):
            ordered = sorted(by_file[file_id])
            for pos, (chunk_index, row) in enumerate(ordered):
                for next_pos in range(pos + 1, min(pos + 1 + cfg.adjacent_window, len(ordered))):
                    delta_index, other_row = ordered[next_pos]
                    delta = delta_index - chunk_index
                    if 1 <= delta <= cfg.adjacent_window:
                        edges.append((row, other_row, cfg.weight_adjacent / float(delta)))
        return edges

    @staticmethod
    def _hub_index(sim_edges: Sequence[tuple[int, int, float]]) -> float:
        """max degree / edge count on the SIM subgraph (hubness indicator)."""
        if not sim_edges:
            return 0.0
        degree: dict[int, int] = {}
        for row, col, _ in sim_edges:
            degree[row] = degree.get(row, 0) + 1
            degree[col] = degree.get(col, 0) + 1
        return max(degree.values()) / float(len(sim_edges))

    # -- query-time materialisation ----------------------------------------

    def to_networkx(
        self,
        *,
        seed_ids: Sequence[str] = (),
        query_terms: Sequence[str] = (),
        exact_tokens: Sequence[str] = (),
        extra_edges: Iterable[tuple[str, str, float]] = (),
    ) -> nx.Graph:
        """Materialise the weighted undirected graph for one query.

        Static SIM/ADJACENT edges are precomputed; LEXICAL/EXACT_ID edges are
        instantiated only for this query's terms/tokens; ``extra_edges`` carry
        hybrid real-graph evidence (already scaled by the caller). networkx
        row-normalises the weighted adjacency internally during PageRank, so
        per-node weights act as transition probabilities. Every edge records
        its evidence ``family`` for the explanation beam.
        """
        graph = nx.Graph()
        graph.add_nodes_from(self.ids)

        def add_edge(row_u: int, row_v: int, weight: float, family: str) -> None:
            if weight <= 0.0 or row_u == row_v:
                return
            u, v = self.ids[row_u], self.ids[row_v]
            if graph.has_edge(u, v):
                existing = graph[u][v]
                if weight > float(existing.get("weight") or 0.0):
                    existing["weight"] = float(weight)
                if _FAMILY_PRIORITY[family] < _FAMILY_PRIORITY.get(str(existing.get("family")), 99):
                    existing["family"] = family
                return
            graph.add_edge(u, v, weight=float(weight), family=family)

        for row_u, row_v, weight in self.sim_edges:
            add_edge(row_u, row_v, weight, "SIMILAR")
        for row_u, row_v, weight in self.adjacent_edges:
            add_edge(row_u, row_v, weight, "ADJACENT")

        seed_rows = {self.index[seed_id] for seed_id in seed_ids if seed_id in self.index}
        for term in query_terms:
            self._add_term_edges(graph, str(term).lower(), add_edge, seed_rows, exact=False)
        for token in exact_tokens:
            self._add_term_edges(graph, str(token).lower(), add_edge, seed_rows, exact=True)

        for u, v, weight in extra_edges:
            if u != v and u in self.index and v in self.index:
                add_edge(self.index[u], self.index[v], float(weight), "REAL")
        return graph

    def _term_matches(self, term: str) -> list[int]:
        """Rows whose content contains ``term`` (deterministic index order)."""
        if not term:
            return []
        return [row for row, content in enumerate(self.contents_lower) if term in content]

    def _add_term_edges(
        self,
        graph: nx.Graph,
        term: str,
        add_edge: Any,
        seed_rows: set[int],
        *,
        exact: bool,
    ) -> None:
        """Connect chunks sharing one query term.

        Small match sets form a clique; large ones connect only to seed rows
        (a star anchored on query evidence) so the edge count stays bounded.
        High-df common terms are dropped for LEXICAL (they carry no signal);
        EXACT_ID tokens are hard links and exempt from the df filter.
        """
        cfg = self.cfg
        matches = self._term_matches(term)
        df = len(matches)
        if df < 2:
            return
        family = "EXACT_ID" if exact else "LEXICAL"
        if exact:
            weight = cfg.weight_exact_id
        else:
            if df / max(len(self.ids), 1) > cfg.lexical_max_df_ratio:
                return
            weight = (
                cfg.weight_lexical
                * (math.log1p(len(self.ids) / df) / math.log1p(len(self.ids)))
                * min(1.0, 8.0 / df)
            )
        if weight <= 0.0:
            return
        if df <= cfg.lexical_clique_cap:
            for pos, row_u in enumerate(matches):
                for row_v in matches[pos + 1 :]:
                    add_edge(row_u, row_v, weight, family)
            return
        for row_u in (row for row in matches if row in seed_rows):
            for row_v in matches:
                if row_u != row_v:
                    add_edge(row_u, row_v, weight, family)

    # -- ranking -------------------------------------------------------------

    def query_cosine(self, query_vector: Sequence[float] | None) -> dict[str, float] | None:
        """Cosine of the query against every chunk (None when unavailable)."""
        if query_vector is None:
            return None
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.size == 0 or self.matrix.ndim != 2 or q.size != self.matrix.shape[1]:
            return None
        norm = float(np.linalg.norm(q))
        if norm <= 0.0:
            return None
        sims = self.matrix @ (q / norm)
        return {chunk_id: float(sim) for chunk_id, sim in zip(self.ids, sims, strict=True)}

    def rank(
        self,
        graph: nx.Graph,
        *,
        query_cosine: Mapping[str, float] | None,
        seed_weights: Mapping[str, float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Personalised PageRank over the implicit graph, max-normalised.

        L5 guarantee: when the PPR outcome is numerically unusable (all-zero /
        non-finite), fall back to the seeds' 1-hop kNN neighbours so the
        channel still contributes instead of returning empty.
        """
        personalization = build_personalization(
            list(graph.nodes),
            query_cosine=query_cosine,
            seed_weights=seed_weights,
            query_mix=self.cfg.query_mix,
            query_pool_size=self.cfg.query_pool_size,
        )
        scores = pagerank_scores(
            graph,
            personalization,
            damping=self.cfg.damping,
            max_iter=self.cfg.max_iter,
            tol=self.cfg.tol,
        )
        ranked = self._validated_scores(scores, top_k)
        if ranked is None:
            ranked = self._seed_knn_fallback(seed_weights, top_k)
        peak = ranked[0][1] if ranked else 0.0
        if peak > 0.0:
            ranked = [(chunk_id, score / peak) for chunk_id, score in ranked]
        return ranked

    def _validated_scores(
        self, scores: Mapping[str, float], top_k: int
    ) -> list[tuple[str, float]] | None:
        """Sorted PPR scores, or None when numerically unusable (L5 trigger)."""
        if not scores:
            return None
        values = [float(v) for v in scores.values()]
        if not values or not all(math.isfinite(v) for v in values) or max(values) < _PPR_ZERO_EPS:
            return None
        ranked = sorted(
            ((chunk_id, float(score)) for chunk_id, score in scores.items() if chunk_id in self.index),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[: max(top_k, 1)]

    def _seed_knn_fallback(
        self, seed_weights: Mapping[str, float], top_k: int
    ) -> list[tuple[str, float]]:
        """L5: seeds' 1-hop kNN neighbours ranked by cosine (never empty)."""
        neighbour_scores: dict[str, float] = {}
        if len(self.ids) < 2:
            # Degenerate single-node corpus: the seed itself is the only
            # possible contribution and argpartition would be out of bounds.
            return [
                (chunk_id, 1.0)
                for chunk_id in sorted(seed_weights, key=lambda cid: (-seed_weights[cid], cid))
                if chunk_id in self.index
            ][: max(top_k, 1)]
        k = min(self.cfg.knn_k, len(self.ids) - 1)
        for seed_id, seed_weight in seed_weights.items():
            row = self.index.get(seed_id)
            if row is None or seed_weight <= 0.0:
                continue
            sims = self.matrix @ self.matrix[row]
            top = np.argpartition(-sims, k)[: k + 1]
            for neighbour in top.tolist():
                if int(neighbour) == row:
                    continue  # never propagate the seed's self-similarity
                chunk_id = self.ids[int(neighbour)]
                score = float(sims[int(neighbour)])
                if score > neighbour_scores.get(chunk_id, 0.0):
                    neighbour_scores[chunk_id] = score
            if not neighbour_scores:
                neighbour_scores[seed_id] = max(neighbour_scores.get(seed_id, 0.0), 1.0)
        ranked = sorted(neighbour_scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[: max(top_k, 1)]

    # -- evidence paths (on demand) ------------------------------------------

    def explain(
        self,
        graph: nx.Graph,
        *,
        seed_weights: Mapping[str, float],
        ppr_scores: Mapping[str, float],
        query_cosine: Mapping[str, float] | None,
    ) -> tuple[MultiHopPath, ...]:
        """Deterministic beam over the implicit graph producing HopExplations.

        Scoring mirrors ``multi_hop.run_beam_search`` (alpha * sim + beta *
        edge_strength + gamma * value_weight) with the value_weight term
        replaced by the PPR score -- the implicit graph has no event
        value_weight, and the PPR score is exactly its analogue. Paths that
        cannot extend are retained (dead ends still explain the seeds).
        """
        beam_width = max(self.cfg.explain_beam_width, 1)
        scored_seeds = sorted(
            ((chunk_id, weight) for chunk_id, weight in seed_weights.items() if chunk_id in self.index),
            key=lambda item: (-item[1], item[0]),
        )
        beam: list[tuple[list[str], float]] = [([seed_id], 0.0) for seed_id, _ in scored_seeds[:beam_width]]
        if not beam:
            return ()

        for _ in range(max(self.cfg.explain_max_hops, 1)):
            extended: list[tuple[list[str], float]] = []
            for path, path_score in beam:
                current = path[-1]
                candidates: list[tuple[float, str, str]] = []
                for neighbour in graph.neighbors(current):
                    if neighbour in path:
                        continue
                    edge = graph[current][neighbour]
                    edge_weight = float(edge.get("weight") or 0.0)
                    sim = float((query_cosine or {}).get(neighbour, 0.0))
                    ppr = float(ppr_scores.get(neighbour, 0.0))
                    hop_score = _EXPLAIN_ALPHA * sim + _EXPLAIN_BETA * edge_weight + _EXPLAIN_GAMMA * ppr
                    candidates.append((hop_score, neighbour, str(edge.get("family"))))
                candidates.sort(key=lambda item: (-item[0], item[1]))
                if not candidates:
                    extended.append((path, path_score))  # dead end: keep for ranking
                    continue
                for hop_score, neighbour, _family in candidates[:beam_width]:
                    extended.append(([*path, neighbour], path_score + hop_score))
            if not extended:
                break
            extended.sort(key=lambda item: (-item[1], item[0][0], len(item[0])))
            beam = extended[:beam_width]

        return self._paths_from_beam([path for path, _ in beam], graph, ppr_scores)

    def _paths_from_beam(
        self, beam: list[list[str]], graph: nx.Graph, ppr_scores: Mapping[str, float]
    ) -> tuple[MultiHopPath, ...]:
        """Convert beam chunk-id paths into MultiHopPath evidence objects."""
        paths: list[MultiHopPath] = []
        for path in beam[: max(self.cfg.explain_top_paths, 1)]:
            hops: list[HopExplanation] = []
            for hop_index, chunk_id in enumerate(path):
                row = self.index.get(chunk_id)
                content = str(self.metas[row].get("content") or "") if row is not None else ""
                if hop_index == 0:
                    family, relation = "SIMILAR", "向量命中种子"
                else:
                    edge = graph.get_edge_data(path[hop_index - 1], chunk_id) or {}
                    family = str(edge.get("family") or "SIMILAR")
                    relation = f"隐式图边 {family}"
                hops.append(
                    HopExplanation(
                        hop_index=hop_index,
                        event_id=chunk_id,
                        summary=content[:120],
                        edge_type=family,  # type: ignore[arg-type] - family comes from the same Literal set
                        relation_to_prev=relation,
                        evidence_chunk_id=chunk_id,
                        evidence_quote=content[:80],
                    )
                )
            if hops:
                paths.append(
                    MultiHopPath(
                        hops=hops,
                        coherence_score=float(ppr_scores.get(path[-1], 0.0)),
                        path_precision=1.0,
                    )
                )
        return tuple(paths)


# -- corpus graph cache -------------------------------------------------------

_GRAPH_CACHE: OrderedDict[str, tuple[str, ImplicitChunkGraph]] = OrderedDict()
_CACHE_LOCK = asyncio.Lock()


def _cache_key(snapshot_ids: Sequence[str], cfg: ImplicitGraphConfig) -> str:
    """Stable cache key: md5 over sorted chunk ids + build signature.

    Chunk ids capture the index version (never Python ``hash()`` -- 
    PYTHONHASHSEED would change it across processes); the config signature
    guards against reusing a graph built with different KNN/threshold params.
    """
    payload = "\x00".join(sorted(snapshot_ids)) + "\x1e" + cfg.build_signature()
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


async def _get_or_build_full_graph(
    *, kb_id: str, snapshot: VectorSnapshot, cfg: ImplicitGraphConfig
) -> ImplicitChunkGraph:
    """Tier A cache lookup / build (double-checked under one asyncio lock)."""
    key = _cache_key(snapshot.ids, cfg)
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        _GRAPH_CACHE.move_to_end(key)
        return cached[1]
    async with _CACHE_LOCK:
        cached = _GRAPH_CACHE.get(key)
        if cached is not None:
            _GRAPH_CACHE.move_to_end(key)
            return cached[1]
        graph = await asyncio.to_thread(
            ImplicitChunkGraph.build_full, kb_id=kb_id, snapshot=snapshot, cfg=cfg
        )
        _GRAPH_CACHE[key] = (kb_id, graph)
        while len(_GRAPH_CACHE) > _CACHE_MAX_ENTRIES:
            _GRAPH_CACHE.popitem(last=False)
        return graph


def invalidate(kb_id: str) -> None:
    """Drop cached corpus graphs of one knowledge base (delete / reindex hook).

    The cache key derives from chunk ids, so mutations naturally change the
    key; this only reclaims memory for the deleted KB's stale entries.
    """
    for key in [key for key, (_, graph) in _GRAPH_CACHE.items() if graph.kb_id == kb_id]:
        _GRAPH_CACHE.pop(key, None)


def _rank_core(
    graph_obj: ImplicitChunkGraph,
    *,
    seed_weights: dict[str, float],
    query_embedding: Sequence[float] | None,
    analysis: QueryAnalysis,
    real_edges: list[tuple[str, str, float]],
    top_k: int,
) -> tuple[list[tuple[str, float]], dict[str, Any]]:
    """Blocking core: materialise the query graph and run PPR (thread offloaded)."""
    started = time.perf_counter()
    query_cosine = graph_obj.query_cosine(query_embedding)
    nx_graph = graph_obj.to_networkx(
        seed_ids=list(seed_weights),
        query_terms=analysis.scoring_terms,
        exact_tokens=analysis.exact_tokens,
        extra_edges=real_edges,
    )
    build_ms = (time.perf_counter() - started) * 1000.0

    ppr_started = time.perf_counter()
    ranked = graph_obj.rank(
        nx_graph,
        query_cosine=query_cosine,
        seed_weights=seed_weights,
        top_k=top_k,
    )
    ppr_ms = (time.perf_counter() - ppr_started) * 1000.0

    paths: tuple[MultiHopPath, ...] = ()
    if graph_obj.cfg.include_explanations and ranked:
        paths = graph_obj.explain(
            nx_graph,
            seed_weights=seed_weights,
            ppr_scores=dict(ranked),
            query_cosine=query_cosine,
        )
    stats = {
        "nodes": nx_graph.number_of_nodes(),
        "edges": nx_graph.number_of_edges(),
        "hub_index": graph_obj.hub_index,
        "build_ms": build_ms,
        "ppr_ms": ppr_ms,
        "paths": paths,
    }
    return ranked, stats


async def retrieve_implicit_chunks(
    *,
    snapshot: VectorSnapshot | None,
    seed_weights: dict[str, float],
    analysis: QueryAnalysis,
    cfg: ImplicitGraphConfig,
    kb_id: str = "",
    query_embedding: Sequence[float] | None = None,
    real_edges: Sequence[tuple[str, str, float]] = (),
    coverage: float = 0.0,
    allowed_file_ids: set[str] | None = None,
    top_k: int = 20,
) -> tuple[list[dict[str, Any]], str | None, ImplicitGraphDebug | None]:
    """Pure entry point of the implicit-graph fallback (full degrade ladder).

    Returns ``(chunks, error, debug)`` and never raises: timeout / exception
    degrade to an error string the caller logs, mirroring the explicit graph
    channel's ``(chunks, error)`` convention. Chunk dicts reuse the explicit
    graph channel's assembly shape so the RRF fusion needs no adaptation.
    """
    started = time.perf_counter()
    if snapshot is None or not snapshot.ids:
        return [], None, ImplicitGraphDebug(mode="skipped")
    if coverage >= 0.999:
        # Fully built corpus: the legacy real-graph path owns retrieval.
        return [], None, ImplicitGraphDebug(mode="real")

    snapshot_ids = set(snapshot.ids)
    seeds = {
        chunk_id: float(weight)
        for chunk_id, weight in seed_weights.items()
        if float(weight) > 0.0 and chunk_id in snapshot_ids
    }
    seeds_total = sum(seeds.values())
    if seeds_total > 0.0:
        seeds = {chunk_id: weight / seeds_total for chunk_id, weight in seeds.items()}
    has_query_signal = query_embedding is not None and cfg.query_mix > 0.0
    if not seeds and not has_query_signal:
        return [], "implicit graph: no seeds and no query signal", ImplicitGraphDebug(mode="skipped")

    try:
        node_count = len(snapshot_ids)
        use_full_tier = node_count <= min(cfg.max_corpus_nodes, _ABSOLUTE_FULL_CAP)
        if use_full_tier:
            graph_obj = await _get_or_build_full_graph(kb_id=kb_id, snapshot=snapshot, cfg=cfg)
            build_ms = (time.perf_counter() - started) * 1000.0
        else:
            build_started = time.perf_counter()
            graph_obj = await asyncio.to_thread(
                ImplicitChunkGraph.build_pool,
                kb_id=kb_id,
                snapshot=snapshot,
                cfg=cfg,
                seed_ids=list(seeds),
                query_vector=query_embedding,
            )
            build_ms = (time.perf_counter() - build_started) * 1000.0

        # Mirror the legacy graph channel: widen the candidate pool when a
        # file filter will cut results afterwards.
        rank_top_k = top_k * 3 if allowed_file_ids is not None else top_k
        ranked, stats = await asyncio.wait_for(
            asyncio.to_thread(
                _rank_core,
                graph_obj,
                seed_weights=seeds,
                query_embedding=query_embedding,
                analysis=analysis,
                real_edges=list(real_edges),
                top_k=rank_top_k,
            ),
            timeout=cfg.timeout_ms / 1000.0,
        )

        mode = "hybrid" if real_edges else "implicit"
        debug = ImplicitGraphDebug(
            mode=mode,
            nodes=int(stats["nodes"]),
            edges=int(stats["edges"]),
            hub_index=float(stats["hub_index"]),
            build_ms=round(build_ms, 3),
            ppr_ms=round(float(stats["ppr_ms"]), 3),
            tier=graph_obj.tier,
            paths=stats["paths"],
        )

        chunks: list[dict[str, Any]] = []
        for chunk_id, score in ranked:
            row = graph_obj.index.get(chunk_id)
            if row is None:
                continue
            meta = graph_obj.metas[row]
            if allowed_file_ids is not None and str(meta.get("file_id") or "") not in allowed_file_ids:
                continue
            chunks.append(
                {
                    "content": str(meta.get("content") or ""),
                    "metadata": {
                        "source": "未知来源",
                        "chunk_id": chunk_id,
                        "file_id": meta.get("file_id"),
                        "chunk_index": meta.get("chunk_index"),
                    },
                    "score": float(score),
                    "graph_score": float(score),
                    "implicit_score": float(score),
                }
            )
            if len(chunks) >= max(top_k, 1):
                break
        return chunks, None, debug
    except TimeoutError:
        logger.warning(f"Implicit graph retrieval timed out for kb={kb_id} after {cfg.timeout_ms}ms")
        return [], f"implicit graph: timeout after {cfg.timeout_ms}ms", ImplicitGraphDebug(mode="skipped")
    except Exception as exc:  # noqa: BLE001 - 回退通道任何异常都必须降级而非中断主检索
        logger.warning(f"Implicit graph retrieval failed for kb={kb_id}, degrading: {exc}")
        return [], f"implicit graph: {exc}", ImplicitGraphDebug(mode="skipped")
