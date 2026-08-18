"""Personalized PageRank (PPR) retrieval over the NetworkX graph storage.

Replaces the igraph/Neo4j PPR of the source project with ``networkx``: a
weighted undirected graph over entity and chunk nodes is built from the
storage, then ``networkx.pagerank`` runs with a personalization vector that
explicitly zeroes non-seed nodes (nx defaults missing keys to 1.0, unlike
igraph's uniform reset). When pagerank yields no usable scores, retrieval
falls back to the storage's 2-hop then 1-hop seed lookups.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage
from yusu_kb.utils.logger import logger

_RELATION_WEIGHT = 1.0
_MENTION_BASE_WEIGHT = 0.3
_DEFAULT_CHUNK_COUNT_WEIGHT = 0.2


def build_ppr_graph(
    storage: NetworkXGraphStorage,
    *,
    directed: bool,
    chunk_count_weight: float = _DEFAULT_CHUNK_COUNT_WEIGHT,
) -> nx.Graph:
    """Build a weighted graph over entity nodes + chunk nodes.

    RELATION edges weight 1.0; MENTIONS edges weight 0.3 plus optional
    ``chunk_count_weight`` times the chunk's normalized mention count (a chunk
    mentioning many entities is a stronger evidence source). In directed mode
    both Chunk->Entity and Entity->Chunk MENTIONS edges are added (equivalent
    in the undirected graph, kept for API parity with the source project).
    """
    graph = nx.Graph()
    inner = storage._graph
    mention_counts = {
        node: sum(
            1 for _, _, eattr in inner.out_edges(node, data=True) if eattr.get("edge_type") == "MENTIONS"
        )
        for node, nattr in inner.nodes(data=True)
        if nattr.get("type") == "chunk"
    }
    max_count = max(mention_counts.values(), default=0)
    for node, nattr in inner.nodes(data=True):
        graph.add_node(node, type=nattr.get("type"))
    for u, v, eattr in inner.edges(data=True):
        if eattr.get("edge_type") == "RELATION":
            graph.add_edge(u, v, weight=_RELATION_WEIGHT)
        elif eattr.get("edge_type") == "MENTIONS":
            count = mention_counts.get(u, 0)
            normalized = count / max_count if max_count > 0 else 0.0
            weight = _MENTION_BASE_WEIGHT + chunk_count_weight * normalized
            graph.add_edge(u, v, weight=weight)
            if directed:
                graph.add_edge(v, u, weight=weight)
    return graph


def rank_chunks_by_ppr(
    storage: NetworkXGraphStorage,
    seed_weights: dict[str, float],
    *,
    top_k: int,
    max_nodes: int,
    damping: float,
    directed: bool,
    chunk_count_weight: float = _DEFAULT_CHUNK_COUNT_WEIGHT,
) -> list[tuple[str, float]]:
    """Rank chunks by personalized pagerank seeded with ``seed_weights``.

    ``networkx.pagerank(G, alpha=damping, personalization=reset,
    weight="weight")``; the personalization vector covers ALL nodes with
    explicit 0.0 for non-seeds. Falls back to ``chunk_lookup_2hop`` (then
    ``chunk_lookup_1hop``) when pagerank is empty or all-zero. Scores are
    normalized to [0, 1] by the max. Returns ``[(chunk_id, score)]`` sorted
    descending.
    """
    if not seed_weights:
        return []
    seeds = {
        entity_id: weight
        for entity_id, weight in seed_weights.items()
        if storage.get_entity_node(entity_id) is not None
    }
    if not seeds:
        return []
    nodes = _collect_ppr_nodes(storage, list(seeds), max_nodes)
    graph = build_ppr_graph(storage, directed=directed, chunk_count_weight=chunk_count_weight)
    graph = graph.subgraph(nodes)
    reset = {node: 0.0 for node in graph.nodes}
    total = sum(seeds.values()) or 1.0
    for seed, weight in seeds.items():
        if seed in reset:
            reset[seed] = weight / total
    try:
        scores = nx.pagerank(
            graph,
            alpha=min(max(damping, 0.1), 0.99),
            personalization=reset,
            weight="weight",
        )
    except Exception as e:  # noqa: BLE001 - pagerank 失败须降级 2hop/1hop，不能阻断检索
        logger.warning("PPR pagerank failed: %s, falling back to seed-based lookup", e)
        return _fallback_chunks(storage, seeds)
    ranked = [
        (node, float(score))
        for node, score in scores.items()
        if graph.nodes[node].get("type") == "chunk"
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    if not ranked or all(abs(score) < 1e-10 for _, score in ranked):
        logger.warning("PPR returned no usable chunk scores, falling back to seed-based lookup")
        return _fallback_chunks(storage, seeds)
    max_score = ranked[0][1]
    if max_score > 0:
        ranked = [(chunk_id, score / max_score) for chunk_id, score in ranked]
    return ranked[:top_k]


def _collect_ppr_nodes(storage: NetworkXGraphStorage, seed_ids: list[str], max_nodes: int) -> list[str]:
    """Collect seeds plus their undirected neighborhood, capped at max_nodes.

    Used only when the full graph exceeds ``max_nodes``: pagerank runs on the
    restricted subgraph so big KBs stay fast. The cap is checked per hop; a
    missing seed is skipped silently.
    """
    inner = storage._graph
    if inner.number_of_nodes() <= max_nodes:
        return list(inner.nodes())
    visited: dict[str, None] = {}
    frontier: list[str] = []
    for seed in seed_ids:
        if seed in inner and seed not in visited:
            visited[seed] = None
            frontier.append(seed)
    while frontier and len(visited) < max_nodes:
        next_frontier: list[str] = []
        for node in frontier:
            if len(visited) >= max_nodes:
                break
            for neighbor in _undirected_neighbors(inner, node):
                if neighbor in visited or len(visited) >= max_nodes:
                    continue
                visited[neighbor] = None
                next_frontier.append(neighbor)
        frontier = next_frontier
    return list(visited)


def _undirected_neighbors(inner: Any, node: str) -> list[str]:
    """All out- and in-neighbors of ``node`` (first-seen order)."""
    seen: dict[str, None] = {}
    for _, target in inner.out_edges(node):
        seen[target] = None
    for source, _ in inner.in_edges(node):
        seen[source] = None
    return list(seen)


def _fallback_chunks(storage: NetworkXGraphStorage, seed_weights: dict[str, float]) -> list[tuple[str, float]]:
    """2-hop lookup first; 1-hop weighted by the max reaching seed weight."""
    two_hop = storage.chunk_lookup_2hop(seed_weights)
    if two_hop:
        return two_hop
    chunk_weights: dict[str, float] = {}
    for seed, weight in seed_weights.items():
        for chunk_id, _ in storage.chunk_lookup_1hop([seed]):
            if weight > chunk_weights.get(chunk_id, float("-inf")):
                chunk_weights[chunk_id] = weight
    return sorted(chunk_weights.items(), key=lambda item: item[1], reverse=True)


__all__ = ["build_ppr_graph", "rank_chunks_by_ppr"]
