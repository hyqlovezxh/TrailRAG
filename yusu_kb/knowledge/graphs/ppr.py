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
_EVENT_MENTION_WEIGHT = 0.6
_CHUNK_EVENT_BASE_WEIGHT = 0.3
_NEXT_LINK_WEIGHT = 0.5
_TEMPORAL_LINK_WEIGHT = 0.5
_LOCATION_LINK_WEIGHT = 0.3
_DEFAULT_CHUNK_COUNT_WEIGHT = 0.2
_DEFAULT_EVENT_VALUE_WINDOW = (0.3, 1.0)  # Chunk↔Event 边权的 value_weight 映射区间


def pagerank_scores(
    graph: nx.Graph,
    personalization: dict[str, float],
    *,
    damping: float,
    max_iter: int | None = None,
    tol: float | None = None,
) -> dict[str, float]:
    """Run personalised PageRank, optionally pinning convergence parameters.

    Shared by the entity-graph path (:func:`rank_chunks_by_ppr`) and the
    query-time implicit chunk graph. ``max_iter``/``tol`` are forwarded only
    when given, so omitting them keeps ``networkx``'s own defaults (the
    entity-graph path is numerically unchanged).

    The implicit graph pins both values (100 / 1e-8): it is cached across
    queries, and ``nx.pagerank``'s defaults differ between the pure-python and
    scipy backends -- which would surface as the same query ranking differently
    across environments.
    """
    options: dict[str, Any] = {}
    if max_iter is not None:
        options["max_iter"] = max_iter
    if tol is not None:
        options["tol"] = tol
    return nx.pagerank(
        graph,
        alpha=min(max(damping, 0.1), 0.99),
        personalization=personalization,
        weight="weight",
        **options,
    )


def build_ppr_graph(
    storage: NetworkXGraphStorage,
    *,
    directed: bool,
    chunk_count_weight: float = _DEFAULT_CHUNK_COUNT_WEIGHT,
) -> nx.Graph:
    """Build a weighted graph over entity / chunk / event nodes.

    RELATION edges weight 1.0; MENTIONS edges weight 0.3 plus optional
    ``chunk_count_weight`` times the chunk's normalized mention count (a chunk
    mentioning many entities is a stronger evidence source). In directed mode
    both Chunk->Entity and Entity->Chunk MENTIONS edges are added (equivalent
    in the undirected graph, kept for API parity with the source project).

    事件路径边权（L2/L4 + G4 value_weight 的第二落点）：
    - CHUNK_EVENT：``(0.3 + ccw*norm) × (0.3 + 0.7*value_weight)``——nx 无向图两端
      共享同一权重，乘 value_weight 同时降低事件从 chunk 分得的质量并提高其
      回吐比例 → chitchat/none "有位置无分量"（参考实现乘在出边上是无效的：
      出边行归一化下不改变事件传出总量）；
    - EVENT_MENTIONS：常数 0.6（锚点跳板边）；
    - EVENT_NEXT / EVENT_TEMPORAL / EVENT_LOCATION：确定性结构边，权重 0.5/0.5/0.3
      （TEMPORAL 按 Δt 指数衰减）。
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
    event_counts = {
        node: sum(
            1 for _, _, eattr in inner.out_edges(node, data=True) if eattr.get("edge_type") == "CHUNK_EVENT"
        )
        for node, nattr in inner.nodes(data=True)
        if nattr.get("type") == "chunk"
    }
    max_count = max(mention_counts.values(), default=0)
    max_event_count = max(event_counts.values(), default=0)
    value_weight_by_id: dict[str, float] = {
        node: float(nattr.get("value_weight") or 1.0)
        for node, nattr in inner.nodes(data=True)
        if nattr.get("type") == "event"
    }
    for node, nattr in inner.nodes(data=True):
        graph.add_node(node, type=nattr.get("type"))
    for u, v, eattr in inner.edges(data=True):
        etype = eattr.get("edge_type")
        if etype == "RELATION":
            graph.add_edge(u, v, weight=_RELATION_WEIGHT)
        elif etype == "MENTIONS":
            count = mention_counts.get(u, 0)
            normalized = count / max_count if max_count > 0 else 0.0
            weight = _MENTION_BASE_WEIGHT + chunk_count_weight * normalized
            graph.add_edge(u, v, weight=weight)
            if directed:
                graph.add_edge(v, u, weight=weight)
        elif etype == "CHUNK_EVENT":
            count = event_counts.get(u, 0)
            normalized = count / max_event_count if max_event_count > 0 else 0.0
            base = _CHUNK_EVENT_BASE_WEIGHT + chunk_count_weight * normalized
            value_weight = value_weight_by_id.get(v, 1.0)
            # G4 第二落点：边权随事件价值缩放（0.3~1.0 映射，保留基础连接）
            scale = _DEFAULT_EVENT_VALUE_WINDOW[0] + (
                _DEFAULT_EVENT_VALUE_WINDOW[1] - _DEFAULT_EVENT_VALUE_WINDOW[0]
            ) * value_weight
            weight = base * scale
            graph.add_edge(u, v, weight=weight)
            if directed:
                graph.add_edge(v, u, weight=weight)
        elif etype == "EVENT_MENTIONS":
            graph.add_edge(u, v, weight=_EVENT_MENTION_WEIGHT)
            if directed:
                graph.add_edge(v, u, weight=_EVENT_MENTION_WEIGHT)
        elif etype == "EVENT_NEXT":
            graph.add_edge(u, v, weight=_NEXT_LINK_WEIGHT)
            if directed:
                graph.add_edge(v, u, weight=_NEXT_LINK_WEIGHT)
        elif etype == "EVENT_TEMPORAL":
            delta = eattr.get("delta_seconds")
            decay = 1.0
            if delta is not None:
                try:
                    decay = max(0.0, 1.0 - min(abs(float(delta)) / 86400.0, 1.0))
                except (TypeError, ValueError):
                    decay = 1.0
            graph.add_edge(u, v, weight=_TEMPORAL_LINK_WEIGHT * decay)
            if directed:
                graph.add_edge(v, u, weight=_TEMPORAL_LINK_WEIGHT * decay)
        elif etype == "EVENT_LOCATION":
            graph.add_edge(u, v, weight=_LOCATION_LINK_WEIGHT)
            if directed:
                graph.add_edge(v, u, weight=_LOCATION_LINK_WEIGHT)
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
    event_seed_weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Rank chunks by personalized pagerank seeded with ``seed_weights``.

    ``networkx.pagerank(G, alpha=damping, personalization=reset,
    weight="weight")``; the personalization vector covers ALL nodes with
    explicit 0.0 for non-seeds. Falls back to ``chunk_lookup_2hop`` (then
    event-path lookup, then ``chunk_lookup_1hop``) when pagerank is empty or
    all-zero. Scores are normalized to [0, 1] by the max. Returns
    ``[(chunk_id, score)]`` sorted descending.

    事件种子（event_seed_weights）：event 节点作 reset 源（G4 第一落点，
    种子质量已乘 value_weight）；事件 PPR 分数按 chunk_id 聚合回抬升来源 chunk。
    """
    event_seed_weights = dict(event_seed_weights or {})
    if not seed_weights and not event_seed_weights:
        return []
    seeds = {
        entity_id: weight
        for entity_id, weight in seed_weights.items()
        if storage.get_entity_node(entity_id) is not None
    }
    event_seeds = {
        event_id: weight
        for event_id, weight in event_seed_weights.items()
        if storage.get_event_node(event_id) is not None
    }
    if not seeds and not event_seeds:
        return []
    all_seed_ids = [*seeds.keys(), *event_seeds.keys()]
    nodes = _collect_ppr_nodes(storage, all_seed_ids, max_nodes)
    graph = build_ppr_graph(storage, directed=directed, chunk_count_weight=chunk_count_weight)
    graph = graph.subgraph(nodes)
    reset = {node: 0.0 for node in graph.nodes}
    total = sum(seeds.values()) + sum(event_seeds.values()) or 1.0
    for seed, weight in seeds.items():
        if seed in reset:
            reset[seed] = weight / total
    for seed, weight in event_seeds.items():
        if seed in reset:
            reset[seed] = weight / total
    try:
        scores = nx.pagerank(
            graph,
            alpha=min(max(damping, 0.1), 0.99),
            personalization=reset,
            weight="weight",
        )
    except Exception as e:  # noqa: BLE001 - pagerank 失败须降级到 2hop/1hop，不能阻断检索
        logger.warning("PPR pagerank failed: %s, falling back to seed-based lookup", e)
        return _fallback_chunks(storage, seeds, event_seed_weights=event_seeds)
    # 事件 PPR 分数聚合到其来源 chunk（事件与 chunk 共享 N1 前缀？不——事件 id 是
    # ev:{chunk_id}#{idx}，chunk_id 是节点属性；直接按 chunk_id 聚合）
    chunk_scores: dict[str, float] = {}
    for node, score in scores.items():
        node_type = graph.nodes[node].get("type")
        if node_type == "chunk":
            chunk_scores[node] = max(chunk_scores.get(node, 0.0), float(score))
        elif node_type == "event":
            chunk_id = storage.get_event_node(node).get("chunk_id")  # type: ignore[union-attr]
            if chunk_id:
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), float(score))
    ranked = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or all(abs(score) < 1e-10 for _, score in ranked):
        logger.warning("PPR returned no usable chunk scores, falling back to seed-based lookup")
        return _fallback_chunks(storage, seeds, event_seed_weights=event_seeds)
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


def _fallback_chunks(
    storage: NetworkXGraphStorage,
    seed_weights: dict[str, float],
    event_seed_weights: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """2-hop entity lookup → event-path lookup → 1-hop entity lookup."""
    if seed_weights:
        two_hop = storage.chunk_lookup_2hop(seed_weights)
        if two_hop:
            return two_hop
    event_path = storage.chunk_lookup_event_paths(event_seed_weights or {}, seed_weights)
    if event_path:
        return event_path
    chunk_weights: dict[str, float] = {}
    for seed, weight in seed_weights.items():
        for chunk_id, _ in storage.chunk_lookup_1hop([seed]):
            if weight > chunk_weights.get(chunk_id, float("-inf")):
                chunk_weights[chunk_id] = weight
    return sorted(chunk_weights.items(), key=lambda item: item[1], reverse=True)


__all__ = ["build_ppr_graph", "rank_chunks_by_ppr"]
