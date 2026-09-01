"""多跳检索的 NetworkX 扩展回调（S7）。

multi_hop.run_beam_search 的 expand 回调注入点：把图存储遍历（共享锚点
MENTIONS 两跳 + L4 事件-事件边）封装为
``{frontier_event_id: [EventExpansion...]}`` 的批量扩展。

- 共享锚点：``(f)-[:EVENT_MENTIONS]->(a)<-[:EVENT_MENTIONS]-(c)``，
  每个候选事件携带经共享锚点跳转的锚点名；
- L4 边：EVENT_NEXT / EVENT_TEMPORAL / EVENT_LOCATION 直达邻居，
  edge_type 与边属性（delta_seconds / location）一并返回。

证据摘录（evidence_preview）：事件节点不存 content_preview（N11 防 JSON
膨胀），按 candidate.chunk_id 回查 chunk 节点内容填充。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage
from yusu_kb.knowledge.graphs.multi_hop import EventExpansion, EventNodeData

_L4_EDGE_TYPES = ("EVENT_NEXT", "EVENT_TEMPORAL", "EVENT_LOCATION")


def _event_node_data(node: dict[str, Any], storage: NetworkXGraphStorage) -> EventNodeData:
    """事件节点属性 → EventNodeData（evidence_preview 回查 chunk 节点）。"""
    preview = ""
    chunk_id = str(node.get("chunk_id") or "")
    if chunk_id and storage._graph.has_node(chunk_id):
        preview = str(storage._graph.nodes[chunk_id].get("content_preview") or "")[:300]
    return EventNodeData.from_storage_node(
        {**node, "evidence_preview": preview}
    )


def build_expand_fn(storage: NetworkXGraphStorage, *, max_per_frontier: int = 200):
    """构造批量扩展回调（单次调用遍历一次存储内部图）。

    Args:
        storage: KB 的图存储
        max_per_frontier: 每个终点事件的最大候选数（防邻居爆炸）
    """
    inner = storage._graph

    def _shared_anchor_candidates(event_id: str) -> list[EventExpansion]:
        """共享锚点两跳扩展：f → anchors → 其他事件。"""
        candidates: dict[str, dict[str, Any]] = {}
        for _, anchor_id, eattr in inner.out_edges(event_id, data=True):
            if eattr.get("edge_type") != "EVENT_MENTIONS":
                continue
            anchor = inner.nodes.get(anchor_id)
            anchor_name = str((anchor or {}).get("name") or anchor_id)
            # in_edges 返回 (source, target, data)：指向该锚点的源才是事件
            for other_event, _tgt, in_eattr in inner.in_edges(anchor_id, data=True):
                if in_eattr.get("edge_type") != "EVENT_MENTIONS":
                    continue
                if other_event == event_id:
                    continue
                entry = candidates.setdefault(other_event, {"anchor_names": [], "node": None})
                entry["anchor_names"].append(anchor_name)
        expansions: list[EventExpansion] = []
        for other_event, entry in candidates.items():
            node = inner.nodes[other_event]
            if node.get("type") != "event":
                continue
            expansions.append(
                EventExpansion(
                    candidate=_event_node_data(dict(node), storage),
                    shared_anchor_names=tuple(sorted(set(entry["anchor_names"]))),
                    edge_type="MENTIONS",
                )
            )
        return expansions

    def _l4_candidates(event_id: str) -> list[EventExpansion]:
        """L4 事件-事件边直达邻居（NEXT/TEMPORAL/LOCATION）。"""
        expansions: list[EventExpansion] = []
        seen: set[str] = set()

        def _walk(neighbor: str, eattr: dict[str, Any]) -> None:
            if neighbor == event_id or neighbor in seen:
                return
            node = inner.nodes.get(neighbor)
            if node is None or node.get("type") != "event":
                return
            seen.add(neighbor)
            edge_type = eattr.get("edge_type")
            if edge_type == "EVENT_NEXT":
                mapped = "NEXT"
            elif edge_type == "EVENT_TEMPORAL":
                mapped = "TEMPORAL"
            else:
                mapped = "LOCATION"
            expansions.append(
                EventExpansion(
                    candidate=_event_node_data(dict(node), storage),
                    edge_type=mapped,
                    edge_attr={
                        "delta_seconds": eattr.get("delta_seconds"),
                        "location": eattr.get("location"),
                    },
                )
            )

        for _, neighbor, eattr in inner.edges(event_id, data=True):
            if eattr.get("edge_type") in _L4_EDGE_TYPES:
                _walk(neighbor, eattr)
        for source, _, eattr in inner.in_edges(event_id, data=True):
            if eattr.get("edge_type") in _L4_EDGE_TYPES:
                _walk(source, eattr)
        return expansions

    async def expand(frontier_ids: Iterable[str]) -> dict[str, list[EventExpansion]]:
        result: dict[str, list[EventExpansion]] = {}
        for event_id in frontier_ids:
            by_mentions = _shared_anchor_candidates(event_id)
            seen_ids = {exp.candidate.event_id for exp in by_mentions}
            merged = list(by_mentions)
            for exp in _l4_candidates(event_id):
                if exp.candidate.event_id not in seen_ids:
                    merged.append(exp)
            result[event_id] = merged[:max_per_frontier]
        return result

    return expand


def seed_events_from_hits(
    storage: NetworkXGraphStorage,
    hits: list[dict[str, Any]],
    *,
    min_value_weight: float = 0.15,
    top_n: int = 8,
) -> list[EventNodeData]:
    """事件向量命中 → 种子 EventNodeData（G4 过滤 + top-N 截断）。

    hits 元素：{"id", "score", "value_weight", ...}（event 向量库 search 返回）。
    """
    seeds: list[EventNodeData] = []
    for hit in hits:
        event_id = str(hit.get("id") or "")
        value_weight = float(hit.get("value_weight") or 1.0)
        if value_weight < min_value_weight:
            continue
        node = storage.get_event_node(event_id)
        if node is None:
            continue
        seeds.append(_event_node_data(dict(node), storage))
        if len(seeds) >= top_n:
            break
    return seeds


__all__ = ["build_expand_fn", "seed_events_from_hits"]
