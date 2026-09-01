"""In-memory knowledge graph storage backed by NetworkX.

Per-KB in-memory ``networkx.MultiDiGraph`` with atomic JSON persistence,
replacing the Neo4j graph of the source project. The GraphService reads and
writes entities, relations and chunk mentions through this class; BFS
subgraphs feed visualization and PPR retrieval.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import networkx as nx

from yusu_kb.knowledge.graphs.graph_utils import DESC_SEPARATOR
from yusu_kb.utils.logger import logger

# v2: 事件驱动重构——新增 event 节点与 EVENT_MENTIONS/CHUNK_EVENT/
# EVENT_NEXT/EVENT_TEMPORAL/EVENT_LOCATION 边。v1 payload 是 v2 的合法子集，
# load() 同时接受 {1,2}，v1 数据无需任何转换即可读取（见 load() 版本检查）。
_STORAGE_VERSION = 2


def _merge_description(old: str, new: str) -> str:
    """Merge ``new`` into ``old``, joining with DESC_SEPARATOR and deduping.

    The separator is shared with ``graph_utils.DESC_SEPARATOR`` so pipeline
    descriptions (already <SEP>-joined by ``build_graph_payload``) and merged
    descriptions keep one parseable format for downstream consumers. An empty
    ``new`` keeps ``old``; a ``new`` already contained in ``old`` is skipped.
    """
    if not new:
        return old
    if not old:
        return new
    if new in old:
        return old
    return f"{old}{DESC_SEPARATOR}{new}"


def _merge_file_ids(old: list[str], new: list[str]) -> list[str]:
    """Append ``new`` file ids to ``old``, preserving order and deduping."""
    return list(dict.fromkeys([*old, *new]))


def _undirected_neighbors(graph: nx.MultiDiGraph, node: str) -> list[str]:
    """All out- and in-neighbors of ``node`` (first-seen order)."""
    seen: dict[str, None] = {}
    for _, target in graph.out_edges(node):
        seen[target] = None
    for source, _ in graph.in_edges(node):
        seen[source] = None
    return list(seen)


def _merge_attributes(old: list[Any], new: list[Any] | None) -> list[Any]:
    """Union attributes by (text, label) pair, preserving order (dedup).

    Matches ``graph_utils.build_graph_payload``'s per-batch attribute union so
    re-upserts across extraction batches never drop earlier attributes.
    """
    if not new:
        return list(old)
    merged = list(old)
    known = {(attr["text"], attr["label"]) for attr in merged}
    for attribute in new:
        pair = (attribute["text"], attribute["label"])
        if pair not in known:
            merged.append(attribute)
            known.add(pair)
    return merged


class NetworkXGraphStorage:
    """In-memory NetworkX MultiDiGraph per KB with JSON persistence.

    Nodes: entity nodes (type="entity", entity_id, normalized_name, label, name,
    attributes, description) and chunk nodes (type="chunk", chunk_id, file_id,
    chunk_index, content_preview).
    Edges: RELATION (entity->entity, key (source_id, target_id, type), attrs:
    triple_id, text, type, file_ids, description) and MENTIONS (chunk->entity,
    key (chunk_id, entity_id), attrs: chunk_id, file_id, entity_id).
    """

    def __init__(self, kb_id: str, work_dir: str | Path) -> None:
        if not kb_id:
            raise ValueError("kb_id must be a non-empty string")
        self._kb_id = kb_id
        self._file = Path(work_dir) / kb_id / "graph_storage.json"
        self._graph = nx.MultiDiGraph()

    # --- persistence ----------------------------------------------------

    def load(self) -> None:
        """Rebuild the in-memory graph from the JSON file (idempotent).

        A missing file silently leaves the graph empty (fresh KB is normal);
        a corrupt or malformed file logs a warning and also leaves it empty.
        """
        self._graph = nx.MultiDiGraph()
        if not self._file.exists():
            return
        try:
            with self._file.open(encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, ValueError) as e:
            logger.warning("Failed to load graph storage %s: %s", self._file, e)
            return
        if not isinstance(payload, dict):
            logger.warning("Invalid graph storage payload in %s", self._file)
            return
        if payload.get("version") not in (1, 2):
            logger.warning(
                "Unsupported graph storage version %r in %s; ignoring",
                payload.get("version"),
                self._file,
            )
            return
        try:
            for node in payload.get("nodes") or []:
                if not isinstance(node, dict):
                    continue
                if node.get("type") == "entity":
                    self._graph.add_node(
                        node["entity_id"],
                        type="entity",
                        entity_id=node["entity_id"],
                        normalized_name=node.get("normalized_name", ""),
                        label=node.get("label", "Entity"),
                        name=node.get("name", ""),
                        attributes=list(node.get("attributes") or []),
                        description=node.get("description", ""),
                    )
                elif node.get("type") == "chunk":
                    self._graph.add_node(
                        node["chunk_id"],
                        type="chunk",
                        chunk_id=node["chunk_id"],
                        file_id=node.get("file_id", ""),
                        chunk_index=node.get("chunk_index", 0),
                        content_preview=node.get("content_preview", ""),
                    )
                elif node.get("type") == "event":
                    self._graph.add_node(
                        node["event_id"],
                        type="event",
                        event_id=node["event_id"],
                        chunk_id=node.get("chunk_id", ""),
                        file_id=node.get("file_id", ""),
                        chunk_index=node.get("chunk_index", 0),
                        event_type=node.get("event_type", ""),
                        summary=node.get("summary", ""),
                        time_expr=node.get("time_expr", ""),
                        time_norm=node.get("time_norm"),
                        time_resolution=node.get("time_resolution", "unknown"),
                        location=node.get("location"),
                        action=node.get("action", ""),
                        participants=list(node.get("participants") or []),
                        objects=list(node.get("objects") or []),
                        amount=node.get("amount"),
                        exact_identifiers=list(node.get("exact_identifiers") or []),
                        text_span=node.get("text_span"),
                        verification=node.get("verification", "unverified"),
                        value_weight=float(node.get("value_weight") or 1.0),
                        duplicate_of=node.get("duplicate_of"),
                    )
            for edge in payload.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                if edge.get("edge_type") == "RELATION":
                    rtype = edge.get("type", "")
                    self._graph.add_edge(
                        edge["source_id"],
                        edge["target_id"],
                        key=(edge["source_id"], edge["target_id"], rtype),
                        edge_type="RELATION",
                        triple_id=edge.get("triple_id", ""),
                        text=edge.get("text", ""),
                        type=rtype,
                        file_ids=list(edge.get("file_ids") or []),
                        description=edge.get("description", ""),
                    )
                elif edge.get("edge_type") == "MENTIONS":
                    chunk_id = edge["chunk_id"]
                    entity_id = edge["entity_id"]
                    self._graph.add_edge(
                        chunk_id,
                        entity_id,
                        key=(chunk_id, entity_id),
                        edge_type="MENTIONS",
                        chunk_id=chunk_id,
                        file_id=edge.get("file_id", ""),
                        entity_id=entity_id,
                    )
                elif edge.get("edge_type") == "EVENT_MENTIONS":
                    event_id = edge["event_id"]
                    entity_id = edge["entity_id"]
                    self._graph.add_edge(
                        event_id,
                        entity_id,
                        key=("EM", event_id, entity_id),
                        edge_type="EVENT_MENTIONS",
                        event_id=event_id,
                        chunk_id=edge.get("chunk_id", ""),
                        file_id=edge.get("file_id", ""),
                        entity_id=entity_id,
                        role=edge.get("role", ""),
                        source=edge.get("source", "participant"),
                    )
                elif edge.get("edge_type") == "CHUNK_EVENT":
                    chunk_id = edge["chunk_id"]
                    event_id = edge["event_id"]
                    self._graph.add_edge(
                        chunk_id,
                        event_id,
                        key=(chunk_id, event_id),
                        edge_type="CHUNK_EVENT",
                        chunk_id=chunk_id,
                        file_id=edge.get("file_id", ""),
                        event_id=event_id,
                        ordinal=edge.get("ordinal", 0),
                    )
                elif edge.get("edge_type") in ("EVENT_NEXT", "EVENT_TEMPORAL", "EVENT_LOCATION"):
                    self._add_event_link_from_dict(edge)
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning("Malformed graph storage payload in %s: %s", self._file, e)
            self._graph = nx.MultiDiGraph()

    def _add_event_link_from_dict(self, edge: dict[str, Any]) -> None:
        """Rehydrate an L4 event-event edge from a serialized dict."""
        edge_type = edge.get("edge_type")
        u = edge.get("source")
        v = edge.get("target")
        file_id = edge.get("file_id", "")
        if edge_type == "EVENT_NEXT":
            key = ("NEXT", file_id)
        elif edge_type == "EVENT_TEMPORAL":
            key = ("TEMPORAL", file_id)
        else:  # EVENT_LOCATION
            key = ("LOC", file_id, edge.get("location") or "")
        self._graph.add_edge(
            u,
            v,
            key=key,
            edge_type=edge_type,
            file_id=file_id,
            delta_seconds=edge.get("delta_seconds"),
            location=edge.get("location"),
        )

    def save(self) -> None:
        """Serialize the graph to JSON via a temp file and atomic replace."""
        self._file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kb_id": self._kb_id,
            "version": _STORAGE_VERSION,
            "nodes": self._serialized_nodes(),
            "edges": self._serialized_edges(),
        }
        tmp_file = self._file.with_name(self._file.name + ".tmp")
        with tmp_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_file, self._file)

    def _serialized_nodes(self) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        for _, nattr in self._graph.nodes(data=True):
            if nattr.get("type") == "entity":
                nodes.append(
                    {
                        "type": "entity",
                        "entity_id": nattr["entity_id"],
                        "normalized_name": nattr.get("normalized_name", ""),
                        "label": nattr.get("label", "Entity"),
                        "name": nattr.get("name", ""),
                        "attributes": list(nattr.get("attributes") or []),
                        "description": nattr.get("description", ""),
                    }
                )
            elif nattr.get("type") == "chunk":
                nodes.append(
                    {
                        "type": "chunk",
                        "chunk_id": nattr["chunk_id"],
                        "file_id": nattr.get("file_id", ""),
                        "chunk_index": nattr.get("chunk_index", 0),
                        "content_preview": nattr.get("content_preview", ""),
                    }
                )
            elif nattr.get("type") == "event":
                nodes.append(
                    {
                        "type": "event",
                        "event_id": nattr["event_id"],
                        "chunk_id": nattr.get("chunk_id", ""),
                        "file_id": nattr.get("file_id", ""),
                        "chunk_index": nattr.get("chunk_index", 0),
                        "event_type": nattr.get("event_type", ""),
                        "summary": nattr.get("summary", ""),
                        "time_expr": nattr.get("time_expr", ""),
                        "time_norm": nattr.get("time_norm"),
                        "time_resolution": nattr.get("time_resolution", "unknown"),
                        "location": nattr.get("location"),
                        "action": nattr.get("action", ""),
                        "participants": list(nattr.get("participants") or []),
                        "objects": list(nattr.get("objects") or []),
                        "amount": nattr.get("amount"),
                        "exact_identifiers": list(nattr.get("exact_identifiers") or []),
                        "text_span": nattr.get("text_span"),
                        "verification": nattr.get("verification", "unverified"),
                        "value_weight": float(nattr.get("value_weight") or 1.0),
                        "duplicate_of": nattr.get("duplicate_of"),
                    }
                )
        return nodes

    def _serialized_edges(self) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        for u, v, eattr in self._graph.edges(data=True):
            if eattr.get("edge_type") == "RELATION":
                edges.append(
                    {
                        "edge_type": "RELATION",
                        "source_id": u,
                        "target_id": v,
                        "triple_id": eattr.get("triple_id", ""),
                        "text": eattr.get("text", ""),
                        "type": eattr.get("type", ""),
                        "file_ids": list(eattr.get("file_ids") or []),
                        "description": eattr.get("description", ""),
                    }
                )
            elif eattr.get("edge_type") == "MENTIONS":
                edges.append(
                    {
                        "edge_type": "MENTIONS",
                        "chunk_id": u,
                        "entity_id": v,
                        "file_id": eattr.get("file_id", ""),
                    }
                )
            elif eattr.get("edge_type") == "EVENT_MENTIONS":
                edges.append(
                    {
                        "edge_type": "EVENT_MENTIONS",
                        "event_id": u,
                        "chunk_id": eattr.get("chunk_id", ""),
                        "file_id": eattr.get("file_id", ""),
                        "entity_id": v,
                        "role": eattr.get("role", ""),
                        "source": eattr.get("source", "participant"),
                    }
                )
            elif eattr.get("edge_type") == "CHUNK_EVENT":
                edges.append(
                    {
                        "edge_type": "CHUNK_EVENT",
                        "chunk_id": u,
                        "file_id": eattr.get("file_id", ""),
                        "event_id": v,
                        "ordinal": eattr.get("ordinal", 0),
                    }
                )
            elif eattr.get("edge_type") in ("EVENT_NEXT", "EVENT_TEMPORAL", "EVENT_LOCATION"):
                edges.append(
                    {
                        "edge_type": eattr.get("edge_type"),
                        "source": u,
                        "target": v,
                        "file_id": eattr.get("file_id", ""),
                        "delta_seconds": eattr.get("delta_seconds"),
                        "location": eattr.get("location"),
                    }
                )
        return edges

    def is_built(self) -> bool:
        """True when at least one entity / chunk / event node exists."""
        return any(
            nattr.get("type") in ("entity", "chunk", "event")
            for _, nattr in self._graph.nodes(data=True)
        )

    # --- writes (MERGE semantics) ---------------------------------------

    def upsert_entity(
        self,
        *,
        entity_id: str,
        normalized_name: str,
        label: str,
        name: str,
        attributes: list[Any] | None = None,
        description: str = "",
    ) -> None:
        """Create or merge an entity node (description and attributes merged).

        label 不覆盖：锚点先建后实体路径命中时，以既有 label 为准
        （AnchorRegistry 单调约束：已落盘的 entity_id 与 label 永不变更）。
        """
        merged_attributes: list[Any] = list(attributes) if attributes else []
        if entity_id in self._graph:
            existing = self._graph.nodes[entity_id]
            if existing.get("type") == "entity":
                label = str(existing.get("label") or label)
                description = _merge_description(existing.get("description", ""), description)
                merged_attributes = _merge_attributes(
                    list(existing.get("attributes") or []), attributes
                )
        self._graph.add_node(
            entity_id,
            type="entity",
            entity_id=entity_id,
            normalized_name=normalized_name,
            label=label,
            name=name,
            attributes=merged_attributes,
            description=description,
        )

    def add_chunk(
        self,
        *,
        chunk_id: str,
        file_id: str,
        chunk_index: int,
        content_preview: str,
    ) -> None:
        """Create or refresh a chunk node (idempotent)."""
        self._graph.add_node(
            chunk_id,
            type="chunk",
            chunk_id=chunk_id,
            file_id=file_id,
            chunk_index=chunk_index,
            content_preview=content_preview,
        )

    def add_mention(self, *, entity_id: str, chunk_id: str, file_id: str) -> None:
        """Add a chunk->entity MENTIONS edge if not already present."""
        key = (chunk_id, entity_id)
        if not self._graph.has_edge(chunk_id, entity_id, key=key):
            self._graph.add_edge(
                chunk_id,
                entity_id,
                key=key,
                edge_type="MENTIONS",
                chunk_id=chunk_id,
                file_id=file_id,
                entity_id=entity_id,
            )

    # --- event writes (事件驱动路径，L1/L2 落地) --------------------------

    def upsert_event(
        self,
        *,
        event_id: str,
        chunk_id: str,
        file_id: str,
        chunk_index: int,
        event_type: str,
        summary: str,
        time_expr: str,
        time_norm: str | None,
        time_resolution: str,
        location: str | None,
        action: str,
        participants: list[str],
        objects: list[str],
        amount: float | None,
        exact_identifiers: list[str],
        text_span: list[int] | None,
        verification: str,
        value_weight: float,
        duplicate_of: str | None,
    ) -> None:
        """Create or refresh an event node (idempotent, key=event_id).

        序列化约束（N2）：time_norm 为 ISO 字符串、text_span 为 list，
        JSON 持久化不接受 datetime / tuple。
        """
        self._graph.add_node(
            event_id,
            type="event",
            event_id=event_id,
            chunk_id=chunk_id,
            file_id=file_id,
            chunk_index=chunk_index,
            event_type=event_type,
            summary=summary,
            time_expr=time_expr,
            time_norm=time_norm,
            time_resolution=time_resolution,
            location=location,
            action=action,
            participants=list(participants),
            objects=list(objects),
            amount=amount,
            exact_identifiers=list(exact_identifiers),
            text_span=text_span,
            verification=verification,
            value_weight=float(value_weight),
            duplicate_of=duplicate_of,
        )

    def add_event_mention(
        self, *, event_id: str, chunk_id: str, file_id: str, entity_id: str, role: str = "", source: str = "participant"
    ) -> None:
        """Add an event->entity EVENT_MENTIONS edge (锚点跳板边).

        用独立边类型 EVENT_MENTIONS（不复用 MENTIONS）：chunk_lookup_1hop 只认
        MENTIONS，复用会把 event_id 当 chunk_id 污染检索（N3）。
        """
        key = ("EM", event_id, entity_id)
        if self._graph.has_edge(event_id, entity_id, key=key):
            eattr = self._graph[event_id][entity_id][key]
            eattr["file_id"] = file_id
            if role:
                eattr["role"] = role
            eattr["source"] = source
        else:
            self._graph.add_edge(
                event_id,
                entity_id,
                key=key,
                edge_type="EVENT_MENTIONS",
                event_id=event_id,
                chunk_id=chunk_id,
                file_id=file_id,
                entity_id=entity_id,
                role=role,
                source=source,
            )

    def add_chunk_event(self, *, event_id: str, chunk_id: str, file_id: str, ordinal: int) -> None:
        """Add a chunk->event CHUNK_EVENT edge（L2：把事件接入 chunk 网络）。"""
        key = (chunk_id, event_id)
        if not self._graph.has_edge(chunk_id, event_id, key=key):
            self._graph.add_edge(
                chunk_id,
                event_id,
                key=key,
                edge_type="CHUNK_EVENT",
                chunk_id=chunk_id,
                file_id=file_id,
                event_id=event_id,
                ordinal=ordinal,
            )

    def iter_events(self) -> list[dict[str, Any]]:
        """All event nodes in insertion order."""
        return [
            dict(nattr)
            for _, nattr in self._graph.nodes(data=True)
            if nattr.get("type") == "event"
        ]

    def _event_edges_by_file(self, file_id: str) -> list[tuple[str, str, Any]]:
        """本文件相关的 L4 事件-事件边（u, v, key），供 clear_event_links 清理。"""
        return [
            (u, v, k)
            for u, v, k, eattr in self._graph.edges(keys=True, data=True)
            if eattr.get("edge_type") in ("EVENT_NEXT", "EVENT_TEMPORAL", "EVENT_LOCATION")
            and eattr.get("file_id") == file_id
        ]

    def clear_event_links(self, file_id: str) -> None:
        """删除某文件的全部 L4 事件-事件边（幂等重算前提）。"""
        for u, v, k in self._event_edges_by_file(file_id):
            self._graph.remove_edge(u, v, key=k)

    def add_event_link(
        self,
        *,
        u: str,
        v: str,
        edge_type: str,
        file_id: str,
        delta_seconds: float | None = None,
        location: str | None = None,
    ) -> None:
        """Add a deterministic event-event link (L4, edge keys deterministic).

        key 构成：("NEXT", file_id) / ("TEMPORAL", file_id) / ("LOC", file_id, location)
        同 key 覆盖 → 重放幂等。delta_seconds 供 TEMPORAL 衰减评分。
        """
        if edge_type == "EVENT_NEXT":
            key = ("NEXT", file_id)
            attrs: dict[str, Any] = {"edge_type": "EVENT_NEXT", "file_id": file_id}
        elif edge_type == "EVENT_TEMPORAL":
            key = ("TEMPORAL", file_id)
            attrs = {"edge_type": "EVENT_TEMPORAL", "file_id": file_id, "delta_seconds": delta_seconds}
        elif edge_type == "EVENT_LOCATION":
            key = ("LOC", file_id, location or "")
            attrs = {"edge_type": "EVENT_LOCATION", "file_id": file_id, "location": location or ""}
        else:
            raise ValueError(f"未知的事件-事件边类型: {edge_type}")
        if self._graph.has_edge(u, v, key=key):
            self._graph[u][v][key].update(attrs)
        else:
            self._graph.add_edge(u, v, key=key, **attrs)

    def event_links(self) -> list[dict[str, Any]]:
        """All L4 event-event edges (NEXT/TEMPORAL/LOCATION) as dicts."""
        out: list[dict[str, Any]] = []
        for u, v, eattr in self._graph.edges(data=True):
            if eattr.get("edge_type") in ("EVENT_NEXT", "EVENT_TEMPORAL", "EVENT_LOCATION"):
                out.append(
                    {
                        "source": u,
                        "target": v,
                        "edge_type": eattr.get("edge_type"),
                        "file_id": eattr.get("file_id", ""),
                        "delta_seconds": eattr.get("delta_seconds"),
                        "location": eattr.get("location"),
                    }
                )
        return out

    def upsert_relation(
        self,
        *,
        triple_id: str,
        source_id: str,
        target_id: str,
        text: str,
        rtype: str,
        file_ids: list[str],
        description: str = "",
    ) -> None:
        """Create or merge a RELATION edge (file ids merged, description appended)."""
        key = (source_id, target_id, rtype)
        if self._graph.has_edge(source_id, target_id, key=key):
            eattr = self._graph[source_id][target_id][key]
            eattr["triple_id"] = triple_id
            eattr["text"] = text
            eattr["file_ids"] = _merge_file_ids(list(eattr.get("file_ids") or []), file_ids)
            eattr["description"] = _merge_description(eattr.get("description", ""), description)
        else:
            self._graph.add_edge(
                source_id,
                target_id,
                key=key,
                edge_type="RELATION",
                triple_id=triple_id,
                text=text,
                type=rtype,
                file_ids=list(file_ids),
                description=description,
            )

    # --- reads ----------------------------------------------------------

    def search_entities_by_name(self, keyword: str, limit: int) -> list[dict[str, Any]]:
        """Return entity nodes whose name matches ``keyword`` as a substring."""
        if not keyword:
            return []
        lowered = keyword.lower()
        matches: list[dict[str, Any]] = []
        for _, nattr in self._graph.nodes(data=True):
            if nattr.get("type") != "entity":
                continue
            haystack = f"{nattr.get('name', '')}\0{nattr.get('normalized_name', '')}".lower()
            if lowered in haystack:
                matches.append(dict(nattr))
                if len(matches) >= limit:
                    break
        return matches

    def get_labels(self) -> list[dict[str, Any]]:
        """Return entity label counts as [{label, count}] sorted by label."""
        counts: dict[str, int] = {}
        for _, nattr in self._graph.nodes(data=True):
            if nattr.get("type") == "entity":
                label = nattr.get("label") or "Entity"
                counts[label] = counts.get(label, 0) + 1
        return [{"label": label, "count": count} for label, count in sorted(counts.items())]

    def get_stats(self) -> dict[str, int]:
        """Return node/edge counts: entities, relations, mentions, chunks, events."""
        entities = chunks = events = 0
        for _, nattr in self._graph.nodes(data=True):
            if nattr.get("type") == "entity":
                entities += 1
            elif nattr.get("type") == "chunk":
                chunks += 1
            elif nattr.get("type") == "event":
                events += 1
        relations = mentions = event_mentions = chunk_events = event_links = 0
        for _, _, eattr in self._graph.edges(data=True):
            etype = eattr.get("edge_type")
            if etype == "RELATION":
                relations += 1
            elif etype == "MENTIONS":
                mentions += 1
            elif etype == "EVENT_MENTIONS":
                event_mentions += 1
            elif etype == "CHUNK_EVENT":
                chunk_events += 1
            elif etype in ("EVENT_NEXT", "EVENT_TEMPORAL", "EVENT_LOCATION"):
                event_links += 1
        return {
            "entities": entities,
            "relations": relations,
            "mentions": mentions,
            "chunks": chunks,
            "events": events,
            "event_mentions": event_mentions,
            "chunk_events": chunk_events,
            "event_links": event_links,
        }

    def bfs_subgraph(
        self,
        seed_entity_ids: list[str],
        *,
        max_depth: int,
        max_nodes: int,
        max_paths_per_hop: int = 5000,
    ) -> dict[str, Any]:
        """Collect the entity subgraph reachable from seeds via RELATION edges.

        Hop-by-hop BFS with a per-frontier-node expansion cap
        (``max_paths_per_hop``) to prevent path explosion, stopping early once
        ``max_nodes`` entities are collected. Only RELATION edges whose both
        endpoints are collected are returned; MENTIONS never participate.
        """
        visited: dict[str, None] = {}
        frontier: list[str] = []
        # Seeds are collected up front, before the max_nodes cap applies to the
        # BFS expansion: requested seeds always stay in the subgraph even when
        # the cap is smaller than the seed list.
        for seed in seed_entity_ids:
            if seed in visited:
                continue
            if self._graph.has_node(seed) and self._graph.nodes[seed].get("type") == "entity":
                visited[seed] = None
                frontier.append(seed)
        for _ in range(max_depth):
            if not frontier or len(visited) >= max_nodes:
                break
            next_frontier: list[str] = []
            for node in frontier:
                if len(visited) >= max_nodes:
                    break
                expanded = 0
                for neighbor in self._relation_neighbors(node):
                    if len(visited) >= max_nodes or expanded >= max_paths_per_hop:
                        break
                    if neighbor in visited:
                        continue
                    visited[neighbor] = None
                    next_frontier.append(neighbor)
                    expanded += 1
            frontier = next_frontier
        node_ids = list(visited)
        node_set = set(node_ids)
        edges = [
            self._relation_edge_dict(u, v, eattr)
            for u, v, eattr in self._graph.edges(data=True)
            if eattr.get("edge_type") == "RELATION" and u in node_set and v in node_set
        ]
        return {"nodes": [dict(self._graph.nodes[nid]) for nid in node_ids], "edges": edges}

    def chunk_lookup_1hop(self, seed_entity_ids: list[str]) -> list[tuple[str, float]]:
        """Chunks directly mentioning any seed entity, each with weight 1.0.

        The caller scales the weight by its own seed weights; 2-hop retrieval
        (``chunk_lookup_2hop``) carries the weights itself instead.
        """
        results: list[tuple[str, float]] = []
        seen: set[str] = set()
        for entity_id in seed_entity_ids:
            if not self._graph.has_node(entity_id):
                continue
            for chunk_id, _, eattr in self._graph.in_edges(entity_id, data=True):
                if eattr.get("edge_type") == "MENTIONS" and chunk_id not in seen:
                    seen.add(chunk_id)
                    results.append((chunk_id, 1.0))
        return results

    def chunk_lookup_2hop(self, seed_weights: dict[str, float]) -> list[tuple[str, float]]:
        """Chunks two hops from seeds (seed -> relation -> mid -> mention).

        Each chunk keeps the maximum seed weight reaching it; results are
        sorted by weight descending.
        """
        chunk_weights: dict[str, float] = {}
        for seed, weight in seed_weights.items():
            if not self._graph.has_node(seed):
                continue
            for mid in self._relation_neighbors(seed):
                for chunk_id, _, eattr in self._graph.in_edges(mid, data=True):
                    if eattr.get("edge_type") != "MENTIONS":
                        continue
                    if weight > chunk_weights.get(chunk_id, float("-inf")):
                        chunk_weights[chunk_id] = weight
        return sorted(chunk_weights.items(), key=lambda item: item[1], reverse=True)

    def real_chunk_edges(
        self,
        seed_chunk_ids: list[str],
        *,
        weight_scale: float = 0.6,
        max_neighbours_per_seed: int = 64,
    ) -> list[tuple[str, str, float]]:
        """Chunk-chunk co-mention evidence edges from the real entity graph.

        Hybrid implicit-graph mode only: two chunks are connected when they
        both mention the same entity (chunk -> MENTIONS -> entity <- MENTIONS
        <- chunk). The raw weight counts shared entities, normalised by the
        strongest pair, then scaled by ``weight_scale`` so real evidence
        reinforces the implicit structure without dominating it. Endpoints
        missing from the implicit graph are filtered by the consumer.
        """
        if weight_scale <= 0.0:
            return []
        shared_counts: dict[tuple[str, str], int] = {}
        for seed in seed_chunk_ids:
            if not self._graph.has_node(seed):
                continue
            entities = [
                target
                for _, target, eattr in self._graph.out_edges(seed, data=True)
                if eattr.get("edge_type") == "MENTIONS"
            ]
            if not entities:
                continue
            neighbours: dict[str, int] = {}
            for entity_id in entities:
                for chunk_id, _, eattr in self._graph.in_edges(entity_id, data=True):
                    if eattr.get("edge_type") != "MENTIONS" or chunk_id == seed:
                        continue
                    neighbours[chunk_id] = neighbours.get(chunk_id, 0) + 1
            for other_id in sorted(neighbours)[:max_neighbours_per_seed]:
                pair = (seed, other_id) if seed < other_id else (other_id, seed)
                shared_counts[pair] = max(shared_counts.get(pair, 0), neighbours[other_id])
        if not shared_counts:
            return []
        peak = max(shared_counts.values())
        return [
            (u, v, weight_scale * (count / peak)) for (u, v), count in sorted(shared_counts.items())
        ]

    def relations_between(self, entity_ids: list[str]) -> list[dict[str, Any]]:
        """RELATION edges whose endpoints both lie in ``entity_ids``."""
        wanted = set(entity_ids)
        return [
            self._relation_edge_dict(u, v, eattr)
            for u, v, eattr in self._graph.edges(data=True)
            if eattr.get("edge_type") == "RELATION" and u in wanted and v in wanted
        ]

    def get_entity_node(self, entity_id: str) -> dict[str, Any] | None:
        """Return a copy of the entity node's attributes, or None."""
        if entity_id not in self._graph:
            return None
        nattr = self._graph.nodes[entity_id]
        if nattr.get("type") != "entity":
            return None
        return dict(nattr)

    def get_event_node(self, event_id: str) -> dict[str, Any] | None:
        """Return a copy of the event node's attributes, or None."""
        if event_id not in self._graph:
            return None
        nattr = self._graph.nodes[event_id]
        if nattr.get("type") != "event":
            return None
        return dict(nattr)

    def chunk_lookup_event_paths(
        self,
        event_seed_weights: dict[str, float],
        entity_seed_weights: dict[str, float] | None = None,
        *,
        max_per_seed: int = 8,
    ) -> list[tuple[str, float]]:
        """事件路径兜底检索：种子（事件或实体）穿过事件节点命中 chunk。

        - 事件种子 → 其 CHUNK_EVENT 来源 chunk；
        - 事件种子 → L4 事件边邻居 → 邻居的来源 chunk；
        - 实体种子 → EVENT_MENTIONS 指向的事件 → 事件的来源 chunk。
        每个 chunk 保留到达它的最大种子权重；结果按权重降序。
        """
        chunk_weights: dict[str, float] = {}
        inner = self._graph

        def _source_chunks(event_id: str, weight: float) -> None:
            node = self.get_event_node(event_id)
            if node is None:
                return
            chunk_id = node.get("chunk_id")
            if chunk_id and weight > chunk_weights.get(chunk_id, float("-inf")):
                chunk_weights[chunk_id] = weight

        for event_id, weight in (event_seed_weights or {}).items():
            _source_chunks(event_id, weight)
            # L4 邻居（NEXT/TEMPORAL/LOCATION）
            seen: dict[str, None] = {}
            for _, neighbor, eattr in inner.edges(event_id, data=True):
                if eattr.get("edge_type") in ("EVENT_NEXT", "EVENT_TEMPORAL", "EVENT_LOCATION"):
                    seen[neighbor] = None
            for source, _, eattr in inner.in_edges(event_id, data=True):
                if eattr.get("edge_type") in ("EVENT_NEXT", "EVENT_TEMPORAL", "EVENT_LOCATION"):
                    seen[source] = None
            for i, neighbor in enumerate(seen):
                if i >= max_per_seed:
                    break
                _source_chunks(neighbor, weight * 0.6)

        for entity_id, weight in (entity_seed_weights or {}).items():
            if not self._graph.has_node(entity_id):
                continue
            seen_events: dict[str, None] = {}
            for _, _, eattr in self._graph.in_edges(entity_id, data=True):
                if eattr.get("edge_type") == "EVENT_MENTIONS" and eattr.get("event_id"):
                    seen_events[eattr["event_id"]] = None
            for i, event_id in enumerate(seen_events):
                if i >= max_per_seed:
                    break
                _source_chunks(event_id, weight * 0.6)

        return sorted(chunk_weights.items(), key=lambda item: item[1], reverse=True)

    def iter_entities(self) -> list[dict[str, Any]]:
        """All entity nodes in insertion order."""
        return [
            dict(nattr)
            for _, nattr in self._graph.nodes(data=True)
            if nattr.get("type") == "entity"
        ]

    def iter_relations(self) -> list[dict[str, Any]]:
        """All RELATION edges in insertion order."""
        return [
            self._relation_edge_dict(u, v, eattr)
            for u, v, eattr in self._graph.edges(data=True)
            if eattr.get("edge_type") == "RELATION"
        ]

    def delete_file(self, file_id: str) -> list[str]:
        """Drop all graph data attributed to a file; return orphan entity ids.

        Removes the file's MENTIONS / EVENT_MENTIONS / CHUNK_EVENT / L4 事件边
        and chunk / event nodes, strips the file id from RELATION ``file_ids``
        (dropping the edge once emptied), and returns the entity ids left
        without any incident edge. Entity nodes themselves are kept; the
        caller decides whether to purge orphans.
        """
        incident_edge_keys = [
            (u, v, k)
            for u, v, k, eattr in self._graph.edges(keys=True, data=True)
            if eattr.get("edge_type") in ("MENTIONS", "EVENT_MENTIONS", "CHUNK_EVENT")
            and eattr.get("file_id") == file_id
        ]
        for u, v, k in incident_edge_keys:
            self._graph.remove_edge(u, v, key=k)

        # L4 事件-事件边：按 file_id 清理
        for u, v, k in self._event_edges_by_file(file_id):
            self._graph.remove_edge(u, v, key=k)

        relation_keys = [
            (u, v, k, eattr)
            for u, v, k, eattr in self._graph.edges(keys=True, data=True)
            if eattr.get("edge_type") == "RELATION" and file_id in (eattr.get("file_ids") or [])
        ]
        for u, v, k, eattr in relation_keys:
            remaining = [fid for fid in eattr["file_ids"] if fid != file_id]
            if remaining:
                eattr["file_ids"] = remaining
            else:
                self._graph.remove_edge(u, v, key=k)

        chunk_ids = [
            node
            for node, nattr in self._graph.nodes(data=True)
            if nattr.get("type") == "chunk" and nattr.get("file_id") == file_id
        ]
        for chunk_id in chunk_ids:
            self._graph.remove_node(chunk_id)

        event_ids = [
            node
            for node, nattr in self._graph.nodes(data=True)
            if nattr.get("type") == "event" and nattr.get("file_id") == file_id
        ]
        for event_id in event_ids:
            self._graph.remove_node(event_id)

        return sorted(
            node
            for node, nattr in self._graph.nodes(data=True)
            if nattr.get("type") == "entity" and self._graph.degree(node) == 0
        )

    def update_entity_description(self, entity_id: str, description: str) -> None:
        """Replace an entity node's description (no-op when missing)."""
        if entity_id in self._graph and self._graph.nodes[entity_id].get("type") == "entity":
            self._graph.nodes[entity_id]["description"] = description

    def update_relation_description(self, triple_id: str, description: str) -> None:
        """Replace a RELATION edge's description (first match by triple_id)."""
        for _, _, eattr in self._graph.edges(data=True):
            if eattr.get("edge_type") == "RELATION" and eattr.get("triple_id") == triple_id:
                eattr["description"] = description
                return

    # --- connectivity metrics（S5 一等指标）------------------------------

    def get_connectivity(self, *, max_lcc_nodes: int = 2_000_000) -> dict[str, Any]:
        """图连通性指标（孤儿率 / 最大连通分量 / 平均度 / 锚点复用）。

        全部基于无向视角的 BFS（不调用 to_undirected()，避免整图拷贝），O(V+E)。
        - orphan_rate：度为 0 的节点占比（事件化重构的核心度量，参考实现的问题）；
        - lcc_ratio：最大连通分量占全部节点比例（≥0.9 视为图成图）；
        - anchor_reuse：事件数 / 锚点数（锚点成 hub 的程度，<2 说明锚点身份分裂）；
        - zero_anchor_event_rate：无 EVENT_MENTIONS 边的事件占比（零参与者事件的度量）。

        max_lcc_nodes 超限时返回 lcc_ratio=None 并标记 truncated（规模化保护）。
        """
        inner = self._graph
        node_ids = list(inner.nodes())
        total = len(node_ids)
        if total == 0:
            return {
                "node_count": 0,
                "orphan_count": 0,
                "orphan_rate": 0.0,
                "lcc_size": 0,
                "lcc_ratio": 0.0,
                "avg_degree": 0.0,
                "event_count": 0,
                "anchor_count": 0,
                "anchor_reuse": 0.0,
                "zero_anchor_event_rate": 0.0,
                "truncated": False,
            }

        degree_sum = 0
        orphan_count = 0
        event_count = 0
        zero_anchor_events = 0
        anchor_ids: set[str] = set()
        # 孤儿率只统计实体与事件（检索侧可见节点）：chunk 节点无事件时度为 0
        # （LLM 对说明性段落返回空抽取是正常现象），把它计入孤儿率会让指标失真
        # ——实测 13 chunk 的小库因 2 个空抽取 chunk 直接触发 1% 阈值。
        retrieval_node_count = 0
        for node in node_ids:
            degree = inner.degree(node)
            degree_sum += degree
            ntype = inner.nodes[node].get("type")
            if ntype in ("entity", "event"):
                retrieval_node_count += 1
                if degree == 0:
                    orphan_count += 1
            if ntype == "event":
                event_count += 1
                if not any(
                    eattr.get("edge_type") == "EVENT_MENTIONS"
                    for _, _, eattr in inner.edges(node, data=True)
                ):
                    zero_anchor_events += 1
        for _, _, eattr in inner.edges(data=True):
            if eattr.get("edge_type") == "EVENT_MENTIONS":
                anchor_ids.add(eattr.get("entity_id"))

        lcc_size, truncated = self._largest_connected_component(node_ids, max_lcc_nodes)
        orphan_denominator = retrieval_node_count or total

        return {
            "node_count": total,
            "orphan_count": orphan_count,
            "orphan_rate": round(orphan_count / orphan_denominator, 6),
            "lcc_size": lcc_size,
            "lcc_ratio": round(lcc_size / total, 6) if lcc_size is not None else None,
            "avg_degree": round(degree_sum / total, 6),
            "event_count": event_count,
            "anchor_count": len(anchor_ids),
            "anchor_reuse": round(event_count / len(anchor_ids), 6) if anchor_ids else 0.0,
            "zero_anchor_event_rate": round(zero_anchor_events / event_count, 6) if event_count else 0.0,
            "truncated": truncated,
        }

    def _largest_connected_component(
        self, node_ids: list[str], max_nodes: int
    ) -> tuple[int | None, bool]:
        """Undirected BFS largest connected component (no graph copy)."""
        visited: set[str] = set()
        largest = 0
        truncated = False
        for start in node_ids:
            if start in visited:
                continue
            size = 0
            stack = [start]
            visited.add(start)
            while stack:
                node = stack.pop()
                size += 1
                if size > max_nodes:
                    truncated = True
                    return None, truncated
                for neighbor in _undirected_neighbors(self._graph, node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            largest = max(largest, size)
        return largest, truncated

    # --- helpers --------------------------------------------------------

    def _relation_neighbors(self, node: str) -> list[str]:
        """RELATION neighbors of ``node`` (both directions), first-seen order."""
        neighbors: dict[str, None] = {}
        for _, target, eattr in self._graph.out_edges(node, data=True):
            if eattr.get("edge_type") == "RELATION":
                neighbors[target] = None
        for source, _, eattr in self._graph.in_edges(node, data=True):
            if eattr.get("edge_type") == "RELATION":
                neighbors[source] = None
        return list(neighbors)

    def _relation_edge_dict(self, u: str, v: str, eattr: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_id": u,
            "target_id": v,
            "triple_id": eattr.get("triple_id", ""),
            "text": eattr.get("text", ""),
            "type": eattr.get("type", ""),
            "file_ids": list(eattr.get("file_ids") or []),
            "description": eattr.get("description", ""),
        }


__all__ = ["NetworkXGraphStorage"]