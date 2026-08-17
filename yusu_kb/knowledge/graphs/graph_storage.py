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

_STORAGE_VERSION = 1


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
        if payload.get("version") != _STORAGE_VERSION:
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
        except (KeyError, TypeError, AttributeError) as e:
            logger.warning("Malformed graph storage payload in %s: %s", self._file, e)
            self._graph = nx.MultiDiGraph()

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
        return edges

    def is_built(self) -> bool:
        """True when at least one entity or chunk node exists."""
        return any(
            nattr.get("type") in ("entity", "chunk")
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
        """Create or merge an entity node (description and attributes merged)."""
        merged_attributes: list[Any] = list(attributes) if attributes else []
        if entity_id in self._graph:
            existing = self._graph.nodes[entity_id]
            if existing.get("type") == "entity":
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
        """Return node/edge counts: entities, relations, mentions, chunks."""
        entities = chunks = 0
        for _, nattr in self._graph.nodes(data=True):
            if nattr.get("type") == "entity":
                entities += 1
            elif nattr.get("type") == "chunk":
                chunks += 1
        relations = mentions = 0
        for _, _, eattr in self._graph.edges(data=True):
            if eattr.get("edge_type") == "RELATION":
                relations += 1
            elif eattr.get("edge_type") == "MENTIONS":
                mentions += 1
        return {"entities": entities, "relations": relations, "mentions": mentions, "chunks": chunks}

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

        Removes the file's MENTIONS edges and chunk nodes, strips the file id
        from RELATION ``file_ids`` (dropping the edge once emptied), and
        returns the entity ids left without any incident edge. Entity nodes
        themselves are kept; the caller decides whether to purge orphans.
        """
        mention_keys = [
            (u, v, k)
            for u, v, k, eattr in self._graph.edges(keys=True, data=True)
            if eattr.get("edge_type") == "MENTIONS" and eattr.get("file_id") == file_id
        ]
        for u, v, k in mention_keys:
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