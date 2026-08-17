"""Tests for the in-memory NetworkX graph storage (JSON persistence)."""

from __future__ import annotations

import json

import pytest

from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage
from yusu_kb.knowledge.graphs.graph_utils import (
    DESC_SEPARATOR,
    compute_entity_id,
    normalize_entity_name,
)


@pytest.fixture()
def storage(tmp_path):
    return NetworkXGraphStorage("kb_test", tmp_path)


def _add_entity(storage, entity_id, name, label="concept"):
    storage.upsert_entity(
        entity_id=entity_id,
        normalized_name=normalize_entity_name(name),
        label=label,
        name=name,
    )


def _add_relation(storage, source_id, target_id, rtype, file_ids, triple_id=None, text="", description=""):
    storage.upsert_relation(
        triple_id=triple_id or f"{source_id}-{target_id}",
        source_id=source_id,
        target_id=target_id,
        text=text or f"{source_id} 关联 {target_id}",
        rtype=rtype,
        file_ids=file_ids,
        description=description,
    )


def _seed_chain(storage):
    _add_entity(storage, "A", "甲")
    _add_entity(storage, "B", "乙")
    _add_entity(storage, "C", "丙")
    _add_relation(storage, "A", "B", "related_to", ["f1"])
    _add_relation(storage, "B", "C", "related_to", ["f1"])


def test_empty_kb_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        NetworkXGraphStorage("", tmp_path)


def test_is_built(storage):
    assert not storage.is_built()
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="块")
    assert storage.is_built()


def test_empty_graph_save_load_roundtrip(tmp_path):
    storage = NetworkXGraphStorage("kb_test", tmp_path)
    storage.save()
    data_file = tmp_path / "kb_test" / "graph_storage.json"
    assert data_file.exists()
    raw = json.loads(data_file.read_text(encoding="utf-8"))
    assert raw["kb_id"] == "kb_test"
    assert raw["version"] == 1
    assert raw["nodes"] == []
    assert raw["edges"] == []

    storage2 = NetworkXGraphStorage("kb_test", tmp_path)
    storage2.load()
    assert not storage2.is_built()
    assert storage2.get_stats() == {"entities": 0, "relations": 0, "mentions": 0, "chunks": 0}


def test_upsert_entity_merges_description_and_attributes(storage):
    storage.upsert_entity(
        entity_id="e1",
        normalized_name="alpha",
        label="concept",
        name="Alpha",
        description="Alpha 描述",
    )
    assert storage.get_stats()["entities"] == 1

    storage.upsert_entity(
        entity_id="e1",
        normalized_name="alpha",
        label="concept",
        name="Alpha",
        description="第二描述",
    )
    assert storage.get_stats()["entities"] == 1
    assert storage.get_entity_node("e1")["description"] == f"Alpha 描述{DESC_SEPARATOR}第二描述"

    # duplicate (exact and substring containment) → not appended again
    storage.upsert_entity(
        entity_id="e1",
        normalized_name="alpha",
        label="concept",
        name="Alpha",
        description="第二描述",
    )
    storage.upsert_entity(
        entity_id="e1",
        normalized_name="alpha",
        label="concept",
        name="Alpha",
        description="描述",
    )
    assert storage.get_entity_node("e1")["description"] == f"Alpha 描述{DESC_SEPARATOR}第二描述"

    # empty description keeps the stored one; attributes are unioned when given
    storage.upsert_entity(
        entity_id="e1",
        normalized_name="alpha",
        label="concept",
        name="Alpha",
        description="",
    )
    storage.upsert_entity(
        entity_id="e1",
        normalized_name="alpha",
        label="concept",
        name="Alpha",
        attributes=[{"text": "身高", "label": "属性"}],
    )
    node = storage.get_entity_node("e1")
    assert node["description"] == f"Alpha 描述{DESC_SEPARATOR}第二描述"
    assert node["attributes"] == [{"text": "身高", "label": "属性"}]


def test_upsert_entity_unions_attributes(storage):
    storage.upsert_entity(
        entity_id="e1",
        normalized_name="alpha",
        label="concept",
        name="Alpha",
        attributes=[{"text": "身高", "label": "属性"}],
    )
    storage.upsert_entity(
        entity_id="e1",
        normalized_name="alpha",
        label="concept",
        name="Alpha",
        attributes=[{"text": "别名", "label": "属性"}, {"text": "身高", "label": "属性"}],
    )
    node = storage.get_entity_node("e1")
    assert node["attributes"] == [
        {"text": "身高", "label": "属性"},
        {"text": "别名", "label": "属性"},
    ]

    # attributes=None keeps the stored union instead of wiping it
    storage.upsert_entity(entity_id="e1", normalized_name="alpha", label="concept", name="Alpha")
    assert storage.get_entity_node("e1")["attributes"] == [
        {"text": "身高", "label": "属性"},
        {"text": "别名", "label": "属性"},
    ]


def test_add_chunk_and_mention_dedup(storage):
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="块一")
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="块一")
    _add_entity(storage, "e1", "Alpha")
    storage.add_mention(entity_id="e1", chunk_id="c1", file_id="f1")
    storage.add_mention(entity_id="e1", chunk_id="c1", file_id="f1")
    assert storage.get_stats() == {"entities": 1, "relations": 0, "mentions": 1, "chunks": 1}


def test_upsert_relation_merges_file_ids_and_description(storage):
    _add_entity(storage, "e1", "Alpha")
    _add_entity(storage, "e2", "Beta")
    _add_relation(storage, "e1", "e2", "related_to", ["f1"], triple_id="t1", description="关系一")
    _add_relation(storage, "e1", "e2", "related_to", ["f1", "f2"], triple_id="t1", description="新描述")
    _add_relation(storage, "e1", "e2", "related_to", ["f2"], triple_id="t1", description="新描述")

    assert storage.get_stats()["relations"] == 1
    rels = storage.iter_relations()
    assert len(rels) == 1
    assert rels[0]["triple_id"] == "t1"
    assert rels[0]["file_ids"] == ["f1", "f2"]
    assert rels[0]["description"] == f"关系一{DESC_SEPARATOR}新描述"

    # different rtype → separate edge
    _add_relation(storage, "e1", "e2", "other", ["f3"], triple_id="t2")
    assert storage.get_stats()["relations"] == 2


def test_bfs_linear_chain_depths(storage):
    _seed_chain(storage)
    result = storage.bfs_subgraph(["A"], max_depth=1, max_nodes=10)
    assert {n["entity_id"] for n in result["nodes"]} == {"A", "B"}
    assert len(result["edges"]) == 1
    assert (result["edges"][0]["source_id"], result["edges"][0]["target_id"]) == ("A", "B")

    result = storage.bfs_subgraph(["A"], max_depth=2, max_nodes=10)
    assert {n["entity_id"] for n in result["nodes"]} == {"A", "B", "C"}
    assert len(result["edges"]) == 2


def test_bfs_max_nodes_truncation(storage):
    _seed_chain(storage)
    result = storage.bfs_subgraph(["A"], max_depth=2, max_nodes=2)
    assert {n["entity_id"] for n in result["nodes"]} == {"A", "B"}
    assert len(result["edges"]) == 1


def test_bfs_star_path_cap(storage):
    _add_entity(storage, "O", "中心")
    for i in range(10):
        _add_entity(storage, f"L{i}", f"叶子{i}")
        _add_relation(storage, "O", f"L{i}", "related_to", ["f1"])
    result = storage.bfs_subgraph(["O"], max_depth=1, max_nodes=100, max_paths_per_hop=3)
    assert len(result["nodes"]) == 4
    assert len(result["edges"]) == 3
    assert all(e["source_id"] == "O" for e in result["edges"])


def test_bfs_edges_keep_only_collected_endpoints(storage):
    _seed_chain(storage)
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="块")
    storage.add_mention(entity_id="B", chunk_id="c1", file_id="f1")
    result = storage.bfs_subgraph(["A"], max_depth=2, max_nodes=2)
    node_ids = {n["entity_id"] for n in result["nodes"]}
    assert node_ids == {"A", "B"}
    assert len(result["edges"]) == 1
    assert all(e["source_id"] in node_ids and e["target_id"] in node_ids for e in result["edges"])
    assert all(n["type"] == "entity" for n in result["nodes"])


def test_chunk_lookup_1hop_dedup(storage):
    _add_entity(storage, "e1", "甲")
    _add_entity(storage, "e2", "乙")
    for chunk_id, file_id in (("c1", "f1"), ("c2", "f1"), ("c3", "f2")):
        storage.add_chunk(chunk_id=chunk_id, file_id=file_id, chunk_index=0, content_preview="块")
    storage.add_mention(entity_id="e1", chunk_id="c1", file_id="f1")
    storage.add_mention(entity_id="e1", chunk_id="c2", file_id="f1")
    storage.add_mention(entity_id="e2", chunk_id="c2", file_id="f1")

    result = storage.chunk_lookup_1hop(["e1", "e2"])
    assert result == [("c1", 1.0), ("c2", 1.0)]
    assert storage.chunk_lookup_1hop(["missing"]) == []


def test_chunk_lookup_2hop_weights_and_max(storage):
    _add_entity(storage, "e1", "甲")
    _add_entity(storage, "e2", "乙")
    _add_entity(storage, "e3", "丙")
    _add_entity(storage, "e4", "丁")
    _add_relation(storage, "e1", "e2", "related_to", ["f1"])
    _add_relation(storage, "e4", "e3", "related_to", ["f2"])
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="块一")
    storage.add_chunk(chunk_id="c2", file_id="f1", chunk_index=1, content_preview="块二")
    storage.add_chunk(chunk_id="c3", file_id="f2", chunk_index=0, content_preview="块三")
    storage.add_mention(entity_id="e2", chunk_id="c1", file_id="f1")
    storage.add_mention(entity_id="e2", chunk_id="c2", file_id="f1")
    storage.add_mention(entity_id="e3", chunk_id="c1", file_id="f1")
    storage.add_mention(entity_id="e3", chunk_id="c3", file_id="f2")

    # c1 is reachable from both seeds → keeps the max weight
    result = storage.chunk_lookup_2hop({"e1": 2.0, "e4": 1.0})
    assert result == [("c1", 2.0), ("c2", 2.0), ("c3", 1.0)]
    assert storage.chunk_lookup_2hop({}) == []


def test_search_entities_by_name(storage):
    chen_id = compute_entity_id("kb_test", normalize_entity_name("陈锦标"), "人物")
    storage.upsert_entity(
        entity_id=chen_id,
        normalized_name=normalize_entity_name("陈锦标"),
        label="人物",
        name="陈锦标",
    )
    storage.upsert_entity(entity_id="e2", normalized_name="zhang wei", label="人物", name="张伟")
    storage.upsert_entity(entity_id="e3", normalized_name="zhang san", label="人物", name="张三")

    assert [e["entity_id"] for e in storage.search_entities_by_name("张", limit=10)] == ["e2", "e3"]
    assert len(storage.search_entities_by_name("张", limit=1)) == 1
    assert [e["entity_id"] for e in storage.search_entities_by_name("陈", limit=10)] == [chen_id]
    assert storage.search_entities_by_name("", limit=10) == []


def test_get_labels_and_get_stats(storage):
    _add_entity(storage, "e1", "张三", label="人物")
    _add_entity(storage, "e2", "李四", label="人物")
    _add_entity(storage, "e3", "622200", label="银行账户")
    labels = storage.get_labels()
    assert {item["label"]: item["count"] for item in labels} == {"人物": 2, "银行账户": 1}
    assert storage.get_stats() == {"entities": 3, "relations": 0, "mentions": 0, "chunks": 0}


def test_relations_between_and_node_reads(storage):
    _add_entity(storage, "e1", "Alpha")
    _add_entity(storage, "e2", "Beta")
    _add_entity(storage, "e3", "Gamma")
    _add_relation(storage, "e1", "e2", "related_to", ["f1"], triple_id="t1", description="关系一")

    assert storage.get_entity_node("e1")["name"] == "Alpha"
    assert storage.get_entity_node("e1")["label"] == "concept"
    assert storage.get_entity_node("missing") is None

    sub = storage.relations_between(["e1", "e2"])
    assert len(sub) == 1
    assert (sub[0]["source_id"], sub[0]["target_id"], sub[0]["triple_id"]) == ("e1", "e2", "t1")
    assert storage.relations_between(["e1"]) == []
    assert storage.relations_between(["e1", "e3"]) == []

    assert [e["entity_id"] for e in storage.iter_entities()] == ["e1", "e2", "e3"]
    assert len(storage.iter_relations()) == 1
    assert storage.iter_relations()[0]["description"] == "关系一"


def test_delete_file_cascade_and_orphans(storage):
    _add_entity(storage, "e1", "甲")
    _add_entity(storage, "e2", "乙")
    _add_entity(storage, "e3", "丙")
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="块一")
    storage.add_chunk(chunk_id="c2", file_id="f1", chunk_index=1, content_preview="块二")
    storage.add_chunk(chunk_id="c3", file_id="f2", chunk_index=0, content_preview="块三")
    storage.add_mention(entity_id="e1", chunk_id="c1", file_id="f1")
    storage.add_mention(entity_id="e2", chunk_id="c2", file_id="f1")
    storage.add_mention(entity_id="e2", chunk_id="c3", file_id="f2")
    _add_relation(storage, "e1", "e2", "related_to", ["f1"], triple_id="t1")
    _add_relation(storage, "e2", "e3", "related_to", ["f1", "f2"], triple_id="t2")

    assert storage.delete_file("f1") == ["e1"]
    assert storage.get_stats() == {"entities": 3, "relations": 1, "mentions": 1, "chunks": 1}
    remaining = storage.iter_relations()[0]
    assert (remaining["source_id"], remaining["target_id"], remaining["file_ids"]) == ("e2", "e3", ["f2"])

    # deleting the same file again is a no-op with the same orphan set
    assert storage.delete_file("f1") == ["e1"]

    # e1 stayed in the graph (orphans are reported, not purged), so the
    # final delete reports it alongside e2/e3
    assert sorted(storage.delete_file("f2")) == ["e1", "e2", "e3"]
    assert storage.get_stats() == {"entities": 3, "relations": 0, "mentions": 0, "chunks": 0}


def test_update_descriptions(storage):
    storage.upsert_entity(
        entity_id="e1",
        normalized_name="alpha",
        label="concept",
        name="Alpha",
        description="旧实体描述",
    )
    _add_entity(storage, "e2", "Beta")
    _add_relation(storage, "e1", "e2", "related_to", ["f1"], triple_id="t1", description="旧关系描述")

    storage.update_entity_description("e1", "新实体描述")
    storage.update_entity_description("missing", "忽略")
    storage.update_relation_description("t1", "新关系描述")
    storage.update_relation_description("t2", "忽略")

    assert storage.get_entity_node("e1")["description"] == "新实体描述"
    assert storage.get_entity_node("e2")["description"] == ""
    assert storage.iter_relations()[0]["description"] == "新关系描述"


def test_load_tolerant_of_missing_and_corrupt(tmp_path):
    storage = NetworkXGraphStorage("kb_test", tmp_path)
    storage.load()
    assert not storage.is_built()

    data_file = tmp_path / "kb_test" / "graph_storage.json"
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data_file.write_text("{ broken json", encoding="utf-8")
    storage.load()
    assert not storage.is_built()

    data_file.write_text('{"kb_id": "kb_test", "version": 1, "nodes": "oops"}', encoding="utf-8")
    storage.load()
    assert not storage.is_built()

    # parseable but structurally invalid: entity node without entity_id
    data_file.write_text(
        '{"kb_id": "kb_test", "version": 1, "nodes": [{"type": "entity", "label": "concept"}], "edges": []}',
        encoding="utf-8",
    )
    storage.load()
    assert not storage.is_built()

    # unsupported version is treated as corrupt
    data_file.write_text('{"kb_id": "kb_test", "version": 99, "nodes": [], "edges": []}', encoding="utf-8")
    storage.load()
    assert not storage.is_built()


def test_load_restores_saved_state(tmp_path):
    storage = NetworkXGraphStorage("kb_test", tmp_path)
    _add_entity(storage, "e1", "Alpha")
    storage.save()
    _add_entity(storage, "e2", "Beta")
    assert storage.get_stats()["entities"] == 2
    storage.load()
    assert storage.get_stats()["entities"] == 1
    assert storage.get_entity_node("e2") is None


def test_persisted_format_and_reload(tmp_path):
    storage = NetworkXGraphStorage("kb_test", tmp_path)
    storage.upsert_entity(
        entity_id="e1",
        normalized_name="alpha",
        label="concept",
        name="Alpha",
        attributes=[{"text": "身高", "label": "属性"}],
        description="Alpha 描述",
    )
    storage.upsert_entity(entity_id="e2", normalized_name="beta", label="concept", name="Beta")
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="块一")
    storage.add_chunk(chunk_id="c2", file_id="f1", chunk_index=1, content_preview="块二")
    storage.add_mention(entity_id="e1", chunk_id="c1", file_id="f1")
    storage.add_mention(entity_id="e2", chunk_id="c2", file_id="f1")
    storage.upsert_relation(
        triple_id="t1",
        source_id="e1",
        target_id="e2",
        text="alpha 关联 beta",
        rtype="related_to",
        file_ids=["f1"],
        description="关系一",
    )
    storage.save()

    raw = json.loads((tmp_path / "kb_test" / "graph_storage.json").read_text(encoding="utf-8"))
    assert raw["kb_id"] == "kb_test"
    assert raw["version"] == 1
    assert len(raw["nodes"]) == 4
    assert len(raw["edges"]) == 3

    entity = next(n for n in raw["nodes"] if n["type"] == "entity")
    assert set(entity) == {"type", "entity_id", "normalized_name", "label", "name", "attributes", "description"}
    chunk = next(n for n in raw["nodes"] if n["type"] == "chunk")
    assert set(chunk) == {"type", "chunk_id", "file_id", "chunk_index", "content_preview"}
    relation = next(e for e in raw["edges"] if e["edge_type"] == "RELATION")
    assert set(relation) == {"edge_type", "source_id", "target_id", "triple_id", "text", "type", "file_ids", "description"}
    mention = next(e for e in raw["edges"] if e["edge_type"] == "MENTIONS")
    assert set(mention) == {"edge_type", "chunk_id", "entity_id", "file_id"}
    assert raw["nodes"][0]["attributes"] == [{"text": "身高", "label": "属性"}]

    storage2 = NetworkXGraphStorage("kb_test", tmp_path)
    storage2.load()
    assert storage2.get_stats() == storage.get_stats()
    assert storage2.iter_entities() == storage.iter_entities()
    assert storage2.iter_relations() == storage.iter_relations()
    assert storage2.chunk_lookup_1hop(["e1"]) == [("c1", 1.0)]
    assert storage2.chunk_lookup_2hop({"e1": 2.0}) == [("c2", 2.0)]
    assert storage2.bfs_subgraph(["e1"], max_depth=2, max_nodes=10) == storage.bfs_subgraph(
        ["e1"], max_depth=2, max_nodes=10
    )