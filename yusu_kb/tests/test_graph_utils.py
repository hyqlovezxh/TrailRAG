"""Tests for graph_utils (entity name normalization, id computation, graph payload build).

Ports the pure-function contract of the source project's graph_utils module:
NFKC normalization, canonical label mapping, deterministic md5 ids, and the
placeholder-entity merge behavior of build_graph_payload.
"""

from __future__ import annotations

from yusu_kb.knowledge.graphs.graph_utils import (
    apply_entity_aliases,
    build_graph_payload,
    compute_entity_id,
    compute_triple_id,
    normalize_entity_name,
    normalize_relation_label,
)


def test_normalize_entity_name_nfkc_trailing_punct_lowercase():
    assert normalize_entity_name(" ＡＢＣ。 ") == "abc"


def test_normalize_entity_name_chinese_punct_removed():
    assert normalize_entity_name("张三，") == "张三"


def test_normalize_entity_name_internal_whitespace_collapsed():
    assert normalize_entity_name("张  三  ") == "张 三"


def test_normalize_entity_name_case_sensitive():
    assert normalize_entity_name("Apple", case_sensitive=True) == "Apple"


def test_compute_entity_id_stable_and_kb_scoped():
    first = compute_entity_id("kb1", "陈锦标", "人物")
    assert first == compute_entity_id("kb1", "陈锦标", "人物")
    assert compute_entity_id("kb2", "陈锦标", "人物") != first
    assert len(first) == 32


def test_compute_triple_id_stable():
    first = compute_triple_id("kb1", "陈锦标", "人物", "资金转账", "张三", "人物")
    assert first == compute_triple_id("kb1", "陈锦标", "人物", "资金转账", "张三", "人物")
    assert compute_triple_id("kb1", "陈锦标", "人物", "通话联系", "张三", "人物") != first


def test_normalize_relation_label():
    assert normalize_relation_label("转账") == "资金转账"
    assert normalize_relation_label(None) == "RELATED_TO"
    assert normalize_relation_label("") == "RELATED_TO"
    assert normalize_relation_label("自定义关系") == "自定义关系"


def test_build_graph_payload_dedups_synonym_labels():
    normalized = {
        "entities": [
            {"text": "13800000000", "label": "电话"},
            {"text": "13800000000", "label": "手机号"},
        ],
        "relations": [],
        "metadata": {"kb_id": "kb1"},
    }
    payload = build_graph_payload(normalized)
    assert len(payload["entities"]) == 1
    assert payload["entities"][0]["label"] == "手机号"


def test_build_graph_payload_merges_placeholder_entity_and_redirects():
    normalized = {
        "entities": [
            {"text": "陈锦标", "label": "人物", "attributes": [{"text": "身高", "label": "属性"}]},
            {"text": "陈锦标", "label": "代号", "attributes": [{"text": "别名", "label": "属性"}]},
        ],
        "relations": [
            {
                "source": {"text": "陈锦标", "label": "代号"},
                "target": {"text": "张三", "label": "人物"},
                "text": "转账",
                "label": "转账",
            }
        ],
        "metadata": {"kb_id": "kb1"},
    }
    payload = build_graph_payload(normalized)
    entities = [e for e in payload["entities"] if e["text"] == "陈锦标"]
    assert len(entities) == 1
    typed = entities[0]
    assert typed["label"] == "人物"
    assert {(a["text"], a["label"]) for a in typed["attributes"]} == {
        ("身高", "属性"),
        ("别名", "属性"),
    }
    rel = payload["relations"][0]
    assert rel["source"] == typed["id"]
    assert rel["label"] == "资金转账"


def test_build_graph_payload_merges_descriptions():
    normalized = {
        "entities": [
            {"text": "张三", "label": "人物", "descriptions": ["描述一"]},
            {"text": "张三", "label": "人物", "descriptions": ["描述二"]},
            {"text": "张三", "label": "代号", "descriptions": ["描述三"]},
        ],
        "relations": [],
        "metadata": {"kb_id": "kb1"},
    }
    payload = build_graph_payload(normalized)
    entities = [e for e in payload["entities"] if e["text"] == "张三"]
    assert len(entities) == 1
    entity = entities[0]
    assert entity["descriptions"] == ["描述一", "描述二", "描述三"]
    assert entity["description"] == "描述一<SEP>描述二<SEP>描述三"


def test_apply_entity_aliases_rewrites_entities_and_endpoints():
    normalized = {
        "entities": [{"text": "db", "label": "代号"}],
        "relations": [
            {
                "source": {"text": "hj", "label": "代号"},
                "target": {"text": "陈锦标", "label": "人物"},
                "text": "通话",
                "label": "通话",
            }
        ],
        "metadata": {"kb_id": "kb1"},
    }
    result = apply_entity_aliases(normalized, {"db": "陈锦标", "hj": "胡杰"})
    assert result is normalized
    assert normalized["entities"][0]["text"] == "陈锦标"
    assert normalized["relations"][0]["source"]["text"] == "胡杰"
    assert normalized["relations"][0]["target"]["text"] == "陈锦标"


def test_apply_entity_aliases_normalization_insensitive():
    normalized = {
        "entities": [{"text": "ＡＢＣ", "label": "代号"}],
        "relations": [],
        "metadata": {"kb_id": "kb1"},
    }
    apply_entity_aliases(normalized, {"abc": "实名"})
    assert normalized["entities"][0]["text"] == "实名"


def test_apply_entity_aliases_noop_without_alias_map():
    normalized = {
        "entities": [{"text": "张三", "label": "人物"}],
        "relations": [],
        "metadata": {"kb_id": "kb1"},
    }
    result = apply_entity_aliases(normalized, None)
    assert result is normalized
    assert normalized["entities"][0]["text"] == "张三"