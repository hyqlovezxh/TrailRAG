"""S2/S5 测试：锚点身份规范化、AnchorRegistry 不变量、L4 事件边、连通性门禁。

覆盖事件化重构的两个关键修复：
- L1 锚点身份必须是真实世界指称的规范函数（Tier-A 纯函数 + Tier-B 单调裁决）；
- L4 确定性事件-事件边 + 连通性一等指标（孤儿节点率的量化）。
"""

from __future__ import annotations

import pytest

from yusu_kb.knowledge.graphs.anchor_registry import (
    AnchorRegistry,
    build_anchor_candidate,
)
from yusu_kb.knowledge.graphs.connectivity import evaluate_gate
from yusu_kb.knowledge.graphs.event_links import generate_file_event_links
from yusu_kb.knowledge.graphs.event_schemas import make_event_id
from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage
from yusu_kb.knowledge.graphs.graph_utils import (
    canonical_anchor_label,
    canonical_anchor_name,
    compute_entity_id,
    normalize_label_form,
    route_extractor_for_chunk,
)

# ---------------------------------------------------------------- L1 纯函数


def test_normalize_label_form_unifies_separators():
    """N9 修复：间隔号/斜杠/空白统一为间隔号系。"""
    assert normalize_label_form("资金/账户") == "资金·账户"
    assert normalize_label_form("通讯/账号") == "通讯·账号"
    assert normalize_label_form("资金·账户") == "资金·账户"
    assert normalize_label_form("人物") == "人物"
    assert normalize_label_form("") == "其他"
    assert normalize_label_form(None) == "其他"


def test_canonical_anchor_label_unifies_synonyms_and_identifiers():
    # label 同义归一（电话/手机/号码 → 手机号）
    assert canonical_anchor_label("手机号") == "手机号"
    # 名称命中标识符规则 → 强制定型
    assert canonical_anchor_label(None, name="13857906361") == "手机号"
    assert canonical_anchor_label("其他", name="6222021234567890123") == "银行账户"
    assert canonical_anchor_label(None, name="浙A·JK345") == "车牌"
    assert canonical_anchor_label(None, name="FB2026-0001") == "单号"
    # 未命中标识符的无类型 → 其他
    assert canonical_anchor_label(None, name="李四") == "其他"


def test_canonical_anchor_name_matches_entity_path():
    """R2 修复：事件路径与实体路径必须对同一名字算出同一 entity_id。"""
    name = "W01"
    event_norm = canonical_anchor_name(name)
    entity_norm = canonical_anchor_name(name)
    assert event_norm == entity_norm == "w01"
    # 两条路径用同一 hash 公式 + 同一规范化名 → 同一 entity_id
    assert compute_entity_id("kb1", event_norm, "人物") == compute_entity_id("kb1", entity_norm, "人物")


def test_build_anchor_candidate_output_shape():
    cand = build_anchor_candidate("陈锦标", "人物")
    assert cand["normalized_name"] == "陈锦标"
    assert cand["label"] == "人物"
    assert cand["display_name"] == "陈锦标"
    # 空名不产候选
    assert build_anchor_candidate("  ", "人物") == {}


def test_route_extractor_for_chunk():
    assert route_extractor_for_chunk("transcript", "") == "event"
    assert route_extractor_for_chunk("chat_record", "") == "event"
    assert route_extractor_for_chunk("general", "") == "event"
    assert route_extractor_for_chunk("csv_table", "") == "llm"
    assert route_extractor_for_chunk("spreadsheet", "") == "llm"
    assert route_extractor_for_chunk("laws", "") == "llm"
    assert route_extractor_for_chunk("qa", "") == "llm"
    assert route_extractor_for_chunk("", "") == "llm"  # 未知→llm（反转参考实现默认）
    # book：命中对话/时间标记 → event，否则 llm
    assert route_extractor_for_chunk("book", "问：你叫什么名字？") == "event"
    assert route_extractor_for_chunk("book", "2025-10-05 会议召开") == "event"
    assert route_extractor_for_chunk("book", "第一章 绪论 本章介绍背景知识。") == "llm"


# ---------------------------------------------------------------- AnchorRegistry


def _storage_with_entities(tmp_path, entities):
    storage = NetworkXGraphStorage("kb1", tmp_path)
    for entity_id, name, label in entities:
        storage.upsert_entity(entity_id=entity_id, normalized_name=name, label=label, name=name)
    return storage


def test_registry_monotonic_accepts_historical_label():
    """不变量②：已落盘 entity_id 与 label 永不变更。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        storage = _storage_with_entities(td, [("e1", "李四", "人物")])
        registry = AnchorRegistry.from_storage("kb1", storage)
        entity_id, label = registry.resolve("李四", "通讯·账号")
        assert label == "人物"  # 历史人物优先，不接受通讯·账号
        assert entity_id == "e1"
        assert "通讯·账号" in registry.conflicts.get("李四", set())


def test_registry_first_seen_wins_across_batches():
    """不变量③：同一输入序列无论批次如何切分，resolve 结果相同。"""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        # 批次 A：先见"人物"
        storage = _storage_with_entities(td, [])
        r1 = AnchorRegistry.from_storage("kb1", storage)
        id_a, label_a = r1.resolve("张三", "人物")
        assert label_a == "人物"

        # 模拟新批次（storage 已含历史，registry 重建）
        storage.upsert_entity(entity_id=id_a, normalized_name="张三", label=label_a, name="张三")
        r2 = AnchorRegistry.from_storage("kb1", storage)
        id_b, label_b = r2.resolve("张三", "其他")
        assert (id_b, label_b) == (id_a, label_a)  # 历史优先，entity_id 不变


def test_registry_resolve_all_prefers_higher_priority_label():
    """批次内同 name 多 label → 取优先级最高者。"""
    candidates = [
        build_anchor_candidate("李四", "通讯·账号"),
        build_anchor_candidate("李四", "人物"),
    ]
    import tempfile

    with tempfile.TemporaryDirectory():
        registry = AnchorRegistry("kb1")
        resolved = registry.resolve_all(candidates, kb_id="kb1")
        assert len(resolved) == 1
        assert resolved[0]["label"] == "人物"  # 人物(0) < 通讯·账号(4)
        assert resolved[0]["normalized_name"] == "李四"


def test_registry_from_storage_recovers_entities():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        storage = _storage_with_entities(td, [("e1", "陈锦标", "人物"), ("e2", "13857906361", "手机号")])
        registry = AnchorRegistry.from_storage("kb1", storage)
        assert registry.label_of("陈锦标") == "人物"
        assert registry.label_of("13857906361") == "手机号"


# ---------------------------------------------------------------- L4 事件边


def _make_event(storage, *, chunk_id, file_id, chunk_index, event_id, summary="事件", time_norm=None, location=None):
    storage.upsert_event(
        event_id=event_id,
        chunk_id=chunk_id,
        file_id=file_id,
        chunk_index=chunk_index,
        event_type="communication",
        summary=summary,
        time_expr="",
        time_norm=time_norm,
        time_resolution="minute" if time_norm else "unknown",
        location=location,
        action="沟通",
        participants=[],
        objects=[],
        amount=None,
        exact_identifiers=[],
        text_span=None,
        verification="verified",
        value_weight=1.0,
        duplicate_of=None,
    )


def _event_ids(storage, edge_type: str) -> list[tuple[str, str]]:
    return sorted(
        (link["source"], link["target"])
        for link in storage.event_links()
        if link["edge_type"] == edge_type
    )


def test_next_edges_follow_file_order(tmp_path):
    storage = NetworkXGraphStorage("kb1", tmp_path)
    # 故意乱序写入，验证按 (chunk_index, text_span) 排序
    _make_event(storage, chunk_id="c1", file_id="f1", chunk_index=1, event_id=make_event_id("c1", 0))
    _make_event(storage, chunk_id="c0", file_id="f1", chunk_index=0, event_id=make_event_id("c0", 0))
    _make_event(storage, chunk_id="c2", file_id="f1", chunk_index=2, event_id=make_event_id("c2", 0))
    count = generate_file_event_links(storage, "f1")
    assert count == 2
    nexts = _event_ids(storage, "EVENT_NEXT")
    assert nexts == [
        (make_event_id("c0", 0), make_event_id("c1", 0)),
        (make_event_id("c1", 0), make_event_id("c2", 0)),
    ]


def test_temporal_edges_sliding_window(tmp_path):
    storage = NetworkXGraphStorage("kb1", tmp_path)
    _make_event(storage, chunk_id="c0", file_id="f1", chunk_index=0, event_id=make_event_id("c0", 0),
                time_norm="2025-10-05T10:00:00")
    _make_event(storage, chunk_id="c1", file_id="f1", chunk_index=1, event_id=make_event_id("c1", 0),
                time_norm="2025-10-05T11:00:00")
    _make_event(storage, chunk_id="c2", file_id="f1", chunk_index=2, event_id=make_event_id("c2", 0),
                time_norm="2025-10-05T12:00:00")
    _make_event(storage, chunk_id="c3", file_id="f1", chunk_index=3, event_id=make_event_id("c3", 0),
                time_norm="2025-10-20T10:00:00")  # 超出 24h 窗口
    generate_file_event_links(storage, "f1", temporal_window_seconds=86400, temporal_top_k=5)
    temporals = _event_ids(storage, "EVENT_TEMPORAL")
    # c3 与前面事件差 15 天，不产生 TEMPORAL 边
    assert all("c3" not in pair for pair in temporals)


def test_location_edges_group_by_location(tmp_path):
    storage = NetworkXGraphStorage("kb1", tmp_path)
    _make_event(storage, chunk_id="c0", file_id="f1", chunk_index=0, event_id=make_event_id("c0", 0),
                location="杭州西湖大道")
    _make_event(storage, chunk_id="c1", file_id="f1", chunk_index=1, event_id=make_event_id("c1", 0),
                location="杭州西湖大道")
    _make_event(storage, chunk_id="c2", file_id="f1", chunk_index=2, event_id=make_event_id("c2", 0),
                location="杭州西湖大道")
    _make_event(storage, chunk_id="c3", file_id="f1", chunk_index=3, event_id=make_event_id("c3", 0),
                location="南宁")
    generate_file_event_links(storage, "f1")
    locations = _event_ids(storage, "EVENT_LOCATION")
    # 西湖大道 3 个事件两两连接（3 条），南宁只有 1 个事件无边
    assert len(locations) == 3
    assert all("c3" not in pair for pair in locations)


def test_event_links_idempotent_regeneration(tmp_path):
    storage = NetworkXGraphStorage("kb1", tmp_path)
    _make_event(storage, chunk_id="c0", file_id="f1", chunk_index=0, event_id=make_event_id("c0", 0))
    _make_event(storage, chunk_id="c1", file_id="f1", chunk_index=1, event_id=make_event_id("c1", 0))
    generate_file_event_links(storage, "f1")
    first = storage.event_links()
    # 重跑（增量/幂等重算场景）→ 边集不变
    generate_file_event_links(storage, "f1")
    assert storage.event_links() == first
    # 删除一个事件后重算 → 旧边清理
    storage._graph.remove_node(make_event_id("c1", 0))
    generate_file_event_links(storage, "f1")
    assert storage.event_links() == []


# ---------------------------------------------------------------- 连通性门禁


def test_connectivity_metrics_basic(tmp_path):
    storage = NetworkXGraphStorage("kb1", tmp_path)
    metrics = storage.get_connectivity()
    assert metrics["node_count"] == 0

    # 一个事件 + 两个锚点（共享同一锚点成 hub）
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="")
    storage.upsert_event(
        event_id=make_event_id("c1", 0), chunk_id="c1", file_id="f1", chunk_index=0,
        event_type="transfer", summary="转账", time_expr="", time_norm=None,
        time_resolution="unknown", location=None, action="转账", participants=["张三", "李四"],
        objects=[], amount=None, exact_identifiers=[], text_span=None,
        verification="verified", value_weight=1.0, duplicate_of=None,
    )
    storage.upsert_entity(entity_id="a1", normalized_name="张三", label="人物", name="张三")
    storage.upsert_entity(entity_id="a2", normalized_name="李四", label="人物", name="李四")
    storage.add_chunk_event(event_id=make_event_id("c1", 0), chunk_id="c1", file_id="f1", ordinal=0)
    storage.add_event_mention(event_id=make_event_id("c1", 0), chunk_id="c1", file_id="f1", entity_id="a1", role="转账方")
    storage.add_event_mention(event_id=make_event_id("c1", 0), chunk_id="c1", file_id="f1", entity_id="a2", role="收款方")

    metrics = storage.get_connectivity()
    assert metrics["node_count"] == 4  # chunk + event + 2 锚点
    assert metrics["orphan_rate"] == 0.0
    assert metrics["lcc_ratio"] == 1.0  # 全连通
    assert metrics["event_count"] == 1
    assert metrics["anchor_count"] == 2
    assert metrics["anchor_reuse"] == pytest.approx(0.5)
    assert metrics["zero_anchor_event_rate"] == 0.0


def test_connectivity_detects_orphan_event(tmp_path):
    storage = NetworkXGraphStorage("kb1", tmp_path)
    # 一个完全孤立的事件（无 chunk、无锚点、无事件边）
    storage.upsert_event(
        event_id="ev:orphan#0", chunk_id="orphan", file_id="f9", chunk_index=0,
        event_type="none", summary="孤立", time_expr="", time_norm=None,
        time_resolution="unknown", location=None, action="", participants=[],
        objects=[], amount=None, exact_identifiers=[], text_span=None,
        verification="unverified", value_weight=0.1, duplicate_of=None,
    )
    metrics = storage.get_connectivity()
    assert metrics["orphan_rate"] == 1.0
    assert metrics["lcc_size"] == 1
    assert metrics["zero_anchor_event_rate"] == 1.0


def test_connectivity_orphan_rate_ignores_chunk_nodes(tmp_path):
    """孤儿率只统计实体与事件：无事件的 chunk（LLM 对说明性段落返回空抽取）
    属正常现象，计入会让小库指标失真（实测 47 节点库因 2 个空 chunk 破 1% 阈值）。"""
    storage = NetworkXGraphStorage("kb1", tmp_path)
    # 孤立的 chunk 节点（无任何事件）
    storage.add_chunk(chunk_id="empty_chunk", file_id="f1", chunk_index=0, content_preview="说明性段落")
    # 一个带锚点的正常事件（经 L2 接入 chunk）
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="")
    storage.upsert_event(
        event_id="ev:c1#0", chunk_id="c1", file_id="f1", chunk_index=0, event_type="transfer",
        summary="转账", time_expr="", time_norm=None, time_resolution="unknown", location=None,
        action="转账", participants=["张三"], objects=[], amount=None, exact_identifiers=[],
        text_span=None, verification="verified", value_weight=1.0, duplicate_of=None,
    )
    storage.upsert_entity(entity_id="a1", normalized_name="张三", label="人物", name="张三")
    storage.add_chunk_event(event_id="ev:c1#0", chunk_id="c1", file_id="f1", ordinal=0)
    storage.add_event_mention(
        event_id="ev:c1#0", chunk_id="c1", file_id="f1", entity_id="a1", role="参与者"
    )

    metrics = storage.get_connectivity()
    # node_count 含全部节点：空 chunk + 事件 chunk + 事件 + 实体 = 4
    assert metrics["node_count"] == 4
    assert metrics["orphan_count"] == 0  # 孤立 chunk 不计入孤儿（分母只取 entity/event）
    assert metrics["orphan_rate"] == 0.0


def test_gate_evaluation():
    ok = {
        "orphan_rate": 0.0,
        "lcc_ratio": 1.0,
        "anchor_reuse": 3.0,
        "zero_anchor_event_rate": 0.0,
    }
    passed, failures = evaluate_gate(ok)
    assert passed
    assert failures == []

    bad = {
        "orphan_rate": 0.35,
        "lcc_ratio": 0.4,
        "anchor_reuse": 1.1,
        "zero_anchor_event_rate": 0.5,
    }
    passed, failures = evaluate_gate(bad)
    assert not passed
    assert len(failures) == 4
    names = {f["metric"] for f in failures}
    assert names == {"orphan_rate", "lcc_ratio", "anchor_reuse", "zero_anchor_event_rate"}

    # lcc_ratio=None（规模化截断）→ 该指标 unknown，不算失败
    truncated = dict(ok)
    truncated["lcc_ratio"] = None
    passed, failures = evaluate_gate(truncated)
    assert passed
    assert failures == []


# ---------------------------------------------------------------- 存储事件持久化


def test_event_storage_roundtrip(tmp_path):
    storage = NetworkXGraphStorage("kb1", tmp_path)
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="块一")
    storage.upsert_event(
        event_id=make_event_id("c1", 0), chunk_id="c1", file_id="f1", chunk_index=0,
        event_type="transfer", summary="张三转账给李四50000元", time_expr="2025-10-05",
        time_norm="2025-10-05T14:24:00", time_resolution="minute", location="南宁",
        action="转账", participants=["张三", "李四"], objects=[], amount=50000.0,
        exact_identifiers=["FB2026-0001"], text_span=None, verification="verified",
        value_weight=1.0, duplicate_of=None,
    )
    storage.add_chunk_event(event_id=make_event_id("c1", 0), chunk_id="c1", file_id="f1", ordinal=0)
    storage.upsert_entity(entity_id="a1", normalized_name="张三", label="人物", name="张三")
    storage.add_event_mention(
        event_id=make_event_id("c1", 0), chunk_id="c1", file_id="f1", entity_id="a1", role="转账方"
    )
    storage.save()

    stats = storage.get_stats()
    assert stats["events"] == 1
    assert stats["event_mentions"] == 1
    assert stats["chunk_events"] == 1

    storage2 = NetworkXGraphStorage("kb1", tmp_path)
    storage2.load()
    events = storage2.iter_events()
    assert len(events) == 1
    assert events[0]["time_norm"] == "2025-10-05T14:24:00"  # ISO 字符串（N2 序列化）
    assert events[0]["amount"] == 50000.0
    assert storage2.get_stats() == stats  # 落盘重载无损
    # 事件 id 与 chunk 不冲突（N1）
    assert make_event_id("c1", 0) != "c1"
    assert storage2._graph.has_node("c1")


def test_delete_file_removes_events_and_links(tmp_path):
    storage = NetworkXGraphStorage("kb1", tmp_path)
    storage.add_chunk(chunk_id="c1", file_id="f1", chunk_index=0, content_preview="")
    _make_event(storage, chunk_id="c1", file_id="f1", chunk_index=0, event_id=make_event_id("c1", 0))
    _make_event(storage, chunk_id="c2", file_id="f1", chunk_index=1, event_id=make_event_id("c2", 0))
    generate_file_event_links(storage, "f1")
    # 另一文件的 chunk/event 不受影响
    storage.add_chunk(chunk_id="d1", file_id="f2", chunk_index=0, content_preview="")
    _make_event(storage, chunk_id="d1", file_id="f2", chunk_index=0, event_id=make_event_id("d1", 0))

    storage.delete_file("f1")
    stats = storage.get_stats()
    assert stats["events"] == 1  # 只剩 f2 的事件
    assert stats["chunks"] == 1
    assert storage.event_links() == []  # f1 的 L4 边已清理
