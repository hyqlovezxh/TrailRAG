"""S7 测试：多跳检索的 NetworkX 扩展回调 + 种子组装。

验证 build_expand_fn 的共享锚点两跳与 L4 边直达扩展，以及
seed_events_from_hits 的 G4 过滤与证据摘录填充。
"""

from __future__ import annotations

from yusu_kb.knowledge.graphs.event_expand import build_expand_fn, seed_events_from_hits
from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage
from yusu_kb.knowledge.graphs.multi_hop import MultiHopConfig, run_beam_search


def _add_chunk(storage, chunk_id, file_id="f1", chunk_index=0, content_preview=""):
    storage.add_chunk(
        chunk_id=chunk_id, file_id=file_id, chunk_index=chunk_index, content_preview=content_preview
    )


def _add_event(storage, *, event_id, chunk_id, chunk_index=0, participants=(), value_weight=1.0, location=None):
    storage.upsert_event(
        event_id=event_id,
        chunk_id=chunk_id,
        file_id="f1",
        chunk_index=chunk_index,
        event_type="communication",
        summary=f"事件 {event_id}",
        time_expr="",
        time_norm=None,
        time_resolution="unknown",
        location=location,
        action="沟通",
        participants=list(participants),
        objects=[],
        amount=None,
        exact_identifiers=[],
        text_span=None,
        verification="verified",
        value_weight=value_weight,
        duplicate_of=None,
    )


def _bind(storage, event_id, chunk_id, anchor_id, *, anchor_name="锚点"):
    storage.upsert_entity(entity_id=anchor_id, normalized_name=anchor_name, label="人物", name=anchor_name)
    storage.add_event_mention(
        event_id=event_id, chunk_id=chunk_id, file_id="f1", entity_id=anchor_id, role="参与者"
    )
    storage.add_mention(entity_id=anchor_id, chunk_id=chunk_id, file_id="f1")


def test_expand_shared_anchor_two_hop():
    storage = NetworkXGraphStorage("kb1", ".")
    _add_chunk(storage, "c1", content_preview="第一个 chunk 原文")
    _add_chunk(storage, "c2", content_preview="第二个 chunk 原文")
    _add_event(storage, event_id="ev:c1#0", chunk_id="c1")
    _add_event(storage, event_id="ev:c2#0", chunk_id="c2")
    # c1 与 c2 的事件共享锚点 a1
    _bind(storage, "ev:c1#0", "c1", "a1", anchor_name="张三")
    _bind(storage, "ev:c2#0", "c2", "a1", anchor_name="张三")

    expand = build_expand_fn(storage)
    import asyncio

    expansion_map = asyncio.run(expand(["ev:c1#0"]))
    expansions = expansion_map["ev:c1#0"]
    assert len(expansions) == 1
    assert expansions[0].candidate.event_id == "ev:c2#0"
    assert expansions[0].shared_anchor_names == ("张三",)
    assert expansions[0].edge_type == "MENTIONS"
    # 证据摘录回查 chunk 节点（N11：事件节点不存 content_preview）
    assert expansions[0].candidate.evidence_preview == "第二个 chunk 原文"


def test_expand_l4_edges():
    storage = NetworkXGraphStorage("kb1", ".")
    _add_chunk(storage, "c1")
    _add_chunk(storage, "c2")
    _add_event(storage, event_id="ev:c1#0", chunk_id="c1")
    _add_event(storage, event_id="ev:c2#0", chunk_id="c2")
    storage.add_event_link(u="ev:c1#0", v="ev:c2#0", edge_type="EVENT_NEXT", file_id="f1")
    storage.add_event_link(
        u="ev:c1#0", v="ev:c2#0", edge_type="EVENT_TEMPORAL", file_id="f1", delta_seconds=1800.0
    )

    expand = build_expand_fn(storage)
    import asyncio

    expansion_map = asyncio.run(expand(["ev:c1#0"]))
    by_type = {exp.edge_type: exp for exp in expansion_map["ev:c1#0"]}
    # 同一邻居经多条 L4 边连接时去重（NEXT 优先），候选只保留一条
    assert by_type["NEXT"].candidate.event_id == "ev:c2#0"
    assert "TEMPORAL" not in by_type
    assert by_type["NEXT"].edge_attr == {"delta_seconds": None, "location": None}


def test_expand_prefers_shared_anchor_over_l4():
    """同一邻居既共享锚点又存在 L4 边时，MENTIONS 优先（去重）。"""
    storage = NetworkXGraphStorage("kb1", ".")
    _add_chunk(storage, "c1")
    _add_chunk(storage, "c2")
    _add_event(storage, event_id="ev:c1#0", chunk_id="c1")
    _add_event(storage, event_id="ev:c2#0", chunk_id="c2")
    _bind(storage, "ev:c1#0", "c1", "a1", anchor_name="张三")
    _bind(storage, "ev:c2#0", "c2", "a1", anchor_name="张三")
    storage.add_event_link(u="ev:c1#0", v="ev:c2#0", edge_type="EVENT_NEXT", file_id="f1")

    expand = build_expand_fn(storage)
    import asyncio

    expansion_map = asyncio.run(expand(["ev:c1#0"]))
    # 只保留 MENTIONS 一条（共享锚点优先），不重复
    assert [exp.edge_type for exp in expansion_map["ev:c1#0"]] == ["MENTIONS"]


def test_seed_events_from_hits_filters_by_value_weight():
    storage = NetworkXGraphStorage("kb1", ".")
    _add_chunk(storage, "c1", content_preview="原始文本")
    _add_event(storage, event_id="ev:c1#0", chunk_id="c1", value_weight=1.0)
    _add_event(storage, event_id="ev:c1#1", chunk_id="c1", value_weight=0.1)

    hits = [
        {"id": "ev:c1#0", "score": 0.9, "value_weight": 1.0},
        {"id": "ev:c1#1", "score": 0.8, "value_weight": 0.1},
    ]
    seeds = seed_events_from_hits(storage, hits, min_value_weight=0.15, top_n=8)
    assert len(seeds) == 1
    assert seeds[0].event_id == "ev:c1#0"
    assert seeds[0].evidence_preview == "原始文本"


def test_beam_search_with_networkx_expand(tmp_path):
    """端到端：图存储 → expand → beam search 输出可解释路径。"""
    storage = NetworkXGraphStorage("kb1", tmp_path)
    _add_chunk(storage, "c1", content_preview="张三与李四在杭州会面")
    _add_chunk(storage, "c2", content_preview="李四随后转账给王五")
    _add_chunk(storage, "c3", content_preview="王五驾车前往边境")
    _add_event(storage, event_id="ev:c1#0", chunk_id="c1", participants=("张三", "李四"))
    _add_event(storage, event_id="ev:c2#0", chunk_id="c2", participants=("李四", "王五"))
    _add_event(storage, event_id="ev:c3#0", chunk_id="c3", participants=("王五",))
    _bind(storage, "ev:c1#0", "c1", "a_张三", anchor_name="张三")
    _bind(storage, "ev:c1#0", "c1", "a_李四", anchor_name="李四")
    _bind(storage, "ev:c2#0", "c2", "a_李四", anchor_name="李四")
    _bind(storage, "ev:c2#0", "c2", "a_王五", anchor_name="王五")
    _bind(storage, "ev:c3#0", "c3", "a_王五", anchor_name="王五")

    from yusu_kb.knowledge.graphs.event_expand import build_expand_fn
    from yusu_kb.knowledge.graphs.event_schemas import make_event_id
    from yusu_kb.knowledge.graphs.multi_hop import EventNodeData

    seed = EventNodeData(
        event_id=make_event_id("c1", 0),
        summary="张三与李四在杭州会面",
        time_norm=None,
        participants=("张三", "李四"),
        value_weight=1.0,
        verification="verified",
        chunk_id="c1",
        evidence_preview="张三与李四在杭州会面",
    )
    import asyncio

    paths = asyncio.run(
        run_beam_search(
            [seed],
            {},
            build_expand_fn(storage),
            MultiHopConfig(max_hops=3, beam_width=4, top_k_paths=2),
        )
    )
    assert paths and paths[0].hop_count >= 2
    hops = paths[0].hops
    # 每跳 edge_type 合法；扩展跳（hop_index>0）的 MENTIONS 必须有 via_anchor 说明；
    # 种子跳（hop 0）edge_type 默认 MENTIONS 但无 via_anchor（无前一跳）
    for hop in hops:
        assert hop.edge_type in ("MENTIONS", "TEMPORAL", "LOCATION", "NEXT")
        if hop.edge_type == "MENTIONS" and hop.hop_index > 0:
            assert hop.via_anchor  # 共享锚点跳必须有锚点说明
    # 终点事件证据指回 chunk
    assert hops[-1].evidence_chunk_id
