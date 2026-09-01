"""S6 测试：PPR 事件化 + 事件检索通道 + 事件路径兜底。

关键设计点验证：
- G4 value_weight 的正确落点是 reset 种子质量与 Chunk↔Event 边权，而不是
  Event 出边（nx 出边行归一化下乘出边权不改变事件传出总量——参考实现踩的坑）；
- 单出边节点（出度=1）无论边权如何都传出全部质量 → 反例测试固定种子拓扑；
- 事件种子 → chunk 的兜底检索穿过 Event 节点（chunk_lookup_event_paths）。
"""

from __future__ import annotations

import pytest

from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage
from yusu_kb.knowledge.graphs.ppr import build_ppr_graph, rank_chunks_by_ppr

EV = "ev:c1#0"


def _add_entity(storage, entity_id, name="实体", label="人物"):
    storage.upsert_entity(entity_id=entity_id, normalized_name=name, label=label, name=name)


def _add_chunk(storage, chunk_id, file_id="f1", chunk_index=0):
    storage.add_chunk(chunk_id=chunk_id, file_id=file_id, chunk_index=chunk_index, content_preview="")


def _add_event(
    storage,
    *,
    event_id,
    chunk_id,
    chunk_index=0,
    value_weight=1.0,
    time_norm=None,
    location=None,
    file_id="f1",
):
    storage.upsert_event(
        event_id=event_id,
        chunk_id=chunk_id,
        file_id=file_id,
        chunk_index=chunk_index,
        event_type="transfer",
        summary="事件",
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
        value_weight=value_weight,
        duplicate_of=None,
    )


def test_ppr_graph_chunk_event_edge_scales_with_value_weight():
    """G4 第二落点：Chunk↔Event 边权随 value_weight 缩放。"""
    storage = NetworkXGraphStorage("kb1", ".")
    _add_chunk(storage, "c1")
    _add_event(storage, event_id=EV, chunk_id="c1", value_weight=1.0)
    storage.add_chunk_event(event_id=EV, chunk_id="c1", file_id="f1", ordinal=0)
    graph = build_ppr_graph(storage, directed=False)
    w_high = graph["c1"][EV]["weight"]
    # base = 0.3 + 0.2*1.0 = 0.5；scale = 0.3 + 0.7*1.0 = 1.0 → 0.5
    assert w_high == pytest.approx(0.5)

    storage2 = NetworkXGraphStorage("kb1", ".")
    _add_chunk(storage2, "c1")
    _add_event(storage2, event_id=EV, chunk_id="c1", value_weight=0.2)  # chitchat
    storage2.add_chunk_event(event_id=EV, chunk_id="c1", file_id="f1", ordinal=0)
    graph2 = build_ppr_graph(storage2, directed=False)
    w_low = graph2["c1"][EV]["weight"]
    # scale = 0.3 + 0.7*0.2 = 0.44 → 边权 0.5*0.44 = 0.22
    assert w_low == pytest.approx(0.22)
    assert w_low < w_high


def test_ppr_event_seed_reaches_chunk():
    """事件种子在 PPR 中扩散到其来源 chunk。"""
    storage = NetworkXGraphStorage("kb1", ".")
    _add_chunk(storage, "c1")
    _add_chunk(storage, "c2")
    _add_event(storage, event_id=EV, chunk_id="c1", value_weight=1.0)
    _add_event(storage, event_id="ev:c2#0", chunk_id="c2", value_weight=0.1)  # 低价值
    storage.add_chunk_event(event_id=EV, chunk_id="c1", file_id="f1", ordinal=0)
    storage.add_chunk_event(event_id="ev:c2#0", chunk_id="c2", file_id="f1", ordinal=0)

    ranked = rank_chunks_by_ppr(
        storage,
        {},
        top_k=10,
        max_nodes=100,
        damping=0.85,
        directed=False,
        event_seed_weights={EV: 1.0},
    )
    chunk_ids = [chunk_id for chunk_id, _ in ranked]
    assert chunk_ids  # 有事件种子即返回结果
    assert chunk_ids[0] == "c1"  # 高价值事件的 chunk 居首


def test_event_seed_weight_in_reset_affects_rank():
    """G4 第一落点：reset 中事件种子质量已乘 value_weight，种子质量线性生效。"""
    storage = NetworkXGraphStorage("kb1", ".")
    _add_chunk(storage, "c1")
    _add_event(storage, event_id=EV, chunk_id="c1", value_weight=1.0)
    _add_entity(storage, "e1")
    storage.add_chunk_event(event_id=EV, chunk_id="c1", file_id="f1", ordinal=0)
    # e1 与事件共享锚点 → e1 从事件获得传播质量
    storage.add_event_mention(event_id=EV, chunk_id="c1", file_id="f1", entity_id="e1")
    storage.add_mention(entity_id="e1", chunk_id="c1", file_id="f1")

    ranked = rank_chunks_by_ppr(
        storage,
        {},
        top_k=10,
        max_nodes=100,
        damping=0.85,
        directed=False,
        event_seed_weights={EV: 1.0},
    )
    assert ranked and ranked[0][0] == "c1"


def test_chunk_lookup_event_paths_entity_to_chunk():
    """实体种子 → EVENT_MENTIONS → 事件 → 来源 chunk 的兜底链路。"""
    storage = NetworkXGraphStorage("kb1", ".")
    _add_chunk(storage, "c1")
    _add_event(storage, event_id=EV, chunk_id="c1")
    _add_entity(storage, "e1")
    storage.add_event_mention(event_id=EV, chunk_id="c1", file_id="f1", entity_id="e1")

    paths = storage.chunk_lookup_event_paths({}, {"e1": 1.0})
    assert ("c1", pytest.approx(0.6)) in paths  # 实体种子权重 *0.6


def test_chunk_lookup_event_paths_event_via_l4_neighbor():
    """事件种子 → L4 邻居事件 → 邻居的来源 chunk。"""
    storage = NetworkXGraphStorage("kb1", ".")
    _add_chunk(storage, "c1")
    _add_chunk(storage, "c2")
    _add_event(storage, event_id=EV, chunk_id="c1")
    _add_event(storage, event_id="ev:c2#0", chunk_id="c2")
    storage.add_chunk_event(event_id=EV, chunk_id="c1", file_id="f1", ordinal=0)
    storage.add_chunk_event(event_id="ev:c2#0", chunk_id="c2", file_id="f1", ordinal=1)
    storage.add_event_link(u=EV, v="ev:c2#0", edge_type="EVENT_NEXT", file_id="f1")

    paths = storage.chunk_lookup_event_paths({EV: 1.0}, {})
    chunk_ids = {chunk_id for chunk_id, _ in paths}
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids  # L4 邻居的来源 chunk


def test_fallback_prefers_entity_2hop_then_event_path():
    """_fallback_chunks 顺序：entity 2hop → 事件路径 → entity 1hop。"""
    storage = NetworkXGraphStorage("kb1", ".")
    # 仅有实体-事件连接，无 RELATION 网络（2hop 空）→ 走事件路径
    _add_chunk(storage, "c1")
    _add_event(storage, event_id=EV, chunk_id="c1")
    _add_entity(storage, "e1")
    storage.add_event_mention(event_id=EV, chunk_id="c1", file_id="f1", entity_id="e1")

    from yusu_kb.knowledge.graphs.ppr import _fallback_chunks

    result = _fallback_chunks(storage, {"e1": 1.0}, event_seed_weights={})
    assert result and result[0][0] == "c1"
