"""事件路径纯逻辑模块单测（S1）：数据模型 / 护栏 / 多跳 / 事件抽取归一化。

覆盖迁移自源项目的纯逻辑代码，全部无 IO 依赖，可脱离图存储单测。
"""

# ruff: noqa: DTZ001 - 时间轴测试构造的是与抽取实现同语义的 naive datetime
# （LLM 输出的 time_norm 无时区，语义上即 naive；测试如实复刻该语义）

from __future__ import annotations

import asyncio
import json
from datetime import datetime

import pytest

from yusu_kb.knowledge.graphs.event_guards import (
    EventDedupIndex,
    fuzzy_contains,
    hamming_distance,
    mmr_select,
    simhash64,
    verify_events,
)
from yusu_kb.knowledge.graphs.event_schemas import (
    EVENT_VALUE_WEIGHTS,
    MAX_EVENTS_PER_CHUNK,
    EventRecord,
    EventType,
    Participant,
    make_event_id,
)
from yusu_kb.knowledge.graphs.extractors.event import (
    EventGraphExtractor,
    normalize_event_result,
    parse_time_norm,
)
from yusu_kb.knowledge.graphs.multi_hop import (
    EventExpansion,
    EventNodeData,
    MultiHopConfig,
    compute_edge_strength,
    is_edge_coherent,
    run_beam_search,
)
from yusu_kb.models.chat import GeneralResponse


def make_event(**overrides) -> EventRecord:
    fields = {
        "event_id": make_event_id("c1", 0),
        "chunk_id": "c1",
        "kb_id": "kb1",
        "file_id": "f1",
        "event_type": EventType.TRANSFER,
        "summary": "张三于2025-10-05转账50000元给李四",
        "participants": [
            Participant(name="张三", role="转账方", entity_type="人物"),
            Participant(name="李四", role="收款方", entity_type="人物"),
        ],
    }
    fields.update(overrides)
    return EventRecord(**fields)


# ---------------------------------------------------------------- schemas


def test_event_id_always_prefixed_with_ev():
    """N1 修复：1:1 与 1:N 的事件 id 都必须带 ev: 前缀，不得等于 chunk_id。"""
    assert make_event_id("c1", 0) == "ev:c1#0"
    assert make_event_id("c1", 3) == "ev:c1#3"
    assert make_event_id("c1", 0) != "c1"
    assert make_event_id("c1", -1) != "c1"


def test_event_record_validates_participants_and_summary():
    with pytest.raises(ValueError):
        EventRecord(event_id="ev:c1#0", chunk_id="c1", summary="  ")
    with pytest.raises(ValueError):
        EventRecord(
            event_id="ev:c1#0",
            chunk_id="c1",
            summary="ok",
            participants=[Participant(name="  ")],
        )


def test_participant_entity_type_falls_back():
    p = Participant(name="李四", entity_type="非法类型")
    assert p.entity_type == "其他"
    assert Participant(name="李四", entity_type="").entity_type == "人物"


def test_value_weight_clamped_and_effective():
    rec = make_event(value_weight=1.0, verification="unverified")
    assert rec.effective_value_weight == pytest.approx(0.5)
    rec.verification = "verified"
    assert rec.effective_value_weight == pytest.approx(1.0)
    assert make_event(value_weight=5.0).value_weight == 1.0


def test_text_span_validation():
    rec = make_event(text_span=(10, 20))
    assert rec.text_span == (10, 20)
    with pytest.raises(ValueError):
        make_event(text_span=(20, 10))


# ---------------------------------------------------------------- guards G1


def test_fuzzy_contains_exact_and_fuzzy():
    content = "陈锦标：到账了。陈 锦 标在群里也说了。"
    assert fuzzy_contains("陈锦标", content)
    assert fuzzy_contains("陈 锦 标", content)  # 空白归一快路径
    # 慢路径（等长滑动窗口）命中：目标形变字恰好匹配原文中相邻的「标在」两字
    assert fuzzy_contains("标在", content)
    assert fuzzy_contains("在群", content)
    assert not fuzzy_contains("王五", content)


def test_verify_events_anchored_against_content():
    content = "【时间段 2025-10-05 14:20 ~ 14:30】陈锦标与李四确认50000元到账，单号FB2026-0001。"
    rec = make_event(
        time_expr="【时间段 2025-10-05 14:20 ~ 14:30】",
        exact_identifiers=["FB2026-0001"],
        participants=[Participant(name="陈锦标", entity_type="人物")],
    )
    [verified] = verify_events([rec], content)
    assert verified.verification == "verified"

    rec2 = make_event(
        time_expr="不存在的时间",
        exact_identifiers=["99999999"],
        participants=[Participant(name="虚构人物", entity_type="人物")],
    )
    [unverified] = verify_events([rec2], content)
    assert unverified.verification == "unverified"


def test_verify_events_participant_is_soft_signal():
    """participants 不再单独降级：笔录正文以「我」自称，chunk 内无被询问人姓名。

    回归：E2E 实测 19 个事件全部因 participant 不在原文被判 unverified，
    而 identifier/time 全部通过——那是 LLM 正确的跨 chunk 主语补全，
    G1 的幻觉护栏只需守住「可逐字回溯的硬证据」（编号 / 时间）。
    """
    content = "问：请介绍你的基本情况。答：我 1979 年出生，2012 年注册宏发建材有限公司。"
    rec = make_event(
        time_expr="2012年",
        exact_identifiers=["宏发建材有限公司"],
        participants=[Participant(name="张三", entity_type="人物")],  # 正文无「张三」
    )
    [verified] = verify_events([rec], content)
    assert verified.verification == "verified"

    # 硬证据缺失仍要降级（真幻觉照拦）
    rec2 = make_event(
        time_expr="2030年",
        exact_identifiers=["不存在的编号"],
        participants=[Participant(name="张三", entity_type="人物")],
    )
    [unverified] = verify_events([rec2], content)
    assert unverified.verification == "unverified"


def test_verify_events_ignores_markdown_space_normalization():
    """markdown 解析会把「3月15日」规范化为「3 月 15 日」——不得误判为幻觉。

    回归：G1 逐字回溯曾经把 LLM 正确抽取的时间/编号判成幻觉并降级 unverified，
    导致整批事件 verification 全掉（E2E 实测：5 个事件全部因此降级）。
    """
    content = "经查，2024 年 3 月 15 日 张三通过建设银行向李四转账 80 万元，单号 FB2026-0001。"
    rec = make_event(
        time_expr="2024年3月15日",  # 无空格（LLM 原始抽取）
        exact_identifiers=["FB2026-0001"],
        participants=[Participant(name="张三", entity_type="人物")],
    )
    [verified] = verify_events([rec], content)
    assert verified.verification == "verified"

    # 真幻觉仍要拦下：年份对不上（剥离空白后也不匹配）
    rec2 = make_event(
        time_expr="2030年5月20日",
        exact_identifiers=["FB2026-0001"],
        participants=[Participant(name="张三", entity_type="人物")],
    )
    [unverified] = verify_events([rec2], content)
    assert unverified.verification == "unverified"


# ---------------------------------------------------------------- guards G2


def test_simhash_similar_summaries_are_close():
    a = simhash64("陈锦标与李四确认50000元资金到账并已转给邓家俊")
    b = simhash64("陈锦标与李四确认50000元资金到账并已转给邓家俊了")
    c = simhash64("王五早上吃了两个包子和一杯豆浆")
    assert hamming_distance(a, b) <= 5
    assert hamming_distance(a, c) > 5


def test_dedup_index_marks_duplicate_once():
    idx = EventDedupIndex(max_hamming=5)
    canonical = make_event(event_id="ev:c1#0", summary="陈锦标与李四确认50000元资金到账并已转给邓家俊")
    dup = make_event(event_id="ev:c2#0", summary="陈锦标与李四确认50000元资金到账并已转给邓家俊了")
    assert idx.check(canonical) is None
    idx.add(canonical)
    assert idx.check(dup) == canonical.event_id
    assert dup.duplicate_of == canonical.event_id
    assert len(idx) == 1  # 重复事件不登记，防链式指向


def test_mmr_select_trades_relevance_vs_diversity():
    ids = ["a", "b", "c"]
    scores = {"a": 1.0, "b": 0.8, "c": 0.6}
    embeddings = {"a": [1.0, 0.0], "b": [0.9999, 0.0001], "c": [0.0, 1.0]}
    picked = mmr_select(ids, scores, embeddings, k=2)
    assert picked[0] == "a"
    # b 与 a 几乎同向（diversity≈1）被重罚，c 与 a 正交——MMR 应选 c 而非 b
    assert picked[1] == "c"


# ---------------------------------------------------------------- 事件归一化


def test_normalize_event_result_basic():
    result = {
        "events": [
            {
                "event_type": "communication",
                "summary": "陈锦标与李四确认资金到账",
                "time_expr": "2025-10-05 14:20",
                "time_norm": "2025-10-05T14:20:00",
                "time_resolution": "minute",
                "action": "确认",
                "participants": [{"name": "陈锦标", "role": "询问", "entity_type": "人物"}],
            }
        ]
    }
    out = normalize_event_result(result, chunk_id="c1", kb_id="kb1", file_id="f1")
    events = out["events"]
    assert len(events) == 1
    assert events[0].event_id == "ev:c1#0"
    assert events[0].time_norm == datetime(2025, 10, 5, 14, 20)
    assert events[0].time_resolution == "second"  # 带秒的完整格式解析为 second
    assert events[0].value_weight == pytest.approx(EVENT_VALUE_WEIGHTS[EventType.COMMUNICATION])
    assert out["metadata"]["extractor_type"] == "event"


def test_normalize_event_result_multiple_events_suffix():
    result = {"events": [{"summary": f"事件{i}"} for i in range(3)]}
    out = normalize_event_result(result, chunk_id="c1")
    events = out["events"]
    assert [e.event_id for e in events] == ["ev:c1#0", "ev:c1#1", "ev:c1#2"]


def test_normalize_event_result_derives_file_id_from_chunk_id():
    """file_id 从 chunk_id 稳定派生（抽取器不透传）。

    回归：Event.file_id 曾全部为空 → delete_file 事件清理失效、
    L4 把所有文件的事件归为同一组。
    """
    out = normalize_event_result({"events": [{"summary": "转账"}]}, chunk_id="file_abc_chunk_2")
    assert out["events"][0].file_id == "file_abc"
    # 显式传入优先
    out2 = normalize_event_result(
        {"events": [{"summary": "转账"}]}, chunk_id="file_abc_chunk_2", file_id="explicit"
    )
    assert out2["events"][0].file_id == "explicit"
    # 无 _chunk_ 标记时退化为 chunk_id 本身
    out3 = normalize_event_result({"events": [{"summary": "转账"}]}, chunk_id="plain_id")
    assert out3["events"][0].file_id == "plain_id"


def test_normalize_event_result_truncates_at_max():
    result = {"events": [{"summary": f"事件{i}"} for i in range(MAX_EVENTS_PER_CHUNK + 3)]}
    out = normalize_event_result(result, chunk_id="c1")
    assert len(out["events"]) == MAX_EVENTS_PER_CHUNK


def test_normalize_event_result_skips_empty_and_dedups_summary():
    result = {"events": [{"summary": ""}, {"summary": "张三 转账"}, {"summary": "张三转账"}]}
    out = normalize_event_result(result, chunk_id="c1")
    assert len(out["events"]) == 1  # 空 summary 跳过；摘要归一化重复只留首个


def test_normalize_event_result_invalid_type_falls_to_none():
    out = normalize_event_result({"events": [{"summary": "纯聊天内容", "event_type": "not-a-type"}]}, chunk_id="c1")
    assert out["events"][0].event_type == EventType.NONE
    assert out["events"][0].value_weight == pytest.approx(0.1)


def test_parse_time_norm_formats():
    assert parse_time_norm("2025-10-05T14:24:00")[1] == "second"
    assert parse_time_norm("2025-10-05 14:24")[1] == "minute"
    assert parse_time_norm("2025-10-05")[1] == "day"
    assert parse_time_norm("2025年10月5日")[1] == "day"
    assert parse_time_norm("无法解析的时间") == (None, "unknown")
    assert parse_time_norm(None) == (None, "unknown")


# ---------------------------------------------------------------- 事件抽取器


class FakeChat:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls: list[list[dict]] = []

    async def __call__(self, messages: list[dict]) -> GeneralResponse:
        self.calls.append(messages)
        return GeneralResponse(self.payload)


def test_event_extractor_builds_prompt_and_parses():
    payload = json.dumps(
        {"events": [{"summary": "测试事件", "event_type": "none", "action": "无"}]}, ensure_ascii=False
    )
    chat = FakeChat(payload)
    extractor = EventGraphExtractor({"model_spec": "fake-model", "chat_model_fn": chat})
    out = asyncio.run(extractor.extract("待抽取文本", chunk_metadata={"chunk_id": "c1", "document_title": "笔录一"}))
    assert out["events"][0]["summary"] == "测试事件"
    assert "事件" in chat.calls[0][-1]["content"]  # 事件抽取 prompt 已注入


def test_event_extractor_rejects_gleaning():
    extractor = EventGraphExtractor({"model_spec": "m", "chat_model_fn": lambda m: None, "gleaning_count": 1})
    with pytest.raises(ValueError, match="不支持 gleaning"):
        extractor.validate_options()


# ---------------------------------------------------------------- 多跳检索


def test_compute_edge_strength_mentions():
    a = EventNodeData(
        "ev:c1#0", "事件A", datetime(2025, 10, 5, 14, 0), ("张三", "李四"), 1.0, "verified", "c1"
    )
    b = EventNodeData(
        "ev:c2#0", "事件B", datetime(2025, 10, 5, 14, 30), ("张三",), 1.0, "verified", "c2"
    )
    strength = compute_edge_strength(a, b, shared_anchor_count=1, window_seconds=86400)
    # b 只有 1 名参与者 → 锚点共享率 1.0（分母=min(2,1)=1）；30 分钟差 → 时间项 0.979
    # strength = 0.5*1.0 + 0.5*0.979
    assert strength == pytest.approx(0.98958, abs=1e-4)
    # 无共享锚点 + 时间缺失 → 锚点项 0、时间项中性 0.5 → 0.25
    d = EventNodeData("ev:c4#0", "事件D", None, ("王五",), 1.0, "verified", "c4")
    assert compute_edge_strength(a, d, shared_anchor_count=0, window_seconds=86400) == pytest.approx(0.25)
    # 同时刻同锚点比例 → 锚点项 0.5、时间项 1.0 → 0.75
    c = EventNodeData(
        "ev:c3#0", "事件C", datetime(2025, 10, 5, 14, 0), ("张三", "李四"), 1.0, "verified", "c3"
    )
    assert compute_edge_strength(a, c, shared_anchor_count=1, window_seconds=86400) == pytest.approx(0.75)


def test_compute_edge_strength_l4_edges():
    a = EventNodeData("ev:c1#0", "事件A", None, (), 1.0, "verified", "c1")
    b = EventNodeData("ev:c1#1", "事件B", None, (), 1.0, "verified", "c1")
    # NEXT 不要求 time_norm，直接给 1.0
    assert compute_edge_strength(a, b, 0, 86400, edge_type="NEXT") == pytest.approx(1.0)
    # TEMPORAL 按 Δt 衰减：0.5 + 0.5*(1 - 3600/86400)
    temporal = compute_edge_strength(a, b, 0, 86400, edge_type="TEMPORAL", delta_seconds=3600)
    assert temporal == pytest.approx(0.979166, abs=1e-5)
    # Δt 远超窗口 → 时间项趋 0
    far = compute_edge_strength(a, b, 0, 86400, edge_type="TEMPORAL", delta_seconds=10 * 86400)
    assert far == pytest.approx(0.5)
    assert compute_edge_strength(a, b, 0, 86400, edge_type="LOCATION") == pytest.approx(0.5)


def test_is_edge_coherent_l4_edges_always_true():
    a = EventNodeData("ev:c1#0", "A", None, ("张三",), 0.1, "unverified", "c1")
    b = EventNodeData("ev:c1#1", "B", None, (), 0.1, "unverified", "c1")
    # L4 边是确定性结构证据，即使无共享锚点、时间缺失也天然连贯
    assert is_edge_coherent(a, b, 0, 86400, edge_type="NEXT")
    assert is_edge_coherent(a, b, 0, 86400, edge_type="TEMPORAL")
    assert is_edge_coherent(a, b, 0, 86400, edge_type="LOCATION")
    # MENTIONS 需要共享锚点或时间窗内共享参与者
    assert not is_edge_coherent(a, b, 0, 86400, edge_type="MENTIONS")


async def _noop_expand(frontier: list[str]) -> dict[str, list[EventExpansion]]:
    return {fid: [] for fid in frontier}


async def test_beam_search_returns_seed_paths_on_disconnected_graph():
    seeds = [
        EventNodeData("ev:c1#0", "种子事件A", None, ("张三",), 1.0, "verified", "c1"),
        EventNodeData("ev:c2#0", "种子事件B", None, ("李四",), 1.0, "verified", "c2"),
    ]
    paths = await run_beam_search(seeds, {}, _noop_expand, MultiHopConfig(max_hops=2))
    assert len(paths) == 2  # 孤立图仍输出种子路径（死端保留语义）
    for path in paths:
        assert path.hop_count == 1
        assert path.hops[0].evidence_chunk_id in ("c1", "c2")


async def test_beam_search_expands_via_shared_anchor():
    c1 = EventNodeData("ev:c1#0", "事件一", None, ("张三",), 1.0, "verified", "c1")
    c2 = EventNodeData("ev:c2#0", "事件二", None, ("张三", "李四"), 1.0, "verified", "c2")
    c3 = EventNodeData("ev:c3#0", "事件三", None, ("李四",), 1.0, "verified", "c3")

    async def expand(frontier: list[str]) -> dict[str, list[EventExpansion]]:
        out: dict[str, list[EventExpansion]] = {}
        if "ev:c1#0" in frontier:
            out["ev:c1#0"] = [EventExpansion(candidate=c2, shared_anchor_names=("张三",))]
        if "ev:c2#0" in frontier:
            out["ev:c2#0"] = [EventExpansion(candidate=c3, shared_anchor_names=("李四",))]
        return out

    paths = await run_beam_search([c1], {}, expand, MultiHopConfig(max_hops=2))
    assert paths[0].hop_count == 3
    assert [hop.event_id for hop in paths[0].hops] == ["ev:c1#0", "ev:c2#0", "ev:c3#0"]
    assert paths[0].hops[1].via_anchor == "张三"
    assert paths[0].hops[1].edge_type == "MENTIONS"


async def test_beam_search_respects_value_weight_and_loop_detection():
    c1 = EventNodeData("ev:c1#0", "事件一", None, ("张三",), 1.0, "verified", "c1")
    low = EventNodeData("ev:c2#0", "闲聊事件", None, ("张三",), 0.1, "unverified", "c2")
    loop = EventNodeData("ev:c1#0", "事件一", None, ("张三",), 1.0, "verified", "c1")

    async def expand(frontier: list[str]) -> dict[str, list[EventExpansion]]:
        out: dict[str, list[EventExpansion]] = {}
        if "ev:c1#0" in frontier:
            out["ev:c1#0"] = [
                EventExpansion(candidate=low, shared_anchor_names=("张三",)),
                EventExpansion(candidate=loop, shared_anchor_names=("张三",)),
            ]
        return out

    paths = await run_beam_search([c1], {}, expand, MultiHopConfig(max_hops=2))
    # 低 value_weight 事件被剪（<0.15 且非 NEXT），回环被检测，路径不扩展
    assert paths[0].hop_count == 1
