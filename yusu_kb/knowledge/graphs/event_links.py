"""L4 确定性事件-事件边生成（零 LLM 成本）。

设计文档 §5.1 断言"不需要事件-事件边，共享锚点即是关系"，参考实现据此不
建任何事件-事件边，结果：锚点身份一旦分裂（R1）或事件无参与者（R3），
图就退化为大量二度"哑铃"与度 0 孤儿。L4 用**确定性结构证据**补齐三种边，
把"图是否成图"从依赖 LLM 锚点质量变为有下限保证：

- EVENT_NEXT：同文件按 (chunk_index, text_span[0], event_id) 排序的相邻对，
  每文件 |E|-1 条——保证同文件事件链不断裂（增量的关键兜底）；
- EVENT_TEMPORAL：按 time_norm 排序后滑动窗口 top-K（K=5，超窗口即 break），
  O(N log N)，附 delta_seconds 供打分衰减；
- EVENT_LOCATION：按归一化地点分组，组内按时间取 top-K 近邻，组上限 50。

关键简化：**只在文件内生成，不跨文件**——删文件只需重算该文件；跨文件
关联由 L1 共享锚点承担（这正是锚点身份的职责）。

生成时机：构建后处理阶段（非 flush 内，worker 完成顺序≠文档顺序）。
"""

from __future__ import annotations

from datetime import datetime
from itertools import pairwise
from typing import Any

from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage

# TEMPORAL 滑动窗口参数
_TEMPORAL_WINDOW_SECONDS = 86400.0  # 24h
_TEMPORAL_TOP_K = 5
# LOCATION 分组上限与近邻数
_LOCATION_GROUP_CAP = 50
_LOCATION_TOP_K = 5
# 单事件 L4 出度硬上限（防低质事件扩散面）
MAX_LINKS_PER_EVENT = 16


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _event_sort_key(event: dict[str, Any]) -> tuple:
    """文件内事件排序键：chunk_index → text_span 起点 → event_id（稳定）。"""
    text_span = event.get("text_span") or []
    span_start = text_span[0] if text_span else 0
    return (int(event.get("chunk_index") or 0), int(span_start or 0), str(event.get("event_id") or ""))


def generate_file_event_links(
    storage: NetworkXGraphStorage,
    file_id: str,
    *,
    temporal_window_seconds: float = _TEMPORAL_WINDOW_SECONDS,
    temporal_top_k: int = _TEMPORAL_TOP_K,
    location_group_cap: int = _LOCATION_GROUP_CAP,
    location_top_k: int = _LOCATION_TOP_K,
    max_links_per_event: int = MAX_LINKS_PER_EVENT,
) -> int:
    """为单文件生成 L4 事件-事件边（先清旧边再重算，幂等）。

    返回生成的边数。调用方保证在 storage 已含该文件全部事件节点后调用。
    """
    events = [
        event
        for event in storage.iter_events()
        if event.get("file_id") == file_id
    ]
    if not events:
        storage.clear_event_links(file_id)
        return 0
    if len(events) == 1:
        storage.clear_event_links(file_id)
        return 0

    storage.clear_event_links(file_id)

    link_count = 0
    # 出度计数：低质事件（chitchat/none）扩散面受限
    out_degree: dict[str, int] = {}

    def _add(u: str, v: str, edge_type: str, **attrs: Any) -> None:
        nonlocal link_count
        if out_degree.get(u, 0) >= max_links_per_event:
            return
        storage.add_event_link(u=u, v=v, edge_type=edge_type, file_id=file_id, **attrs)
        out_degree[u] = out_degree.get(u, 0) + 1
        link_count += 1

    # 1) NEXT：同文件相邻链（无 LLM 依赖的连通性兜底）
    ordered = sorted(events, key=_event_sort_key)
    for prev, nxt in pairwise(ordered):
        _add(prev["event_id"], nxt["event_id"], "EVENT_NEXT")

    # 2) TEMPORAL：time_norm 排序 + 滑动窗口 top-K（超窗口即 break）
    timed = [
        (event, _parse_time(event.get("time_norm")))
        for event in events
        if _parse_time(event.get("time_norm")) is not None
    ]
    timed.sort(key=lambda pair: pair[1])  # type: ignore[arg-type]
    for i, (event_a, t_a) in enumerate(timed):
        for j in range(i + 1, len(timed)):
            if j - i - 1 >= temporal_top_k:
                break
            _, t_b = timed[j]
            delta = (t_b - t_a).total_seconds()  # type: ignore[operator]
            if delta > temporal_window_seconds:
                break  # 已超窗口，后续更远，直接剪枝
            if delta <= 0:
                continue
            _add(
                event_a["event_id"],
                timed[j][0]["event_id"],
                "EVENT_TEMPORAL",
                delta_seconds=delta,
            )

    # 3) LOCATION：按归一化地点分组，组内按时间取近邻
    groups: dict[str, list[tuple[Any, Any]]] = {}
    for event in events:
        location = str(event.get("location") or "").strip()
        if not location:
            continue
        groups.setdefault(location, []).append((event, _parse_time(event.get("time_norm"))))
    for location, members in groups.items():
        if len(members) > location_group_cap:
            members = members[:location_group_cap]
        members.sort(
            key=lambda pair: (pair[1] is None, pair[1] or datetime.min, pair[0].get("event_id"))  # noqa: DTZ901 - 纯排序哨兵
        )
        for i, (event_a, t_a) in enumerate(members):
            for j in range(i + 1, len(members)):
                if j - i - 1 >= location_top_k:
                    break
                _, t_b = members[j]
                delta = (t_b - t_a).total_seconds() if t_a and t_b else None
                _add(
                    event_a["event_id"],
                    members[j][0]["event_id"],
                    "EVENT_LOCATION",
                    location=location,
                    delta_seconds=delta,
                )

    return link_count
