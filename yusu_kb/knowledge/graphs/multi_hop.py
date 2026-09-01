"""Beam Search 多跳事件检索（设计文档 §5.2）。

叙事类语义多跳通道：从 event summary 向量召回的种子事件出发，逐跳经
Anchor（ENTITY 节点的 EVENT_MENTIONS 边）或确定性事件-事件边（EVENT_NEXT /
EVENT_TEMPORAL / EVENT_LOCATION，L4 修复）扩展到候选事件，按

    score = α·sim(q, e.summary) + β·edge_strength + γ·e.value_weight

打分保留 top-B 条路径，输出带结构化解释（HopExplanation）的 MultiHopPath。

本模块只承载纯逻辑（打分/剪枝/路径组装），图存储 IO 由调用方经 expand
回调注入，保证核心逻辑可脱离存储层单测。

与源项目差异（NetworkX 适配）：
- EventNodeData.from_neo4j_row → from_storage_node（直接消费存储节点属性 dict，
  time_norm 为 ISO 字符串或 None，解析逻辑相同）。
- content_preview 不再作为节点属性存储（避免 JSON 膨胀）；证据摘录由调用方
  按 chunk_id 回查 chunk 节点后填充到 HopExplanation（see from_storage_node
  的 evidence_preview 参数）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Any

from yusu_kb.knowledge.graphs.event_schemas import HopExplanation, MultiHopPath


@dataclass
class MultiHopConfig:
    """Beam 多跳检索参数（默认值即设计文档 §5.2 的规格）。"""

    beam_width: int = 8
    max_hops: int = 5
    # G4 阈值：0.3→0.15。chitchat(0.2)/none(0.1) 仍被剪，但 unverified 的
    # 次要事件（如 statement 0.9×0.5=0.45、knowledge 0.7×0.5=0.35）可进入。
    min_value_weight: float = 0.15
    alpha: float = 0.5  # 查询相关性权重
    beta: float = 0.3  # 边强度权重
    gamma: float = 0.2  # 事件价值权重
    temporal_window_seconds: float = 86400.0  # 连贯性时间窗（默认 24h）
    seed_top_n: int = 8  # 向量召回种子数上限
    sim_pool_size: int = 64  # 向量召回候选池（种子 + 全程 sim 查表）
    top_k_paths: int = 5  # 输出路径数上限


@dataclass(frozen=True)
class EventNodeData:
    """多跳检索视角下的事件节点数据（存储节点属性投影）。"""

    event_id: str
    summary: str
    time_norm: datetime | None
    participants: tuple[str, ...]
    value_weight: float
    verification: str
    chunk_id: str
    evidence_preview: str = ""

    @classmethod
    def from_storage_node(cls, node: Mapping[str, Any]) -> EventNodeData:
        """从图存储的事件节点属性 dict 构造（time_norm 为 ISO 字符串或 None）。"""
        value_weight = node.get("value_weight")
        return cls(
            event_id=str(node.get("event_id") or ""),
            summary=str(node.get("summary") or ""),
            time_norm=_parse_time_norm(node.get("time_norm")),
            participants=tuple(str(p) for p in (node.get("participants") or []) if str(p).strip()),
            value_weight=float(value_weight) if value_weight is not None else 1.0,
            verification=str(node.get("verification") or "unverified"),
            chunk_id=str(node.get("chunk_id") or ""),
            evidence_preview=str(node.get("evidence_preview") or ""),
        )

    @classmethod
    def from_event_row(cls, row: Mapping[str, Any]) -> EventNodeData:
        """兼容事件-边扩展查询返回的行形态（含 summary 等候选事件字段）。"""
        return EventNodeData.from_storage_node(
            {
                "event_id": row.get("event_id"),
                "summary": row.get("summary"),
                "time_norm": row.get("time_norm"),
                "participants": row.get("participants"),
                "value_weight": row.get("value_weight"),
                "verification": row.get("verification"),
                "chunk_id": row.get("chunk_id"),
                "evidence_preview": row.get("evidence_preview"),
            }
        )


@dataclass(frozen=True)
class EventExpansion:
    """一次扩展：候选事件 + 连接信息（共享锚点名或事件-事件边类型）。"""

    candidate: EventNodeData
    shared_anchor_names: tuple[str, ...] = ()
    # L4 事件-事件边的连接类型；空表示经共享锚点（MENTIONS）跳转
    edge_type: str = "MENTIONS"
    edge_attr: dict[str, Any] | None = None  # TEMPORAL 边携带 delta_seconds / location


def compute_edge_strength(
    from_event: EventNodeData,
    to_event: EventNodeData,
    shared_anchor_count: int,
    window_seconds: float,
    edge_type: str = "MENTIONS",
    delta_seconds: float | None = None,
) -> float:
    """边强度 = 0.5·锚点共享率 + 0.5·时间邻近度。

    锚点共享率以两端事件参与者数较小者为分母（参与者即锚点来源）；
    任一端时间缺失时时间项取中性值 0.5，不因数据缺失作结构性惩罚。

    L4 边（确定性结构证据）不依赖锚点共享率，时间项权重更高：
    - NEXT：文件内相邻（chunk 顺序），时间项取 1.0（不要求 time_norm）；
    - TEMPORAL：时间项 = decay(Δt)（调用方已给出 delta_seconds）；
    - LOCATION：同地点事件，时间项取中性 0.5。
    """
    if edge_type == "NEXT":
        return 1.0
    if edge_type == "TEMPORAL" and delta_seconds is not None:
        temporal = 1.0 - min(abs(delta_seconds) / max(window_seconds, 1e-9), 1.0)
        return 0.5 + 0.5 * temporal
    if edge_type == "LOCATION":
        return 0.5
    # MENTIONS（共享锚点）
    denominator = max(1, min(len(from_event.participants), len(to_event.participants)))
    anchor_ratio = min(1.0, shared_anchor_count / denominator)
    if from_event.time_norm and to_event.time_norm:
        delta = abs((from_event.time_norm - to_event.time_norm).total_seconds())
        temporal = 1.0 - min(delta / max(window_seconds, 1e-9), 1.0)
    else:
        temporal = 0.5
    return 0.5 * anchor_ratio + 0.5 * temporal


def is_edge_coherent(
    from_event: EventNodeData,
    to_event: EventNodeData,
    shared_anchor_count: int,
    window_seconds: float,
    edge_type: str = "MENTIONS",
) -> bool:
    """连贯性硬约束（防止语义漂移）。

    - MENTIONS（共享锚点）：共享 ≥1 个锚点，或 time_norm 相差在窗口内且共享
      ≥1 个参与者；
    - L4 边（NEXT/TEMPORAL/LOCATION）：**视为天然连贯**——确定性结构证据
      （同一文件相邻 / 时间邻近 / 同地点），不是 LLM 语义猜测，不需要二次守卫。
    """
    if edge_type != "MENTIONS":
        return True
    if shared_anchor_count >= 1:
        return True
    if from_event.time_norm and to_event.time_norm:
        delta = abs((from_event.time_norm - to_event.time_norm).total_seconds())
        if delta <= window_seconds and set(from_event.participants) & set(to_event.participants):
            return True
    return False


async def run_beam_search(
    seeds: Sequence[EventNodeData],
    sim_lookup: dict[str, float],
    expand: Callable[[list[str]], Awaitable[dict[str, list[EventExpansion]]]],
    config: MultiHopConfig,
) -> list[MultiHopPath]:
    """Beam Search 主循环，复杂度 O(H × B × avg_degree)。

    每跳对当前 beam 中路径终点事件做一次批量扩展（expand 收到去重后的
    终点 id 列表），逐候选过滤（环/G4 阈值/连贯性硬约束）→ 打分 →
    与本跳无法扩展的死端路径合并后保留 top-B。终点无法扩展的路径保留
    在 beam 中参与排序，保证孤立图上仍有种子路径可输出。
    """
    if not seeds:
        return []
    path_ids = count()
    beam = [
        _BeamPath(
            id=next(path_ids),
            events=[seed],
            explanations=[_build_seed_hop(seed)],
            edge_strengths=[],
            visited={seed.event_id},
            score=0.0,
        )
        for seed in seeds
    ]

    for hop_index in range(1, config.max_hops + 1):
        frontier_ids = list(dict.fromkeys(path.events[-1].event_id for path in beam))
        expansion_map = await expand(frontier_ids)
        extended: list[_BeamPath] = []
        dead_ends: list[_BeamPath] = []
        for path in beam:
            end = path.events[-1]
            path_extended = False
            for expansion in expansion_map.get(end.event_id, ()):
                candidate = expansion.candidate
                if candidate.event_id in path.visited:
                    continue  # 环检测
                # G4 阈值剪枝：NEXT 相邻事件豁免——文件内相邻是硬结构信号，
                # 不应被低 value_weight 剪掉（chitchat 相邻事件可能承载上下文）。
                if candidate.value_weight < config.min_value_weight and expansion.edge_type != "NEXT":
                    continue
                shared_count = len(expansion.shared_anchor_names)
                if not is_edge_coherent(
                    end,
                    candidate,
                    shared_count,
                    config.temporal_window_seconds,
                    edge_type=expansion.edge_type,
                ):
                    continue
                edge_strength = compute_edge_strength(
                    end,
                    candidate,
                    shared_count,
                    config.temporal_window_seconds,
                    edge_type=expansion.edge_type,
                    delta_seconds=_delta_seconds(expansion.edge_attr),
                )
                sim = sim_lookup.get(candidate.event_id, 0.0)
                hop_score = (
                    config.alpha * sim + config.beta * edge_strength + config.gamma * candidate.value_weight
                )
                extended.append(path.extend(candidate, expansion, edge_strength, hop_score, next(path_ids)))
                path_extended = True
            if not path_extended:
                # 死端路径（无候选或候选全被过滤）：过滤条件随路径单调收紧
                # （visited 只增不减），此后不可能再扩展，保留参与最终排序。
                dead_ends.append(path)
        if not extended:
            break
        beam = _select_beam(extended + dead_ends, config.beam_width)

    ranked = sorted(beam, key=lambda p: (-p.score, -len(p.events), p.events[0].event_id))
    return [_finalize_path(path) for path in ranked[: config.top_k_paths]]


def _delta_seconds(edge_attr: dict[str, Any]) -> float | None:
    """从边属性取 delta_seconds（TEMPORAL 边由生成侧写入）。"""
    if not edge_attr:
        return None
    value = edge_attr.get("delta_seconds")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class _BeamPath:
    """beam 中的一条进行中路径（内部状态，输出时转为 MultiHopPath）。"""

    id: int
    events: list[EventNodeData]
    explanations: list[HopExplanation]
    edge_strengths: list[float]
    visited: set[str]
    score: float

    def extend(
        self,
        candidate: EventNodeData,
        expansion: EventExpansion,
        edge_strength: float,
        hop_score: float,
        new_id: int,
    ) -> _BeamPath:
        hop_index = len(self.events)
        return _BeamPath(
            id=new_id,
            events=[*self.events, candidate],
            explanations=[
                *self.explanations,
                _build_expansion_hop(self.events[-1], candidate, expansion, hop_index),
            ],
            edge_strengths=[*self.edge_strengths, edge_strength],
            visited={*self.visited, candidate.event_id},
            score=self.score + hop_score,
        )


def _select_beam(paths: list[_BeamPath], beam_width: int) -> list[_BeamPath]:
    """按分数保留 top-B，并以事件序列去重（不同锚点到达同序列只留最高分）。"""
    ranked = sorted(paths, key=lambda p: (-p.score, -len(p.events), p.events[0].event_id))
    selected: list[_BeamPath] = []
    seen_sequences: set[tuple[str, ...]] = set()
    for path in ranked:
        sequence = tuple(event.event_id for event in path.events)
        if sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)
        selected.append(path)
        if len(selected) >= beam_width:
            break
    return selected


def _finalize_path(path: _BeamPath) -> MultiHopPath:
    verified = sum(1 for event in path.events if event.verification == "verified")
    coherence = sum(path.edge_strengths) / len(path.edge_strengths) if path.edge_strengths else 0.0
    return MultiHopPath(
        hops=path.explanations,
        coherence_score=round(coherence, 6),
        path_precision=round(verified / len(path.events), 6),
    )


def _build_seed_hop(event: EventNodeData) -> HopExplanation:
    """种子跳（hop 0）：无 via_anchor，指回 chunk 证据。"""
    return HopExplanation(
        event_id=event.event_id,
        summary=event.summary,
        time_norm=event.time_norm,
        participants=list(event.participants),
        evidence_chunk_id=event.chunk_id,
        evidence_quote=_truncate_quote(event.evidence_preview),
    )


def _build_expansion_hop(
    prev: EventNodeData,
    candidate: EventNodeData,
    expansion: EventExpansion,
    hop_index: int,
) -> HopExplanation:
    """扩展跳：可解释性四要素（via_anchor/边类型/关系说明/时间差）+ 证据引用。"""
    anchor_name = expansion.shared_anchor_names[0] if expansion.shared_anchor_names else None
    edge_type = expansion.edge_type
    time_delta = None
    if prev.time_norm and candidate.time_norm:
        time_delta = _format_time_delta((candidate.time_norm - prev.time_norm).total_seconds())
    if edge_type == "MENTIONS":
        relation_to_prev = f"共享参与者 {anchor_name}" if anchor_name else ""
    elif edge_type == "NEXT":
        relation_to_prev = "同一文件相邻事件"
    elif edge_type == "TEMPORAL":
        relation_to_prev = "时间邻近事件"
    else:  # LOCATION
        location = (expansion.edge_attr or {}).get("location") or ""
        relation_to_prev = f"同一地点 {location}" if location else "同一地点事件"
    return HopExplanation(
        hop_index=hop_index,
        event_id=candidate.event_id,
        summary=candidate.summary,
        time_norm=candidate.time_norm,
        participants=list(candidate.participants),
        via_anchor=anchor_name,
        edge_type=edge_type,  # type: ignore[arg-type]  # Literal 覆盖见 event_schemas
        relation_to_prev=relation_to_prev,
        time_delta=time_delta,
        evidence_chunk_id=candidate.chunk_id,
        evidence_quote=_truncate_quote(candidate.evidence_preview),
    )


def _format_time_delta(delta_seconds: float) -> str:
    """Δt 人类可读："2小时15分钟后" / "3天前"（方向相对前一跳事件）。"""
    seconds = abs(int(delta_seconds))
    direction = "后" if delta_seconds >= 0 else "前"
    if seconds < 60:
        return f"{seconds}秒{direction}"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}分钟{direction}"
    hours, remainder_minutes = divmod(seconds // 60, 60)
    if hours < 24:
        return f"{hours}小时{remainder_minutes}分钟{direction}" if remainder_minutes else f"{hours}小时{direction}"
    days, remainder_hours = divmod(hours, 24)
    return f"{days}天{remainder_hours}小时{direction}" if remainder_hours else f"{days}天{direction}"


def _truncate_quote(content_preview: str, max_chars: int = 100) -> str:
    quote = content_preview.strip()
    if len(quote) <= max_chars:
        return quote
    return quote[:max_chars] + "…"


def _parse_time_norm(value: Any) -> datetime | None:
    """time_norm 落库为 ISO8601 字符串（存储层序列化），读回按需解析。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


__all__ = [
    "EventExpansion",
    "EventNodeData",
    "MultiHopConfig",
    "compute_edge_strength",
    "is_edge_coherent",
    "run_beam_search",
]
