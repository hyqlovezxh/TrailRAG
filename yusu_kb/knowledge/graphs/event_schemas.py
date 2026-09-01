"""事件驱动知识表示的数据模型（迁移自源项目事件化改造，适配 NetworkX 单命名空间）。

Event 为独立图节点（非 Chunk 双标签），Chunk 节点零改动；Anchor 复用现有
Entity 节点。本模块只承载纯数据契约，不涉及 IO；图写入/检索层按需序列化。

与源项目差异（NetworkX 适配）：
- ``make_event_id`` 恒返回 ``f"ev:{chunk_id}#{idx}"`` 形态（1:1 也带 ``#0``）。
  源项目在 1:1 时返回 chunk_id 本身，Neo4j 靠标签隔离所以无事；但 NetworkX
  是单命名空间，event_id 与 chunk_id 相同会直接覆盖 chunk 节点（图损坏）。
  ``ev:`` 前缀同时保证事件节点 id 与实体/Chunk 永不冲突。
- 节点属性序列化约束：time_norm 落库为 ISO 字符串、text_span 落库为 list
  （NetworkX JSON 持久化不接受 datetime / tuple），见 graph_storage 写入层。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# 1 chunk → N 事件时的上限：防 LLM 碎片化（预检门实测校准，超限截断于归一化层）
MAX_EVENTS_PER_CHUNK = 5
# 单事件参与者上限：笔录/群聊场景的合理上界，防锚点度数爆炸
MAX_PARTICIPANTS_PER_EVENT = 12

# 事件 id 前缀：NetworkX 单命名空间下与 chunk_id 隔离（见模块 docstring）
EVENT_ID_PREFIX = "ev:"


class EventType(str, Enum):
    """事件类型：领域无关基类 + 公安领域扩展 + 噪声显式类型（护栏 G3）。"""

    TRANSFER = "transfer"  # 资金流转
    COMMUNICATION = "communication"  # 通讯联络
    MOVEMENT = "movement"  # 人员/车辆移动
    MEETING = "meeting"  # 会面聚集
    TRANSACTION = "transaction"  # 物品交易
    STATEMENT = "statement"  # 供述/证言（笔录）
    STATUTE = "statute"  # 法条/规范（通用领域）
    KNOWLEDGE = "knowledge"  # 通用知识陈述（通用领域）
    CHITCHAT = "chitchat"  # 无实质内容闲聊
    NONE = "none"  # 判定为非事件


# 护栏 G4（对应风险 R12）：按 event_type 赋传播权重——
# "让噪声有位置，但不让噪声有分量"：chitchat/none 仍建节点保覆盖率，
# 在 PPR 传播与多跳扩展中结构性降权。
EVENT_VALUE_WEIGHTS: dict[EventType, float] = {
    EventType.TRANSFER: 1.0,
    EventType.COMMUNICATION: 1.0,
    EventType.MOVEMENT: 1.0,
    EventType.MEETING: 1.0,
    EventType.TRANSACTION: 1.0,
    EventType.STATEMENT: 0.9,
    EventType.STATUTE: 0.8,
    EventType.KNOWLEDGE: 0.7,
    EventType.CHITCHAT: 0.2,
    EventType.NONE: 0.1,
}

DEFAULT_EVENT_VALUE_WEIGHT = 1.0

# 参与者类型枚举：与实体路径建库规范对齐（注意 N9：事件枚举用间隔号，
# 实体 schema 用斜杠，两条路径合图前必须经 normalize_label_form 归一）。
PARTICIPANT_ENTITY_TYPES = ("人物", "组织机构", "地点", "资金·账户", "通讯·账号", "其他")

TimeResolution = Literal["second", "minute", "hour", "day", "month", "unknown"]


class Participant(BaseModel):
    """事件参与者。name 为原文逐字（锚点与 G1 校验的依据）。"""

    name: str = Field(min_length=1)
    role: str = ""  # 角色描述：嫌疑人/受害人/主叫/驾驶 等
    entity_type: str = "人物"

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("participant.name 不能为空")
        return v

    @field_validator("entity_type")
    @classmethod
    def _validate_entity_type(cls, v: str) -> str:
        v = v.strip() or "人物"
        if v not in PARTICIPANT_ENTITY_TYPES:
            v = "其他"
        return v


class EventRecord(BaseModel):
    """事件节点数据契约。

    event_id = ``ev:{chunk_id}#{idx}``（1:1 时 idx=0，见 make_event_id）；
    text_span 仅 1:N 时非空，记录事件在 chunk 内的字符区间。
    summary 为向量化对象，必须是完整命题（主体+动作+客体+关键限定）。
    """

    event_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    kb_id: str = ""
    file_id: str = ""

    # —— n 元语义场（公理 A2：槽位是一等字段，不是属性碎片）——
    event_type: EventType = EventType.NONE
    summary: str
    time_expr: str = ""  # 原文时间表达逐字（含【时间段 X ~ Y】前缀），回溯依据
    time_norm: datetime | None = None
    time_resolution: TimeResolution = "unknown"
    location: str | None = None
    action: str = ""  # 核心动作谓词
    participants: list[Participant] = Field(default_factory=list, max_length=MAX_PARTICIPANTS_PER_EVENT)
    objects: list[str] = Field(default_factory=list)  # 涉及物品/工具/标的
    amount: float | None = None
    exact_identifiers: list[str] = Field(default_factory=list)  # 手机号/账号/车牌/单号，可正则验证

    # —— 工程字段 ——
    text_span: tuple[int, int] | None = None
    verification: Literal["verified", "unverified"] = "unverified"  # 护栏 G1 结果
    value_weight: float = DEFAULT_EVENT_VALUE_WEIGHT  # 护栏 G4 传播权重
    duplicate_of: str | None = None  # 护栏 G2：SimHash 近重复指向的规范事件 id

    @field_validator("summary")
    @classmethod
    def _summary_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("event.summary 不能为空")
        return v

    @field_validator("exact_identifiers")
    @classmethod
    def _dedup_identifiers(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in v:
            item = str(item).strip()
            if item and item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped

    @field_validator("value_weight")
    @classmethod
    def _clamp_weight(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("text_span")
    @classmethod
    def _validate_text_span(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is None:
            return None
        start, end = int(v[0]), int(v[1])
        if start < 0 or end < start:
            raise ValueError(f"text_span 非法: {v}")
        return (start, end)

    @property
    def effective_value_weight(self) -> float:
        """G1 校验失败时传播权重减半（unverified 降权，与 verification 字段联动）。"""
        return self.value_weight * (0.5 if self.verification == "unverified" else 1.0)


def make_event_id(chunk_id: str, idx: int) -> str:
    """事件 id 生成规则：恒为 ``ev:{chunk_id}#{idx}``（1:1 时 idx=0）。

    NetworkX 单命名空间适配：源项目 1:1 返回 chunk_id 本身，会覆盖 chunk 节点。
    按 chunk_id 归组删除的语义不变（event_id 携带 chunk_id 前缀）。
    """
    return f"{EVENT_ID_PREFIX}{chunk_id}#{idx}"


class HopExplanation(BaseModel):
    """多跳路径的单跳解释（公理 A5：每一跳都能指回原文证据）。"""

    hop_index: int = 0
    event_id: str
    summary: str
    time_norm: datetime | None = None
    participants: list[str] = Field(default_factory=list)
    # —— 可解释性四要素 ——
    via_anchor: str | None = None  # 经哪个锚点跳到此事件
    # MENTIONS=共享锚点跳；TEMPORAL/LOCATION/NEXT=确定性事件-事件边（L4 修复）；
    # SIMILAR/ADJACENT/LEXICAL/EXACT_ID/REAL=检索时隐式 chunk 图边（implicit_graph），
    # 其中 REAL 为混合模式下真实图的 chunk-chunk 证据边。
    edge_type: Literal[
        "MENTIONS",
        "TEMPORAL",
        "LOCATION",
        "NEXT",
        "SIMILAR",
        "ADJACENT",
        "LEXICAL",
        "EXACT_ID",
        "REAL",
    ] = "MENTIONS"
    relation_to_prev: str = ""  # 人类可读："共享参与者 李四"
    time_delta: str | None = None  # "2h15m 后"
    evidence_chunk_id: str = ""
    evidence_quote: str = ""


class MultiHopPath(BaseModel):
    """Beam 多跳检索的完整路径输出。hop_count 由 hops 长度同步，不接受外部赋值。"""

    hops: list[HopExplanation] = Field(min_length=1)
    hop_count: int = 1
    coherence_score: float = 0.0  # 路径整体连贯性
    path_precision: float = 0.0  # 每跳经证据校验的比例

    @model_validator(mode="after")
    def _sync_hop_count(self) -> MultiHopPath:
        self.hop_count = len(self.hops)
        return self
