"""图谱构建相关的纯函数工具集。

将数据变换逻辑从图存储服务中抽离，
使 service 类专注于 I/O 和业务编排。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from yusu_kb.utils.hash_utils import hashstr

# GKB-10: 专用描述分隔符，替代易与自然文本冲突的 "; "
# 与 LightRAG 的 <SEP> 约定一致，避免误拆含分号的正文
DESC_SEPARATOR = "<SEP>"

# 末尾常见标点（中英文句号、逗号、分号、感叹号、问号、冒号）
_TRAILING_PUNCT_PATTERN = re.compile(r"[。.，,；;！!？?：:]+$")


def normalize_entity_name(text: str, *, case_sensitive: bool = False) -> str:
    """统一实体名称：NFKC 归一化 + 去首尾空白/末尾标点 + 压缩内部空白。

    - NFKC：全角→半角（"ＡＢＣ"→"ABC"）、兼容等价形式合并
    - 去末尾标点：清理 LLM 抽取时附带的句末标点（"阿里巴巴。"→"阿里巴巴"）
    - 压缩内部空白：合并连续空格

    Args:
        text: 原始实体名称
        case_sensitive: 是否保留大小写。默认 False（向后兼容），
            启用时不做小写化，以区分大小写敏感的专有名词。
    """
    if not text:
        return ""
    # NFKC 归一化：处理全角/半角混用、兼容等价形式
    normalized = unicodedata.normalize("NFKC", text)
    # 去首尾空白
    normalized = normalized.strip()
    # 去末尾常见标点（LLM 抽取的句末实体可能附带标点）
    normalized = _TRAILING_PUNCT_PATTERN.sub("", normalized)
    # 压缩内部连续空白
    normalized = " ".join(normalized.split())
    if not case_sensitive:
        normalized = normalized.lower()
    return normalized


def compute_entity_id(kb_id: str, normalized_name: str, label: str) -> str:
    return hashstr(f"{kb_id}:{normalized_name}:{label}", length=32)


# S2-B2 关系 label 代码级归一化（与抽取 prompt 第 6 条双保险）：
# LLM 对同类关系可能产出同义 label（"转账给/汇款至/打入"），label 进入 triple_id
# 后同义分裂为不同边，破坏跨 chunk 图路径一致性。此处仅做严格等值映射，不做子串替换。
_RELATION_LABEL_CANONICAL: dict[str, str] = {
    "转账": "资金转账",
    "转账给": "资金转账",
    "汇款": "资金转账",
    "汇款至": "资金转账",
    "汇入": "资金转账",
    "汇出": "资金转账",
    "转入": "资金转账",
    "转出": "资金转账",
    "打入": "资金转账",
    "转给": "资金转账",
    "付款": "资金转账",
    "支付": "资金转账",
    "收款": "资金转账",
    "通话": "通话联系",
    "呼叫": "通话联系",
    "拨打": "通话联系",
    "主叫": "通话联系",
    "被叫": "通话联系",
    "通联": "通话联系",
    "经过": "驾车经过",
    "途经": "驾车经过",
    "过卡": "驾车经过",
    "驶经": "驾车经过",
    "驶过": "驾车经过",
}


def normalize_relation_label(label: str | None) -> str:
    """归一化关系 label：严格等值映射同义 label 到统一枚举，未命中保持原文。"""
    if not label:
        return "RELATED_TO"
    return _RELATION_LABEL_CANONICAL.get(label.strip(), label.strip())


# S4 实体 label 代码级归一化：编号类实体的同义 label 统一，避免同名不同 label 分裂节点
# （compute_entity_id 含 label）。严格等值映射。
_ENTITY_LABEL_CANONICAL: dict[str, str] = {
    "电话": "手机号",
    "手机": "手机号",
    "号码": "手机号",
    "电话号码": "手机号",
    "手机号码": "手机号",
    "账号": "银行账户",
    "银行卡": "银行账户",
    "卡号": "银行账户",
    "银行账号": "银行账户",
}

# S4 可被吸收的别名侧 label：与 "Entity" 占位实体同等对待，同名类型化实体存在时合并
_ALIAS_SIDE_LABELS: tuple[str, ...] = ("Entity", "代号", "昵称", "别名")


def apply_entity_aliases(normalized_result: dict[str, Any], alias_map: dict[str, str] | None) -> dict[str, Any]:
    """按 KB 级别名表将实体名重写到规范名（S4 代号-实名跨源对齐）。

    alias_map 形如 {"db": "陈锦标", "hj": "胡杰"}；匹配基于 normalize_entity_name
    （大小写/全角/空白不敏感），重写保留规范名原文。就地修改并返回同一对象。
    构建期后处理，不触碰抽取缓存——别名表变更后重建图谱即生效（新库灰度策略）。
    """
    if not alias_map:
        return normalized_result
    lookup = {normalize_entity_name(str(k)): str(v) for k, v in alias_map.items() if str(v).strip()}
    if not lookup:
        return normalized_result

    def resolve(text: Any) -> str | None:
        if not isinstance(text, str):
            return None
        return lookup.get(normalize_entity_name(text))

    for entity in normalized_result.get("entities") or []:
        canonical = resolve(entity.get("text"))
        if canonical is not None and canonical != entity.get("text"):
            entity["text"] = canonical

    for relation in normalized_result.get("relations") or []:
        for endpoint in ("source", "target"):
            target = relation.get(endpoint)
            if isinstance(target, dict):
                canonical = resolve(target.get("text"))
                if canonical is not None and canonical != target.get("text"):
                    target["text"] = canonical

    return normalized_result


def compute_triple_id(
    kb_id: str,
    source_normalized_name: str,
    source_label: str,
    relation_type: str,
    target_normalized_name: str,
    target_label: str,
) -> str:
    return hashstr(
        f"{kb_id}:{source_normalized_name}:{source_label}:{relation_type}:{target_normalized_name}:{target_label}",
        length=32,
    )


def build_graph_payload(normalized_result: dict[str, Any], *, case_sensitive: bool = False) -> dict[str, Any]:
    """将抽取器产出的标准化结果转换为 Neo4j 写入所需的图结构。

    返回的 entities 已完成去重合并：同名同 label 的实体只保留一份，
    属性（attributes）取并集。

    Args:
        normalized_result: 抽取器产出的标准化结果
        case_sensitive: 是否保留实体名称大小写。默认 False（向后兼容）。
            启用时不做小写化，以区分大小写敏感的专有名词。
    """
    entities: list[dict[str, Any]] = []
    entity_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def add_entity(entity: dict[str, Any]) -> str:
        # S4：实体 label 同义归一（电话/手机/号码 → 手机号），进入去重 key 前统一
        entity_label = _ENTITY_LABEL_CANONICAL.get(entity.get("label") or "Entity", entity.get("label") or "Entity")
        key = (normalize_entity_name(entity["text"], case_sensitive=case_sensitive), entity_label)
        existing = entity_by_key.get(key)
        if existing is not None:
            known_attributes = {(attr["text"], attr["label"]) for attr in existing.get("attributes") or []}
            for attribute in entity.get("attributes") or []:
                attribute_key = (attribute["text"], attribute["label"])
                if attribute_key not in known_attributes:
                    existing.setdefault("attributes", []).append(attribute)
                    known_attributes.add(attribute_key)
            # V2: 累积 descriptions（同名实体在不同 relation 出现处可能有不同描述）
            new_descs = entity.get("descriptions") or []
            if new_descs:
                existing_descs = existing.setdefault("descriptions", [])
                for d in new_descs:
                    if d not in existing_descs:
                        existing_descs.append(d)
            return existing["id"]

        graph_entity = {
            "id": f"e{len(entities) + 1}",
            "text": entity["text"],
            # S4：存储 label 同步归一化，保证与去重 key 一致，
            # 下游 compute_entity_id/Neo4j label 不再因同义 label 分裂节点
            "label": entity_label,
            "attributes": list(entity.get("attributes") or []),
        }
        # V2: 透传 descriptions 列表，供 _build_entity_records → DescriptionMerger 使用
        descriptions = entity.get("descriptions") or []
        if descriptions:
            graph_entity["descriptions"] = list(descriptions)
            # 同步设置 description string，供 _build_entity_records 读取并写入 Milvus content 和 Neo4j
            # 修复字段名不匹配 bug：此前仅设置 descriptions(list)，_build_entity_records 读取 description(string) 永远为空
            graph_entity["description"] = DESC_SEPARATOR.join(descriptions)
        entities.append(graph_entity)
        entity_by_key[key] = graph_entity
        return graph_entity["id"]

    for entity in normalized_result["entities"]:
        add_entity(entity)

    relations = []
    for relation in normalized_result["relations"]:
        rel_dict = {
            "source": add_entity(relation["source"]),
            "target": add_entity(relation["target"]),
            "text": relation["text"],
            # S2-B2：同义 label 归一化，保证 triple_id 去重与图路径 label 一致性
            "label": normalize_relation_label(relation.get("label")),
        }
        # V2: 透传 description，供 _build_triple_records → DescriptionMerger 使用
        rel_desc = relation.get("description")
        if rel_desc:
            rel_dict["description"] = rel_desc
        relations.append(rel_dict)

    _merge_placeholder_entities(entities, relations, entity_by_key, case_sensitive=case_sensitive)

    return {"entities": entities, "relations": relations, "metadata": normalized_result["metadata"]}


def _merge_placeholder_entities(
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    entity_by_key: dict[tuple[str, str], dict[str, Any]],
    *,
    case_sensitive: bool = False,
) -> None:
    """将 label="Entity" 的占位实体（及 S4 扩展：代号/昵称/别名声侧实体）合并入同名类型化实体。

    当 _normalize_relation_endpoint 无法匹配已有实体时，会创建 label="Entity" 的占位实体。
    若同名类型化实体已存在（如 label="人物"），占位实体与之分裂为两个节点，
    导致类型化实体成为孤儿（关系全挂在占位实体上）。

    S4 扩展：别名表重写后，代号/昵称侧实体与实名实体同名但 label 不同（代号 vs 人物），
    同样吸收进类型化实体，完成代号-实名的节点级归一（替代改动 compute_entity_id 的重建方案）。

    此处将占位实体合并到类型化实体：
    - 合并占位实体的 descriptions/attributes 到类型化实体
    - 重定向关系端点：占位实体 id → 类型化实体 id
    - 从 entities 列表移除占位实体

    同时修复历史 L1 缓存数据：重建时经过 build_graph_payload 即自动纠正，无需重新调用 LLM。
    """
    placeholder_redirect: dict[str, str] = {}
    for entity in entities:
        if entity.get("label") not in _ALIAS_SIDE_LABELS:
            continue
        norm_name = normalize_entity_name(entity["text"], case_sensitive=case_sensitive)
        typed_target = None
        for (existing_name, existing_label), existing_entity in entity_by_key.items():
            if (
                existing_name == norm_name
                and existing_label not in _ALIAS_SIDE_LABELS
                and existing_entity["id"] != entity["id"]
            ):
                typed_target = existing_entity
                break
        if typed_target is None:
            continue
        # 合并 attributes（与 add_entity 内合并逻辑一致）
        known_attrs = {(a["text"], a["label"]) for a in typed_target.get("attributes") or []}
        for attr in entity.get("attributes") or []:
            attr_key = (attr["text"], attr["label"])
            if attr_key not in known_attrs:
                typed_target.setdefault("attributes", []).append(attr)
                known_attrs.add(attr_key)
        # 合并 descriptions 并同步 description string
        new_descs = entity.get("descriptions") or []
        if new_descs:
            target_descs = typed_target.setdefault("descriptions", [])
            for d in new_descs:
                if d not in target_descs:
                    target_descs.append(d)
            typed_target["description"] = DESC_SEPARATOR.join(target_descs)
        placeholder_redirect[entity["id"]] = typed_target["id"]

    if not placeholder_redirect:
        return
    entities[:] = [e for e in entities if e["id"] not in placeholder_redirect]
    for rel in relations:
        if rel["source"] in placeholder_redirect:
            rel["source"] = placeholder_redirect[rel["source"]]
        if rel["target"] in placeholder_redirect:
            rel["target"] = placeholder_redirect[rel["target"]]


# ─── 事件路径锚点身份规范化（L1 修复：锚点必须是真实世界指称的规范函数）───
# 根因：参考实现把 LLM 每次自由输出的 entity_type 直接塞进 compute_entity_id
# 的哈希，同一实体在不同事件被标不同类型 → 分裂为多个锚点 → 事件间隐式边
# （共享锚点）全部断裂 → 图退化为二度"哑铃"。
#
# Tier-A（纯函数，worker 内并行）在此完成：label 标点归一（N9：事件枚举用
# 间隔号 ·、实体 schema 用斜杠 /，必须统一才能跨路径合图）+ 名称规范化 +
# 实体 label 同义归一 + 标识符正则定型。产出 (canonical_name, tier_a_label)。

# 标识符正则定型：命中即强制为对应类型（L3 兜底锚点与 participants 正则校验共用）。
# 顺序即优先级：手机号 → 银行账户 → 车牌 → 单号 → 其他。
_IDENTIFIER_LABEL_RULES: tuple[tuple[str, str], ...] = (
    (r"^1[3-9]\d{9}$", "手机号"),
    (r"^\d{16,19}$", "银行账户"),
    (r"^[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z](?:[·・]?[A-Z0-9]){5,6}$", "车牌"),
    (r"^[A-Z]{2,}\d{3,}", "单号"),
)

# 锚点 label 类型优先级（Tier-B 裁决）：值越小越优先。
# 人物 0 > 组织机构 1 > 地点 2 > 资金·账户/银行账户/手机号/车牌/单号 3 > 通讯·账号 4 > 其他 9。
LABEL_PRIORITY: dict[str, int] = {
    "人物": 0,
    "组织机构": 1,
    "地点": 2,
    "资金·账户": 3,
    "银行账户": 3,
    "手机号": 3,
    "车牌": 3,
    "单号": 3,
    "通讯·账号": 4,
    "其他": 9,
    "Entity": 9,
}


def normalize_label_form(label: str | None) -> str:
    """统一锚点/实体 label 的标点形式（N9 修复：间隔号/斜杠/空白）。

    产出稳定形态：`资金·账户` / `通讯·账号`（保留间隔号）——LABEL_PRIORITY
    与 _ENTITY_LABEL_CANONICAL 均按该形态登记。
    """
    if not label:
        return "其他"
    label = label.strip()
    if not label:
        return "其他"
    # 斜杠系 → 间隔号系：资金/账户 → 资金·账户，通讯/账号 → 通讯·账号
    if "/" in label or "／" in label:
        parts = [p for p in re.split(r"[/／]", label) if p.strip()]
        label = "·".join(parts)
    # 多余空白压缩
    label = " ".join(label.split())
    return label or "其他"


def _entity_label_canonical(label: str) -> str:
    """实体 label 同义归一（电话/手机/号码→手机号，账号/银行卡→银行账户）。"""
    return _ENTITY_LABEL_CANONICAL.get(label, label)


def canonical_anchor_name(name: str) -> str:
    """锚点名规范化：与实体路径完全一致的 normalize_entity_name(case_sensitive=False)。

    参考实现硬编码 case_sensitive=True，导致含拉丁字符的名称（W01 / FB2026-0001）
    在事件路径与实体路径算出不同 entity_id，跨路径永远不合图（R2）。
    """
    return normalize_entity_name(name, case_sensitive=False)


def canonical_anchor_label(entity_type: str | None, *, name: str = "") -> str:
    """锚点 label 规范化：标点统一 + 同义归一 + 标识符正则定型。

    - 无类型或未知类型 → 其他/Entity（保留原语义）；
    - 名称本身命中标识符规则（如 participants 把手机号当名字标"其他"）→ 强制定型；
    - 其余走标点统一 + 实体 label 同义归一。
    """
    label = normalize_label_form(entity_type)
    label = _entity_label_canonical(label)
    if label in ("其他", "Entity"):
        # 标识符规则基于原文（大小写敏感：车牌/单号含大写字母），
        # canonical_anchor_name 会小写化，不能用于正则匹配。
        original = name or ""
        stripped = original.strip()
        for pattern, fixed_label in _IDENTIFIER_LABEL_RULES:
            if re.match(pattern, stripped):
                return fixed_label
    return label


# 叙事/结构化文档类型的路由（chunk 级）：参考实现把 book/laws/qa/general 全路由
# 到 event，法条/知识陈述天然无参与者 → 100% 产出度 0 孤儿（R3）。本表修正：
# - 笔录/聊天 → event（天然 n 元叙事，时间窗已提供锚）
# - laws/qa/csv_table/spreadsheet → llm（价值在概念网络/结构化聚合，FR-2）
# - general → event（笔录/纪要检测失败时的落点）
# - book → 仅当 chunk 命中对话/时间标记才走 event，否则 llm（说明性章节无事件）
# - 未知/缺失 → llm（反转参考实现的默认，实体路径是经过验证的基线）
_EVENT_ROUTED_DOC_TYPES = frozenset({"transcript", "chat_record", "general"})
# book 走 event 的启发式标记：对话（问：/说"）或时间戳
_BOOK_EVENT_MARKERS = (re.compile(r"[问：:]|说[：“」]|答[:：]"), re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}"))


def route_extractor_for_chunk(doc_type: str | None, content: str | None = "") -> str:
    """按 chunk 的 doc_type（与文本启发式）路由抽取路径："event" 或 "llm"。

    doc_type 缺失/未知 → "llm"（安全默认，实体路径为已验证基线）。
    """
    dtype = (doc_type or "").strip().lower()
    if dtype in _EVENT_ROUTED_DOC_TYPES:
        return "event"
    if dtype == "book":
        text = content or ""
        if any(marker.search(text) for marker in _BOOK_EVENT_MARKERS):
            return "event"
        return "llm"
    return "llm"