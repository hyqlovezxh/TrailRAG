# ruff: noqa: TRY004 - 移植自源项目：对 LLM 输出数据校验统一抛 ValueError，
# 与下方跳过逻辑 except (ValueError, TypeError) 契约一致，保持逐字移植不改语义
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..graph_utils import normalize_entity_name

logger = logging.getLogger(__name__)


class GraphExtractor(ABC):
    extractor_type: str

    def __init__(self, options: dict[str, Any] | None = None):
        self.options = options or {}

    @abstractmethod
    async def extract(self, text: str, *, chunk_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        pass

    def validate_options(self) -> None:
        return None


def normalize_extraction_result(result: dict[str, Any], extractor_type: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("extraction_result 必须是对象")

    entities = result.get("entities") or []
    relations = result.get("relations") or []
    if not isinstance(entities, list) or not isinstance(relations, list):
        raise ValueError("extraction_result.entities 和 relations 必须是数组")

    normalized_entities_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    entity_refs: dict[str, dict[str, Any]] = {}

    def add_entity(entity: Any, path: str) -> dict[str, Any]:
        normalized_entity = _normalize_entity(entity, path)
        key = _entity_key(normalized_entity)
        existing = normalized_entities_by_key.get(key)
        if existing is None:
            normalized_entities_by_key[key] = normalized_entity
            existing = normalized_entity
        else:
            _merge_attributes(existing, normalized_entity)
            # V2: 收集同一实体的多条 description（来自不同 relation 出现处）
            new_desc = normalized_entity.get("descriptions") or []
            if new_desc:
                existing.setdefault("descriptions", []).extend(new_desc)

        for ref in _entity_refs(entity, existing):
            entity_refs[ref] = existing
        return existing

    skipped_entities = 0
    for index, entity in enumerate(entities):
        try:
            add_entity(entity, f"entities[{index}]")
        except (ValueError, TypeError) as e:
            skipped_entities += 1
            logger.warning(f"跳过非法 entity[{index}]: {e}")

    normalized_relations = []
    skipped_relations = 0
    for index, relation in enumerate(relations):
        try:
            if not isinstance(relation, dict):
                raise ValueError("relations 元素必须是对象")
            source = _normalize_relation_endpoint(
                relation.get("source"),
                entity_refs,
                add_entity,
                f"relations[{index}].source",
            )
            target = _normalize_relation_endpoint(
                relation.get("target"),
                entity_refs,
                add_entity,
                f"relations[{index}].target",
            )
            text = str(relation.get("text") or "").strip()
            if not text:
                raise ValueError("relations[].text 不能为空")
            normalized_relation = {
                "source": source,
                "target": target,
                "text": text,
                "label": str(relation.get("label") or "RELATED_TO").strip() or "RELATED_TO",
            }
            # V2: 提取关系 description（V1 输出无此字段，留空字符串，向后兼容）
            # 兼容 LLM 输出变体：description (string) | descriptions (list)
            relation_descs: list[str] = []
            rel_desc_singular = str(relation.get("description") or "").strip()
            if rel_desc_singular:
                relation_descs.append(rel_desc_singular)
            rel_desc_list = relation.get("descriptions")
            if isinstance(rel_desc_list, list):
                for d in rel_desc_list:
                    d_text = str(d or "").strip()
                    if d_text and d_text not in relation_descs:
                        relation_descs.append(d_text)
            if relation_descs:
                # 单条直接使用 string，多条 join（与 _build_triple_records 期望一致）
                normalized_relation["description"] = (
                    relation_descs[0] if len(relation_descs) == 1 else "; ".join(relation_descs)
                )
            normalized_relations.append(normalized_relation)
        except (ValueError, TypeError) as e:
            skipped_relations += 1
            logger.warning(f"跳过非法 relation[{index}]: {e}")

    if skipped_entities or skipped_relations:
        logger.info(
            f"normalize_extraction_result: 跳过 {skipped_entities} 个非法 entity、"
            f"{skipped_relations} 条非法 relation，保留 {len(normalized_entities_by_key)} 个 entity、"
            f"{len(normalized_relations)} 条 relation"
        )

    # 兜底：为无 description 的实体/关系生成模板化最小描述，确保 description 字段非空
    # 这不是"用防御掩盖设计缺陷"，而是"在 LLM 未输出时提供最小可用值"以支撑向量检索召回
    for entity in normalized_entities_by_key.values():
        if not entity.get("descriptions"):
            text = entity["text"]
            label = entity.get("label") or "Entity"
            entity["descriptions"] = [f"{text}，类型：{label}。"]
    for relation in normalized_relations:
        if not relation.get("description"):
            source_text = relation["source"]["text"]
            target_text = relation["target"]["text"]
            rel_label = relation.get("label") or "RELATED_TO"
            relation["description"] = f"{source_text} {rel_label} {target_text}。"

    metadata = dict(result.get("metadata") or {})
    metadata.setdefault("extractor_type", extractor_type)
    metadata.setdefault("schema_version", 1)
    return {
        "entities": list(normalized_entities_by_key.values()),
        "relations": normalized_relations,
        "metadata": metadata,
    }


def _normalize_relation_endpoint(
    endpoint: Any,
    entity_refs: dict[str, dict[str, Any]],
    add_entity: Callable[[Any, str], dict[str, Any]],
    path: str,
) -> dict[str, Any]:
    if isinstance(endpoint, dict):
        return add_entity(endpoint, path)

    endpoint_ref = str(endpoint or "").strip()
    if not endpoint_ref:
        raise ValueError(f"{path} 不能为空字符串")

    # 1. 精确匹配：entity_refs 按 text/id 建立
    entity = entity_refs.get(endpoint_ref)
    if entity is not None:
        return entity

    # 2. 归一化匹配：NFKC + 去末尾标点 + 大小写折叠后比较，
    #    容忍 LLM 输出的轻微差异（全角/半角、末尾标点、大小写）
    normalized_ref = normalize_entity_name(endpoint_ref)
    seen_ids: set[int] = set()
    for ent in entity_refs.values():
        if id(ent) in seen_ids:
            continue
        seen_ids.add(id(ent))
        if normalize_entity_name(ent["text"]) == normalized_ref:
            return ent

    # 3. 未命中：自动注册为最小实体（text + label "Entity"），
    #    避免 LLM 引用文本不一致导致关系丢失、损失召回
    #    同时生成最小 description，确保 description 字段非空以支撑向量检索召回
    logger.warning(
        f"relations[].source/target 引用的实体未在 entities 中找到，自动注册为最小实体: {path}={endpoint_ref}"
    )
    return add_entity(
        {
            "text": endpoint_ref,
            "label": "Entity",
            "descriptions": [f"{endpoint_ref}，作为关系端点出现于当前文本。"],
        },
        path,
    )


def _normalize_entity(entity: Any, path: str) -> dict[str, Any]:
    if not isinstance(entity, dict):
        raise ValueError(f"{path} 必须是对象")

    text = str(entity.get("text") or "").strip()
    if not text:
        raise ValueError(f"{path}.text 不能为空")

    attributes = entity.get("attributes") or []
    if not isinstance(attributes, list):
        logger.warning(f"{path}.attributes 不是数组，强制转为空数组")
        attributes = []

    normalized_attributes = []
    for attribute in attributes:
        if not isinstance(attribute, dict):
            continue
        attr_text = str(attribute.get("text") or "").strip()
        if not attr_text:
            continue
        normalized_attributes.append(
            {
                "text": attr_text,
                "label": str(attribute.get("label") or "Attribute").strip() or "Attribute",
            }
        )

    normalized = {
        "text": text,
        "label": str(entity.get("label") or "Entity").strip() or "Entity",
        "attributes": normalized_attributes,
    }
    # V2: 提取实体 description（V1 输出无此字段，向后兼容）
    # 兼容 LLM 输出变体：
    # - description (string): prompt 期望格式
    # - descriptions (list): 部分 LLM（如 GLM-5.2）会输出复数列表形式
    # 两种形式均收集到 descriptions 列表，去重保序
    descriptions: list[str] = []
    desc_singular = str(entity.get("description") or "").strip()
    if desc_singular:
        descriptions.append(desc_singular)
    desc_list = entity.get("descriptions")
    if isinstance(desc_list, list):
        for d in desc_list:
            d_text = str(d or "").strip()
            if d_text and d_text not in descriptions:
                descriptions.append(d_text)
    if descriptions:
        normalized["descriptions"] = descriptions
    return normalized


def _entity_key(entity: dict[str, Any]) -> tuple[str, str]:
    return (normalize_entity_name(entity["text"]), entity["label"])


def _entity_refs(raw_entity: Any, entity: dict[str, Any]) -> list[str]:
    refs = [entity["text"]]
    if isinstance(raw_entity, dict):
        entity_id = str(raw_entity.get("id") or "").strip()
        if entity_id:
            refs.append(entity_id)
    return refs


def _merge_attributes(target: dict[str, Any], source: dict[str, Any]) -> None:
    known_attributes = {(attr["text"], attr["label"]) for attr in target.get("attributes") or []}
    for attribute in source.get("attributes") or []:
        attribute_key = (attribute["text"], attribute["label"])
        if attribute_key not in known_attributes:
            target.setdefault("attributes", []).append(attribute)
            known_attributes.add(attribute_key)