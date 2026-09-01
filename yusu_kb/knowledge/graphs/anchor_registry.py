"""Tier-B 锚点身份裁决器（L1 修复的核心状态组件）。

问题：同一 normalized_name 在不同事件可能被 LLM 标不同 entity_type
（"李四"→人物 / 通讯·账号），参考实现直接按 name+label 算 entity_id 导致
同一实体分裂为多个锚点、事件间隐式边断裂。

方案：worker（Tier-A 纯函数）只产出 ``(canonical_name, tier_a_label)`` 候选，
不计算 entity_id；本 Registry 仅在 flusher 单协程内裁决（零锁），为每个
canonical_name 定一个规范 label，再统一计算 entity_id。

不变量（写进单测）：
1. 任意 worker 对同一 name 产出的 Tier-A 结果相同（纯函数，见 graph_utils）；
2. 同一 KB 内 entity_id 一经写入永不变更（monotonic）——已落盘节点的 label
   历史优先，新批次接受历史 canonical label，冲突只记录不覆盖；
3. registry 对同一输入序列无论批次如何切分，resolve 结果相同（首见优先 +
   稳定 supply 顺序）。

持久化：registry 不单独落盘，从 NetworkXGraphStorage 的 entity 节点 O(N)
重建（单一事实源，无漂移）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from yusu_kb.knowledge.graphs.graph_utils import (
    LABEL_PRIORITY,
    canonical_anchor_label,
    canonical_anchor_name,
    compute_entity_id,
)

if TYPE_CHECKING:
    from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage


def _label_rank(label: str) -> int:
    return LABEL_PRIORITY.get(label, 9)


@dataclass
class AnchorRegistry:
    """KB 级锚点 label 裁决器（单协程内使用，非线程安全）。"""

    kb_id: str
    # canonical_name -> (entity_id, canonical_label, name)
    _by_name: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    # canonical_name -> set[被历史 label 拒绝的 label]（连通性报告用）
    conflicts: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_storage(cls, kb_id: str, storage: NetworkXGraphStorage) -> AnchorRegistry:
        """从现有图的 entity 节点重建 registry（增量/重启时继承历史 label）。

        历史分裂（同名多 label 双节点）时按优先级取一作为 canonical，
        其余保留为影子节点（不删，避免既有边断裂），记入 conflicts。
        """
        registry = cls(kb_id=kb_id)
        for node in storage.iter_entities():
            name = str(node.get("normalized_name") or "").strip()
            label = str(node.get("label") or "Entity").strip()
            if not name:
                continue
            registry._register(name, node.get("entity_id"), label, str(node.get("name") or name))
        return registry

    def _register(self, name: str, entity_id: Any, label: str, display_name: str) -> None:
        existing = self._by_name.get(name)
        if existing is None:
            self._by_name[name] = (str(entity_id), label, display_name)
            return
        _, existing_label, _ = existing
        if existing_label != label:
            # 同 name 出现第二 label：历史 label 优先（monotonic，entity_id 不可改），
            # 新 label 记入 conflicts 供连通性报告展示，不覆盖。
            self.conflicts.setdefault(name, set()).add(label)

    def resolve(self, name: str, tier_a_label: str, *, display_name: str = "") -> tuple[str, str]:
        """为锚点候选裁决规范 label 并返回 (entity_id, canonical_label)。

        - 已见 name：接受历史 canonical label（entity_id 不变更）；
        - 未见 name：登记首见 label（若该 label 优先级更低——如"其他"——且
          同批稍后有更高优先级候选，由 caller 以 resolve_all 二次收敛）。
        """
        canonical_name = name
        existing = self._by_name.get(canonical_name)
        if existing is not None:
            entity_id, label, _ = existing
            if label != tier_a_label:
                # 历史 label 优先（monotonic，entity_id 不可改），冲突记录供连通性报告
                self.conflicts.setdefault(canonical_name, set()).add(tier_a_label)
            return entity_id, label
        label = tier_a_label
        entity_id = compute_entity_id(self.kb_id, canonical_name, label)
        self._by_name[canonical_name] = (entity_id, label, display_name or canonical_name)
        return entity_id, label

    def resolve_all(
        self, candidates: list[dict[str, Any]], *, kb_id: str
    ) -> list[dict[str, Any]]:
        """批量裁决：先按优先级收敛同 name 的 label（批次内多数/最高优先级），
        再逐名 resolve。返回含 entity_id / label / display_name 的锚点行。

        candidates 结构：{"name", "normalized_name", "label", "display_name"}。
        收敛规则：同 normalized_name 在批次内出现多个 label 时取优先级最高者；
        同优先级取首见。
        """
        # 1) 批次内收敛：name -> (最佳 label, display_name, 优先级, 首见序)
        best: dict[str, dict[str, Any]] = {}
        for cand in candidates:
            name = str(cand.get("normalized_name") or "").strip()
            if not name:
                continue
            label = str(cand.get("label") or "其他").strip()
            display = str(cand.get("display_name") or cand.get("name") or name)
            current = best.get(name)
            if current is None:
                best[name] = {"label": label, "display_name": display, "rank": _label_rank(label), "seen": 0}
            else:
                rank = _label_rank(label)
                if rank < current["rank"]:
                    best[name] = {"label": label, "display_name": display, "rank": rank, "seen": current["seen"]}

        # 2) 逐个 resolve（Tier-B 裁决，monotonic）
        resolved: list[dict[str, Any]] = []
        for name, entry in best.items():
            entity_id, label = self.resolve(name, entry["label"], display_name=entry["display_name"])
            resolved.append(
                {
                    "entity_id": entity_id,
                    "normalized_name": name,
                    "label": label,
                    "name": entry["display_name"],
                    "kb_id": kb_id,
                }
            )
        return resolved

    def label_of(self, name: str) -> str | None:
        existing = self._by_name.get(name)
        return existing[1] if existing else None


def build_anchor_candidate(
    participant_name: str,
    participant_type: str | None,
) -> dict[str, str]:
    """Tier-A：participant → 锚点候选（纯函数，worker 内调用，无锁）。

    canonical_anchor_label 内部已做标点统一（N9）+ label 同义归一 +
    标识符正则定型；canonical_anchor_name 与实体路径同规（case_sensitive=False）。
    """
    normalized_name = canonical_anchor_name(participant_name)
    if not normalized_name:
        return {}
    label = canonical_anchor_label(participant_type, name=participant_name)
    return {
        "name": participant_name,
        "normalized_name": normalized_name,
        "label": label,
        "display_name": participant_name,
    }
