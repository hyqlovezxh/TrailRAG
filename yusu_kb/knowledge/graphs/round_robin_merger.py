"""双路检索 seed 交替合并模块（移植自 LightRAG operate.py:4456-4475 的 round-robin 逻辑）。

LightRAG 在 hybrid/mix 模式下并行执行 local（LL 关键词→entity VDB）和 global（HL 关键词→relation VDB）
两路检索，然后按位交错合并 entities 列表，去重时保留首次出现的位置。

本模块将 LightRAG 内联的 round-robin 逻辑提取为纯函数，作用于 entity_id→weight 的 seed 列表，
供 Yuxi V2 检索路径在 PPR 之前合并双路检索结果使用。

设计要点：
- 纯函数，无副作用，无 I/O，便于单元测试
- 按位交错：local[0] → global[0] → local[1] → global[1] → ...
- 同 entity_id 出现时取 max weight，保留首次出现的位置（PPR seed 顺序敏感）
"""

from __future__ import annotations


def merge_seed_lists_round_robin(
    local_seeds: list[tuple[str, float]],
    global_seeds: list[tuple[str, float]],
) -> dict[str, float]:
    """双路检索 seed 交替合并。

    Args:
        local_seeds: LL 关键词→entity VDB 检索结果，按相似度降序排列，元素为 (entity_id, weight)
        global_seeds: HL 关键词→triple VDB 检索提取的 entity 列表，按相似度降序排列

    Returns:
        合并后的 entity_id → weight 字典（按交错顺序插入，同 entity_id 取 max weight）

    Examples:
        >>> merge_seed_lists_round_robin([("A", 0.9), ("B", 0.8)], [("B", 0.95), ("C", 0.7)])
        {'A': 0.9, 'B': 0.95, 'C': 0.7}
    """
    merged: dict[str, float] = {}
    max_len = max(len(local_seeds), len(global_seeds))
    for i in range(max_len):
        if i < len(local_seeds):
            entity_id, weight = local_seeds[i]
            _merge_seed(merged, entity_id, weight)
        if i < len(global_seeds):
            entity_id, weight = global_seeds[i]
            _merge_seed(merged, entity_id, weight)
    return merged


def _merge_seed(merged: dict[str, float], entity_id: str, weight: float) -> None:
    """合并单个 seed：首次出现时插入，后续取 max weight（保留首次位置）。"""
    if entity_id in merged:
        if weight > merged[entity_id]:  # noqa: PLR1730 - 逐字移植自源项目 round_robin_merger.py，保留源逻辑
            merged[entity_id] = weight
    else:
        merged[entity_id] = weight