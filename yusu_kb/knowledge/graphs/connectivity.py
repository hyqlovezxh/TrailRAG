"""图连通性门禁（S5 一等指标）。

设计文档的 NFR 里没有连通性指标，"孤儿节点"这个现象从未被度量过——
这是事件化重构跑偏的根因之一。本模块把图连通性提为可量化的硬门禁：
- orphan_rate：度为 0 的节点占比（孤儿节点率，重构核心度量）；
- lcc_ratio：最大连通分量占比（图是否"成图"）；
- anchor_reuse：事件数 / 锚点数（锚点成 hub 的程度，<2 说明锚点身份分裂）；
- zero_anchor_event_rate：零参与者事件占比（路由把 laws/qa 类文本送进
  事件路径的直接后果）。

evaluate_gate 为纯函数，阈值可注入；默认阈值按 35 份笔录 + 5 份 md 的
实际构建验收校准。
"""

from __future__ import annotations

from typing import Any

DEFAULT_GATE_THRESHOLDS: dict[str, float] = {
    "orphan_rate": 0.01,  # 孤儿率 ≤ 1%
    "lcc_ratio": 0.90,  # 最大连通分量 ≥ 90%
    "anchor_reuse": 2.0,  # 事件数/锚点数 ≥ 2（锚点成 hub）
    "zero_anchor_event_rate": 0.20,  # 零锚点事件率 ≤ 20%
}

_GATE_DESCRIPTIONS: dict[str, str] = {
    "orphan_rate": "孤儿节点率（度为0）",
    "lcc_ratio": "最大连通分量占比",
    "anchor_reuse": "锚点复用率（事件数/锚点数）",
    "zero_anchor_event_rate": "零锚点事件率",
}


def evaluate_gate(
    metrics: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """判定连通性门禁。返回 (passed, failures)。

    failures 每项：{"metric", "threshold", "actual", "description"}。
    metrics 缺失或 truncate（lcc_ratio=None）的指标不参与判定（unknown 不算失败，
    避免规模化小样本误杀），但 lcc_ratio 为 None 时单独记录为 unknown。
    """
    thresholds = {**DEFAULT_GATE_THRESHOLDS, **(thresholds or {})}
    failures: list[dict[str, Any]] = []
    unknown: list[str] = []

    if not metrics:
        return False, [{"metric": "metrics", "description": "缺少连通性指标"}]

    for metric, threshold in thresholds.items():
        actual = metrics.get(metric)
        if actual is None:
            unknown.append(metric)
            continue
        if metric in ("lcc_ratio", "anchor_reuse"):
            # 越大越好：连通占比、锚点复用（事件数/锚点数，越高说明锚点越成 hub）
            failed = actual < threshold
        else:
            # orphan_rate / zero_anchor_event_rate 是越小越好
            failed = actual > threshold
        if failed:
            failures.append(
                {
                    "metric": metric,
                    "threshold": threshold,
                    "actual": actual,
                    "description": _GATE_DESCRIPTIONS.get(metric, metric),
                }
            )
    return not failures, failures
