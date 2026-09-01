"""事件护栏 G1/G2（迁移自源项目事件化改造，纯逻辑零 IO）。

- G1 锚定校验：exact_identifiers 逐字回溯、participants 模糊匹配、time_expr
  逐字校验；不过校验的事件标记 unverified（写入时传播权重减半）。
- G2 去重：写入期 SimHash 近重复粗筛（hamming ≤5 ≈ 相似度 ≥0.92）+ 检索期 MMR。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from yusu_kb.knowledge.graphs.event_schemas import EventRecord
from yusu_kb.utils.logger import logger

# 参与者名模糊匹配阈值：容忍 LLM 输出的轻微形变（全半角、空格、个别字差异）
PARTICIPANT_FUZZY_THRESHOLD = 0.8

_WS_PATTERN = re.compile(r"\s+")


def fuzzy_contains(name: str, content: str, *, threshold: float = PARTICIPANT_FUZZY_THRESHOLD) -> bool:
    """判断参与名是否以逐字或高相似形式出现在原文中。

    先做空白归一后的逐字包含（快路径，覆盖"陈 锦标"类形变）；再对原文做
    等长滑动窗口 SequenceMatcher 比对（慢路径，容忍个别字符差异）。
    窗口长度等于名称长度，保证比率不被长文本稀释。
    """
    if not name or not content:
        return False
    compact_name = _WS_PATTERN.sub("", name)
    compact_content = _WS_PATTERN.sub("", content)
    if not compact_name:
        return False
    if compact_name in compact_content:
        return True
    n = len(compact_name)
    for i in range(len(compact_content) - n + 1):
        if SequenceMatcher(None, compact_name, compact_content[i : i + n]).ratio() >= threshold:
            return True
    return False


def verify_event(record: EventRecord, chunk_content: str) -> EventRecord:
    """G1 锚定校验：就地更新 record.verification 并返回。

    硬证据（任一失败 → unverified）：
    - exact_identifiers 逐字存在于 chunk 原文；
    - time_expr 逐字存在于 chunk 原文（为空视为通过——none/chitchat 事件可无时间表达）。

    软信号（不单独降级，仅记日志）：participants 每个名字逐字或模糊（>=0.8）
    出现在原文。**笔录场景实测修正**：嫌疑人笔录正文以"我"自称，chunk 内没有
    被询问人姓名，LLM 从文档标题/上下文补全主语是正确行为（正是笔录场景的核心
    价值），逐字回溯会系统性误伤——19 事件实测全部因此降级 unverified，而
    identifier/time 全部通过，说明这不是幻觉而是主语补全。
    """
    unverified_reasons: list[str] = []
    # 空格无关比较：markdown 解析会把"3月15日"规范化为"3 月 15 日"（数字与汉字间
    # 插空格），逐字包含会把 LLM 的正确抽取误判为幻觉。剥离空白后再回溯，
    # 既不漏掉真幻觉（"2025年" 在只提 2024 的文本里仍不匹配），也不误伤规范化形变。
    compact_content = _WS_PATTERN.sub("", chunk_content)
    for ident in record.exact_identifiers:
        if _WS_PATTERN.sub("", ident) not in compact_content:
            unverified_reasons.append(f"identifier 不在原文: {ident}")
    if record.time_expr and _WS_PATTERN.sub("", record.time_expr) not in compact_content:
        unverified_reasons.append(f"time_expr 不在原文: {record.time_expr}")

    # participants 软信号：不参与降级判定，仅记录（不命中即"跨 chunk 主语补全"候选）
    unanchored_participants = [
        participant.name
        for participant in record.participants
        if not fuzzy_contains(participant.name, chunk_content)
    ]

    record.verification = "unverified" if unverified_reasons else "verified"
    if unverified_reasons:
        logger.info(f"G1 锚定校验未通过 event_id={record.event_id}: {'; '.join(unverified_reasons[:3])}")
    elif unanchored_participants:
        logger.debug(
            f"G1 participants 跨 chunk 补全（主语未在正文出现）event_id={record.event_id}: "
            f"{'; '.join(unanchored_participants[:3])}"
        )
    return record


def verify_events(records: list[EventRecord], chunk_content: str) -> list[EventRecord]:
    """对单 chunk 的全部事件执行 G1 校验。"""
    return [verify_event(record, chunk_content) for record in records]


# —— G2 SimHash 去重 ——

_SIMHASH_BITS = 64
# hamming 距离阈值：64 位 SimHash 下 ≤5 相当于相似度 ≥0.92（设计文档口径）
_SIMHASH_MAX_HAMMING = 5
# 中文字符 token 化：单字 + 字母数字串按 bigram 切分（语义噪声小、对语序不敏感）
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """字符级 unigram + 数字/字母词 bigram 混合 token 化（面向中文叙事文本）。"""
    normalized = re.sub(r"\s+", "", text)
    if not normalized:
        return ["<empty>"]
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(normalized):
        token = match.group(0)
        if len(token) == 1:
            tokens.append(token)
        else:
            tokens.extend(token[i : i + 2] for i in range(len(token) - 1))  # 拆 bigram
    return tokens or [normalized[:16]]


def simhash64(text: str) -> int:
    """64 位 SimHash：token 哈希加权叠加，符号位聚合。

    中文叙事摘要的经验参数；无外部依赖（hashlib md5 足够，召回质量由
    hamming 阈值控制，不追求密码学强度）。
    """
    vector = [0] * _SIMHASH_BITS
    tokens = _tokenize(text)
    seen: dict[str, int] = {}
    for token in tokens:
        seen[token] = seen.get(token, 0) + 1
    for token, weight in seen.items():
        digest = hashlib.md5(token.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big")
        for bit in range(_SIMHASH_BITS):
            if (value >> bit) & 1:
                vector[bit] += weight
            else:
                vector[bit] -= weight
    fingerprint = 0
    for bit in range(_SIMHASH_BITS):
        if vector[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


@dataclass
class EventDedupIndex:
    """单次构建运行内的 KB 级事件去重索引（SimHash 粗筛）。

    生命周期 = 一次构建运行：跨批累积已写入事件；多 KB 构建各自持有一个实例。
    """

    max_hamming: int = _SIMHASH_MAX_HAMMING
    _fingerprints: dict[str, int] = field(default_factory=dict)  # event_id -> simhash

    def check(self, record: EventRecord) -> str | None:
        """返回重复指向的 event_id（duplicate_of），无重复返回 None。

        重复事件仍返回 record（由调用方决定写入策略），仅标记。
        """
        fingerprint = simhash64(record.summary)
        for event_id, existing_fp in self._fingerprints.items():
            if hamming_distance(fingerprint, existing_fp) <= self.max_hamming:
                record.duplicate_of = event_id
                return event_id
        return None

    def add(self, record: EventRecord) -> None:
        """登记规范事件（重复事件不登记，防链式指向）。"""
        self._fingerprints.setdefault(record.event_id, simhash64(record.summary))

    def __len__(self) -> int:
        return len(self._fingerprints)


# —— 检索期 MMR ——


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def mmr_select(
    candidate_ids: list[str],
    query_scores: dict[str, float],
    embeddings: dict[str, list[float]],
    k: int,
    lambda_mult: float = 0.7,
) -> list[str]:
    """检索期 MMR（最大边际相关性）：兼顾查询相关性与候选间差异。

    candidate_ids 已按 query_scores 降序排列；返回 ≤k 个 id。
    缺 embedding 的候选按零向量处理（只由相关性驱动）。
    """
    if not candidate_ids:
        return []
    selected: list[str] = []
    candidates = list(dict.fromkeys(candidate_ids))
    while candidates and len(selected) < k:
        best_id, best_score = None, float("-inf")
        for cid in candidates:
            relevance = query_scores.get(cid, 0.0)
            diversity = max(
                (_cosine(embeddings.get(cid, []), embeddings.get(sid, [])) for sid in selected),
                default=0.0,
            )
            score = lambda_mult * relevance - (1 - lambda_mult) * diversity
            if score > best_score:
                best_id, best_score = cid, score
        selected.append(best_id)  # type: ignore[arg-type]
        candidates.remove(best_id)  # type: ignore[arg-type]
    return selected
