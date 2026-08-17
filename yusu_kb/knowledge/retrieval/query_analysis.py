"""查询侧分析：精确 token 提取 + 中文分词词项扩展。

定位（线索证据挖掘优化 S1-A2，吸收 SAG 查询分析实践）：
- exact_tokens：从查询中按白名单正则提取的精确标识符（编号/手机号/证件号/
  含字母数字的 ID 等）。数字串在 embedding 空间几乎无语义，纯向量检索对
  "CDR-2026-001 通话详情"类查询存在结构性盲区（实测 gold 排 7-10 位、
  recall@1=0）。exact token 由确定性词法通道逐字校验命中后获得排序保障。
- scoring_terms：jieba 分词词项（去噪后），参与词法通道的 BM25 候选召回，
  改善无空格中文串的组合词召回（SAG v1.6.0 同款机制）。
- phrase：去噪归一化整句，供词法整句匹配使用。

韧性要求（SAG 原则）：检索必须 survive 分词器失败——jieba 不可用时
静默回退正则词项，exact token 提取纯正则、不依赖任何分词器。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 查询中的指令性/客套噪声词：只影响 phrase 与 scoring_terms 的构造，
# 不影响 exact_tokens 提取（编号/证件号永远保留）。
_NOISE_WORDS: tuple[str, ...] = (
    "知识库",
    "帮我",
    "请问",
    "请",
    "告诉我",
    "查询一下",
    "查询",
    "查一下",
    "查下",
    "搜一下",
    "搜索一下",
    "搜索",
    "查找一下",
    "查找",
    "找一下",
    "找出",
    "一下",
    "所有",
    "全部",
    "相关",
    "有关",
    "里面",
    "文档",
    "文件",
    "资料",
)

# 手机号：1[3-9] 开头 11 位（前后无数字防截断误匹配）
_PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 身份证号：18 位（末位可为 X/x）
_ID_CARD_PATTERN = re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)")
# 字母数字标识符候选（含连字符/下划线），后续按"含数字"规则过滤
_TOKEN_CANDIDATE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_]{2,}")
# 引号包裹内容（强精确检索意图）：中英文引号
# CR2-1: 反向引用配对开闭引号，混配引号（如 John's order "ORD-991"）不再截出垃圾 token
_QUOTED_PATTERN = re.compile(r"([\"'「『“])((?:(?!\1)[^\"'」』”]){2,30})\1")

_MAX_EXACT_TOKENS = 4
_MAX_SCORING_TERMS = 4
# 纯数字 token 的最小长度（排除年份"2026"等弱信号，保留 5 位以上长编号）
_MIN_PURE_DIGIT_LEN = 5
# 分词词项最小长度（过滤单字噪声）
_MIN_TERM_LEN = 2

_jieba_available: bool | None = None


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    """查询分析结果。

    Attributes:
        phrase: 去噪归一化后的整句（保留原有语序，仅移除指令性噪声词与多余空白）。
        exact_tokens: 精确标识符 token（有序去重），白名单正则提取。
        scoring_terms: 分词词项（有序去重），用于词法通道 BM25 候选召回。
        segmentation_used: 是否使用了 jieba 分词（False 表示正则回退）。
    """

    phrase: str
    exact_tokens: tuple[str, ...]
    scoring_terms: tuple[str, ...]
    segmentation_used: bool


def _segment(query: str) -> list[str]:
    """对去噪后的查询分词：jieba 精确模式，失败/不可用回退正则词项。"""
    global _jieba_available
    if _jieba_available is not False:
        try:
            import jieba

            words = [w.strip() for w in jieba.lcut(query) if len(w.strip()) >= _MIN_TERM_LEN]
            _jieba_available = True
            return words
        except Exception:  # noqa: BLE001 - 分词器任何故障都不应影响检索
            _jieba_available = False
    # 正则回退：连续拉丁字母数字串或连续 CJK 串作为词项
    return [m for m in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+", query) if len(m) >= _MIN_TERM_LEN]


def _strip_noise_words(query: str) -> str:
    """移除指令性噪声词（长词优先，避免子串误删）。"""
    text = query
    for word in sorted(_NOISE_WORDS, key=len, reverse=True):
        text = text.replace(word, " ")
    return re.sub(r"\s+", " ", text).strip()


def _dedup_keep_order(items: list[str], limit: int) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
            if len(ordered) >= limit:
                break
    return tuple(ordered)


def _is_valid_exact_token(token: str) -> bool:
    """标识符白名单规则：必须含数字；纯数字要求长编号；排除常见弱信号。"""
    if not token:
        return False
    has_digit = any(ch.isdigit() for ch in token)
    has_letter = any(ch.isalpha() for ch in token)
    if not has_digit:
        return False
    if not has_letter and len(token) < _MIN_PURE_DIGIT_LEN:  # noqa: SIM103 - 逐字移植自 YUSU
        return False
    return True


def extract_exact_tokens(query: str) -> tuple[str, ...]:
    """提取精确标识符 token：手机号/身份证号优先，再提取含数字的字母数字标识符与引号内容（需含数字）。"""
    tokens: list[str] = []
    consumed_spans: list[tuple[int, int]] = []

    for pattern in (_PHONE_PATTERN, _ID_CARD_PATTERN):
        for match in pattern.finditer(query):
            consumed_spans.append((match.start(), match.end()))
            tokens.append(match.group())

    for match in _QUOTED_PATTERN.finditer(query):
        # CR2-1: 引号内容同样须通过标识符白名单校验（含数字），否则无数字短语
        # 会绕过校验进入 exact_tokens，在 milvus substring 命中后被 0.85 分数下限
        # 强推 top-1——与模块"编号/手机号/证件号确定性兜底"定位不符。
        token = match.group(2).strip()
        if _is_valid_exact_token(token):
            consumed_spans.append((match.start(), match.end()))
            tokens.append(token)

    for match in _TOKEN_CANDIDATE_PATTERN.finditer(query):
        # 已被手机号/证件号/引号覆盖的区间不重复提取
        start, end = match.start(), match.end()
        if any(start < e and end > s for s, e in consumed_spans):
            continue
        token = match.group().strip("-_")
        if _is_valid_exact_token(token):
            tokens.append(token)

    return _dedup_keep_order(tokens, _MAX_EXACT_TOKENS)


def analyze_query(query: str, *, segmentation_enabled: bool = True) -> QueryAnalysis:
    """分析查询，产出确定性词法通道所需的 phrase / exact_tokens / scoring_terms。"""
    query = (query or "").strip()
    if not query:
        return QueryAnalysis(phrase="", exact_tokens=(), scoring_terms=(), segmentation_used=False)

    exact_tokens = extract_exact_tokens(query)
    phrase = _strip_noise_words(query)

    if not segmentation_enabled:
        return QueryAnalysis(phrase=phrase, exact_tokens=exact_tokens, scoring_terms=(), segmentation_used=False)

    segmented = _segment(phrase) if phrase else []
    # 引号内容与 exact token 不重复计入词项
    exact_set = {t.lower() for t in exact_tokens}
    terms = [w for w in segmented if w.lower() not in exact_set]
    return QueryAnalysis(
        phrase=phrase,
        exact_tokens=exact_tokens,
        scoring_terms=_dedup_keep_order(terms, _MAX_SCORING_TERMS),
        segmentation_used=_jieba_available is not False,
    )
