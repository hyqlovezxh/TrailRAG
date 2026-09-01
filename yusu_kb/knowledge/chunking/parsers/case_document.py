from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from yusu_kb.knowledge.chunking import nlp
from yusu_kb.knowledge.chunking.parsers import general, semantic
from yusu_kb.utils.logger import logger

_TRANSCRIPT_Q_PATTERN = re.compile(r"^问[：:]\s*", re.MULTILINE)
# 兼容两种聊天记录格式：
# 1. HH:MM 昵称: 内容（案件库测试/导出格式）
# 2. MM-DD HH:MM 发送者→接收者: 内容（md_writer 生产格式，parse_time_line 输出）
# 可选的 MM-DD 日期前缀使跨天场景下日期随消息体保留，避免后续 chunk 丢失日期上下文（CD-2 修复）
_CHAT_MSG_PATTERN = re.compile(r"^(\d{1,2}-\d{1,2}\s+)?\d{1,2}:\d{2}\s+\S+?(?:→\S+?)?[:：]\s*", re.MULTILINE)
# S3 多格式时间戳：微信/QQ 导出与方括号时间戳变体（与上面两种共同参与消息切分与类型检测）
_CHAT_MSG_PATTERN_FULL_DATE = re.compile(
    r"^\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?\s+\S+?(?:→\S+?)?[:：]\s*", re.MULTILINE
)
_CHAT_MSG_PATTERN_BRACKET = re.compile(
    r"^\[\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?\]\s*\S+?(?:→\S+?)?[:：]\s*", re.MULTILINE
)
_CHAT_MSG_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    _CHAT_MSG_PATTERN,
    _CHAT_MSG_PATTERN_FULL_DATE,
    _CHAT_MSG_PATTERN_BRACKET,
)
_MD_TABLE_HEADER_PATTERN = re.compile(r"^\|(.+)\|\s*$")
_MD_TABLE_SEPARATOR_PATTERN = re.compile(r"^\|[\s\-:|]+\|\s*$")
_MD_TABLE_ROW_PATTERN = re.compile(r"^\|(.+)\|\s*$")

# S3 格式模板注册：按 文件名/内容 正则将文档路由到专用子分块器，未知格式安全回退 general。
# 内置模板覆盖常见案件格式；parser_config["format_overrides"] 可零代码扩展新格式
# （经 chunk_parser_config 三级合并：KB → 文件 → 请求），结构与内置项一致。
CASE_FORMAT_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "transcript_default",
        "doc_type": "transcript",
        "filename_patterns": ("笔录", "讯问", "询问"),
        "content_patterns": (r"^问[：:]",),
    },
    {
        "id": "chat_default",
        "doc_type": "chat_record",
        "filename_patterns": ("私聊", "群聊", "聊天", "微信"),
        "content_patterns": (),
    },
    {
        "id": "chat_wechat_export",
        "doc_type": "chat_record",
        "filename_patterns": (),
        "content_patterns": (r"^\d{4}-\d{1,2}-\d{1,2}[ T]\d{1,2}:\d{2}(?::\d{2})?\s+\S+[:：]",),
    },
]

# S3 笔录头身份信号：命中才注入正文 chunk 前缀（防误检文件注入噪声）
_HEADER_IDENTITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|\n)姓名[：:]\s*(\S{2,20})"),
    re.compile(r"(?:^|\n)身份证号[：:]?\s*([0-9Xx]{15,18})"),
    re.compile(r"(?:^|\n)联系电话[：:]\s*(1[3-9]\d{9})"),
)

# S3 时间戳解析（用于聊天时间窗口断块与时间段锚定）
_CHAT_TS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})(?::\d{2})?"),
    re.compile(r"^\[(\d{4})-(\d{1,2})-(\d{1,2})[ T](\d{1,2}):(\d{2})(?::\d{2})?\]"),
    re.compile(r"^(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})"),
    re.compile(r"^(\d{1,2}):(\d{2})"),
)


# A-11: 时间戳日期粒度——full 携带真实年月日；month_day/time_only 以 1900 为
# 占位年份，与 full 相邻时需按同日推算对齐后再计算时间差
_DATE_RESOLUTION_FULL = "full"
_DATE_RESOLUTION_MONTH_DAY = "month_day"
_DATE_RESOLUTION_TIME_ONLY = "time_only"
_RESOLUTION_RANK = {_DATE_RESOLUTION_TIME_ONLY: 0, _DATE_RESOLUTION_MONTH_DAY: 1, _DATE_RESOLUTION_FULL: 2}


def _parse_chat_timestamp(line: str) -> tuple[datetime, str] | None:
    """解析消息行首时间戳，返回 (时间戳, 日期粒度)；无法解析返回 None。"""
    for pattern in _CHAT_TS_PATTERNS:
        match = pattern.match(line)
        if not match:
            continue
        groups = match.groups()
        if len(groups) >= 5:
            year, month, day, hour, minute = (int(g) for g in groups[:5])
            resolution = _DATE_RESOLUTION_FULL
        elif len(groups) == 4:
            month, day, hour, minute = (int(g) for g in groups)
            year = 1900
            resolution = _DATE_RESOLUTION_MONTH_DAY
        else:
            hour, minute = int(groups[0]), int(groups[1])
            year, month, day = 1900, 1, 1
            resolution = _DATE_RESOLUTION_TIME_ONLY
        try:
            # 聊天时间戳无时区（文本来源，占位年份 1900），语义上即 naive datetime
            return datetime(year, month, day, hour, minute), resolution  # noqa: DTZ001
        except ValueError:
            return None
    return None


def parse_chat_timestamp(line: str) -> datetime | None:
    """从消息行首解析时间戳，供时间窗口断块计算时间差。

    无日期格式（HH:MM / MM-DD HH:MM）以 1900 年为占位年份；调用方以
    时间差为负视为跨天边界。无法解析返回 None。
    """
    parsed = _parse_chat_timestamp(line)
    return parsed[0] if parsed else None


def timestamp_text(line: str) -> str | None:
    """提取消息行首的时间戳原文（用于时间段锚定前缀）。"""
    for pattern in _CHAT_TS_PATTERNS:
        match = pattern.match(line)
        if match:
            return match.group(0).strip()
    return None


# S2-B1 列裁剪默认黑名单：坐标类列对检索与图谱无价值，反而稀释向量与 LLM 抽取
# （18 列 CDR/卡口宽表实测单 chunk 抽取 30+ 实体导致超时）。可通过 chunk_parser_config
# 的 csv_drop_columns / csv_keep_columns 覆盖；csv_drop_columns=[] 可显式关闭裁剪。
_DEFAULT_CSV_DROP_COLUMNS: tuple[str, ...] = ("lat", "lon", "latitude", "longitude", "经度", "纬度")


def _column_filter(parser_config: dict[str, Any]):
    """构造列保留判定函数（S2-B1 列裁剪）。

    - csv_keep_columns 非空：白名单模式，仅保留指定列（大小写不敏感）
    - 否则 csv_drop_columns：黑名单模式，默认裁剪坐标类列；显式传 [] 关闭裁剪
    """
    keep_columns = [str(c).strip().lower() for c in parser_config.get("csv_keep_columns") or [] if str(c).strip()]
    if keep_columns:
        keep_set = set(keep_columns)
        return lambda header: header.strip().lower() in keep_set
    drop_config = parser_config.get("csv_drop_columns")
    drop_columns = (
        [str(c).strip().lower() for c in drop_config if str(c).strip()]
        if drop_config is not None
        else [c.lower() for c in _DEFAULT_CSV_DROP_COLUMNS]
    )
    if not drop_columns:
        return lambda header: True
    drop_set = set(drop_columns)
    return lambda header: header.strip().lower() not in drop_set


def _resolve_format_templates(parser_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """合并内置格式模板与 parser_config["format_overrides"] 用户扩展（S3 注册机制）。"""
    templates = list(CASE_FORMAT_TEMPLATES)
    for override in (parser_config or {}).get("format_overrides") or []:
        if isinstance(override, dict) and override.get("doc_type"):
            templates.insert(0, override)  # 用户模板优先于内置
    return templates


def _detect_document_type(filename: str, content: str, parser_config: dict[str, Any] | None = None) -> str:
    """根据格式模板与文件名/内容特征检测文档类型。

    优先级：用户格式模板（format_overrides）→ 内置模板 → 内置启发式 → general 兜底。
    未知格式安全回退 general（不丢数据、不崩溃）。
    """
    fname = (filename or "").lower()
    text = content or ""
    templates = _resolve_format_templates(parser_config)

    # 审查修复(M5): 表格类文件跳过模板 filename 关键词匹配——"微信转账流水.csv" 等命名
    # 不应被 chat_default 的 "微信" 模板劫持为 chat_record 而丢失 csv_table 处理；
    # 内容正则判定保留，真实聊天导出表仍可命中消息行模式
    fname_is_table = fname.endswith((".csv", ".xlsx", ".xls"))
    for template in templates:
        doc_type = template.get("doc_type")
        if doc_type not in ("transcript", "chat_record"):
            continue
        if not fname_is_table and any(pat and re.search(pat, fname) for pat in template.get("filename_patterns") or ()):
            return doc_type
    for template in templates:
        doc_type = template.get("doc_type")
        if doc_type not in ("transcript", "chat_record"):
            continue
        if any(pat and re.search(pat, text, re.MULTILINE) for pat in template.get("content_patterns") or ()):
            return doc_type

    if "笔录" in fname or text.strip().startswith("笔录"):
        return "transcript"

    if any(kw in fname for kw in ("私聊", "群聊", "聊天")):
        return "chat_record"

    if fname.endswith(".csv"):
        return "csv_table"

    if fname.endswith((".xlsx", ".xls")):
        return "spreadsheet"

    if any(pattern.search(text) for pattern in _CHAT_MSG_LINE_PATTERNS):
        return "chat_record"

    if _TRANSCRIPT_Q_PATTERN.search(text):
        return "transcript"

    return "general"


def _build_transcript_header_prefix(header: str) -> str:
    """从笔录元信息头提取身份信号构建短前缀（S3 笔录头注入，CD-4 表头注入同模式）。

    元信息头独立成块后，证件号/联系电话与正文检索语义分离，"笔录身份证号"类
    精确查询 top-1 失准。命中身份信号（≥2 项）才注入，防止误检文件注入噪声。
    """
    signals: list[str] = []
    for pattern in _HEADER_IDENTITY_PATTERNS:
        match = pattern.search(header)
        if match:
            value = match.group(1)
            label = "姓名" if "姓名" in pattern.pattern else ("身份证号" if "身份证" in pattern.pattern else "联系电话")
            signals.append(f"{label}{value}")
    if len(signals) < 2:
        return ""
    return "〔笔录头：" + "｜".join(signals[:3]) + "〕\n"


def _chunk_transcript(content: str, parser_config: dict[str, Any]) -> list[str]:
    """笔录分块：元信息头独立成块 + 问答对按 token 上限打包。

    S3：transcript_header_inject（默认开启）将身份信号短前缀注入每个正文 chunk，
    使证件号/联系电话类精确查询可直接命中正文 chunk。
    """
    chunk_token_num = int(parser_config.get("chunk_token_num", 1200) or 1200)
    overlapped_percent = int(parser_config.get("overlapped_percent", 15) or 15)
    header_inject = bool(parser_config.get("transcript_header_inject", True))
    text = content.strip()
    if not text:
        return []

    first_q_match = _TRANSCRIPT_Q_PATTERN.search(text)
    if not first_q_match:
        return general.chunk_markdown(text, parser_config)

    header = text[: first_q_match.start()].strip()
    qa_body = text[first_q_match.start() :]

    # 按 "问：" 切分问答对，保留分隔符
    parts = _TRANSCRIPT_Q_PATTERN.split(qa_body)
    # split 后第一个元素是空字符串（因为 text 以 "问：" 开头）
    qa_pairs: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            qa_pairs.append(f"问：{part}")

    chunks: list[str] = []

    if not qa_pairs:
        if header:
            chunks.append(header)
        return chunks

    # 用 naive_merge 将问答对按 token 上限打包
    sections = [(pair, "") for pair in qa_pairs]
    merged = nlp.naive_merge(
        sections, chunk_token_num=chunk_token_num, delimiter="\n", overlapped_percent=overlapped_percent
    )

    # CD-1 修复：对超长问答对做 token 上限兜底（与 general.chunk_markdown 一致）
    if header:
        chunks.append(header)
    body_chunks = general._ensure_chunk_token_limit(merged, chunk_token_num)

    header_prefix = _build_transcript_header_prefix(header) if header_inject and header else ""
    if header_prefix:
        body_chunks = [header_prefix + chunk for chunk in body_chunks]

    chunks.extend(body_chunks)

    return [c.strip() for c in chunks if c and c.strip()]


def _split_chat_messages(msg_body: str) -> list[str]:
    """按全部已知时间戳行格式切分消息（S3 多格式支持）。"""
    # CR2-7: 原元组第二位（模式序号）恒为 0 的死字段已删除，仅保留起始位置
    starts: list[int] = []
    seen: set[int] = set()
    for pattern in _CHAT_MSG_LINE_PATTERNS:
        for match in pattern.finditer(msg_body):
            if match.start() not in seen:
                seen.add(match.start())
                starts.append(match.start())
    starts.sort()

    messages: list[str] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(msg_body)
        msg = msg_body[start:end].strip()
        if msg:
            messages.append(msg)
    return messages


def _align_timestamp_same_day(ts: datetime, resolution: str, anchor: datetime) -> datetime:
    """A-11: 同日推算——将较粗粒度时间戳的缺失部分对齐到较细粒度 anchor。

    time_only 借 anchor 的年月日；month_day 借 anchor 的年份。占位时间戳的
    2 月 29 日在 1900 占位年下构造即失败（返回 None），对齐不会产生非法日期。
    """
    if resolution == _DATE_RESOLUTION_TIME_ONLY:
        return anchor.replace(hour=ts.hour, minute=ts.minute)
    return ts.replace(year=anchor.year)


def _split_messages_by_time_gap(
    messages: list[str], gap_minutes: int
) -> list[tuple[list[str], str | None, str | None]]:
    """按相邻消息时间差断块（S3 时间窗口分块）。

    gap > gap_minutes 或时间回退（跨天，无日期格式下时间差为负）时强制断块，
    提升单 chunk 话题纯度、缓解向量稀释。返回 [(messages, first_ts_text, last_ts_text)]。
    gap_minutes <= 0 时禁用断块（返回单组）。
    A-11: 相邻消息时间戳格式粒度不一致时（如全日期导出格式与 HH:MM 生产格式
    混排），占位年份 1900 会造成巨大负差被误判为跨天——较粗粒度一侧先按同日
    推算对齐到较细粒度一侧，再计算时间差；同粒度相邻保持原语义不变。
    """
    if gap_minutes <= 0:
        ts_first = timestamp_text(messages[0]) if messages else None
        ts_last = timestamp_text(messages[-1]) if messages else None
        return [(messages, ts_first, ts_last)]

    groups: list[tuple[list[str], str | None, str | None]] = []
    current: list[str] = []
    prev: tuple[datetime, str] | None = None

    def close_group() -> None:
        nonlocal current
        if current:
            first_text = timestamp_text(current[0])
            last_text = timestamp_text(current[-1])
            groups.append((current, first_text, last_text))
        current = []

    for msg in messages:
        parsed = _parse_chat_timestamp(msg)
        if parsed is not None and prev is not None:
            ts, resolution = parsed
            prev_ts, prev_resolution = prev
            effective_ts, effective_prev = ts, prev_ts
            if resolution != prev_resolution:
                if _RESOLUTION_RANK[resolution] > _RESOLUTION_RANK[prev_resolution]:
                    effective_prev = _align_timestamp_same_day(prev_ts, prev_resolution, ts)
                else:
                    effective_ts = _align_timestamp_same_day(ts, resolution, prev_ts)
            delta_minutes = (effective_ts - effective_prev).total_seconds() / 60
            if delta_minutes > gap_minutes or delta_minutes < 0:
                close_group()
        if parsed is not None:
            prev = parsed
        current.append(msg)
    close_group()
    return groups if groups else [(messages, None, None)]


def _chunk_chat_record(content: str, parser_config: dict[str, Any]) -> list[str]:
    """聊天记录分块：元信息头独立成块 + 时间窗口断块 + 消息按 token 上限打包。

    S3：chat_time_gap_minutes（默认 30）在相邻消息时间差超阈值或跨天处强制断块，
    每块注入时间段锚定前缀，支持"某时间段聊了什么"类查询；设为 0 禁用。
    """
    chunk_token_num = int(parser_config.get("chunk_token_num", 1200) or 1200)
    overlapped_percent = int(parser_config.get("overlapped_percent", 15) or 15)
    gap_minutes = int(parser_config.get("chat_time_gap_minutes", 30) or 0)
    text = content.strip()
    if not text:
        return []

    first_msg_match = None
    for pattern in _CHAT_MSG_LINE_PATTERNS:
        first_msg_match = pattern.search(text)
        if first_msg_match:
            break
    if not first_msg_match:
        return general.chunk_markdown(text, parser_config)

    header = text[: first_msg_match.start()].strip()
    msg_body = text[first_msg_match.start() :]

    messages = _split_chat_messages(msg_body)

    chunks: list[str] = []

    if not messages:
        if header:
            chunks.append(header)
        return chunks

    if header:
        chunks.append(header)

    for group_messages, ts_first, ts_last in _split_messages_by_time_gap(messages, gap_minutes):
        sections = [(msg, "") for msg in group_messages]
        merged = nlp.naive_merge(
            sections, chunk_token_num=chunk_token_num, delimiter="\n", overlapped_percent=overlapped_percent
        )
        # CD-1 修复：对超长单条消息做 token 上限兜底（与 general.chunk_markdown 一致）
        protected = general._ensure_chunk_token_limit(merged, chunk_token_num)
        if ts_first:
            time_anchor = f"【时间段 {ts_first}" + (f" ~ {ts_last}" if ts_last and ts_last != ts_first else "") + "】\n"
            protected = [time_anchor + chunk for chunk in protected]
        chunks.extend(protected)

    return [c.strip() for c in chunks if c and c.strip()]


def _parse_markdown_tables(content: str) -> tuple[list[str], list[dict[str, list[str]]]]:
    """从 row-by-row markdown 表格中解析出表头和数据行。

    CSV 经 unified.py 解析后，每行变成一个独立的 markdown 表格（含表头+分隔行+数据行）。
    本函数将其还原为统一的表头 + 数据行列表。

    BUG-33 修复：每张表独立使用自己的表头，避免多表头错位。
    """
    lines = content.strip().splitlines()
    headers: list[str] = []
    seen_header_keys: set[str] = set()
    data_rows: list[dict[str, list[str]]] = []
    current_headers: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not line or not line.startswith("|"):
            i += 1
            continue

        header_match = _MD_TABLE_HEADER_PATTERN.match(line)
        if header_match and i + 1 < len(lines):
            sep_line = lines[i + 1].strip()
            if _MD_TABLE_SEPARATOR_PATTERN.match(sep_line):
                # 这是表头行
                current_headers = [c.strip() for c in header_match.group(1).split("|")]
                # CD-4 修复：收集所有表的表头并集（保序去重），用于 header_context 注入
                for h in current_headers:
                    if h and h not in seen_header_keys:
                        seen_header_keys.add(h)
                        headers.append(h)
                # 跳过表头和分隔行，看下一行是否是数据行
                i += 2
                while i < len(lines):
                    row_line = lines[i].strip()
                    if not row_line.startswith("|"):
                        break
                    row_match = _MD_TABLE_ROW_PATTERN.match(row_line)
                    if not row_match or _MD_TABLE_SEPARATOR_PATTERN.match(row_line):
                        break
                    row_cells = [c.strip() for c in row_match.group(1).split("|")]
                    # BUG-33 修复：每行携带所属表的表头，避免多表头错位
                    data_rows.append({"headers": current_headers[:], "cells": row_cells})
                    i += 1
                continue

        i += 1

    return headers, data_rows


def _chunk_csv_table(content: str, parser_config: dict[str, Any]) -> list[str]:
    """CSV 表格分块：将 row-by-row markdown 表格转为键值对记录，按 token 上限打包。

    S2-B1 列裁剪：按 csv_keep_columns/csv_drop_columns 过滤低价值列后再生成键值对，
    缓解宽表向量稀释与图谱抽取过载（仅影响新入库文件）。
    """
    chunk_token_num = int(parser_config.get("chunk_token_num", 1200) or 1200)
    overlapped_percent = int(parser_config.get("overlapped_percent", 15) or 15)
    headers, data_rows = _parse_markdown_tables(content)

    if not headers or not data_rows:
        return general.chunk_markdown(content, parser_config)

    keep_column = _column_filter(parser_config)

    # 将每行转为键值对格式（S2-B1：跳过被裁剪的列）
    records: list[str] = []
    for row in data_rows:
        # BUG-33 修复：每行使用自己的表头，避免多表头错位
        row_headers = row["headers"]
        cells = row["cells"]
        parts: list[str] = []
        for j, val in enumerate(cells):
            if j < len(row_headers) and row_headers[j]:
                key = row_headers[j]
                if val and keep_column(key):
                    parts.append(f"{key}：{val}")
        if parts:
            records.append("；".join(parts) + "；")

    if not records:
        return general.chunk_markdown(content, parser_config)

    # 表头上下文注入到每个 chunk，确保向量检索时字段语义完整（S2-B1：同步过滤裁剪列）
    # CR2-7: 删除 retained_headers 空回退——所有列被裁剪时 records 必空、已在上方提前
    # return，该回退不可达
    retained_headers = [h for h in headers if h and keep_column(h)]
    header_context = f"表头字段：{'；'.join(retained_headers)}；\n"

    sections = [(record, "") for record in records]
    merged = nlp.naive_merge(
        sections, chunk_token_num=chunk_token_num, delimiter="\n", overlapped_percent=overlapped_percent
    )
    # CD-1 修复：先对 merged（未加 header_context）做 token 上限兜底，再加 header_context，
    # 避免 header_context 被硬切导致字段语义残缺
    protected = general._ensure_chunk_token_limit(merged, chunk_token_num)

    chunks: list[str] = []
    for chunk in protected:
        chunk = chunk.strip()
        if not chunk:
            continue
        chunks.append(header_context + chunk)

    return chunks


def _chunk_spreadsheet(content: str, parser_config: dict[str, Any]) -> list[str]:
    """表格分块：委托 semantic 解析器，并注入表头上下文以保持字段语义完整。

    与 CSV 表格策略一致：从 markdown 表格中提取表头，注入到每个 chunk 前部，
    确保向量检索时每个 chunk 都携带字段语义，提升召回率。
    S2-B1：表头上下文同步过滤被裁剪的列（正文 chunk 由 semantic 解析器处理，
    完整列裁剪请使用 CSV 导入路径）。
    """
    chunks = semantic.chunk_markdown(content, parser_config)
    if not chunks:
        return chunks
    headers, _ = _parse_markdown_tables(content)
    if not headers:
        return chunks
    keep_column = _column_filter(parser_config)
    retained_headers = [h for h in headers if h and keep_column(h)] or [h for h in headers if h]
    header_context = f"表头字段：{'；'.join(retained_headers)}；\n"
    return [header_context + chunk for chunk in chunks]


def chunk_markdown(filename: str, markdown_content: str, parser_config: dict[str, Any] | None = None) -> list[str]:
    """案例文档分块入口：自动检测文档类型并分派到专用子解析器。"""
    parser_config = parser_config or {}
    content = markdown_content or ""
    doc_type = _detect_document_type(filename, content, parser_config)

    logger.info(f"case_document chunking: filename={filename}, detected_type={doc_type}, content_len={len(content)}")

    if doc_type == "transcript":
        return _chunk_transcript(content, parser_config)
    if doc_type == "chat_record":
        return _chunk_chat_record(content, parser_config)
    if doc_type == "csv_table":
        return _chunk_csv_table(content, parser_config)
    if doc_type == "spreadsheet":
        return _chunk_spreadsheet(content, parser_config)

    return general.chunk_markdown(content, parser_config)
