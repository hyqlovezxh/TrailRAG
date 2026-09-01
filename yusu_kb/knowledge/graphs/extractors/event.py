"""事件抽取器（设计文档 §4.2）。

EventGraphExtractor 继承 LLMGraphExtractor 以复用 429/瞬时错误重试
（_call_llm_with_retry），但抽取语义完全不同：输出 n 元事件（不是
实体/关系二元组）。归一化由 normalize_event_result 承担，与实体路径的
normalize_extraction_result 平行。

与源项目差异（yusu_kb 适配）：
- 无 select_model / use_llm_func_with_cache / disable_thinking_model_params：
  LLM 调用经 options["chat_model_fn"] 注入（OpenAIChatAdapter.call_collect 流式收集器）。
- content_preview 不作为 EventRecord 字段（避免节点属性膨胀，见 event_schemas）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import json_repair

from yusu_kb.knowledge.graphs.event_schemas import (
    EVENT_VALUE_WEIGHTS,
    MAX_EVENTS_PER_CHUNK,
    EventRecord,
    EventType,
    Participant,
    make_event_id,
)
from yusu_kb.knowledge.graphs.extractors.llm import (
    CONTEXT_INSTRUCTION,
    LLMGraphExtractor,
)
from yusu_kb.utils.logger import logger

# 事件抽取读超时（秒）：n 元槽位 JSON 输出远长于三元组，客户端默认 90s 不够
EVENT_EXTRACTION_READ_TIMEOUT = 240.0

EVENT_EXTRACTION_PROMPT = """你是公安办案知识图谱构建专家。请从下面文本中抽取**事件**，返回严格 JSON，不要输出解释。

事件是"谁在什么时间什么地点对谁做了什么"的完整命题。事件必须保留 n 元绑定关系：时间、地点、参与者、金额、标的都属于事件本身，不可拆散为孤立属性。

抽取要求：
1. **事件覆盖**：识别文本中所有独立事件。同一文本含多个独立事件（时间或主体明显断裂）时拆分为多个 events，**最多 5 个**；单一连续行为合并为 1 个事件。
2. **完整命题 summary**：一句话完整命题，必须包含主体+动作+客体+关键限定（时间/地点/金额/方式），如"陈锦标于2025-10-05 14:24主叫邓家俊，通话295秒，商议境外窝点战略会议"。summary 将用于语义向量检索，请确保语义信息丰富以便召回。
3. **time_expr 逐字**：time_expr 必须从原文**逐字摘录**时间表达（若文本含【时间段 X ~ Y】前缀，优先使用该前缀区间）。禁止改写、翻译或推算 time_expr。
4. **time_norm 归一化**：基于 time_expr 与【时间段】前缀锚定输出 ISO8601 格式（如 2025-10-05T14:24:00），time_resolution 按可确定的最细粒度填 second/minute/hour/day/month；无法解析时 time_norm 输出 null、time_resolution 填 unknown。相对时间（如"当晚""三天后"）结合上下文锚定，无法锚定时输出 null。
5. **participants 逐字**：参与者 name 必须从原文**逐字摘录**，禁止改写或使用原文未出现的代称；entity_type 枚举：人物、组织机构、地点、资金·账户、通讯·账号、其他；role 描述参与角色（如 主叫/被叫/转账方/收款方/驾驶人/嫌疑人）。最多 12 人，选取关键参与者。
6. **exact_identifiers 逐字**：手机号、银行账号、车牌、单号、IMEI 等编号标识必须**逐字摘录**（如 13857906361、FB2026-0001、浙A·JK345）。这些标识将用于逐字回溯校验，任何改写都会导致事件被判定为不可信。
7. **event_type 枚举**：transfer（资金流转）/ communication（通讯联络）/ movement（人员车辆移动）/ meeting（会面聚集）/ transaction（物品交易）/ statement（供述证言）/ chitchat（无实质内容闲聊）/ none（非事件）。纯闲聊必须标 chitchat 并产出 1 个概括闲聊主题的事件（如"陈锦标与李四闲聊日常问候"），不得硬凑成实质事件；完全无信息的段落标 none。
8. **金额 amount**：涉及资金时输出数值（单位：元），否则 null。
9. **text_span**：多事件拆分时输出每个事件在原文中的字符区间 [起始, 结束]；单一事件输出 null。
10. **location**：事件发生地点，逐字摘录原文表达（如"杭州西湖大道""边境基站"）；无地点信息输出 null。
11. **JSON 转义**：确保 JSON 字符串中的双引号（\\"）、反斜杠（\\\\）、换行符（\\n）正确转义。
12. **输出约束**：events 数组中每个对象必须包含 event_type/summary/time_expr/time_norm/time_resolution/action 字段；participants/exact_identifiers/objects 无内容时输出空数组。

JSON 格式：
{
  "events": [
    {
      "event_type": "transfer",
      "summary": "完整命题一句话",
      "time_expr": "原文时间表达逐字",
      "time_norm": "2025-10-05T14:24:00 或 null",
      "time_resolution": "minute",
      "location": "地点或 null",
      "action": "核心动作谓词（如 转账/主叫/会面/驾车经过/供述）",
      "participants": [{"name": "逐字原文", "role": "角色", "entity_type": "人物"}],
      "objects": ["涉及物品/标的"],
      "amount": 50000.0,
      "exact_identifiers": ["编号逐字"],
      "text_span": [0, 120]
    }
  ]
}

示例：
文本：【时间段 2025-10-05 14:20 ~ 14:30】
陈锦标：刚才那笔 5 万到账了没？我用的尾号 6688 的卡。
李四：到了，FB2026-0001，已转给上面的邓家俊。
{
  "events": [
    {
      "event_type": "communication",
      "summary": "2025-10-05 14:20~14:30 陈锦标与李四在聊天中确认一笔50000元资金已到账并已转给邓家俊",
      "time_expr": "【时间段 2025-10-05 14:20 ~ 14:30】",
      "time_norm": "2025-10-05T14:20:00",
      "time_resolution": "minute",
      "location": null,
      "action": "确认资金到账",
      "participants": [
        {"name": "陈锦标", "role": "询问方", "entity_type": "人物"},
        {"name": "李四", "role": "确认方", "entity_type": "人物"},
        {"name": "邓家俊", "role": "资金接收方", "entity_type": "人物"}
      ],
      "objects": ["尾号6688的银行卡"],
      "amount": 50000.0,
      "exact_identifiers": ["FB2026-0001"],
      "text_span": null
    }
  ]
}
"""


class EventGraphExtractor(LLMGraphExtractor):
    """事件抽取器：n 元语义场输出，供事件图谱路径消费。"""

    extractor_type = "event"

    def validate_options(self) -> None:
        if not self.options.get("model_spec"):
            raise ValueError("事件抽取器需要 model_spec")
        # 注入式 chat_model_fn 契约与 LLMGraphExtractor 一致（流式收集器）
        if not callable(self.options.get("chat_model_fn")):
            raise ValueError(  # noqa: TRY004 - 与 LLMGraphExtractor 契约文案一致（任务约定错误类型）
                "chat_model_fn is required (async streaming collector "
                "(messages: list[dict]) -> GeneralResponse)"
            )
        if self.options.get("prompt"):
            raise ValueError("事件抽取器不支持自定义完整 Prompt")
        if self.options.get("gleaning_count"):
            raise ValueError("事件抽取器不支持 gleaning（事件粒度由 MAX_EVENTS_PER_CHUNK 控制）")
        concurrency_count = self.options.get("concurrency_count", 1)
        try:
            concurrency_count = int(concurrency_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("事件抽取器 concurrency_count 必须是整数") from exc
        if concurrency_count < 1 or concurrency_count > 128:
            raise ValueError("事件抽取器 concurrency_count 必须在 1 到 128 之间")
        model_params = self.options.get("model_params")
        if model_params is not None and not isinstance(model_params, dict):
            raise ValueError("事件抽取器 model_params 必须是对象")

    async def extract(
        self,
        text: str,
        *,
        chunk_metadata: dict[str, Any] | None = None,
        cache_repo: Any | None = None,
    ) -> dict[str, Any]:
        """抽取事件列表，返回原始 dict（{"events": [...]}），归一化由服务层负责。"""
        self.validate_options()
        chat_model_fn = self.options["chat_model_fn"]
        model_spec = self.options["model_spec"]
        model_params = dict(self.options.get("model_params") or {})
        # 事件抽取提示长（n 元槽位 + 枚举约束）、输出 token 多（JSON 对象数组），
        # 客户端默认 90s 读超时在慢模型上会整体 ReadTimeout。timeout 由
        # chat_model_fn 提取后交给 httpx（不进 JSON body），这里给出场景默认值。
        model_params.setdefault("timeout", EVENT_EXTRACTION_READ_TIMEOUT)
        prompt = self._build_prompt(text, chunk_metadata=chunk_metadata)
        chunk_id = chunk_metadata.get("chunk_id", "unknown") if chunk_metadata else "unknown"

        result, _ = await self._call_llm_with_retry(
            chat_model_fn,
            prompt,
            chunk_id=chunk_id,
            model_spec=model_spec,
            cache_repo=cache_repo,
            cache_type="extract_event",
            chunk_id_for_cache=chunk_id,
        )
        parsed = json_repair.loads(result or "")
        if not isinstance(parsed, dict) or not isinstance(parsed.get("events") or [], list):
            return {"events": []}
        return parsed

    def _build_prompt(self, text: str, *, chunk_metadata: dict[str, Any] | None = None) -> str:
        prompt = EVENT_EXTRACTION_PROMPT
        document_title = (chunk_metadata or {}).get("document_title") or ""
        if document_title:
            prompt = f"{prompt}\n{CONTEXT_INSTRUCTION.format(document_title=document_title)}"
        return f"{prompt}\n\n文本：\n{text}"


# —— 归一化（与实体路径 normalize_extraction_result 平行）——

# (格式, time_resolution) 对：按可确定的最细粒度标注
_TIME_NORM_FORMATS: tuple[tuple[str, str], ...] = (
    ("%Y-%m-%dT%H:%M:%S", "second"),
    ("%Y-%m-%d %H:%M:%S", "second"),
    ("%Y-%m-%dT%H:%M", "minute"),
    ("%Y-%m-%d %H:%M", "minute"),
    ("%Y/%m/%d %H:%M", "minute"),
    ("%Y年%m月%d日 %H:%M", "minute"),
    ("%Y-%m-%d", "day"),
    ("%Y/%m/%d", "day"),
    ("%Y年%m月%d日", "day"),
)


def parse_time_norm(raw: Any) -> tuple[datetime | None, str]:
    """解析 LLM 输出的 time_norm 字符串；失败返回 (None, "unknown")，不抛异常。"""
    if isinstance(raw, dict) or raw is None:
        return None, "unknown"
    text = str(raw).strip()
    if not text or text.lower() in ("null", "none", "unknown"):
        return None, "unknown"
    for fmt, resolution in _TIME_NORM_FORMATS:
        try:
            # LLM 输出的 time_norm 无时区（文本来源），语义上即 naive datetime
            return datetime.strptime(text, fmt), resolution  # noqa: DTZ007
        except ValueError:
            continue
    return None, "unknown"


def normalize_event_result(
    result: dict[str, Any],
    *,
    chunk_id: str,
    kb_id: str = "",
    file_id: str | None = None,
) -> dict[str, Any]:
    """将 LLM 原始事件输出归一化为 EventRecord 列表。

    - 非法事件跳过（summary 空、结构错）并告警，不阻塞整 chunk；
    - 事件数超 MAX_EVENTS_PER_CHUNK 截断（防碎片化，护栏 G3 配套）；
    - value_weight 按 EVENT_VALUE_WEIGHTS 赋值（护栏 G4）；
    - 同 chunk 内 summary 归一化重复的事件去重（保序）。
    返回 {"events": [EventRecord...], "metadata": {"extractor_type": "event", ...}}。
    """
    raw_events = (result or {}).get("events") or []
    if not isinstance(raw_events, list):
        raw_events = []

    # file_id 稳定派生：抽取器不透传 file_id，而 Event 节点的 file_id 是
    # delete_file 事件清理与 L4 文件内分组的唯一依据（实测曾全部为空 →
    # 事件按文件删除失效、所有文件事件被 L4 归为同一组）。派生后不依赖下游透传。
    if not file_id:
        marker = "_chunk_"
        pos = chunk_id.rfind(marker)
        file_id = chunk_id[:pos] if pos > 0 else chunk_id

    events: list[EventRecord] = []
    skipped = 0
    seen_summaries: set[str] = set()

    for index, raw in enumerate(raw_events):
        if len(events) >= MAX_EVENTS_PER_CHUNK:
            dropped = len(raw_events) - index
            logger.warning(
                f"normalize_event_result: 事件数达到上限 {MAX_EVENTS_PER_CHUNK}，丢弃后续 {dropped} 个候选，"
                f"chunk_id={chunk_id}"
            )
            break
        if not isinstance(raw, dict):
            skipped += 1
            logger.warning(f"normalize_event_result: 跳过非法 events[{index}]（非对象）")
            continue
        summary = str(raw.get("summary") or "").strip()
        if not summary:
            skipped += 1
            logger.warning(f"normalize_event_result: 跳过非法 events[{index}]（summary 为空）")
            continue
        try:
            event_type = EventType(str(raw.get("event_type") or "none").strip().lower())
        except ValueError:
            event_type = EventType.NONE
            logger.warning(f"normalize_event_result: events[{index}].event_type 非法，降级为 none")

        time_norm, time_resolution = parse_time_norm(raw.get("time_norm"))

        participants: list[Participant] = []
        for p in raw.get("participants") or []:
            if isinstance(p, dict) and str(p.get("name") or "").strip():
                try:
                    participants.append(Participant(**p))
                except Exception as exc:  # noqa: BLE001 - 单个参与者非法不应阻塞事件
                    logger.warning(f"normalize_event_result: 跳过非法 participant: {exc}")
            elif isinstance(p, str) and p.strip():
                # 兼容 LLM 直接输出字符串名单
                participants.append(Participant(name=p.strip()))

        text_span_raw = raw.get("text_span")
        text_span = None
        if (
            isinstance(text_span_raw, (list, tuple))
            and len(text_span_raw) == 2
            and all(isinstance(x, (int, float)) for x in text_span_raw)
        ):
            text_span = (int(text_span_raw[0]), int(text_span_raw[1]))

        amount_raw = raw.get("amount")
        amount = float(amount_raw) if isinstance(amount_raw, (int, float)) else None

        location_raw = raw.get("location")
        location = str(location_raw).strip() if location_raw else None

        try:
            record = EventRecord(
                event_id=chunk_id,  # 占位；1:N 时在收集完成后统一改写为 #idx 后缀
                chunk_id=chunk_id,
                kb_id=kb_id,
                file_id=file_id,
                event_type=event_type,
                summary=summary,
                time_expr=str(raw.get("time_expr") or "").strip(),
                time_norm=time_norm,
                time_resolution=time_resolution,
                location=location or None,
                action=str(raw.get("action") or "").strip(),
                participants=participants,
                objects=[str(o).strip() for o in (raw.get("objects") or []) if str(o).strip()],
                amount=amount,
                exact_identifiers=[
                    str(x).strip() for x in (raw.get("exact_identifiers") or []) if str(x).strip()
                ],
                text_span=text_span,
                value_weight=EVENT_VALUE_WEIGHTS[event_type],
            )
        except Exception as exc:  # noqa: BLE001 - 单个事件非法不应阻塞整 chunk
            skipped += 1
            logger.warning(f"normalize_event_result: 跳过非法 events[{index}]: {exc}")
            continue

        dedup_key = summary.replace(" ", "").lower()
        if dedup_key in seen_summaries:
            continue
        seen_summaries.add(dedup_key)
        events.append(record)

    if skipped:
        logger.info(f"normalize_event_result: chunk {chunk_id} 跳过 {skipped} 个非法事件，保留 {len(events)} 个")

    # 1:N 事件 id 规则（设计文档 §4.3 + N1 修复）：整组统一带 ev: 前缀与 #idx 后缀，
    # id 形态即可区分 1:1 与 1:N，且与 chunk/entity 命名空间隔离。
    if len(events) > 1:
        for idx, record in enumerate(events):
            record.event_id = make_event_id(chunk_id, idx)
    elif events:
        events[0].event_id = make_event_id(chunk_id, 0)

    return {
        "events": events,
        "metadata": {"extractor_type": "event", "schema_version": 1},
    }
