from __future__ import annotations

import asyncio
import random
from typing import Any

import json_repair

from yusu_kb.utils.logger import logger

from ..graph_utils import normalize_entity_name
from ..token_utils import count_tokens, truncate_section_context
from .base import GraphExtractor

DEFAULT_TRIPLE_EXTRACTION_PROMPT = """你是一个知识图谱构建专家。请从下面文本中抽取实体和实体关系，并生成简明描述，返回严格 JSON，不要输出解释。

抽取要求：
1. 实体覆盖：提取文本中所有关键实体，包括人名、地名、机构、概念、事件、技术、产品等。不要遗漏任何重要实体。优先为每个实体找出文本支持的关系，仅在文本确无关系表述时才作为独立实体列出。
2. 关系完整性：不仅抽取显式关系（如"张三任职于公司"），还要识别隐含关系：
   - 因果关系（导致、引发、促成）
   - 组成关系（包含、属于、组成部分）
   - 时间关系（先于、后于、同时）
   - 空间关系（位于、源于、流向）
   - 从属关系（隶属、管辖、归属）
   - 相似/对立关系（类似、对比、竞争）
3. 实体一致性：同一实体在所有出现处必须使用完全相同的 text 和 label。
4. 关系粒度：尽量使用具体的 label（如 任职于、位于、导致），而非笼统的相关。
5. 违法线索识别：如果文本涉及资金流转、通讯联络、人员聚集、物品交易、组织层级等内容，需识别出其中的资金关系、通联关系、组织关系和时空关联关系。
6. **表格/流水数据专项**：文本含"字段：值"键值对记录（资金流水、通话记录、卡口轨迹、维表等）时：
   - 数值与度量（金额、时长、速度、经纬度）不作为实体，一律放入相关实体的 attributes
   - 关系 label 使用统一枚举，避免同义 label 分裂同一类关系：资金往来用"资金转账"（方向由 source→target 表达）、通讯联络用"通话联系"、车辆轨迹用"驾车经过"、账户操作用"账户交易"
   - 编号类标识（手机号、账号、车牌、单号、IMEI）作为实体时，label 统一为其载体类型（手机号/银行账户/车牌/单号），text 保持编号原文逐字
   - 单条流水记录优先归约为 1 条核心关系（如转账方→收款方的"资金转账"），不要为每个字段各建一个实体
   - 代号/别名对应：同一人可能以实名、昵称、代号（如 db、W01）出现。若文本明确表明对应关系
     （自称、他称、元信息标注），抽取 source=实名实体、target=代号实体、label="代号为"的关系，
     代号实体 label 统一用"代号"；无法确认对应关系时不要猜测
7. **实体描述（description）**：为每个实体生成简明描述，基于**仅当前 chunk 内**的信息，涵盖实体的关键属性与活动。禁止跨 chunk 推断。description **必须非空**，即使信息不足也必须基于实体名称和所在 chunk 上下文生成至少一句话描述（如"张三，出现于当前文本。"）。description 为第三人称，1-2 句话，约 60 字以内，开头提及实体名称（如"张三，阿里巴巴高级工程师，负责支付系统开发。"）。description 将用于语义向量检索，请确保语义信息丰富以便召回。
8. **关系描述（description）**：为每条关系生成简明描述，**必须非空**，1 句话，约 50 字以内，解释 source 与 target 之间的连接语义，保留时间、地点、方式等关键细节。信息不足时也必须基于 source/target/relation_type 生成语义描述（如"张三 任职于 阿里巴巴。"）。同时保留 text 字段作为关系显示文本（如"供述"、"商议"）。关系 description 同样用于向量检索，请确保语义信息丰富以便召回。
9. **数量限制**：单次抽取实体不超过 30 个，关系不超过 50 条。若文本内容丰富，优先抽取最关键的实体和关系。
10. **JSON 转义**：确保 JSON 字符串中的双引号（\\"）、反斜杠（\\\\）、换行符（\\n）正确转义，避免 JSON 解析失败。
11. **关系端点引用约束**：relations 中的 source/target **必须是字符串**，直接引用 entities 数组中已出现实体的 text 字段值，**不要**在 source/target 中重复输出实体对象、label、attributes 或 description。source/target 必须**逐字复制** entities[].text 的原文，不得使用别名、简称或改写（如 entities 中 text="阿里巴巴集团"，则 source/target 必须写 "阿里巴巴集团"，不得写 "阿里巴巴" 或 "阿里"）。所有 relation 引用的 source/target 实体必须出现在 entities 数组中。

---实体类型约束---
__SCHEMA_BLOCK__
---

JSON 格式：
{
  "entities": [
    {
      "text": "独立实体文本（未参与关系也要列出）",
      "label": "实体类型",
      "attributes": [{"text": "属性值", "label": "属性名称"}],
      "description": "基于当前 chunk 信息的实体简明而全面的描述"
    }
  ],
  "relations": [
    {
      "source": "entities[].text 中的实体文本字符串",
      "target": "entities[].text 中的实体文本字符串",
      "text": "关系显示文本",
      "label": "关系类型",
      "description": "关系语义的自然语言描述，保留关键细节"
    }
  ]
}

示例：
文本：张三是阿里巴巴的高级工程师，负责支付系统的开发。该公司总部位于杭州。
{
  "entities": [
    {"text": "张三", "label": "人物", "attributes": [{"text": "高级工程师", "label": "职位"}], "description": "张三，阿里巴巴高级工程师，负责支付系统开发。"},
    {"text": "阿里巴巴", "label": "机构", "description": "阿里巴巴，总部位于杭州的科技公司。"},
    {"text": "支付系统", "label": "系统", "description": "支付系统，由张三负责开发。"},
    {"text": "杭州", "label": "地点", "description": "杭州，阿里巴巴总部所在地。"}
  ],
  "relations": [
    {"source": "张三", "target": "阿里巴巴", "text": "任职于", "label": "任职于", "description": "张三任职于阿里巴巴，担任高级工程师职位。"},
    {"source": "张三", "target": "支付系统", "text": "负责开发", "label": "开发", "description": "张三负责开发阿里巴巴支付系统。"},
    {"source": "阿里巴巴", "target": "杭州", "text": "总部位于", "label": "位于", "description": "阿里巴巴总部位于杭州。"}
  ]
}
"""

CONTEXT_INSTRUCTION = """文档上下文：
- 文档标题：{document_title}
"""

SECTION_CONTEXT_TEMPLATE = """---
文档章节路径：{heading_path}
---

"""

# Gleaning 追问 prompt：让 LLM 检查遗漏并补充。参考 LightRAG prompt.py:143-159
GLEANING_CONTINUE_PROMPT = """基于上一轮的实体关系抽取结果，请仔细复查是否有遗漏或格式错误的实体和关系。
仅输出上一轮**遗漏或格式错误**的实体和关系，**不要**重复输出已正确抽取的内容。
特别关注：
- 是否遗漏了隐含关系（因果、组成、时间、空间、从属、相似/对立）
- 是否遗漏了未参与关系的独立实体
- 关系描述是否保留了时间、地点、方式等关键细节

如果上一轮抽取已经完整且正确，请直接返回 {"entities": [], "relations": []}。

输出格式与首次抽取一致（严格 JSON，无解释）。relations 的 source/target 直接使用实体 text 字符串，不要输出嵌套实体对象。
"""

# Gleaning 守卫：单次 gleaning 的最大输入 token 上限，避免上下文超限
MAX_EXTRACT_INPUT_TOKENS = 20480

# 429 / 限流相关标记
_RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "too many requests",
)

# 其他可重试的瞬时错误标记
_TRANSIENT_MARKERS = (
    "timed out",
    "timeout",
    "connection error",
    "502",
    "503",
    "504",
    # 网络层异常类型名（chat adapter 以 "TypeName: msg" 格式保留空消息异常的类型，
    # 例如 sensenova 高并发下服务端断连产生的 httpx.ReadError，其 str() 为空）
    "readerror",
    "connecterror",
    "remoteprotocolerror",
    "writeerror",
    "localprotocolerror",
    "connection reset",
    "connection aborted",
    "broken pipe",
)

# 429 重试参数：全抖动指数退避，避免 1000 worker 同步重试风暴
# v3: 重试 5→3 次，退避上限 60→30s，减少最后 chunk 卡住时间
_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_BASE_BACKOFF = 2.0
_RATE_LIMIT_MAX_BACKOFF = 30.0

# 其他瞬时错误重试参数
# KG-9: stream_chunk_timeout 触发的「流式中途长时间无 chunk」属瞬时错误（服务端节流暂停），
# 重试 2 次作为兜底（实测 sensenova flash-lite 单次暂停可超 120s，1 次重试仍可能再超时）；
# 主修复是下方 EXTRACTION_STREAM_CHUNK_TIMEOUT 放宽暂停容忍度。
_TRANSIENT_MAX_RETRIES = 2
_TRANSIENT_BASE_BACKOFF = 1.0
_TRANSIENT_MAX_BACKOFF = 10.0

# KG-9: 流式抽取/合并大型 JSON 时，sensenova 等服务端会中途节流暂停（实测单次暂停可超 120s）。
# langchain_openai 默认 stream_chunk_timeout=120s 会在暂停时中止调用导致抽取 100% 失败；
# 提高到 300s 容忍服务端节流，使长生成能完成（实测 300s 下 30 实体/44 关系抽取在 402s 内成功）。
EXTRACTION_STREAM_CHUNK_TIMEOUT = 300.0


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _RATE_LIMIT_MARKERS)


def _is_transient(exc: Exception) -> bool:
    # asyncio.TimeoutError 的 str() 为空，须按类型直接判定
    if isinstance(exc, TimeoutError):
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def _is_retryable(exc: Exception) -> bool:
    return _is_rate_limit(exc) or _is_transient(exc)


def _full_jitter_backoff(attempt: int, base: float, cap: float) -> float:
    """全抖动指数退避：wait = uniform(0, min(cap, base * 2**attempt))。

    相比固定退避+小抖动，全抖动能有效分散 1000 worker 的重试时间点，
    避免同步重试风暴加剧 LLM 供应商限流。
    """
    expo = min(cap, base * (2**attempt))
    return random.uniform(0, expo)


def _build_section_context(chunk_metadata: dict[str, Any] | None) -> str:
    """构建章节路径面包屑（h1 → h2 → h3），超 256 tokens 折叠。

    优先使用 chunk_metadata.heading_path；缺失时退化为 document_title → section_title。
    参考 LightRAG `_truncate_section_context` + `format_heading_context`。
    """
    if not chunk_metadata:
        return ""

    heading_path = (chunk_metadata.get("heading_path") or "").strip()
    if not heading_path:
        parts: list[str] = []
        if chunk_metadata.get("document_title"):
            parts.append(str(chunk_metadata["document_title"]))
        if chunk_metadata.get("section_title"):
            parts.append(str(chunk_metadata["section_title"]))
        heading_path = " → ".join(parts)

    if not heading_path:
        return ""

    heading_path = truncate_section_context(heading_path, max_tokens=256)
    return SECTION_CONTEXT_TEMPLATE.format(heading_path=heading_path)


def _merge_gleaning_result(initial: dict[str, Any], gleaning: dict[str, Any]) -> dict[str, Any]:
    """合并 gleaning 结果到 initial。

    策略（参考 LightRAG operate.py:3660-3700）：
    - 实体按 (name, label) 去重；新实体直接加入；已有实体按 description 字符长度取较长者
    - 关系按 (src_text, tgt_text, label) 去除完全重复；已有关系按 description 长度取较长者

    GKB-12 修复：合并键使用 normalize_entity_name 归一化，与入库键一致，
    避免 gleaning 轮与初始轮因大小写/全半角/空白差异被视为不同实体，
    导致 Milvus 记录与 description 分裂（依赖下游 Neo4j MERGE 兜底但向量已分裂）。
    """
    if not isinstance(gleaning, dict):
        return initial

    initial_entities = initial.get("entities") or []
    initial_relations = initial.get("relations") or []
    glean_entities = gleaning.get("entities") or []
    glean_relations = gleaning.get("relations") or []

    # 实体合并：按 (normalized_name, label) 索引
    entity_index: dict[tuple[str, str], dict[str, Any]] = {}
    for ent in initial_entities:
        key = (normalize_entity_name(str(ent.get("text", ""))), str(ent.get("label", "")))
        entity_index[key] = dict(ent)
    for ent in glean_entities:
        key = (normalize_entity_name(str(ent.get("text", ""))), str(ent.get("label", "")))
        if key not in entity_index:
            entity_index[key] = dict(ent)
            continue
        existing = entity_index[key]
        merged = dict(existing)
        # description 取较长者（参考 LightRAG operate.py:3660-3700）
        existing_desc = str(existing.get("description", ""))
        glean_desc = str(ent.get("description", ""))
        if len(glean_desc) > len(existing_desc):
            merged["description"] = glean_desc
        # attributes 始终并集合并（独立于 description 长度比较）
        existing_attrs = existing.get("attributes") or []
        glean_attrs = ent.get("attributes") or []
        seen = {(str(a.get("text", "")), str(a.get("label", ""))) for a in existing_attrs}
        for attr in glean_attrs:
            attr_key = (str(attr.get("text", "")), str(attr.get("label", "")))
            if attr_key not in seen:
                existing_attrs.append(attr)
                seen.add(attr_key)
        merged["attributes"] = existing_attrs
        entity_index[key] = merged

    # 关系合并：按 (normalized_src, normalized_tgt, label) 索引
    def _endpoint_text(endpoint: Any) -> str:
        """从关系端点提取 text，兼容字符串引用与嵌套 dict 两种格式。"""
        if isinstance(endpoint, dict):
            return str(endpoint.get("text") or "")
        return str(endpoint or "")

    relation_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for rel in initial_relations:
        key = (
            normalize_entity_name(_endpoint_text(rel.get("source"))),
            normalize_entity_name(_endpoint_text(rel.get("target"))),
            str(rel.get("label") or ""),
        )
        relation_index[key] = dict(rel)
    for rel in glean_relations:
        key = (
            normalize_entity_name(_endpoint_text(rel.get("source"))),
            normalize_entity_name(_endpoint_text(rel.get("target"))),
            str(rel.get("label") or ""),
        )
        if key not in relation_index:
            relation_index[key] = dict(rel)
            continue
        existing = relation_index[key]
        existing_desc = str(existing.get("description", ""))
        glean_desc = str(rel.get("description", ""))
        if len(glean_desc) > len(existing_desc):
            merged = dict(existing)
            merged["description"] = glean_desc
            relation_index[key] = merged

    return {
        "entities": list(entity_index.values()),
        "relations": list(relation_index.values()),
    }


class LLMGraphExtractor(GraphExtractor):
    extractor_type = "llm"

    def validate_options(self) -> None:
        if not self.options.get("model_spec"):
            raise ValueError("LLM 抽取器需要 model_spec")
        # Adapted: the model call is injected via options instead of being
        # built with select_model (no langchain/openai SDK dependency here).
        # The injected fn is an async callable accepting a messages list and
        # returning a GeneralResponse-like object with a .content attribute.
        # Contract: the injected fn MUST be a streaming collector (e.g.
        # OpenAIChatAdapter.call_collect) so read timeouts reset per token for
        # large extraction JSONs (KG-9). model_params (incl. stream_chunk_timeout)
        # are intentionally not forwarded through this seam.
        # 校验失败按任务约定抛 ValueError（与源项目 validate_options 风格一致），
        # 类型检查场景下 ruff 建议 TypeError，此处以契约文案为准
        if not callable(self.options.get("chat_model_fn")):
            raise ValueError(  # noqa: TRY004 - 任务约定错误类型与文案，见上方说明
                "chat_model_fn is required (async streaming collector (messages: list[dict]) "
                "-> GeneralResponse, e.g. OpenAIChatAdapter.call_collect)"
            )
        if self.options.get("prompt"):
            raise ValueError("LLM 图谱抽取器不支持自定义完整 Prompt，请使用 schema 配置抽取约束")
        concurrency_count = self.options.get("concurrency_count", 1)
        try:
            concurrency_count = int(concurrency_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("LLM 抽取器 concurrency_count 必须是整数") from exc
        if concurrency_count < 1 or concurrency_count > 128:
            raise ValueError("LLM 抽取器 concurrency_count 必须在 1 到 128 之间（与执行层 cap 一致）")
        if self.options.get("model_params") is not None and not isinstance(self.options["model_params"], dict):
            raise ValueError("LLM 抽取器 model_params 必须是对象")
        # Gleaning 次数校验（默认 0，最大化索引速度；1-3 参考 LightRAG DEFAULT_MAX_GLEANING）
        gleaning_count = self.options.get("gleaning_count", 0)
        try:
            gleaning_count = int(gleaning_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("LLM 抽取器 gleaning_count 必须是整数") from exc
        if gleaning_count < 0 or gleaning_count > 3:
            raise ValueError("LLM 抽取器 gleaning_count 必须在 0 到 3 之间")
        # 诊断日志：打印实际生效的 gleaning_count，便于排查 DB 配置残留
        logger.info(f"LLM extractor gleaning_count={gleaning_count} (0=disabled, 1-3=enabled)")

    @property
    def gleaning_count(self) -> int:
        """gleaning 追问轮次（0=禁用（默认，最大化索引速度），1-3=参考 LightRAG）"""
        return int(self.options.get("gleaning_count", 0))

    async def extract(
        self,
        text: str,
        *,
        chunk_metadata: dict[str, Any] | None = None,
        cache_repo: Any | None = None,
    ) -> dict[str, Any]:
        """抽取实体与关系，可选 gleaning 追问补漏。

        Args:
            text: chunk 文本
            chunk_metadata: chunk 元数据（document_title / section_title / heading_path / chunk_id）
            cache_repo: LLM 缓存仓库，None 则不缓存
        """
        self.validate_options()
        # Adapted: no model object is constructed here — the LLM call is
        # injected via options["chat_model_fn"]; model_spec is kept for
        # validation/metadata only. cache_repo is accepted but ignored
        # (caching is handled by the GraphService layer in this project).
        chat_model_fn = self.options["chat_model_fn"]
        model_spec = self.options["model_spec"]
        model_params = dict(self.options.get("model_params") or {})
        # KG-9: 流式生成大型抽取 JSON 时容忍服务端节流暂停（见 EXTRACTION_STREAM_CHUNK_TIMEOUT）
        model_params.setdefault("stream_chunk_timeout", EXTRACTION_STREAM_CHUNK_TIMEOUT)
        prompt = self._build_prompt(text, chunk_metadata=chunk_metadata)
        chunk_id = chunk_metadata.get("chunk_id", "unknown") if chunk_metadata else "unknown"

        # === 初始抽取（带 LLM 缓存）===
        initial_result, _ = await self._call_llm_with_retry(
            chat_model_fn,
            prompt,
            chunk_id=chunk_id,
            model_spec=model_spec,
            cache_repo=cache_repo,
            cache_type="extract",
            chunk_id_for_cache=chunk_id,
        )
        parsed = json_repair.loads(initial_result or "")
        if not isinstance(parsed, dict):
            parsed = {"entities": [], "relations": []}

        gleaning_count = self.gleaning_count
        if gleaning_count <= 0:
            return parsed

        # === Gleaning 守卫：MAX_EXTRACT_INPUT_TOKENS=20480 ===
        history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": initial_result},
        ]
        gleaning_tokens = count_tokens(prompt) + count_tokens(initial_result) + count_tokens(GLEANING_CONTINUE_PROMPT)
        if gleaning_tokens > MAX_EXTRACT_INPUT_TOKENS:
            logger.warning(
                f"Gleaning skipped: gleaning_tokens={gleaning_tokens} > {MAX_EXTRACT_INPUT_TOKENS}, chunk_id={chunk_id}"
            )
            return parsed

        # === Gleaning 追问（单次执行，参考 LightRAG 实际实现）===
        for round_idx in range(gleaning_count):
            glean_result, _ = await self._call_llm_with_retry(
                chat_model_fn,
                GLEANING_CONTINUE_PROMPT,
                chunk_id=chunk_id,
                model_spec=model_spec,
                cache_repo=cache_repo,
                cache_type="extract_gleaning",
                chunk_id_for_cache=chunk_id,
                history_messages=history,
            )
            glean_parsed = json_repair.loads(glean_result or "")
            if not isinstance(glean_parsed, dict):
                # 解析失败不阻塞，保留初始结果
                logger.warning(
                    f"Gleaning round {round_idx + 1} parse failed, chunk_id={chunk_id}, keeping initial result"
                )
                break

            before_entities = len(parsed.get("entities") or [])
            before_relations = len(parsed.get("relations") or [])
            parsed = _merge_gleaning_result(parsed, glean_parsed)
            after_entities = len(parsed.get("entities") or [])
            after_relations = len(parsed.get("relations") or [])

            logger.info(
                f"Gleaning round {round_idx + 1} done, chunk_id={chunk_id}, "
                f"entities {before_entities} -> {after_entities}, "
                f"relations {before_relations} -> {after_relations}"
            )

            # 若 gleaning 没有补充任何内容，提前结束
            if after_entities == before_entities and after_relations == before_relations:
                logger.info(f"Gleaning round {round_idx + 1} no new entities/relations, stop, chunk_id={chunk_id}")
                break

            # 更新 history 供下一轮使用
            history.append({"role": "user", "content": GLEANING_CONTINUE_PROMPT})
            history.append({"role": "assistant", "content": glean_result})

            # 守卫：检查下一轮 token 是否超限
            next_tokens = (
                count_tokens(GLEANING_CONTINUE_PROMPT)
                + count_tokens(glean_result)
                + count_tokens(GLEANING_CONTINUE_PROMPT)
            )
            if gleaning_tokens + next_tokens > MAX_EXTRACT_INPUT_TOKENS:
                logger.warning(
                    f"Gleaning round {round_idx + 2} skipped: cumulative tokens exceed limit, chunk_id={chunk_id}"
                )
                break
            gleaning_tokens += next_tokens

        return parsed

    async def _call_llm_with_retry(
        self,
        chat_model_fn: Any,
        prompt: str,
        *,
        chunk_id: str,
        model_spec: str,
        cache_repo: Any | None = None,
        cache_type: str = "extract",
        chunk_id_for_cache: str | None = None,
        history_messages: list[dict] | None = None,
    ) -> tuple[str, int]:
        """带 429/瞬时错误重试的 LLM 调用。

        Adapted: the original LLM cache wrapper (use_llm_func_with_cache) is
        not portable — caching is handled by the GraphService layer in this
        project. cache_repo / cache_type / chunk_id_for_cache / model_spec
        are kept for signature compatibility and ignored; the timestamp in
        the return tuple is a placeholder (0).

        Returns:
            (content, timestamp) - LLM 响应内容与创建/缓存时间戳
        """

        async def _llm_call(
            user_prompt: str,
            *,
            system_prompt: str | None = None,
            history_messages: list[dict] | None = None,
            **_kwargs: Any,
        ) -> str:
            """实际 LLM 调用（带重试）。

            Fix-8: 当 history_messages 非空时，构造真正的多轮对话消息列表传给 model.call，
            而非将 history 拼接为文本块。LangChainChatAdapter.call 支持传入消息列表
            （内部经 convert_to_messages 转换为 BaseMessage），LLM 能正确理解追问上下文。
            """
            attempt = 0
            while True:
                try:
                    # Fix-8: 构造多轮对话消息列表
                    if history_messages:
                        messages: list[dict[str, str]] = [
                            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                            for msg in history_messages
                        ]
                        messages.append({"role": "user", "content": user_prompt})
                        # 流式收集：read timeout 随每个 token 重置，根治大型抽取 JSON
                        # 在非流式 120s 硬超时下失败（KG-9）
                        response = await chat_model_fn(messages)
                    else:
                        # Adapted: the injected fn takes a messages list, so a
                        # bare prompt string is normalized to a single user message.
                        response = await chat_model_fn([{"role": "user", "content": user_prompt}])
                    return response.content if response else ""
                except Exception as exc:
                    is_rl = _is_rate_limit(exc)
                    is_tr = _is_transient(exc)

                    if not (is_rl or is_tr):
                        raise

                    if is_rl:
                        if attempt >= _RATE_LIMIT_MAX_RETRIES:
                            raise
                        wait = _full_jitter_backoff(attempt, _RATE_LIMIT_BASE_BACKOFF, _RATE_LIMIT_MAX_BACKOFF)
                    else:
                        if attempt >= _TRANSIENT_MAX_RETRIES:
                            raise
                        wait = _full_jitter_backoff(attempt, _TRANSIENT_BASE_BACKOFF, _TRANSIENT_MAX_BACKOFF)

                    logger.warning(
                        f"图谱抽取重试 attempt={attempt + 1}, wait={wait:.1f}s, "
                        f"reason={'429' if is_rl else 'transient'}, chunk_id={chunk_id}"
                    )
                    await asyncio.sleep(wait)
                    attempt += 1

        content = await _llm_call(prompt, history_messages=history_messages)
        return content, 0

    def _build_prompt(self, text: str, *, chunk_metadata: dict[str, Any] | None = None) -> str:
        extraction_prompt = DEFAULT_TRIPLE_EXTRACTION_PROMPT

        # 注入 schema 到内嵌占位符（已包含 ---实体类型约束--- 段落）
        schema = str(self.options.get("schema") or "").strip()
        schema_block = (
            schema
            if schema
            else (
                "默认实体类型：人物、组织机构、地点、时间、事件、物品、资金/账户、通讯/账号、文书、概念术语。"
                "可标注「其他」表示不在上述类型中的实体。"
            )
        )
        extraction_prompt = extraction_prompt.replace("__SCHEMA_BLOCK__", schema_block)

        # Section Context（标题面包屑，超 256 tokens 折叠）
        section_context = _build_section_context(chunk_metadata)
        if section_context:
            extraction_prompt = f"{section_context}{extraction_prompt}"

        # 文档标题上下文
        document_title = (chunk_metadata or {}).get("document_title") or ""
        if document_title:
            extraction_prompt = f"{extraction_prompt}\n{CONTEXT_INSTRUCTION.format(document_title=document_title)}"

        return f"{extraction_prompt}\n\n文本：\n{text}"