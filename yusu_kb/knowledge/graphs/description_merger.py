"""Description map-reduce 合并模块（移植自 LightRAG _handle_entity_relation_summary）。

将同一 entity/relation 在多 chunk 中被抽取出的多条 description 通过 LLM map-reduce
合并为一条综合 description，提升 entity 向量检索的语义召回率。

设计要点：
- 独立 asyncio.Semaphore(max_async)，不与全局 Semaphore(10) 竞争（合并是只读+LLM 调用）
- LLM 失败降级为 "; ".join(descriptions)，warning 日志，不阻塞索引流程
- Prompt 复用 LightRAG 的 summarize_entity_descriptions（prompt.py:295）
- map-reduce 阈值：total_tokens > summary_context_size 或 len >= force_llm_summary_on_merge
- L2 缓存：由 GraphService 层负责（本模块剥离 use_llm_func_with_cache，cache_repo 参数保留兼容但忽略）
- Token 预算：batch 级 entity/relation/total 三级预算，超预算项降级为 join 不丢数据
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from yusu_kb.knowledge.graphs import token_utils
from yusu_kb.knowledge.graphs.extractors.llm import (
    _RATE_LIMIT_BASE_BACKOFF,
    _RATE_LIMIT_MAX_BACKOFF,
    _RATE_LIMIT_MAX_RETRIES,
    _full_jitter_backoff,
    _is_rate_limit,
)
from yusu_kb.utils.logger import logger

# ADV-01: prompt 占用上下文窗口的安全阈值（超过则截断 descriptions）
_PROMPT_CONTEXT_SAFE_RATIO = 0.8

# LightRAG summarize_entity_descriptions prompt（prompt.py:295，逐字复用）
_SUMMARIZE_PROMPT_TEMPLATE = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

----Task---
Your task is to synthesize a list of descriptions of a given entity or relation into a single, comprehensive, and cohesive summary.

----Instructions---
1. Input Format: The description list is provided in JSON format. Each JSON object (representing a single description) appears on a new line within the `Description List` section.
2. Output Format: The merged description will be returned as plain text, presented in multiple paragraphs, without any additional formatting or extraneous comments before or after the summary.
3. Comprehensiveness: The summary must integrate all key information from *every* provided description. Do not omit any important facts or details.
4. Context: Ensure the summary is written from an objective, third-person perspective; explicitly mention the name of the entity or relation for full clarity and context.
5. Context & Objectivity:
  - Write the summary from an objective, third-person perspective.
  - Explicitly mention the full name of the entity or relation at the beginning of the summary to ensure immediate clarity and context.
6. Conflict Handling:
  - In cases of conflicting or inconsistent descriptions, first determine if these conflicts arise from multiple, distinct entities or relationships that share the same name.
  - If distinct entities/relations are identified, summarize each one *separately* within the overall output.
  - If conflicts within a single entity/relation (e.g., historical discrepancies) exist, attempt to reconcile them or present both viewpoints with noted uncertainty.
7. Length Constraint:The summary's total length must not exceed {summary_length} tokens, while still maintaining depth and completeness.
8. Language: The entire output must be written in {language}. Proper nouns (e.g., personal names, place names, organization names) may in their original language if proper translation is not available.
  - The entire output must be written in {language}.
  - Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

---Input---
{description_type} Name: {description_name}

Description List:

```
{description_list}
```

---Output---
"""


# 单条 description 在 JSON list 中的格式（ADV-02: 改用 json.dumps 序列化，覆盖所有控制字符）
def _build_description_list(descriptions: list[str]) -> str:
    """构造 JSON-line 格式的 description 列表（与 LightRAG 一致）。

    ADV-02 修复：使用 json.dumps 替代手工转义，确保所有特殊字符（\\t/\\r/\\f 等）正确转义。
    """
    lines = [json.dumps({"description": desc}, ensure_ascii=False) for desc in descriptions]
    return "\n".join(lines)


class DescriptionMerger:
    """合并同实体/关系在多 chunk 中的多条 description（移植自 LightRAG _handle_entity_relation_summary）。

    规则（逐个 entity/triple 应用）：
    - 0 条 description → 返回空字符串
    - 1 条 description → 直接返回
    - total_tokens < summary_context_size 且 len < force_llm_summary_on_merge → 直接 join，不调 LLM
    - total_tokens < summary_max_tokens → 单次 LLM 合并
    - 否则 → map-reduce 分批合并后再合并
    - LLM 失败 → 降级为 "; ".join(descriptions)，warning 日志
    """

    def __init__(
        self,
        model_spec: str,
        chat_model_fn: Callable[..., Awaitable[Any]] | None = None,
        request_timeout: float = 120.0,
        max_async: int = 4,
        summary_context_size: int = 12000,
        summary_max_tokens: int = 1200,
        force_llm_summary_on_merge: int = 8,
        language: str = "Chinese",
        cache_repo: Any | None = None,
        model_max_context: int | None = None,
        entity_token_budget: int = 6000,
        relation_token_budget: int = 8000,
        total_token_budget: int = 30000,
    ):
        # Adapted: the model call is injected via chat_model_fn instead of
        # being built with yuxi.models.chat.select_model (no langchain/openai
        # SDK, no registry/config-center dependency here). Contract matches
        # extractors/llm.py options["chat_model_fn"] and KeywordExtractor:
        # an async callable accepting a messages list and returning a
        # GeneralResponse-like object with a .content attribute (e.g.
        # OpenAIChatAdapter.call_collect). model_params (incl. the KG-9
        # stream_chunk_timeout) are intentionally not forwarded through this
        # seam — the injected fn owns its own streaming/timeout behavior.
        if chat_model_fn is None or not callable(chat_model_fn):
            raise ValueError("chat_model_fn is required (async (messages: list[dict]) -> GeneralResponse)")
        self.chat_model_fn = chat_model_fn
        self.model_spec = model_spec
        # Adapted: request_timeout kept for signature compatibility and
        # ignored — timeout is owned by the injected chat_model_fn.
        self.request_timeout = request_timeout
        self.max_async = max_async
        self.summary_context_size = summary_context_size
        self.summary_max_tokens = summary_max_tokens
        self.force_llm_summary_on_merge = force_llm_summary_on_merge
        self.language = language
        # L2 缓存仓库（None 表示不缓存；Adapted: 缓存由 GraphService 层提供，本模块忽略该参数）
        self.cache_repo = cache_repo
        # Adapted: 源项目经 model_cache.get_model_info(model_spec).max_context 查询模型
        # 上下文窗口（ADV-01 截断用）；本模块改为构造参数注入，None/<=0 表示未知（跳过截断）
        self.model_max_context = model_max_context
        # Token 预算（batch 级，参考 LightRAG token_budget 配置）
        self.entity_token_budget = entity_token_budget
        self.relation_token_budget = relation_token_budget
        self.total_token_budget = total_token_budget
        # 独立 Semaphore，不与全局 Semaphore(10) 竞争
        self._semaphore = asyncio.Semaphore(max_async)
        # Adapted: 源项目在此初始化 tiktoken cl100k_base 编码器（失败回退字符数/4）；
        # 本模块复用 token_utils 的模块级 tokenizer 单例（同样容错回退），见 _count_tokens

    async def merge_batch(
        self,
        entities: list[dict[str, Any]],
        triples: list[dict[str, Any]],
        existing_descriptions: dict[str, list[str]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """批量合并 entities 与 triples 的 description。

        Args:
            entities: [{entity_id, name, label, descriptions: [str, ...], ...}, ...]
            triples: [{triple_id, ..., descriptions: [str, ...], ...}, ...]
            existing_descriptions: {id: [desc, ...]} 已存在的 description，与新抽取的合并

        Returns:
            (merged_entities, merged_triples)，每个 item 含单条 description 字段（merged 或空字符串）

        Token 预算策略（batch 级，参考 LightRAG token_budget）：
        - entity_token_budget/relation_token_budget：单批 LLM 合并的 token 上限
        - total_token_budget：entity+relation 合并的总 token 上限，超限时按比例缩减
        - 超预算的 item 不丢弃，降级为 join 拼接（无 LLM 调用），保证数据完整性
        """
        existing_descriptions = existing_descriptions or {}

        # Token 预算分配：计算 entity/relation 的实际可用预算
        entity_budget, relation_budget = self._allocate_token_budgets(entities, triples)

        # 按 token 预算拆分：预算内走 LLM map-reduce，超预算走 join 降级
        llm_entities, fallback_entities = self._split_by_token_budget(entities, entity_budget)
        llm_triples, fallback_triples = self._split_by_token_budget(triples, relation_budget)

        if fallback_entities or fallback_triples:
            logger.info(
                f"DescriptionMerger token budget split: "
                f"entity LLM={len(llm_entities)}/fallback={len(fallback_entities)} (budget={entity_budget}), "
                f"triple LLM={len(llm_triples)}/fallback={len(fallback_triples)} (budget={relation_budget})"
            )

        async def process_entity(entity: dict[str, Any]) -> dict[str, Any]:
            entity_id = entity.get("entity_id") or ""
            new_descs = entity.get("descriptions") or []
            existing_descs = existing_descriptions.get(entity_id) or []
            all_descs = list(existing_descs) + [d for d in new_descs if d]
            # 去重保序
            seen: set[str] = set()
            unique_descs: list[str] = []
            for d in all_descs:
                if d not in seen:
                    seen.add(d)
                    unique_descs.append(d)
            merged = await self._merge_one(
                "Entity",
                entity.get("name") or entity.get("normalized_name") or entity_id,
                unique_descs,
            )
            merged_entity = dict(entity)
            merged_entity["description"] = merged
            # descriptions 列表已合并为单条 description，删除中间字段
            merged_entity.pop("descriptions", None)
            return merged_entity

        async def process_triple(triple: dict[str, Any]) -> dict[str, Any]:
            triple_id = triple.get("triple_id") or ""
            new_descs = triple.get("descriptions") or []
            existing_descs = existing_descriptions.get(triple_id) or []
            all_descs = list(existing_descs) + [d for d in new_descs if d]
            seen: set[str] = set()
            unique_descs: list[str] = []
            for d in all_descs:
                if d not in seen:
                    seen.add(d)
                    unique_descs.append(d)
            # 三元组的显示名：source → type → target
            display_name = (
                f"{triple.get('source_name', '')} → {triple.get('relation_type', '')} → {triple.get('target_name', '')}"
            )
            merged = await self._merge_one("Relation", display_name, unique_descs)
            merged_triple = dict(triple)
            merged_triple["description"] = merged
            merged_triple.pop("descriptions", None)
            return merged_triple

        def fallback_process(item: dict[str, Any]) -> dict[str, Any]:
            """超预算 item 的降级处理：直接 join descriptions，不调 LLM。"""
            result = dict(item)
            descs = item.get("descriptions") or []
            result["description"] = "; ".join(d for d in descs if d)
            result.pop("descriptions", None)
            return result

        # 并发处理 LLM 候选项（受 self._semaphore 控制），降级项同步 join
        merged_llm_entities = await asyncio.gather(*(process_entity(e) for e in llm_entities))
        merged_llm_triples = await asyncio.gather(*(process_triple(t) for t in llm_triples))
        merged_fb_entities = [fallback_process(e) for e in fallback_entities]
        merged_fb_triples = [fallback_process(t) for t in fallback_triples]

        return list(merged_llm_entities) + merged_fb_entities, list(merged_llm_triples) + merged_fb_triples

    def _allocate_token_budgets(
        self,
        entities: list[dict[str, Any]],
        triples: list[dict[str, Any]],
    ) -> tuple[int, int]:
        """根据 total_token_budget 按比例分配 entity/relation 子预算。

        - 若 entity+relation 总 token 未超 total_token_budget，直接返回原预算
        - 若超出，entity/relation 子预算按同一全局缩放比例等比缩减（保持两者原始比例）
        """
        entity_tokens = sum(self._count_tokens(d) for e in entities for d in (e.get("descriptions") or []) if d)
        relation_tokens = sum(self._count_tokens(d) for t in triples for d in (t.get("descriptions") or []) if d)
        total_tokens = entity_tokens + relation_tokens
        if total_tokens <= self.total_token_budget or total_tokens == 0:
            return self.entity_token_budget, self.relation_token_budget
        # 按比例缩减
        scale = self.total_token_budget / total_tokens
        scaled_entity = max(1, int(self.entity_token_budget * scale))
        scaled_relation = max(1, int(self.relation_token_budget * scale))
        logger.warning(
            f"Token budget overflow: total={total_tokens} > {self.total_token_budget}, "
            f"scale={scale:.3f}, entity_budget {self.entity_token_budget}->{scaled_entity}, "
            f"relation_budget {self.relation_token_budget}->{scaled_relation}"
        )
        return scaled_entity, scaled_relation

    def _split_by_token_budget(
        self,
        items: list[dict[str, Any]],
        budget: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """按 token 预算拆分 items 为 (llm_candidates, fallback_items)。

        累计 descriptions token 达到 budget 后，剩余 item 进 fallback（join 降级）。
        """
        if budget <= 0 or not items:
            return [], list(items)
        llm_items: list[dict[str, Any]] = []
        fallback_items: list[dict[str, Any]] = []
        tokens = 0
        budget_exceeded = False
        for item in items:
            if budget_exceeded:
                fallback_items.append(item)
                continue
            descs = item.get("descriptions") or []
            item_tokens = sum(self._count_tokens(d) for d in descs if d)
            if tokens + item_tokens > budget and llm_items:
                budget_exceeded = True
                fallback_items.append(item)
            else:
                tokens += item_tokens
                llm_items.append(item)
        return llm_items, fallback_items

    async def _merge_one(
        self,
        description_type: str,
        name: str,
        descriptions: list[str],
    ) -> str:
        """合并单个 entity/relation 的多条 description（map-reduce）。

        移植自 LightRAG _handle_entity_relation_summary，简化为同步 token 计算 + 异步 LLM 调用。
        """
        if not descriptions:
            return ""
        if len(descriptions) == 1:
            return descriptions[0]

        current_list = list(descriptions)
        while True:
            total_tokens = sum(self._count_tokens(d) for d in current_list)

            # 终止条件 1：总量在上下文窗口内 或 仅剩 2 条以下
            if total_tokens <= self.summary_context_size or len(current_list) <= 2:
                # 终止条件 1a：数量少且 token 数小 → 直接 join，不调 LLM
                if len(current_list) < self.force_llm_summary_on_merge and total_tokens < self.summary_max_tokens:
                    return "; ".join(current_list)
                # 终止条件 1b：调 LLM 做最终合并
                return await self._llm_summarize(description_type, name, current_list)

            # Map 阶段：分批，每批不超过 summary_context_size tokens
            chunks: list[list[str]] = []
            current_chunk: list[str] = []
            current_tokens = 0
            for desc in current_list:
                desc_tokens = self._count_tokens(desc)
                if current_tokens + desc_tokens > self.summary_context_size and current_chunk:
                    # 保证每批至少 2 条以推进进度
                    if len(current_chunk) == 1 and current_list:
                        current_chunk.append(desc)
                        # GKB-7: desc 已补入旧批，跳过本轮末尾的 append，避免同一条 desc 进入两个批次
                        chunks.append(current_chunk)
                        current_chunk = []
                        current_tokens = 0
                        continue
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                current_chunk.append(desc)
                current_tokens += desc_tokens
            if current_chunk:
                chunks.append(current_chunk)

            # Reduce 阶段：并发合并每批，递归处理
            reduced = await asyncio.gather(*(self._llm_summarize(description_type, name, chunk) for chunk in chunks))
            current_list = [r for r in reduced if r]
            if not current_list:
                return "; ".join(descriptions)

    async def _llm_summarize(
        self,
        description_type: str,
        name: str,
        descriptions: list[str],
    ) -> str:
        """调用 LLM 合并 description（带 429 退避重试），失败降级为 join。"""
        if not descriptions:
            return ""
        async with self._semaphore:
            try:
                # ADV-01: prompt token 预检，超模型上下文窗口 80% 时按比例截断 descriptions
                # 触发场景：LLM 幻觉产生超长 description，或 entity 在大量 chunk 中重复出现
                # Adapted: 源项目直接重绑 descriptions，若截断后 LLM 失败，降级 join 只含截断
                # 子集（静默丢数据，违背"降级 join 不丢数据"承诺）；此处先保留原始列表供降级路径使用
                original_descriptions = list(descriptions)
                descriptions = self._truncate_descriptions_for_context(description_type, name, descriptions)

                prompt = _SUMMARIZE_PROMPT_TEMPLATE.format(
                    summary_length=self.summary_max_tokens,
                    language=self.language,
                    description_type=description_type,
                    description_name=name,
                    description_list=_build_description_list(descriptions),
                )

                async def _use_llm_func(user_prompt: str) -> str:
                    """实际 LLM 调用闭包（含 429 全抖动退避重试）。

                    Adapted: 源项目经 use_llm_func_with_cache 包装 + select_model 构建模型
                    （KG-9 流式收集中断容忍依赖 model_params.stream_chunk_timeout）；
                    本模块直接调用注入的 chat_model_fn，L2 缓存由 GraphService 层负责，
                    流式超时行为由注入函数自身拥有。复用 extractors/llm.py 的
                    _full_jitter_backoff 与 _is_rate_limit，429 最多重试 3 次，
                    重试仍失败抛出由外层降级 join。
                    """
                    attempt = 0
                    while True:
                        try:
                            # Adapted: select_model + model.call_collect 替换为注入的
                            # chat_model_fn；裸 prompt 归一化为单条 user 消息
                            response = await self.chat_model_fn([{"role": "user", "content": user_prompt}])
                            content = (response.content if response else "").strip()
                            if not content:
                                raise ValueError("LLM 返回空内容")
                            # ADV-13: LLM 输出质量校验，过滤空/过短/过长异常内容
                            self._validate_summary_quality(content, description_type, name)
                            return content
                        except Exception as exc:  # 重试判定需捕获全部异常，非限流异常原样上抛由外层降级 join（本分支必然 re-raise，ruff 不适用 BLE001）
                            if not _is_rate_limit(exc):
                                raise
                            if attempt >= _RATE_LIMIT_MAX_RETRIES:
                                raise
                            wait = _full_jitter_backoff(attempt, _RATE_LIMIT_BASE_BACKOFF, _RATE_LIMIT_MAX_BACKOFF)
                            logger.warning(
                                f"DescriptionMerger 429 重试 attempt={attempt + 1}, wait={wait:.1f}s, "
                                f"{description_type} '{name}'"
                            )
                            await asyncio.sleep(wait)
                            attempt += 1

                # Adapted: 源项目 use_llm_func_with_cache(user_prompt, ..., cache_type="description_merge",
                # enable_cache=cache_repo is not None) 的 L2 缓存由 GraphService 层替代，此处直接调用闭包；
                # cache_repo 参数保留兼容但忽略
                content = await _use_llm_func(prompt)
                return content
            except Exception as exc:  # noqa: BLE001 - LLM 合并失败须降级 join 而非抛出（源项目语义：不阻塞索引流程）
                logger.warning(
                    f"DescriptionMerger LLM summarize failed for {description_type} '{name}': {exc}, degrade to join"
                )
                # Adapted: join 截断前的完整描述列表，保证降级不丢数据（见上方 original_descriptions 说明）
                return "; ".join(original_descriptions)

    def _truncate_descriptions_for_context(
        self,
        description_type: str,
        name: str,
        descriptions: list[str],
    ) -> list[str]:
        """ADV-01: 按模型上下文窗口截断 descriptions 列表。

        - 计算单条 description 平均 token，估算完整 prompt 的 token 占用
        - 若超过模型 max_context * _PROMPT_CONTEXT_SAFE_RATIO，按比例保留前 N 条
        - 至少保留 2 条以保证合并语义；截断后记 warning 日志
        """
        if len(descriptions) <= 2:
            return descriptions
        # Adapted: 源项目经 model_cache.get_model_info(self.model_spec) 获取 max_context，
        # 本模块使用构造参数注入（None/<=0 视为未知，跳过截断）
        max_context = self.model_max_context or 0
        if not max_context or max_context <= 0:
            return descriptions
        safe_budget = int(max_context * _PROMPT_CONTEXT_SAFE_RATIO)
        # 估算完整 prompt token：模板固定部分 + description_list
        template_overhead = self._count_tokens(
            _SUMMARIZE_PROMPT_TEMPLATE.format(
                summary_length=self.summary_max_tokens,
                language=self.language,
                description_type=description_type,
                description_name=name,
                description_list="",
            )
        )
        desc_total = sum(self._count_tokens(d) for d in descriptions)
        estimated_prompt = template_overhead + desc_total
        if estimated_prompt <= safe_budget:
            return descriptions
        # 按比例截断：保留 safe_budget 内可容纳的 description 数（至少 2 条）
        available = max(0, safe_budget - template_overhead)
        if available <= 0:
            return descriptions[:2]
        avg_per_desc = desc_total / len(descriptions) if descriptions else 0
        max_keep = max(2, int(available / avg_per_desc)) if avg_per_desc > 0 else len(descriptions)
        if max_keep >= len(descriptions):
            return descriptions
        logger.warning(
            f"ADV-01: prompt estimated {estimated_prompt} tokens exceeds {safe_budget} "
            f"(model={self.model_spec}, max_context={max_context}), "
            f"truncating {len(descriptions)} -> {max_keep} descriptions for {description_type} '{name}'"
        )
        return descriptions[:max_keep]

    def _validate_summary_quality(self, content: str, description_type: str, name: str) -> None:
        """ADV-13: LLM summary 质量校验，异常内容抛 ValueError 触发降级 join。

        - 长度过短（<10）：可能是 LLM 幻觉或无意义回复
        - 长度过长（>summary_max_tokens*3）：可能是 LLM 未遵循长度约束
        - 不影响正常输出，仅过滤极端异常情况
        - Adapted: 空内容分支已移除——上游 _use_llm_func 对空 content 先抛 "LLM 返回空内容"，
          本方法仅收到非空内容，原 if not content 为死分支
        """
        if len(content) < 10:
            raise ValueError(
                f"LLM summary too short ({len(content)} chars) for {description_type} '{name}': {content[:50]!r}"
            )
        max_chars = self.summary_max_tokens * 3
        if len(content) > max_chars:
            raise ValueError(
                f"LLM summary too long ({len(content)} chars > {max_chars}) for {description_type} '{name}'"
            )

    def _count_tokens(self, text: str) -> int:
        """计算 token 数，tiktoken 不可用时回退到字符数 / 4 的粗略估计。

        Adapted: 复用 token_utils.count_tokens（模块级 cl100k_base 单例 + 字符/4 回退），
        替代源项目实例级 tiktoken 编码器；max(1, ...) 保持源项目"每条 description 至少计 1 token"语义。
        """
        return max(1, token_utils.count_tokens(text))