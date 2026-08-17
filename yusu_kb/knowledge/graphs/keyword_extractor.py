"""HL/LL 双层关键词抽取模块（移植自 LightRAG extract_keywords_only）。

从用户 query 中抽取两类关键词：
- high_level_keywords（HL）：概括性概念/主题，用于 triple VDB 检索（global 路径）
- low_level_keywords（LL）：具体实体/专有名词，用于 entity VDB 检索（local 路径）

设计要点：
- 独立 asyncio.Semaphore(max_async)，不与全局 Semaphore(10) 竞争（检索期只读+LLM 调用）
- LLM 失败/JSON 解析失败 返回 ([], [])，warning 日志，不抛异常（检索路径自动降级 v1）
- JSON 解析三级容错：json.loads → 去 markdown fence → json_repair
- 进程内 LRU 缓存（capacity=256），同 query 不重复调用 LLM
- Prompt 复用 LightRAG 的 keywords_extraction（prompt.py:484）
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any

import json_repair

from yusu_kb.utils.logger import logger

# LightRAG keywords_extraction prompt（prompt.py:484-515，逐字复用）
# 注意：原模板使用 {{ }} 转义字面 { }，本模板保留转义，format 时仅填充 {language}/{examples}/{query}
_KEYWORDS_EXTRACTION_PROMPT = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), comments, or any other text before or after the JSON.
2. **Exact JSON Shape**: The JSON object must contain exactly these two keys:
   - `"high_level_keywords"`: an array of strings
   - `"low_level_keywords"`: an array of strings
3. **JSON Boundary**: The first character of your response must be `{{` and the last character must be `}}`.
4. **Source of Truth**: All keywords must be explicitly derived only from the `User Query` in the `---Real Data---` section. Do not infer unsupported facts. Do not invent entities, products, organizations, dates, or technical terms that are not grounded in the query.
5. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept instead of splitting meaningful phrases into isolated words.
6. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), return:
   `{{"high_level_keywords": [], "low_level_keywords": []}}`
7. **No Duplicates**: Do not repeat the same keyword within a list. Keep the lists short and high-signal.
8. **Language**: All extracted keywords MUST be in {language}. Proper nouns (e.g., personal names, place names, organization names) should be kept in their original language.
9. **Output Format Template Safety**: The `---Output Format Template---` section contains an output JSON template only. It is never source text. Do not extract, infer, or copy keywords from the template. Angle-bracket tokens such as `<high_level_keyword>` are placeholders; replace them only with keywords derived from the current `User Query` and never output the placeholders literally.

---Output Format Template---
The following content is an output JSON format template only. It is not source text and must never be used as keyword extraction content.

{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

# LightRAG keywords_extraction_examples（prompt.py:517-523）
_KEYWORDS_EXTRACTION_EXAMPLES = [
    """{
  "high_level_keywords": ["<high_level_keyword>"],
  "low_level_keywords": ["<low_level_keyword>"]
}
""",
]

# ADV-03: query 长度上限，防止超长 query 导致 LLM 上下文溢出和缓存污染
_MAX_QUERY_LENGTH = 8192


def _strip_markdown_code_fence(text: str) -> str:
    """去除 LLM 输出可能包裹的 markdown 代码块围栏（```json ... ```）。"""
    match = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return text


def _normalize_keyword_list(raw_values: Any, field_name: str) -> list[str]:
    """将 LLM 输出的关键词字段归一化为干净的字符串列表。

    - 字符串：按换行/逗号/分号拆分
    - 列表：保留多词短语不拆分
    - 其他类型：返回空列表
    """
    if raw_values is None:
        return []

    if isinstance(raw_values, str):
        raw_values = [part.strip() for part in re.split(r"[\n,;]+", raw_values) if part and part.strip()]

    if not isinstance(raw_values, list):
        logger.warning(f"Keyword extraction field '{field_name}' is not a list: {raw_values!r}")
        return []

    normalized: list[str] = []
    for value in raw_values:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                normalized.append(cleaned)
    return normalized


def _parse_keywords_payload(result: Any) -> tuple[list[str], list[str]]:
    """三级容错解析 LLM 关键词输出，失败返回 ([], [])。

    1. 直接 dict / 有 model_dump 的对象
    2. 字符串：json.loads
    3. 字符串：去 markdown fence 后 json_repair.loads
    """
    if result is None:
        return [], []

    payload: Any
    if hasattr(result, "model_dump") and callable(result.model_dump):
        payload = result.model_dump()
    elif isinstance(result, dict):
        payload = result
    elif isinstance(result, str):
        cleaned = _strip_markdown_code_fence(result).strip()
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                payload = json_repair.loads(cleaned)
                logger.warning(f"Keyword extraction response required JSON repair: response={cleaned[:500]!r}")
            except Exception as repair_exc:  # noqa: BLE001 - JSON 修复失败须降级返回空关键词而非抛出（源项目语义：检索路径自动降级）
                logger.warning(f"Keyword extraction JSON parse failed: {repair_exc}; response={cleaned[:500]!r}")
                return [], []
    else:
        logger.warning(f"Unsupported keyword extraction response type: {type(result).__name__}")
        return [], []

    if not isinstance(payload, dict):
        logger.warning(f"Keyword extraction payload is not a JSON object: {type(payload).__name__}")
        return [], []

    hl_keywords = _normalize_keyword_list(payload.get("high_level_keywords"), "high_level_keywords")
    ll_keywords = _normalize_keyword_list(payload.get("low_level_keywords"), "low_level_keywords")
    return hl_keywords, ll_keywords


class _LRUCache:
    """轻量进程内 LRU 缓存（避免引入 functools.lru_cache 的全局污染）。"""

    def __init__(self, capacity: int = 256):
        self.capacity = max(1, capacity)
        self._store: OrderedDict[str, tuple[list[str], list[str]]] = OrderedDict()

    def get(self, key: str) -> tuple[list[str], list[str]] | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        # Adapted: return copies so callers cannot mutate (and thus poison)
        # the shared module-level cache through the returned keyword lists.
        high, low = self._store[key]
        return (list(high), list(low))

    def put(self, key: str, value: tuple[list[str], list[str]]) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


# P1-2 修复：模块级单例缓存。原实现将 cache 挂在实例上，但 _retrieve_graph_chunks_v2
# 每次查询都新建 KeywordExtractor 实例，导致缓存形同虚设。改为模块级单例后，
# 所有实例共享同一缓存，命中相同 query 时 0 次 LLM 调用（兑现设计 5.2 节承诺）。
_GLOBAL_KEYWORD_CACHE = _LRUCache(capacity=256)


class KeywordExtractor:
    """HL/LL 双层关键词抽取（移植自 LightRAG extract_keywords_only）。

    使用方式：
        extractor = KeywordExtractor(model_spec=..., chat_model_fn=...)
        hl_keywords, ll_keywords = await extractor.extract_hl_ll(query)
    """

    def __init__(
        self,
        model_spec: str,
        chat_model_fn: Callable[..., Awaitable[Any]] | None = None,
        request_timeout: float = 60.0,
        max_async: int = 4,
        cache_capacity: int = 256,
        language: str = "Chinese",
    ):
        # Adapted: the model call is injected via chat_model_fn instead of
        # being built with yuxi.models.chat.select_model (no langchain/openai
        # SDK, no registry/config-center dependency here). Contract matches
        # extractors/llm.py options["chat_model_fn"]: an async callable
        # accepting a messages list and returning a GeneralResponse-like
        # object with a .content attribute (e.g. OpenAIChatAdapter.call_collect).
        if chat_model_fn is None or not callable(chat_model_fn):
            raise ValueError("chat_model_fn is required (async (messages: list[dict]) -> GeneralResponse)")
        self.chat_model_fn = chat_model_fn
        # Adapted: model_spec no longer selects the model (the fn is injected),
        # so the fn identity must partition the cache key — two extractors with
        # the same model_spec string but different fns must not cross-contaminate
        # the shared module-level cache. module+qualname is stable in-process and
        # discriminates module-level functions and distinct callable classes.
        self._fn_identity = (
            f"{type(self.chat_model_fn).__module__}."
            f"{getattr(self.chat_model_fn, '__qualname__', type(self.chat_model_fn).__name__)}"
        )
        self.model_spec = model_spec
        # Adapted: request_timeout kept for signature compatibility and
        # ignored — timeout is owned by the injected chat_model_fn.
        self.request_timeout = request_timeout
        self.max_async = max_async
        self.language = language
        # 独立 Semaphore，不与全局 Semaphore(10) 竞争
        self._semaphore = asyncio.Semaphore(max_async)
        # P1-2：使用模块级单例缓存，跨实例共享（cache_capacity 参数保留向后兼容，
        # 实际容量由 _GLOBAL_KEYWORD_CACHE 决定）
        self._cache = _GLOBAL_KEYWORD_CACHE
        _ = cache_capacity  # 参数保留以维持向后兼容

    async def extract_hl_ll(self, query: str) -> tuple[list[str], list[str]]:
        """抽取高层（HL）和低层（LL）关键词。

        Returns:
            (hl_keywords, ll_keywords) —— 失败返回 ([], [])，调用方应据此降级 v1 路径

        P1-3 修复：LLM 调用失败或解析结果为空时重试 1 次（设计 5.2 节"LLM 返回非法
        JSON → 重试 1 次"）。重试仅在第 1 次尝试失败时触发，最多 2 次 LLM 调用。
        """
        if not query or not query.strip():
            return [], []

        # ADV-03: 超长 query 截断，防止 LLM 上下文溢出 + 缓存 key 内存膨胀
        if len(query) > _MAX_QUERY_LENGTH:
            logger.warning(
                "ADV-03: KeywordExtractor query too long (%d chars), truncating to %d",
                len(query),
                _MAX_QUERY_LENGTH,
            )
            query = query[:_MAX_QUERY_LENGTH]

        # ADV-09: cache key 加入 language/model_spec/fn 身份分区，避免不同配置下缓存污染
        cache_key = f"{self.language}:{self.model_spec}:{self._fn_identity}:{query.strip()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"KeywordExtractor cache hit for query: {cache_key[:80]!r}")
            return cached

        examples = "\n".join(_KEYWORDS_EXTRACTION_EXAMPLES)
        prompt = _KEYWORDS_EXTRACTION_PROMPT.format(
            query=query,
            examples=examples,
            language=self.language,
        )

        result: tuple[list[str], list[str]] = ([], [])
        async with self._semaphore:
            # P1-3：最多 2 次尝试（1 次重试）。重试触发条件：LLM 异常、空内容、解析结果为空。
            for attempt in range(2):
                try:
                    # Adapted: the original select_model(model_spec, timeout, model_params)
                    # + model.call(prompt, stream=False) is replaced by the injected
                    # chat_model_fn; a bare prompt is normalized to a single user message.
                    response = await self.chat_model_fn([{"role": "user", "content": prompt}])
                    content = (response.content if response else "").strip()
                    if not content:
                        if attempt == 0:
                            logger.warning(
                                f"KeywordExtractor LLM returned empty content for query {query[:80]!r}, retrying"
                            )
                            continue
                        logger.warning(
                            f"KeywordExtractor LLM returned empty content for query {query[:80]!r} after retry"
                        )
                        result = ([], [])
                        break
                    parsed = _parse_keywords_payload(content)
                    if parsed[0] or parsed[1]:
                        result = parsed
                        break
                    # 解析结果为空（可能是非法 JSON 或 trivial query），重试 1 次
                    if attempt == 0:
                        logger.warning(
                            f"KeywordExtractor parsed empty keywords for query {query[:80]!r}, retrying; "
                            f"response={content[:200]!r}"
                        )
                        continue
                    result = parsed
                    break
                except Exception as exc:  # noqa: BLE001 - LLM 调用失败须重试/降级为 ([], []) 而非抛出（源项目语义：检索路径自动降级 v1）
                    if attempt == 0:
                        logger.warning(f"KeywordExtractor LLM call failed for query {query[:80]!r}: {exc}, retrying")
                        continue
                    logger.warning(
                        f"KeywordExtractor LLM call failed for query {query[:80]!r} after retry: {exc}, "
                        f"degrade to empty keywords"
                    )
                    result = ([], [])
                    break

        # GKB-8: 失败降级结果不写缓存，避免 LLM 偶发故障导致同查询永久空关键词
        if result != ([], []):
            self._cache.put(cache_key, result)
        return result