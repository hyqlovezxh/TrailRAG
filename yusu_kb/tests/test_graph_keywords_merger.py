"""Tests for graph pipeline utilities: token_utils / keyword_extractor / round_robin_merger.

Ported from YUSU source. LLM calls in KeywordExtractor are exercised through an
injected chat_model_fn (FakeChat, same contract as test_extractors.py) — no
network, no env config required. token_utils tests pin deterministic behavior by
forcing either the char/4 fallback or a char-counting fake tokenizer.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from yusu_kb.knowledge.graphs import keyword_extractor as ke
from yusu_kb.knowledge.graphs import round_robin_merger as rrm
from yusu_kb.knowledge.graphs import token_utils as tu
from yusu_kb.models.chat import GeneralResponse


class FakeChat:
    """Injected chat_model_fn stand-in: async (messages: list[dict]) -> GeneralResponse.

    Serves payloads in order (last one repeats); records every call;
    raises queued errors in order for retry tests.
    """

    def __init__(self, payloads: list[str]):
        self.payloads = payloads
        self.calls: list[list[dict]] = []
        self.errors: list[Exception] = []

    async def __call__(self, messages: list[dict]) -> GeneralResponse:
        self.calls.append(messages)
        if self.errors:
            raise self.errors.pop(0)
        idx = min(len(self.calls) - 1, len(self.payloads) - 1)
        return GeneralResponse(self.payloads[idx])


def make_extractor(chat: FakeChat, **kwargs) -> ke.KeywordExtractor:
    opts: dict = {"model_spec": "fake-model", "chat_model_fn": chat}
    opts.update(kwargs)
    return ke.KeywordExtractor(**opts)


# ================================================================ token_utils


@pytest.fixture(autouse=True)
def clear_keyword_cache():
    """Reset the module-level keyword cache between tests (test isolation).

    The cache is a module-level singleton shared by every KeywordExtractor
    instance, so tests must not leak entries into each other.
    """
    ke._GLOBAL_KEYWORD_CACHE._store.clear()


@pytest.fixture()
def fallback_tokenizer(monkeypatch):
    """Force the char/4 fallback path (no tiktoken in this project's deps)."""
    monkeypatch.setattr(tu, "_tokenizer", None)
    monkeypatch.setattr(tu, "_tokenizer_init_failed", True)


@pytest.fixture()
def char_tokenizer(monkeypatch):
    """Deterministic tokenizer: every character counts as one token."""

    class _CharTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(text)

    monkeypatch.setattr(tu, "_tokenizer", _CharTokenizer())
    monkeypatch.setattr(tu, "_tokenizer_init_failed", False)
    return _CharTokenizer


def test_get_tokenizer_returns_singleton(monkeypatch):
    tokenizer = object()
    monkeypatch.setattr(tu, "_tokenizer", tokenizer)
    assert tu.get_tokenizer() is tokenizer


def test_get_tokenizer_init_failure_falls_back_and_memorializes(monkeypatch):
    monkeypatch.setattr(tu, "_tokenizer", None)
    monkeypatch.setattr(tu, "_tokenizer_init_failed", False)
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    assert tu.get_tokenizer() is None
    assert tu._tokenizer_init_failed is True
    # flag prevents a second import attempt
    assert tu.get_tokenizer() is None


def test_count_tokens_empty():
    assert tu.count_tokens("") == 0


def test_count_tokens_fallback_chars_div4(fallback_tokenizer):
    assert tu.count_tokens("abcdefgh") == 2
    assert tu.count_tokens("abc") == 0


def test_count_tokens_uses_tokenizer_when_available(char_tokenizer):
    assert tu.count_tokens("abc") == 3
    assert tu.count_tokens("你好世界") == 4


def test_count_tokens_falls_back_when_encode_raises(monkeypatch):
    """token_utils.py:46 path — encode failure degrades to char/4 silently."""

    class _RaisingTokenizer:
        def encode(self, text: str) -> list[int]:
            raise RuntimeError("encode exploded")

    monkeypatch.setattr(tu, "_tokenizer", _RaisingTokenizer())
    monkeypatch.setattr(tu, "_tokenizer_init_failed", False)
    assert tu.count_tokens("abcdefgh") == 2


def test_truncate_list_empty_and_nonpositive_budget(fallback_tokenizer):
    assert tu.truncate_list_by_token_size([], lambda s: s, 10) == []
    assert tu.truncate_list_by_token_size(["a", "b"], lambda s: s, 0) == []
    assert tu.truncate_list_by_token_size(["a", "b"], lambda s: s, -1) == []


def test_truncate_list_accumulates_until_budget(fallback_tokenizer):
    items = ["aaaa", "bbbb", "cccc", "dddd"]  # 每项 1 token（4 字符）
    assert tu.truncate_list_by_token_size(items, lambda s: s, 2) == ["aaaa", "bbbb"]


def test_truncate_list_breaks_before_overshooting_budget(fallback_tokenizer):
    items = ["aaaa", "bbbb", "cccc"]
    # 预算 1：第一项 1 token 恰好；第二项累计 2 > 1 → 截断
    assert tu.truncate_list_by_token_size(items, lambda s: s, 1) == ["aaaa"]


def test_truncate_list_first_item_over_budget_truncates_string(fallback_tokenizer):
    result = tu.truncate_list_by_token_size(["aaaaaaaaaaaaaaaa"], lambda s: s, 1)
    assert result == ["aaaa"]


def test_truncate_list_first_item_over_budget_keeps_non_string(fallback_tokenizer):
    item = {"text": "aaaaaaaaaaaaaaaa", "weight": 0.9}
    result = tu.truncate_list_by_token_size([item], lambda d: d["text"], 1)
    assert result == [item]


def test_truncate_list_uses_tokenizer_when_available(char_tokenizer):
    items = ["a", "bb", "ccc"]  # 每字符 1 token；预算 3 → 保留 "a" + "bb"
    assert tu.truncate_list_by_token_size(items, lambda s: s, 3) == ["a", "bb"]


def test_truncate_section_context_empty_and_within_budget(fallback_tokenizer):
    assert tu.truncate_section_context("") == ""
    assert tu.truncate_section_context("第一章 → 第一节") == "第一章 → 第一节"


def test_truncate_section_context_zero_budget_returns_unchanged(fallback_tokenizer):
    assert tu.truncate_section_context("abc", max_tokens=0) == "abc"


def test_truncate_section_context_folds_three_levels(char_tokenizer):
    path = "第一章 → 第一节 → 第一小节 → 细节"
    # 折叠后 "第一章 → … → 细节" 仍超预算（14 > 6）但长度不足 50，不再硬截断
    assert tu.truncate_section_context(path, max_tokens=6) == "第一章 → … → 细节"


def test_truncate_section_context_hard_truncates_dense_path(char_tokenizer):
    path = "很长的章节标题" * 30  # 90 chars = 90 tokens，单层无折叠机会
    result = tu.truncate_section_context(path, max_tokens=10)
    assert result.endswith("…")
    assert len(result) <= 51


def test_truncate_section_context_fallback_hard_truncate(fallback_tokenizer):
    path = "第一节" * 50  # 150 chars → 37 tokens（字符/4）
    result = tu.truncate_section_context(path, max_tokens=5)
    assert result.endswith("…")


# ============================================================ keyword_extractor


def test_keyword_extractor_requires_callable_chat_model_fn():
    with pytest.raises(ValueError, match="chat_model_fn"):
        ke.KeywordExtractor(model_spec="fake-model", chat_model_fn=None)
    with pytest.raises(ValueError, match="chat_model_fn"):
        ke.KeywordExtractor(model_spec="fake-model", chat_model_fn="not callable")


async def test_extract_hl_ll_parses_payload():
    payload = {"high_level_keywords": ["主题概念"], "low_level_keywords": ["张三", "阿里巴巴"]}
    chat = FakeChat([json.dumps(payload, ensure_ascii=False)])
    hl, ll = await make_extractor(chat).extract_hl_ll("张三在阿里巴巴做什么")
    assert hl == ["主题概念"]
    assert ll == ["张三", "阿里巴巴"]
    assert len(chat.calls) == 1
    message = chat.calls[0][-1]
    assert message["role"] == "user"
    assert "张三在阿里巴巴做什么" in message["content"]
    assert "high_level_keywords" in message["content"]


async def test_extract_hl_ll_strips_markdown_fence():
    payload = '```json\n{"high_level_keywords": ["a"], "low_level_keywords": ["b"]}\n```'
    chat = FakeChat([payload])
    hl, ll = await make_extractor(chat).extract_hl_ll("q")
    assert (hl, ll) == (["a"], ["b"])


async def test_extract_hl_ll_splits_string_keywords():
    payload = '{"high_level_keywords": "主题1\\n主题2,主题3", "low_level_keywords": ["b"]}'
    chat = FakeChat([payload])
    hl, ll = await make_extractor(chat).extract_hl_ll("split-query-1")
    assert hl == ["主题1", "主题2", "主题3"]
    assert ll == ["b"]


async def test_extract_hl_ll_recovers_repairable_json():
    # 缺右括号的 JSON，json_repair 应能修复解析（三级容错的第三级）
    payload = '{"high_level_keywords": ["a"], "low_level_keywords": ["b"]'
    chat = FakeChat([payload])
    hl, ll = await make_extractor(chat).extract_hl_ll("repair-query-1")
    assert (hl, ll) == (["a"], ["b"])
    assert len(chat.calls) == 1


async def test_extract_hl_ll_garbage_retries_then_empty_and_no_cache():
    chat = FakeChat(["这不是 JSON"])
    extractor = make_extractor(chat)
    assert await extractor.extract_hl_ll("garbage-query-1") == ([], [])
    assert len(chat.calls) == 2
    # GKB-8: 失败降级结果不写缓存 → 同 query 再次调用仍走 LLM
    assert await extractor.extract_hl_ll("garbage-query-1") == ([], [])
    assert len(chat.calls) == 4


async def test_extract_hl_ll_empty_content_retries_then_empty():
    chat = FakeChat([""])
    extractor = make_extractor(chat)
    assert await extractor.extract_hl_ll("empty-query-1") == ([], [])
    assert len(chat.calls) == 2


async def test_extract_hl_ll_empty_content_then_success_on_retry():
    payload = '{"high_level_keywords": ["ok"], "low_level_keywords": ["k"]}'
    chat = FakeChat(["", payload])
    hl, ll = await make_extractor(chat).extract_hl_ll("empty-retry-query-1")
    assert (hl, ll) == (["ok"], ["k"])
    assert len(chat.calls) == 2


async def test_extract_hl_ll_retries_after_exception():
    chat = FakeChat(['{"high_level_keywords": ["ok"], "low_level_keywords": []}'])
    chat.errors = [RuntimeError("boom")]
    hl, ll = await make_extractor(chat).extract_hl_ll("retry-query-1")
    assert hl == ["ok"]
    assert ll == []
    assert len(chat.calls) == 2


async def test_extract_hl_ll_gives_up_after_two_exceptions():
    chat = FakeChat([""])
    chat.errors = [RuntimeError("boom"), RuntimeError("boom again")]
    extractor = make_extractor(chat)
    assert await extractor.extract_hl_ll("fail-query-1") == ([], [])
    assert len(chat.calls) == 2


async def test_extract_hl_ll_cache_hit_skips_llm():
    payload = '{"high_level_keywords": ["c"], "low_level_keywords": []}'
    chat = FakeChat([payload])
    extractor = make_extractor(chat)
    query = "cache-query-1"
    assert await extractor.extract_hl_ll(query) == (["c"], [])
    assert len(chat.calls) == 1
    # 同 query 命中缓存，不再调用 LLM
    assert await extractor.extract_hl_ll(query) == (["c"], [])
    assert len(chat.calls) == 1
    # P1-2: 缓存为模块级单例，跨实例共享 → 新实例 0 次调用
    chat2 = FakeChat([payload])
    extractor2 = make_extractor(chat2)
    assert await extractor2.extract_hl_ll(query) == (["c"], [])
    assert len(chat2.calls) == 0


async def test_extract_hl_ll_cache_key_partitions_by_model_and_language():
    payload = '{"high_level_keywords": ["x"], "low_level_keywords": []}'
    chat = FakeChat([payload])
    query = "partition-query-1"
    await make_extractor(chat, model_spec="model-a").extract_hl_ll(query)
    assert len(chat.calls) == 1
    await make_extractor(chat, model_spec="model-b").extract_hl_ll(query)
    assert len(chat.calls) == 2
    await make_extractor(chat, model_spec="model-a", language="English").extract_hl_ll(query)
    assert len(chat.calls) == 3
    # 相同 (language, model_spec, query) 再次命中缓存
    await make_extractor(chat, model_spec="model-a", language="English").extract_hl_ll(query)
    assert len(chat.calls) == 3


async def test_extract_hl_ll_cache_key_partitions_by_fn_identity():
    class OtherChat:
        """A distinct injected fn class (different qualname from FakeChat)."""

        def __init__(self):
            self.calls = 0

        async def __call__(self, messages: list[dict]) -> GeneralResponse:
            self.calls += 1
            return GeneralResponse('{"high_level_keywords": ["m"], "low_level_keywords": []}')

    payload = '{"high_level_keywords": ["x"], "low_level_keywords": []}'
    chat = FakeChat([payload])
    query = "fn-partition-query-1"
    await make_extractor(chat).extract_hl_ll(query)
    assert len(chat.calls) == 1
    # 同 model_spec + 不同注入函数（不同 fn 身份）→ 不命中 FakeChat 的缓存条目
    other = OtherChat()
    await make_extractor(other).extract_hl_ll(query)
    assert other.calls == 1
    # 同 fn 身份（同类新实例）→ 命中自己的条目
    await make_extractor(other).extract_hl_ll(query)
    assert other.calls == 1
    assert len(chat.calls) == 1


async def test_extract_hl_ll_empty_query_returns_empty_without_llm():
    chat = FakeChat(['{"high_level_keywords": ["x"], "low_level_keywords": []}'])
    extractor = make_extractor(chat)
    assert await extractor.extract_hl_ll("") == ([], [])
    assert await extractor.extract_hl_ll("   ") == ([], [])
    assert len(chat.calls) == 0


async def test_extract_hl_ll_truncates_oversized_query():
    chat = FakeChat(['{"high_level_keywords": [], "low_level_keywords": []}'])
    extractor = make_extractor(chat)
    long_query = "长" * 10000
    assert await extractor.extract_hl_ll(long_query) == ([], [])
    content = chat.calls[0][-1]["content"]
    assert "长" * ke._MAX_QUERY_LENGTH in content
    assert "长" * (ke._MAX_QUERY_LENGTH + 1) not in content


async def test_extract_hl_ll_truncation_boundary_at_max_query_length():
    chat = FakeChat(['{"high_level_keywords": [], "low_level_keywords": []}'])
    extractor = make_extractor(chat)
    exact = "长" * ke._MAX_QUERY_LENGTH
    assert await extractor.extract_hl_ll(exact) == ([], [])
    content = chat.calls[0][-1]["content"]
    assert "长" * ke._MAX_QUERY_LENGTH in content
    # 超上限一个字符 → 截断到上限（前缀不同，不会命中上面的缓存条目）
    over = "X" * (ke._MAX_QUERY_LENGTH + 1)
    assert await extractor.extract_hl_ll(over) == ([], [])
    content2 = chat.calls[-1][-1]["content"]
    assert "X" * ke._MAX_QUERY_LENGTH in content2
    assert "X" * (ke._MAX_QUERY_LENGTH + 1) not in content2


def test_parse_keywords_payload_dict_and_model_dump():
    raw = {"high_level_keywords": ["a"], "low_level_keywords": ["b"]}
    assert ke._parse_keywords_payload(raw) == (["a"], ["b"])
    obj = SimpleNamespace(model_dump=lambda: dict(raw))
    assert ke._parse_keywords_payload(obj) == (["a"], ["b"])


def test_parse_keywords_payload_string_fences_garbage_and_types():
    raw = '```json\n{"high_level_keywords": ["a"], "low_level_keywords": ["b"]}\n```'
    assert ke._parse_keywords_payload(raw) == (["a"], ["b"])
    assert ke._parse_keywords_payload("not json at all") == ([], [])
    assert ke._parse_keywords_payload(None) == ([], [])
    assert ke._parse_keywords_payload(123) == ([], [])
    assert ke._parse_keywords_payload("42") == ([], [])  # 合法 JSON 但非对象


def test_normalize_keyword_list():
    assert ke._normalize_keyword_list(None, "f") == []
    assert ke._normalize_keyword_list("a\nb,c;d", "f") == ["a", "b", "c", "d"]
    assert ke._normalize_keyword_list([" a ", "", "b"], "f") == ["a", "b"]
    assert ke._normalize_keyword_list([1, "x"], "f") == ["x"]
    assert ke._normalize_keyword_list(42, "f") == []


def test_strip_markdown_code_fence():
    assert ke._strip_markdown_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert ke._strip_markdown_code_fence("```\nhello\n```") == "hello"
    assert ke._strip_markdown_code_fence("plain") == "plain"


def test_lru_cache_eviction_and_recency():
    cache = ke._LRUCache(capacity=2)
    cache.put("a", (["a"], []))
    cache.put("b", (["b"], []))
    cache.put("c", (["c"], []))
    assert cache.get("a") is None  # 最久未使用被淘汰
    assert cache.get("b") == (["b"], [])
    assert cache.get("c") == (["c"], [])
    cache.get("b")  # 提升 b 的 recency
    cache.put("d", (["d"], []))
    assert cache.get("c") is None  # c 变为最久未使用被淘汰
    assert cache.get("b") == (["b"], [])


# =========================================================== round_robin_merger


def test_merge_seed_lists_round_robin_interleaves_and_max_weight():
    merged = rrm.merge_seed_lists_round_robin(
        [("A", 0.9), ("B", 0.8)],
        [("B", 0.95), ("C", 0.7)],
    )
    assert merged == {"A": 0.9, "B": 0.95, "C": 0.7}
    # 交错顺序保留首次出现位置
    assert list(merged) == ["A", "B", "C"]


def test_merge_seed_lists_round_robin_keeps_first_position_on_weight_update():
    merged = rrm.merge_seed_lists_round_robin([("X", 0.3), ("Y", 0.8)], [("X", 0.9)])
    assert merged == {"X": 0.9, "Y": 0.8}
    assert list(merged) == ["X", "Y"]


def test_merge_seed_lists_round_robin_unequal_lengths():
    merged = rrm.merge_seed_lists_round_robin(
        [("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6)],
        [("X", 0.5)],
    )
    assert list(merged) == ["A", "X", "B", "C", "D"]

    merged2 = rrm.merge_seed_lists_round_robin([("A", 0.9)], [("X", 0.5), ("Y", 0.4), ("Z", 0.3)])
    assert list(merged2) == ["A", "X", "Y", "Z"]


def test_merge_seed_lists_round_robin_empty_inputs():
    assert rrm.merge_seed_lists_round_robin([], []) == {}
    assert rrm.merge_seed_lists_round_robin([("A", 0.9)], []) == {"A": 0.9}
    assert rrm.merge_seed_lists_round_robin([], [("A", 0.9)]) == {"A": 0.9}


def test_merge_seed_lists_round_robin_duplicate_within_list_keeps_max():
    merged = rrm.merge_seed_lists_round_robin([("A", 0.5), ("A", 0.9)], [])
    assert merged == {"A": 0.9}
    assert list(merged) == ["A"]
