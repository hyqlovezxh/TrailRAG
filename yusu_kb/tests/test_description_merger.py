"""Tests for description_merger: entity/relation description map-reduce merging.

Ported from YUSU source. LLM calls are exercised through an injected
chat_model_fn (FakeChat, same contract as test_graph_keywords_merger.py) — no
network, no env config required. Token accounting is pinned with a
char-counting fake tokenizer or the char/4 fallback; backoff sleeps are
zeroed by monkeypatching.
"""

from __future__ import annotations

import json

import pytest

from yusu_kb.knowledge.graphs import description_merger as dm
from yusu_kb.knowledge.graphs import token_utils as tu
from yusu_kb.models.chat import GeneralResponse


class FakeChat:
    """Injected chat_model_fn stand-in: async (messages: list[dict]) -> GeneralResponse.

    Serves payloads in order (last one repeats); records every call;
    raises queued errors in order for retry tests; tracks max concurrency.
    """

    def __init__(self, payloads: list[str]):
        self.payloads = payloads
        self.calls: list[list[dict]] = []
        self.errors: list[Exception] = []
        self._inflight = 0
        self.max_inflight = 0

    async def __call__(self, messages: list[dict]) -> GeneralResponse:
        self._inflight += 1
        self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            self.calls.append(messages)
            if self.errors:
                raise self.errors.pop(0)
            idx = min(len(self.calls) - 1, len(self.payloads) - 1)
            return GeneralResponse(self.payloads[idx])
        finally:
            self._inflight -= 1


def make_merger(chat: FakeChat, **kwargs) -> dm.DescriptionMerger:
    opts: dict = {"model_spec": "fake-model", "chat_model_fn": chat}
    opts.update(kwargs)
    return dm.DescriptionMerger(**opts)


def ent(entity_id: str = "e1", name: str = "张三", descs: list[str] | None = None, **extra) -> dict:
    item = {"entity_id": entity_id, "name": name, "label": "人物", "descriptions": list(descs or [])}
    item.update(extra)
    return item


def trip(
    triple_id: str = "t1",
    source: str = "张三",
    rtype: str = "任职于",
    target: str = "阿里巴巴",
    descs: list[str] | None = None,
    **extra,
) -> dict:
    item = {
        "triple_id": triple_id,
        "source_name": source,
        "relation_type": rtype,
        "target_name": target,
        "descriptions": list(descs or []),
    }
    item.update(extra)
    return item


@pytest.fixture()
def char_tokenizer(monkeypatch):
    """Deterministic tokenizer: every character counts as one token."""

    class _CharTokenizer:
        def encode(self, text: str) -> list[int]:
            return list(text)

    monkeypatch.setattr(tu, "_tokenizer", _CharTokenizer())
    monkeypatch.setattr(tu, "_tokenizer_init_failed", False)
    return _CharTokenizer


@pytest.fixture()
def no_backoff(monkeypatch):
    """Zero the 429 backoff so retry tests never sleep."""
    monkeypatch.setattr(dm, "_full_jitter_backoff", lambda *_: 0.0)


# ========================================================== module helpers


def test_build_description_list_json_lines_and_escaping():
    assert dm._build_description_list(["a", "b"]) == '{"description": "a"}\n{"description": "b"}'
    assert dm._build_description_list([]) == ""
    # ADV-02: json.dumps 转义控制字符与引号，round-trip 必须还原
    raw = ['引号"与\t制表符\r回车\n换行\\反斜杠']
    out = dm._build_description_list(raw)
    assert "\\t" in out and "\\r" in out and "\\n" in out and '\\"' in out
    assert [json.loads(line)["description"] for line in out.splitlines()] == raw


def test_constructor_requires_callable_chat_model_fn():
    with pytest.raises(ValueError, match="chat_model_fn"):
        dm.DescriptionMerger(model_spec="fake-model", chat_model_fn=None)
    with pytest.raises(ValueError, match="chat_model_fn"):
        dm.DescriptionMerger(model_spec="fake-model", chat_model_fn="not callable")


# =============================================================== merge_batch


async def test_merge_batch_empty_inputs():
    merger = make_merger(FakeChat(["x"] * 10))
    assert await merger.merge_batch([], []) == ([], [])
    assert len(merger.chat_model_fn.calls) == 0


async def test_merge_batch_item_without_descriptions_gets_empty_string():
    chat = FakeChat(["x"])
    entities, triples = await make_merger(chat).merge_batch([ent(descs=[])], [trip(descs=[])])
    assert entities == [{"entity_id": "e1", "name": "张三", "label": "人物", "description": ""}]
    assert triples == [
        {
            "triple_id": "t1",
            "source_name": "张三",
            "relation_type": "任职于",
            "target_name": "阿里巴巴",
            "description": "",
        }
    ]
    assert len(chat.calls) == 0


async def test_merge_batch_single_description_returned_as_is():
    chat = FakeChat(["x"])
    entities, _ = await make_merger(chat).merge_batch([ent(descs=["唯一描述"])], [])
    assert entities[0]["description"] == "唯一描述"
    assert len(chat.calls) == 0


async def test_merge_batch_small_list_joined_without_llm():
    chat = FakeChat(["x"])
    entities, _ = await make_merger(chat).merge_batch([ent(descs=["描述一", "描述二"])], [])
    assert entities[0]["description"] == "描述一; 描述二"
    assert len(chat.calls) == 0


async def test_merge_batch_many_descriptions_single_llm_summary():
    descs = [f"描述文本{i}" for i in range(8)]
    payload = "这是合并后的综合描述文本"
    chat = FakeChat([payload])
    entities, _ = await make_merger(chat).merge_batch([ent(descs=descs)], [])
    assert entities[0]["description"] == payload
    assert entities[0]["entity_id"] == "e1"
    assert "descriptions" not in entities[0]
    assert len(chat.calls) == 1
    content = chat.calls[0][-1]["content"]
    assert "Entity Name: 张三" in content
    assert "描述文本0" in content and "描述文本7" in content


async def test_merge_batch_dedup_and_existing_descriptions():
    chat = FakeChat(["x"])
    existing = {"e1": ["已有描述A", "已有描述B"]}
    entities, _ = await make_merger(chat).merge_batch(
        [ent(descs=["已有描述B", "新描述C"])], [], existing_descriptions=existing
    )
    # 去重保序：existing 在前，重复的 已有描述B 只保留一次
    assert entities[0]["description"] == "已有描述A; 已有描述B; 新描述C"
    assert len(chat.calls) == 0


async def test_merge_batch_does_not_mutate_input_items():
    chat = FakeChat(["x"])
    entity = ent(descs=["描述一", "描述二"])
    await make_merger(chat).merge_batch([entity], [])
    assert entity["descriptions"] == ["描述一", "描述二"]
    assert "description" not in entity


async def test_merge_batch_triple_display_name_in_prompt():
    descs = [f"关系描述{i}" for i in range(8)]
    chat = FakeChat(["这是合并后的关系描述文本"])
    _, triples = await make_merger(chat).merge_batch([], [trip(descs=descs)])
    assert triples[0]["description"] == "这是合并后的关系描述文本"
    assert "descriptions" not in triples[0]
    content = chat.calls[0][-1]["content"]
    assert "Relation Name: 张三 → 任职于 → 阿里巴巴" in content


async def test_merge_batch_entity_name_fallback_chain():
    descs = [f"d{i}" for i in range(8)]
    # name 为空 → 用 normalized_name
    chat = FakeChat(["这是第一个合并结果文本"])
    _, _ = await make_merger(chat).merge_batch([ent(name="", descs=descs) | {"normalized_name": "李四"}], [])
    assert "Entity Name: 李四" in chat.calls[0][-1]["content"]
    # name 与 normalized_name 都缺失 → 用 entity_id
    chat2 = FakeChat(["这是第二个合并结果文本"])
    await make_merger(chat2).merge_batch([ent(name="", descs=descs) | {"normalized_name": ""}], [])
    assert "Entity Name: e1" in chat2.calls[0][-1]["content"]


async def test_merge_batch_llm_failure_degrades_to_join():
    descs = [f"描述文本{i}" for i in range(8)]
    chat = FakeChat(["x"])
    chat.errors = [RuntimeError("boom")]
    entities, _ = await make_merger(chat).merge_batch([ent(descs=descs)], [])
    assert entities[0]["description"] == "; ".join(descs)
    assert len(chat.calls) == 1


async def test_merge_batch_truncation_plus_llm_failure_keeps_all_descriptions(char_tokenizer):
    """ADV-01 截断 + LLM 失败：降级 join 必须基于截断前的完整描述列表，不丢数据。"""
    descs = [f"x{i:02d}" + "y" * 97 for i in range(40)]  # 40 条互不相同的 100 字符描述
    tmpl = _template_overhead_tokens(make_merger(FakeChat(["x"])))
    # safe_budget 不足以容纳 40 条 → 截断必然触发（确认前置条件成立）
    merger = make_merger(FakeChat(["x"]), model_max_context=tmpl + 4000)
    assert len(merger._truncate_descriptions_for_context("Entity", "张三", descs)) < 40
    chat = FakeChat(["x"])
    chat.errors = [RuntimeError("boom")]
    merger = make_merger(chat, model_max_context=tmpl + 4000)
    entities, _ = await merger.merge_batch([ent(descs=descs)], [])
    merged = entities[0]["description"]
    assert merged == "; ".join(descs)
    assert merged.count("y" * 97) == 40
    assert len(chat.calls) == 1


async def test_merge_batch_max_async_limits_concurrency():
    descs = [f"d{i}" for i in range(8)]
    chat = FakeChat(["这是并发合并结果文本"])
    merger = make_merger(chat, max_async=1)
    entities, _ = await merger.merge_batch([ent("e1", descs=descs), ent("e2", descs=descs)], [])
    assert len(entities) == 2
    assert len(chat.calls) == 2
    assert chat.max_inflight == 1


# ============================================================ token budgets


async def test_allocate_token_budgets_default_within_total(char_tokenizer):
    merger = make_merger(FakeChat(["x"]))
    entities = [ent(f"e{i}", descs=["x" * 100]) for i in range(10)]  # 1000 tokens
    triples = [trip(f"t{i}", descs=["y" * 100]) for i in range(5)]  # 500 tokens
    assert merger._allocate_token_budgets(entities, triples) == (6000, 8000)


async def test_allocate_token_budgets_scales_on_overflow(char_tokenizer):
    merger = make_merger(FakeChat(["x"]), total_token_budget=1000)
    entities = [ent(f"e{i}", descs=["x" * 100]) for i in range(10)]  # 1000 tokens
    triples = [trip(f"t{i}", descs=["y" * 100]) for i in range(5)]  # 500 tokens
    # scale = 1000/1500；6000*scale=4000.0，8000*scale=5333.33→5333
    assert merger._allocate_token_budgets(entities, triples) == (4000, 5333)


async def test_allocate_token_budgets_zero_tokens_defaults():
    merger = make_merger(FakeChat(["x"]), total_token_budget=1)
    assert merger._allocate_token_budgets([], []) == (6000, 8000)


def test_split_by_token_budget_accumulates_then_falls_back(char_tokenizer):
    merger = make_merger(FakeChat(["x"]))
    items = [ent(f"e{i}", descs=["x" * 10]) for i in range(4)]  # 每项 10 tokens
    llm_items, fallback = merger._split_by_token_budget(items, 25)
    assert [i["entity_id"] for i in llm_items] == ["e0", "e1"]
    assert [i["entity_id"] for i in fallback] == ["e2", "e3"]


def test_split_by_token_budget_edge_cases(char_tokenizer):
    merger = make_merger(FakeChat(["x"]))
    # 预算耗尽后全部进 fallback
    items = [ent(f"e{i}", descs=["x" * 100]) for i in range(3)]
    llm_items, fallback = merger._split_by_token_budget(items, 50)
    assert len(llm_items) == 1 and len(fallback) == 2
    # 首项即使超预算也进 LLM 列表（避免整批被降级）
    assert merger._split_by_token_budget([items[0]], 10) == ([items[0]], [])
    # budget <= 0 → 全部 fallback
    assert merger._split_by_token_budget(items, 0) == ([], items)
    # 空输入
    assert merger._split_by_token_budget([], 10) == ([], [])


async def test_merge_batch_token_budget_fallback_join_without_llm(char_tokenizer):
    e1 = ent("e1", descs=list("abcdefgh"))  # 8 tokens
    e2 = ent("e2", descs=list("ijklmnop"))  # 8 tokens
    chat = FakeChat(["这是预算内合并结果文本"])
    entities, _ = await make_merger(chat, entity_token_budget=15).merge_batch([e1, e2], [])
    # e1 在预算内走 LLM，e2 超预算降级 join 且不调 LLM
    assert entities[0]["description"] == "这是预算内合并结果文本"
    assert entities[1]["description"] == "; ".join("ijklmnop")
    assert len(chat.calls) == 1


# ================================================================= _merge_one


async def test_merge_one_empty_and_single():
    merger = make_merger(FakeChat(["x"]))
    assert await merger._merge_one("Entity", "n", []) == ""
    assert await merger._merge_one("Entity", "n", ["仅此一条"]) == "仅此一条"
    assert len(merger.chat_model_fn.calls) == 0


async def test_merge_one_map_reduce_multi_round(char_tokenizer):
    """8 条 4-token 描述 + context=10 → map 4 批 → reduce 2 批 → join 收敛（6 次 LLM）。"""
    descs = ["aaaa", "bbbb", "cccc", "dddd", "eeee", "ffff", "gggg", "hhhh"]
    chat = FakeChat(["ABCDEFGHIJ"])
    merger = make_merger(chat, summary_context_size=10, summary_max_tokens=200)
    result = await merger._merge_one("Entity", "张三", descs)
    # 第 2 轮 reduce 后剩 2 条、总 token 20 < 200 → join 分支
    assert result == "ABCDEFGHIJ; ABCDEFGHIJ"
    assert len(chat.calls) == 6


async def test_merge_one_llm_failure_degrades_to_join():
    descs = [f"d{i}" for i in range(8)]
    chat = FakeChat(["x"])
    chat.errors = [RuntimeError("boom")]
    merger = make_merger(chat)
    assert await merger._merge_one("Entity", "张三", descs) == "; ".join(descs)


# =========================================================== _llm_summarize


async def test_llm_summarize_prompt_format_and_return():
    payload = "这是合并后的单次合并结果文本"
    chat = FakeChat([payload])
    merger = make_merger(chat)
    result = await merger._llm_summarize("Entity", "张三", ["描述一", "描述二"])
    assert result == payload
    assert len(chat.calls) == 1
    content = chat.calls[0][-1]["content"]
    assert "Entity Name: 张三" in content
    assert "Description List" in content
    assert '{"description": "描述一"}' in content
    assert "{summary_length}" not in content and "{language}" not in content
    assert "1200" in content and "Chinese" in content


async def test_llm_summarize_retries_on_rate_limit(no_backoff):
    payload = "这是重试后成功的合并结果文本"
    chat = FakeChat([payload])
    chat.errors = [RuntimeError("429 rate limit exceeded")]
    merger = make_merger(chat)
    assert await merger._llm_summarize("Entity", "张三", ["a"] * 8) == payload
    assert len(chat.calls) == 2


async def test_llm_summarize_gives_up_after_max_retries(no_backoff):
    descs = [f"d{i}" for i in range(8)]
    chat = FakeChat(["x"])
    chat.errors = [RuntimeError("429 too many requests")] * (dm._RATE_LIMIT_MAX_RETRIES + 1)
    merger = make_merger(chat)
    assert await merger._llm_summarize("Entity", "张三", descs) == "; ".join(descs)
    assert len(chat.calls) == dm._RATE_LIMIT_MAX_RETRIES + 1


async def test_llm_summarize_empty_content_degrades_to_join():
    descs = [f"d{i}" for i in range(8)]
    chat = FakeChat([""])
    merger = make_merger(chat)
    assert await merger._llm_summarize("Entity", "张三", descs) == "; ".join(descs)
    assert len(chat.calls) == 1


def test_validate_summary_quality_edges():
    merger = make_merger(FakeChat(["x"]))
    # 正常内容通过
    merger._validate_summary_quality("这是合法的合并结果文本", "Entity", "张三")
    # 过短（<10 字符）
    with pytest.raises(ValueError, match="too short"):
        merger._validate_summary_quality("太短", "Entity", "张三")
    # 过长（> summary_max_tokens*3 字符）
    short_budget = make_merger(FakeChat(["x"]), summary_max_tokens=10)
    with pytest.raises(ValueError, match="too long"):
        short_budget._validate_summary_quality("x" * 31, "Entity", "张三")
    short_budget._validate_summary_quality("x" * 30, "Entity", "张三")


# ============================================== _truncate_descriptions_for_context


def _template_overhead_tokens(merger: dm.DescriptionMerger) -> int:
    """formatted 模板（description_list 为空）的 token 数（char_tokenizer 下即字符数）。"""
    return len(
        dm._SUMMARIZE_PROMPT_TEMPLATE.format(
            summary_length=merger.summary_max_tokens,
            language=merger.language,
            description_type="Entity",
            description_name="张三",
            description_list="",
        )
    )


async def test_truncate_descriptions_for_context_no_model_info(char_tokenizer):
    merger = make_merger(FakeChat(["x"]))  # model_max_context=None
    descs = ["x" * 100 for _ in range(40)]
    assert merger._truncate_descriptions_for_context("Entity", "张三", descs) == descs


async def test_truncate_descriptions_for_context_small_prompt_unchanged(char_tokenizer):
    descs = ["x" * 10 for _ in range(5)]
    merger = make_merger(FakeChat(["x"]), model_max_context=100000)
    assert merger._truncate_descriptions_for_context("Entity", "张三", descs) == descs


async def test_truncate_descriptions_for_context_partial_keep(char_tokenizer):
    descs = ["x" * 100 for _ in range(40)]
    tmpl = _template_overhead_tokens(make_merger(FakeChat(["x"])))
    # safe_budget = 0.8 * max_context；预留空间 = safe_budget - tmpl
    max_context = tmpl + 4000
    merger = make_merger(FakeChat(["x"]), model_max_context=max_context)
    result = merger._truncate_descriptions_for_context("Entity", "张三", descs)
    expected_keep = int(0.8 * max_context - tmpl) // 100
    assert 2 <= expected_keep < 40
    assert result == descs[:expected_keep]


async def test_truncate_descriptions_for_context_min_keep_two(char_tokenizer):
    descs = ["x" * 100 for _ in range(5)]
    tmpl = _template_overhead_tokens(make_merger(FakeChat(["x"])))
    # safe_budget < tmpl → available <= 0 → 保留前 2 条
    merger = make_merger(FakeChat(["x"]), model_max_context=tmpl)
    assert merger._truncate_descriptions_for_context("Entity", "张三", descs) == descs[:2]


async def test_truncate_descriptions_for_context_two_or_fewer_unchanged(char_tokenizer):
    merger = make_merger(FakeChat(["x"]), model_max_context=1)
    assert merger._truncate_descriptions_for_context("Entity", "张三", ["a", "b"]) == ["a", "b"]
    assert merger._truncate_descriptions_for_context("Entity", "张三", ["a"]) == ["a"]