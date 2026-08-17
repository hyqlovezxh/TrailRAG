"""Tests for graph extractors (normalization + LLM extraction layer).

Covers the ported normalize_extraction_result contract and the
LLMGraphExtractor flow with an injected FakeChat chat_model_fn
(async (messages: list[dict]) -> GeneralResponse).
"""

from __future__ import annotations

import json

import pytest

from yusu_kb.knowledge.graphs.extractors import llm as llm_mod
from yusu_kb.knowledge.graphs.extractors.base import normalize_extraction_result
from yusu_kb.knowledge.graphs.extractors.llm import (
    DEFAULT_TRIPLE_EXTRACTION_PROMPT,
    GLEANING_CONTINUE_PROMPT,
    LLMGraphExtractor,
)
from yusu_kb.models.chat import ChatModelError, GeneralResponse


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


def make_extractor(chat: FakeChat, **options) -> LLMGraphExtractor:
    opts: dict = {"model_spec": "fake-model", "chat_model_fn": chat}
    opts.update(options)
    return LLMGraphExtractor(opts)


# ---------------------------------------------------------------- normalize


def test_normalize_requires_dict_result():
    with pytest.raises(ValueError, match="必须是对象"):
        normalize_extraction_result("not a dict", "llm")


def test_normalize_requires_list_fields():
    with pytest.raises(ValueError, match="必须是数组"):
        normalize_extraction_result({"entities": "not a list", "relations": []}, "llm")


def test_normalize_expands_dict_endpoints_into_entities():
    result = normalize_extraction_result(
        {
            "entities": [],
            "relations": [
                {
                    "source": {"text": "张三", "label": "人物"},
                    "target": {"text": "阿里巴巴", "label": "机构"},
                    "text": "任职于",
                    "label": "任职于",
                }
            ],
        },
        "llm",
    )
    assert len(result["entities"]) == 2
    assert result["relations"][0]["source"]["text"] == "张三"
    assert result["relations"][0]["target"]["text"] == "阿里巴巴"


def test_normalize_auto_registers_placeholder_entity_for_unmatched_endpoint():
    result = normalize_extraction_result(
        {
            "entities": [],
            "relations": [
                {"source": "张三", "target": "阿里巴巴", "text": "任职于", "label": "任职于"}
            ],
        },
        "llm",
    )
    assert len(result["entities"]) == 2
    placeholder = next(e for e in result["entities"] if e["text"] == "阿里巴巴")
    assert placeholder["label"] == "Entity"
    assert placeholder["descriptions"] == ["阿里巴巴，作为关系端点出现于当前文本。"]


def test_normalize_skips_invalid_entities_and_relations():
    result = normalize_extraction_result(
        {
            "entities": [
                {"text": "  ", "label": "人物"},
                {"text": "张三", "label": "人物"},
                {"text": "李四", "label": "人物"},
                "not a dict",
            ],
            "relations": [
                {"source": "张三", "target": "李四", "text": "通话", "label": "通话联系"},
                {"source": "张三", "target": "李四", "text": " ", "label": "任职于"},
                {"source": "", "target": "李四", "text": "通话", "label": "通话联系"},
            ],
        },
        "llm",
    )
    assert [e["text"] for e in result["entities"]] == ["张三", "李四"]
    assert len(result["relations"]) == 1
    assert result["relations"][0]["text"] == "通话"


def test_normalize_dedups_entities_and_merges_attributes_and_descriptions():
    result = normalize_extraction_result(
        {
            "entities": [
                {
                    "text": "张三",
                    "label": "人物",
                    "attributes": [{"text": "工程师", "label": "职位"}],
                    "descriptions": ["张三，工程师。"],
                },
                {
                    "text": "张三。",
                    "label": "人物",
                    "attributes": [{"text": "阿里巴巴", "label": "公司"}],
                    "descriptions": ["张三，阿里巴巴员工。"],
                },
            ],
            "relations": [],
        },
        "llm",
    )
    assert len(result["entities"]) == 1
    entity = result["entities"][0]
    assert {(a["text"], a["label"]) for a in entity["attributes"]} == {
        ("工程师", "职位"),
        ("阿里巴巴", "公司"),
    }
    assert entity["descriptions"] == ["张三，工程师。", "张三，阿里巴巴员工。"]


def test_normalize_joins_relation_descriptions():
    result = normalize_extraction_result(
        {
            "entities": [],
            "relations": [
                {
                    "source": "张三",
                    "target": "阿里巴巴",
                    "text": "任职于",
                    "label": "任职于",
                    "description": "张三在阿里巴巴工作。",
                    "descriptions": ["张三在阿里巴巴工作。", "担任高级工程师。"],
                }
            ],
        },
        "llm",
    )
    rel = result["relations"][0]
    assert rel["description"] == "张三在阿里巴巴工作。; 担任高级工程师。"


def test_normalize_metadata_contains_extractor_type_and_schema_version():
    result = normalize_extraction_result(
        {"entities": [], "relations": [], "metadata": {"kb_id": "kb1"}},
        "llm",
    )
    assert result["metadata"]["extractor_type"] == "llm"
    assert result["metadata"]["schema_version"] == 1
    assert result["metadata"]["kb_id"] == "kb1"


def test_normalize_entity_without_description_gets_template_fallback():
    result = normalize_extraction_result(
        {"entities": [{"text": "张三", "label": "人物"}], "relations": []},
        "llm",
    )
    assert result["entities"][0]["descriptions"] == ["张三，类型：人物。"]


def test_normalize_relation_without_description_gets_template_fallback():
    result = normalize_extraction_result(
        {
            "entities": [],
            "relations": [{"source": "张三", "target": "阿里巴巴", "text": "任职于", "label": "任职于"}],
        },
        "llm",
    )
    assert result["relations"][0]["description"] == "张三 任职于 阿里巴巴。"


def test_normalize_skips_attribute_entries_without_text():
    result = normalize_extraction_result(
        {
            "entities": [
                {
                    "text": "张三",
                    "label": "人物",
                    "attributes": [{"text": "", "label": "职位"}, {"label": "无文本"}, "bad"],
                }
            ],
            "relations": [],
        },
        "llm",
    )
    assert result["entities"][0]["attributes"] == []


# ------------------------------------------------------------- validate_options


def test_validate_options_requires_model_spec():
    with pytest.raises(ValueError, match="model_spec"):
        LLMGraphExtractor({"chat_model_fn": FakeChat(['"x"'])}).validate_options()


def test_validate_options_requires_chat_model_fn():
    with pytest.raises(ValueError, match="chat_model_fn is required"):
        LLMGraphExtractor({"model_spec": "fake-model"}).validate_options()


def test_validate_options_rejects_gleaning_count_out_of_range():
    with pytest.raises(ValueError, match="gleaning_count"):
        make_extractor(FakeChat(['"x"']), gleaning_count=5).validate_options()


def test_validate_options_rejects_custom_prompt():
    with pytest.raises(ValueError, match="不支持自定义"):
        make_extractor(FakeChat(['"x"']), prompt="自定义").validate_options()


# -------------------------------------------------------------------- extract


async def test_extract_parses_fake_chat_json():
    payload = {
        "entities": [{"text": "张三", "label": "人物"}],
        "relations": [
            {"source": "张三", "target": "阿里巴巴", "text": "任职于", "label": "任职于"}
        ],
    }
    chat = FakeChat([json.dumps(payload, ensure_ascii=False)])
    extractor = make_extractor(chat)
    result = await extractor.extract("张三是阿里巴巴的高级工程师。")
    assert result["entities"] == payload["entities"]
    assert result["relations"] == payload["relations"]
    assert len(chat.calls) == 1
    assert chat.calls[0][0]["role"] == "user"
    assert "文本：" in chat.calls[0][0]["content"]


async def test_extract_garbage_response_falls_back_to_empty_result():
    # json_repair.loads returns "" for non-JSON garbage (verified against
    # the installed json-repair), and extract maps non-dict parses to an
    # empty entities/relations result — source behavior, no exception.
    chat = FakeChat(["这不是 JSON 输出"])
    extractor = make_extractor(chat)
    result = await extractor.extract("文本")
    assert result == {"entities": [], "relations": []}


async def test_extract_gleaning_round_merges_and_dedups():
    initial = {
        "entities": [{"text": "张三", "label": "人物", "description": "短描述"}],
        "relations": [
            {"source": "张三", "target": "阿里巴巴", "text": "任职于", "label": "任职于", "description": "d1"}
        ],
    }
    glean = {
        "entities": [
            {"text": "张三", "label": "人物", "description": "更长的描述"},
            {"text": "李四", "label": "人物", "description": "李四，人物。"},
        ],
        "relations": [
            {"source": "张三", "target": "阿里巴巴", "text": "任职于", "label": "任职于", "description": "d2 更长"},
            {"source": "张三", "target": "杭州", "text": "位于", "label": "位于", "description": "张三位居杭州。"},
        ],
    }
    chat = FakeChat([json.dumps(initial, ensure_ascii=False), json.dumps(glean, ensure_ascii=False)])
    extractor = make_extractor(chat, gleaning_count=1)
    result = await extractor.extract("文本")
    entities = {e["text"]: e for e in result["entities"]}
    assert set(entities) == {"张三", "李四"}
    assert entities["张三"]["description"] == "更长的描述"
    assert len(result["relations"]) == 2
    rel_by_target = {r["target"]: r for r in result["relations"]}
    assert rel_by_target["阿里巴巴"]["description"] == "d2 更长"
    # Gleaning round sends the full multi-turn history
    assert len(chat.calls) == 2
    roles = [m["role"] for m in chat.calls[1]]
    assert roles == ["user", "assistant", "user"]
    assert chat.calls[1][2]["content"] == GLEANING_CONTINUE_PROMPT


async def test_extract_builds_prompt_with_section_breadcrumb_and_title():
    chat = FakeChat(['{"entities": [], "relations": []}'])
    extractor = make_extractor(chat)
    await extractor.extract(
        "文本",
        chunk_metadata={
            "document_title": "案件卷宗",
            "heading_path": "卷一 → 第二章 → 资金流水",
        },
    )
    prompt = chat.calls[0][0]["content"]
    assert "文档章节路径：卷一 → 第二章 → 资金流水" in prompt
    assert "文档标题：案件卷宗" in prompt
    assert "实体不超过 30 个" in DEFAULT_TRIPLE_EXTRACTION_PROMPT
    assert "关系不超过 50 条" in DEFAULT_TRIPLE_EXTRACTION_PROMPT


# ------------------------------------------------------------- retry behavior


async def test_call_llm_with_retry_succeeds_after_429s(monkeypatch):
    monkeypatch.setattr(llm_mod, "_full_jitter_backoff", lambda attempt, base, cap: 0.0)
    chat = FakeChat(['{"entities": [], "relations": []}'])
    chat.errors = [ValueError("429 rate limit exceeded"), ValueError("429 too many requests")]
    extractor = make_extractor(chat)
    content, _ = await extractor._call_llm_with_retry(
        chat,
        "prompt",
        chunk_id="chunk-1",
        model_spec="fake-model",
    )
    assert content == '{"entities": [], "relations": []}'
    assert len(chat.calls) == 3


async def test_call_llm_with_retry_succeeds_after_transient_error(monkeypatch):
    monkeypatch.setattr(llm_mod, "_full_jitter_backoff", lambda attempt, base, cap: 0.0)
    chat = FakeChat(['{"entities": [], "relations": []}'])
    chat.errors = [ValueError("connection timed out")]
    extractor = make_extractor(chat)
    content, _ = await extractor._call_llm_with_retry(
        chat,
        "prompt",
        chunk_id="chunk-1",
        model_spec="fake-model",
    )
    assert content == '{"entities": [], "relations": []}'
    assert len(chat.calls) == 2


async def test_call_llm_with_retry_raises_on_non_retryable_error(monkeypatch):
    monkeypatch.setattr(llm_mod, "_full_jitter_backoff", lambda attempt, base, cap: 0.0)
    chat = FakeChat(['{"entities": [], "relations": []}'])
    chat.errors = [ValueError("boom")]
    extractor = make_extractor(chat)
    with pytest.raises(ValueError, match="boom"):
        await extractor._call_llm_with_retry(
            chat,
            "prompt",
            chunk_id="chunk-1",
            model_spec="fake-model",
        )
    assert len(chat.calls) == 1


async def test_call_llm_with_retry_gives_up_after_max_429_retries(monkeypatch):
    monkeypatch.setattr(llm_mod, "_full_jitter_backoff", lambda attempt, base, cap: 0.0)
    chat = FakeChat(['{"entities": [], "relations": []}'])
    chat.errors = [
        ValueError("429 rate limit"),
        ValueError("429 rate limit"),
        ValueError("429 rate limit"),
        ValueError("429 rate limit"),
    ]
    extractor = make_extractor(chat)
    with pytest.raises(ValueError, match="429"):
        await extractor._call_llm_with_retry(
            chat,
            "prompt",
            chunk_id="chunk-1",
            model_spec="fake-model",
        )
    # 1 initial attempt + 3 retries, then re-raise
    assert len(chat.calls) == 4


async def test_call_llm_with_retry_succeeds_after_empty_message_network_error(monkeypatch):
    """空消息的网络层异常（httpx.ReadError 的 str() 为空）须按类型名判定为瞬时错误重试。"""
    monkeypatch.setattr(llm_mod, "_full_jitter_backoff", lambda attempt, base, cap: 0.0)
    chat = FakeChat(['{"entities": [], "relations": []}'])
    chat.errors = [ChatModelError("Error calling model: ReadError: , URL: https://x, Model: m")]
    extractor = make_extractor(chat)
    content, _ = await extractor._call_llm_with_retry(
        chat,
        "prompt",
        chunk_id="chunk-1",
        model_spec="fake-model",
    )
    assert content == '{"entities": [], "relations": []}'
    assert len(chat.calls) == 2


async def test_call_llm_with_retry_succeeds_after_bare_timeout_error(monkeypatch):
    """str() 为空的 TimeoutError 须按类型判定为瞬时错误重试。"""
    monkeypatch.setattr(llm_mod, "_full_jitter_backoff", lambda attempt, base, cap: 0.0)
    chat = FakeChat(['{"entities": [], "relations": []}'])
    chat.errors = [TimeoutError()]
    extractor = make_extractor(chat)
    content, _ = await extractor._call_llm_with_retry(
        chat,
        "prompt",
        chunk_id="chunk-1",
        model_spec="fake-model",
    )
    assert content == '{"entities": [], "relations": []}'
    assert len(chat.calls) == 2