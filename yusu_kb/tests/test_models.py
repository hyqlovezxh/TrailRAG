"""Tests for the self-contained model layer (embed / chat / rerank).

All HTTP traffic goes through ``httpx.MockTransport`` — no external service
is contacted.
"""

from __future__ import annotations

import json

import httpx
import pytest

from yusu_kb.models.chat import GeneralResponse, OpenAIChatAdapter
from yusu_kb.models.embed import OtherEmbedding
from yusu_kb.models.rerank import DashscopeReranker, OpenAIReranker


def _embeddings_response(dim: int = 4, count: int = 1) -> dict:
    vectors = [[round(i * 0.1, 3) for i in range(dim)] for _ in range(count)]
    return {"data": [{"embedding": vectors[i], "index": i} for i in range(count)]}


class TestOtherEmbedding:
    @pytest.fixture()
    async def model(self):
        instance = OtherEmbedding(
            model="embed-test",
            base_url="https://example.com/v1/embeddings",
            api_key="key",
            dimension=4,
        )
        yield instance
        await instance.close_async_client()

    async def test_aencode_payload_and_extract(self, model):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=_embeddings_response(dim=4, count=2))

        model._transport = httpx.MockTransport(handler)
        result = await model.aencode(["a", "b"])
        assert seen["body"] == {"model": "embed-test", "input": ["a", "b"]}
        assert len(result) == 2
        assert all(len(vec) == 4 for vec in result)

    async def test_aencode_retries_on_429_then_succeeds(self, model):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, json=_embeddings_response(dim=4))

        model._transport = httpx.MockTransport(handler)
        result = await model.aencode(["a"])
        assert result == [[0.0, 0.1, 0.2, 0.3]]
        assert len(calls) == 2

    async def test_aencode_retries_transient_503_then_succeeds(self, model):
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            if len(calls) == 1:
                return httpx.Response(503)
            return httpx.Response(200, json=_embeddings_response(dim=4))

        model._transport = httpx.MockTransport(handler)
        result = await model.aencode(["a"])
        assert len(result) == 1
        assert len(calls) == 2

    async def test_aencode_raises_on_400(self, model):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "bad"})

        model._transport = httpx.MockTransport(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await model.aencode(["a"])

    async def test_probe_dimension(self, model):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_embeddings_response(dim=64))

        model._transport = httpx.MockTransport(handler)
        assert await model.probe_dimension() == 64

    async def test_test_connection_dim_mismatch(self, model):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_embeddings_response(dim=64))

        model._transport = httpx.MockTransport(handler)
        ok, message = await model.test_connection()
        assert ok is False
        assert "维度不一致" in message

    async def test_batch_encode_chunks_large_inputs(self, model):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            return httpx.Response(200, json=_embeddings_response(dim=4, count=len(body["input"])))

        model._transport = httpx.MockTransport(handler)
        result = await model.abatch_encode([f"doc{i}" for i in range(5)], batch_size=2)
        assert len(result) == 5
        assert len(model.embed_state) <= 1  # completed tasks cleaned up


class TestOpenAIChatAdapter:
    @pytest.fixture()
    def adapter(self):
        return OpenAIChatAdapter(
            "chat-test",
            model_name="chat-test",
            base_url="https://example.com/v1/chat/completions",
            api_key="key",
        )

    async def test_call_non_stream(self, adapter):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["stream"] is False
            assert body["model"] == "chat-test"
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "你好", "reasoning_content": None}}]},
            )

        adapter._transport = httpx.MockTransport(handler)
        response = await adapter.call([{"role": "user", "content": "hi"}], stream=False)
        assert isinstance(response, GeneralResponse)
        assert response.content == "你好"

    async def test_call_reasoning_content(self, adapter):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "answer",
                                "reasoning_content": "thinking...",
                            }
                        }
                    ]
                },
            )

        adapter._transport = httpx.MockTransport(handler)
        response = await adapter.call("hi", stream=False)
        assert response.content == "answer"
        assert response.reasoning_content == "thinking..."

    async def test_call_stream(self, adapter):
        chunks = [
            {"choices": [{"delta": {"content": "深"}}]},
            {"choices": [{"delta": {"content": "度"}}]},
        ]
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

        adapter._transport = httpx.MockTransport(handler)
        stream = await adapter.call("hi", stream=True)
        collected = [part async for part in stream]
        assert [part.content for part in collected] == ["深", "度"]

    async def test_call_stream_sends_stream_flag(self, adapter):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["stream"] is True
            return httpx.Response(
                200,
                text="data: {\"choices\": [{\"delta\": {\"content\": \"a\"}}]}\n\n"
                "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )

        adapter._transport = httpx.MockTransport(handler)
        stream = await adapter.call("hi", stream=True)
        collected = [part async for part in stream]
        assert [part.content for part in collected] == ["a"]

    async def test_call_stream_skips_empty_choices_frames(self, adapter):
        # 部分厂商会在流末尾发送仅含 usage 的空 choices 帧，须跳过而非报错
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text="data: {\"choices\": [{\"delta\": {\"content\": \"a\"}}]}\n\n"
                "data: {\"choices\": [], \"usage\": {\"total_tokens\": 1}}\n\n"
                "data: {\"choices\": [{\"delta\": {\"content\": \"b\"}}]}\n\n"
                "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )

        adapter._transport = httpx.MockTransport(handler)
        stream = await adapter.call("hi", stream=True)
        collected = [part async for part in stream]
        assert [part.content for part in collected] == ["a", "b"]

    async def test_call_stream_reasoning_field_alias(self, adapter):
        # 兼容厂商自研字段名 reasoning（如商汤），与 OpenAI 惯例 reasoning_content 二选一
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text="data: {\"choices\": [{\"delta\": {\"reasoning\": \"想\"}}]}\n\n"
                "data: {\"choices\": [{\"delta\": {\"content\": \"答\"}}]}\n\n"
                "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )

        adapter._transport = httpx.MockTransport(handler)
        stream = await adapter.call("hi", stream=True)
        collected = [part async for part in stream]
        assert [part.reasoning_content for part in collected] == ["想", None]
        assert [part.content for part in collected] == ["", "答"]

    async def test_call_non_stream_reasoning_field_alias(self, adapter):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "answer", "reasoning": "thinking..."}}]},
            )

        adapter._transport = httpx.MockTransport(handler)
        response = await adapter.call("hi", stream=False)
        assert response.content == "answer"
        assert response.reasoning_content == "thinking..."

    async def test_call_collect(self, adapter):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                text="data: {\"choices\": [{\"delta\": {\"content\": \"abc\"}}]}\n\n"
                "data: {\"choices\": [{\"delta\": {\"content\": \"def\"}}]}\n\n"
                "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )

        adapter._transport = httpx.MockTransport(handler)
        response = await adapter.call_collect("hi")
        assert response.content == "abcdef"

    async def test_call_error_mentions_url_and_model(self, adapter):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        adapter._transport = httpx.MockTransport(handler)
        with pytest.raises(Exception) as exc_info:
            await adapter.call("hi", stream=False)
        assert "chat-test" in str(exc_info.value)
        assert "chat/completions" in str(exc_info.value)

    async def test_call_forwards_tools_payload(self, adapter):
        # tools 经 model_kwargs 透传到请求体，模型返回 tool_calls 后原样携带
        tools = [{"type": "function", "function": {"name": "query_kb", "parameters": {"type": "object"}}}]
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "query_kb", "arguments": '{"query": "张三转账"}'},
            }
        ]
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": None, "tool_calls": tool_calls}}]},
            )

        adapter._transport = httpx.MockTransport(handler)
        response = await adapter.call([{"role": "user", "content": "查一下"}], stream=False, tools=tools)
        assert seen["body"]["tools"] == tools
        assert response.content == ""
        assert response.tool_calls == tool_calls

    async def test_call_non_stream_no_tool_calls_defaults_none(self, adapter):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"choices": [{"message": {"content": "普通回答"}}]})

        adapter._transport = httpx.MockTransport(handler)
        response = await adapter.call("hi", stream=False)
        assert response.tool_calls is None

    def test_general_response_tool_calls_default(self):
        assert GeneralResponse("x").tool_calls is None
        calls = [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
        assert GeneralResponse("x", tool_calls=calls).tool_calls == calls


class TestReranker:
    def test_openai_payload_shape(self):
        reranker = OpenAIReranker(model_name="rerank-test", api_key="key", base_url="https://example.com/rerank")
        payload = reranker._build_payload("q", ["d1", "d2"], max_length=128)
        assert payload == {
            "model": "rerank-test",
            "query": "q",
            "documents": ["d1", "d2"],
            "max_chunks_per_doc": 128,
        }

    async def test_acompute_score_order_and_values(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            docs = body["documents"]
            scores = {"d1": 0.9, "d2": 0.7}
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": i, "relevance_score": scores.get(doc, 0.0)}
                        for i, doc in enumerate(docs)
                    ]
                },
            )

        reranker = OpenAIReranker(
            model_name="rerank-test",
            api_key="key",
            base_url="https://example.com/rerank",
            transport=httpx.MockTransport(handler),
        )
        scores = await reranker.acompute_score([["q"], ["d1", "d2"]], batch_size=1)
        assert scores == [0.9, 0.7]
        await reranker.aclose()

    async def test_failed_batch_uses_minus_one_sentinel(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        reranker = OpenAIReranker(
            model_name="rerank-test",
            api_key="key",
            base_url="https://example.com/rerank",
            transport=httpx.MockTransport(handler),
        )
        scores = await reranker.acompute_score([["q"], ["d1", "d2"]], batch_size=1)
        assert scores == [-1.0, -1.0]
        await reranker.aclose()

    async def test_normalize_only_when_outside_unit_range(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 0, "relevance_score": 2.0},
                        {"index": 1, "relevance_score": 1.0},
                    ]
                },
            )

        reranker = OpenAIReranker(
            model_name="rerank-test",
            api_key="key",
            base_url="https://example.com/rerank",
            transport=httpx.MockTransport(handler),
        )
        scores = await reranker.acompute_score([["q"], ["d1", "d2"]], batch_size=1)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)
        await reranker.aclose()

    def test_dashscope_payload_and_extract(self):
        reranker = DashscopeReranker(
            model_name="rerank-test",
            api_key="key",
            base_url="https://example.com/rerank",
            parameters={"instruct": "x"},
        )
        payload = reranker._build_payload("q", ["d1"], max_length=128)
        assert payload["model"] == "rerank-test"
        assert payload["input"] == {"query": "q", "documents": ["d1"]}
        assert payload["parameters"]["top_n"] == 1
        assert payload["parameters"]["instruct"] == "x"
        results = reranker._extract_results({"output": {"results": [{"index": 0}]}})
        assert results == [{"index": 0}]


def test_create_embedding_model_appends_embeddings_suffix(monkeypatch):
    monkeypatch.setenv("YUSU_EMBED_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("YUSU_EMBED_API_KEY", "k")
    monkeypatch.setenv("YUSU_EMBED_MODEL", "m")
    monkeypatch.setenv("YUSU_EMBED_DIM", "8")
    from yusu_kb.models.embed import create_embedding_model

    model = create_embedding_model()
    assert model.base_url == "https://example.com/v1/embeddings"
    assert model.dimension == 8
    assert model.model == "m"


def test_create_embedding_model_requires_model(monkeypatch):
    monkeypatch.setenv("YUSU_EMBED_BASE_URL", "https://example.com/v1")
    monkeypatch.delenv("YUSU_EMBED_MODEL", raising=False)
    from yusu_kb.models.embed import create_embedding_model

    with pytest.raises(ValueError):
        create_embedding_model()