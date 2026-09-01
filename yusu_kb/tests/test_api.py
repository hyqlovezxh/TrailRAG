"""API tests: FastAPI app over a fake embedding + fake chat model.

End-to-end HTTP via httpx ASGITransport; no external services are contacted.
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx
import numpy as np
import pytest

from yusu_kb.api import deps
from yusu_kb.api.app import create_app
from yusu_kb.knowledge.eval.service import EvaluationService
from yusu_kb.knowledge.implementations.local_kb import set_default_graph_chat_model_fn
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.models.chat import ChatModelError, GeneralResponse
from yusu_kb.models.embed import EmbeddingFunc, wrap_embedding_func_with_attrs
from yusu_kb.repositories import configure_repositories
from yusu_kb.storage.sqlite.engine import create_engine, dispose, init_db

FAKE_DIM = 64


async def _fake_embed(texts: list[str], **kwargs) -> np.ndarray:
    del kwargs
    vecs = []
    for text in texts:
        vec = np.zeros(FAKE_DIM)
        for ch in text:
            vec[ord(ch) % FAKE_DIM] += 1.0
        norm = np.linalg.norm(vec)
        vecs.append(vec / norm if norm > 0 else vec)
    return np.array(vecs)


fake_embedding_func: EmbeddingFunc = wrap_embedding_func_with_attrs(embedding_dim=FAKE_DIM)(_fake_embed)


class FakeChat:
    """Stands in for OpenAIChatAdapter without network access."""

    model = "fake-chat"

    def __init__(self, chunks: tuple[str, ...] = ("你好，", "这是测试回答。")):
        self.chunks = chunks
        self.last_messages: list[dict] | None = None

    async def call(self, message, stream=False, **kwargs):
        del kwargs
        self.last_messages = message if isinstance(message, list) else [{"role": "user", "content": str(message)}]
        if stream:
            async def gen():
                for chunk in self.chunks:
                    yield GeneralResponse(chunk)

            return gen()
        return GeneralResponse("".join(self.chunks))

    async def call_collect(self, messages, **kwargs):
        """Streaming collector contract: takes a messages list, returns one response."""
        del kwargs
        return await self.call(messages, stream=False)


GRAPH_PAYLOAD = {
    "entities": [
        {"text": "甲公司", "label": "机构", "description": "甲公司，主营支付系统开发。"},
        {"text": "王五", "label": "人物", "description": "王五，甲公司员工。"},
    ],
    "relations": [
        {"source": "王五", "target": "甲公司", "text": "任职于", "label": "任职于", "description": "王五任职于甲公司。"}
    ],
}

GRAPH_PAYLOAD_B = {
    "entities": [
        {"text": "乙公司", "label": "机构", "description": "乙公司，主营风控系统开发。"},
        {"text": "王五", "label": "人物", "description": "王五，乙公司顾问。"},
    ],
    "relations": [
        {"source": "王五", "target": "乙公司", "text": "任职于", "label": "任职于", "description": "王五任职于乙公司。"}
    ],
}


class GraphFakeChat(FakeChat):
    """FakeChat serving extraction JSON selected by the chunk's marker text."""

    async def call(self, message, stream=False, **kwargs):
        del kwargs
        self.last_messages = message if isinstance(message, list) else [{"role": "user", "content": str(message)}]
        content = self.last_messages[-1]["content"] if self.last_messages else ""
        payload = GRAPH_PAYLOAD_B if "乙公司" in content else GRAPH_PAYLOAD
        text = json.dumps(payload, ensure_ascii=False)
        if stream:
            async def gen():
                yield GeneralResponse(text)

            return gen()
        return GeneralResponse(text)


class EvalFakeChat(FakeChat):
    """FakeChat dispatching benchmark / answer / judge prompts for evaluation flows.

    - prompt contains 问题生成要求 (benchmark generation) -> JSON item with a
      counter-unique query (dedup-safe) and gold ids taken from 片段ID= markers
    - prompt contains 请判断AI生成的答案 (judge) -> JSON score 1.0
    - otherwise (answer generation) -> a fixed plain-text answer
    """

    def __init__(self):
        super().__init__(chunks=("这是评估生成的回答。",))
        self.benchmark_calls = 0

    async def call(self, message, stream=False, **kwargs):
        del kwargs
        self.last_messages = message if isinstance(message, list) else [{"role": "user", "content": str(message)}]
        content = self.last_messages[-1]["content"] if self.last_messages else ""
        if "问题生成要求" in content:
            self.benchmark_calls += 1
            chunk_ids = re.findall(r"片段ID=(\S+)", content)
            payload = {
                "query": f"2024年3月第{self.benchmark_calls}笔记录中张三向李四转账的金额是多少？",
                "gold_answer": "五十万元",
                "gold_chunk_ids": chunk_ids[:1],
            }
            text = json.dumps(payload, ensure_ascii=False)
        elif "请判断AI生成的答案" in content:
            text = json.dumps({"score": 1.0, "reasoning": "标准答案与生成答案一致"}, ensure_ascii=False)
        else:
            text = "这是评估生成的回答。"
        if stream:
            async def gen():
                yield GeneralResponse(text)

            return gen()
        return GeneralResponse(text)


class ToolFakeChat(FakeChat):
    """FakeChat that requests one query_kb tool call, then answers normally."""

    def __init__(self, chunks: tuple[str, ...] = ("工具回答。",)):
        super().__init__(chunks=chunks)
        self.non_stream_calls = 0
        self.kb_id = ""

    async def call(self, message, stream=False, **kwargs):
        del kwargs
        self.last_messages = message if isinstance(message, list) else [{"role": "user", "content": str(message)}]
        if stream:
            async def gen():
                for chunk in self.chunks:
                    yield GeneralResponse(chunk)

            return gen()
        self.non_stream_calls += 1
        if self.non_stream_calls == 1:
            return GeneralResponse(
                "",
                tool_calls=[
                    {
                        "id": "call_query_1",
                        "type": "function",
                        "function": {
                            "name": "query_kb",
                            "arguments": json.dumps(
                                {"kb_id": self.kb_id, "query_text": "苹果"}, ensure_ascii=False
                            ),
                        },
                    }
                ],
            )
        return GeneralResponse("基于工具结果的最终回答。")


class LoopToolFakeChat(FakeChat):
    """FakeChat that always requests list_kbs (no kb_id needed) to exhaust the loop."""

    def __init__(self, chunks: tuple[str, ...] = ("循环上限回答。",)):
        super().__init__(chunks=chunks)
        self.non_stream_calls = 0

    async def call(self, message, stream=False, **kwargs):
        del kwargs
        self.last_messages = message if isinstance(message, list) else [{"role": "user", "content": str(message)}]
        if stream:
            async def gen():
                for chunk in self.chunks:
                    yield GeneralResponse(chunk)

            return gen()
        self.non_stream_calls += 1
        return GeneralResponse(
            "",
            tool_calls=[
                {
                    "id": f"call_loop_{self.non_stream_calls}",
                    "type": "function",
                    "function": {"name": "list_kbs", "arguments": "{}"},
                }
            ],
        )


class ToolsRejectingChat(FakeChat):
    """FakeChat rejecting non-stream calls that carry a tools payload (400)."""

    def __init__(self, chunks: tuple[str, ...] = ("降级回答。",)):
        super().__init__(chunks=chunks)
        self.rejected_tools = False

    async def call(self, message, stream=False, **kwargs):
        self.last_messages = message if isinstance(message, list) else [{"role": "user", "content": str(message)}]
        if not stream and kwargs.get("tools") is not None:
            self.rejected_tools = True
            raise ChatModelError(
                "Error calling model: Client error '400 Bad Request': Unsupported parameter: tools"
            )
        if stream:
            async def gen():
                for chunk in self.chunks:
                    yield GeneralResponse(chunk)

            return gen()
        return GeneralResponse("".join(self.chunks))


@pytest.fixture()
def fake_chat() -> FakeChat:
    return FakeChat()


@pytest.fixture()
async def client(tmp_path, monkeypatch, fake_chat):
    monkeypatch.setenv("YUSU_DATA_DIR", str(tmp_path))
    configure_repositories(None)
    engine = create_engine(db_path=tmp_path / "test.db")
    await init_db(engine)
    configure_repositories(engine)

    manager = KnowledgeBaseManager(str(tmp_path))
    await manager.load_all_metadata()

    app = create_app()
    app.dependency_overrides[deps.get_manager] = lambda: manager
    app.dependency_overrides[deps.get_chat_model] = lambda: fake_chat
    # 复用同一 EvaluationService 实例：后台生成/评估任务挂在实例的 _tasks 上，
    # 换新实例会导致 list 轮询时把 running 任务误收敛为 failed。
    eval_chat = EvalFakeChat()
    eval_service = EvaluationService(select_model_fn=lambda model_spec: eval_chat, kb_manager=manager)
    app.dependency_overrides[deps.get_evaluation_service] = lambda: eval_service
    set_default_graph_chat_model_fn(fake_chat.call_collect)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client, manager

    set_default_graph_chat_model_fn(None)
    await manager.close()
    await dispose(engine)
    configure_repositories(None)
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def default_fake_embedding():
    from yusu_kb.knowledge.implementations.local_kb import set_default_embedding_func

    set_default_embedding_func(fake_embedding_func)
    yield
    set_default_embedding_func(None)


async def _create_kb(client) -> str:
    response = await client.post("/api/knowledge/databases", json={"name": "测试库", "description": "e2e"})
    assert response.status_code == 200
    return response.json()["kb_id"]


async def _upload_and_index(client, kb_id: str, name: str = "note.md", content: str = "") -> str:
    content = content or "# 笔记\n\n苹果是红色的水果。香蕉是黄色的水果。"
    response = await client.post(
        f"/api/knowledge/databases/{kb_id}/files",
        files={"file": (name, content.encode("utf-8"), "text/markdown")},
    )
    assert response.status_code == 200
    file_id = response.json()["file_id"]
    parse_resp = await client.post(f"/api/knowledge/databases/{kb_id}/files/{file_id}/parse")
    assert parse_resp.status_code == 200
    assert parse_resp.json()["status"] == "parsed"
    index_resp = await client.post(f"/api/knowledge/databases/{kb_id}/files/{file_id}/index")
    assert index_resp.status_code == 200
    assert index_resp.json()["status"] == "indexed"
    return file_id


class TestHealth:
    async def test_health_ok(self, client):
        http_client, _ = client
        response = await http_client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestKnowledgeFlow:
    async def test_full_flow(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)

        listing = await http_client.get("/api/knowledge/databases")
        assert kb_id in [db["kb_id"] for db in listing.json()["databases"]]

        info = await http_client.get(f"/api/knowledge/databases/{kb_id}")
        assert info.status_code == 200
        assert info.json()["name"] == "测试库"

        file_id = await _upload_and_index(http_client, kb_id)

        files = await http_client.get(f"/api/knowledge/databases/{kb_id}/files")
        assert files.status_code == 200
        assert [f["file_id"] for f in files.json()["files"]] == [file_id]

        detail = await http_client.get(f"/api/knowledge/databases/{kb_id}/files/{file_id}")
        assert detail.status_code == 200

        query_resp = await http_client.post(
            "/api/knowledge/query",
            json={"kb_id": kb_id, "query": "苹果是什么颜色", "params": {"search_mode": "vector"}},
        )
        assert query_resp.status_code == 200
        results = query_resp.json()["results"]
        assert results, "向量检索应返回命中"
        assert "苹果" in results[0]["content"]

        await http_client.delete(f"/api/knowledge/databases/{kb_id}/files/{file_id}")
        files = await http_client.get(f"/api/knowledge/databases/{kb_id}/files")
        assert files.json()["files"] == []

        await http_client.delete(f"/api/knowledge/databases/{kb_id}")
        missing = await http_client.get(f"/api/knowledge/databases/{kb_id}")
        assert missing.status_code == 404

    async def test_query_params_roundtrip(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)

        response = await http_client.get(f"/api/knowledge/databases/{kb_id}/query-params")
        assert response.status_code == 200
        effective = response.json()["effective"]["options"]
        assert "final_top_k" in effective

        update = await http_client.put(
            f"/api/knowledge/databases/{kb_id}/query-params",
            json={"params": {"final_top_k": 5, "unknown_param": 1}},
        )
        assert update.status_code == 200
        assert update.json()["options"]["final_top_k"] == 5
        assert "unknown_param" not in update.json()["options"]

    async def test_404_on_missing_kb(self, client):
        http_client, _ = client
        missing = await http_client.get("/api/knowledge/databases/no-such-kb")
        assert missing.status_code == 404
        query = await http_client.post("/api/knowledge/query", json={"query": "x", "kb_id": "no-such-kb"})
        assert query.status_code == 404

    async def test_upload_requires_existing_kb(self, client):
        http_client, _ = client
        response = await http_client.post(
            "/api/knowledge/databases/no-such-kb/files",
            files={"file": ("a.md", b"x", "text/markdown")},
        )
        assert response.status_code == 404


class TestChatStream:
    async def _stream_events(self, http_client, payload: dict) -> list[dict]:
        response = await http_client.post("/api/chat/stream", json=payload)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        return [
            json.loads(line[len("data:") :].strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]

    async def test_chat_stream_sse(self, client, fake_chat):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(http_client, kb_id)

        events = await self._stream_events(
            http_client,
            {
                "kb_id": kb_id,
                "messages": [{"role": "user", "content": "苹果是什么颜色"}],
                "query_params": {"search_mode": "vector"},
            },
        )
        types = [event["type"] for event in events]
        assert "sources" in types
        sources = next(event for event in events if event["type"] == "sources")
        assert sources["chunks"], "应携带检索到的资料"
        assert "苹果" in sources["chunks"][0]["content"]
        # sources 事件携带 citation_source（kb:// 格式，前端据此去重编号）
        assert sources["chunks"][0]["citation_source"].startswith("kb://")

        # 默认 tools_enabled=True：模型未返回 tool_calls 时直接给出完整回答（单 delta）
        deltas = [event["content"] for event in events if event["type"] == "delta"]
        assert deltas == ["你好，这是测试回答。"]
        assert "tool" not in types
        assert events[-1]["type"] == "done"

        # LLM 提示应包含检索上下文（system 消息）+ 引用要求 + 当前日期
        assert any(msg["role"] == "system" and "参考资料" in msg["content"] for msg in fake_chat.last_messages or [])
        system_content = next(
            msg["content"] for msg in fake_chat.last_messages or [] if msg["role"] == "system"
        )
        assert "<cite>" in system_content
        assert "当前日期" in system_content

    async def test_chat_stream_custom_system_prompt(self, client, fake_chat):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(http_client, kb_id)

        await self._stream_events(
            http_client,
            {
                "kb_id": kb_id,
                "messages": [{"role": "user", "content": "苹果是什么颜色"}],
                "system_prompt": "请用粤语回答。",
            },
        )
        system_content = next(
            msg["content"] for msg in fake_chat.last_messages or [] if msg["role"] == "system"
        )
        assert "请用粤语回答。" in system_content

    async def test_chat_stream_tools_disabled(self, client, fake_chat):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(http_client, kb_id)

        events = await self._stream_events(
            http_client,
            {
                "kb_id": kb_id,
                "messages": [{"role": "user", "content": "苹果是什么颜色"}],
                "tools_enabled": False,
            },
        )
        types = [event["type"] for event in events]
        # 关闭工具后走旧路径：流式分块输出，无 tool 事件
        deltas = [event["content"] for event in events if event["type"] == "delta"]
        assert deltas == list(fake_chat.chunks)
        assert "tool" not in types
        assert events[-1]["type"] == "done"

    async def test_chat_stream_tool_call_flow(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(http_client, kb_id)

        # 局部 override 换入 ToolFakeChat，不影响其他测试的 fake_chat
        from yusu_kb.api import deps as api_deps
        from yusu_kb.api.app import create_app as _create_app

        app = _create_app()
        manager = client[1]
        app.dependency_overrides[api_deps.get_manager] = lambda: manager
        tool_chat = ToolFakeChat()
        tool_chat.kb_id = kb_id
        app.dependency_overrides[api_deps.get_chat_model] = lambda: tool_chat
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            events = await self._stream_events(
                http,
                {
                    "kb_id": kb_id,
                    "messages": [{"role": "user", "content": "苹果是什么颜色"}],
                },
            )
        types = [event["type"] for event in events]
        assert types == ["sources", "tool", "delta", "done"]
        tool_event = next(event for event in events if event["type"] == "tool")
        assert tool_event["name"] == "query_kb"
        assert tool_event["args"]["kb_id"] == kb_id
        assert tool_event["args"]["query_text"] == "苹果"
        assert isinstance(tool_event["summary"], str) and tool_event["summary"]
        deltas = [event["content"] for event in events if event["type"] == "delta"]
        assert deltas == ["基于工具结果的最终回答。"]
        # 工具结果已作为 tool 消息追加到后续调用（消息含 citation_source 结果）
        assert tool_chat.non_stream_calls == 2
        assert any(msg["role"] == "tool" for msg in tool_chat.last_messages or [])

    async def test_chat_stream_tool_loop_cap(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(http_client, kb_id)

        from yusu_kb.api import deps as api_deps
        from yusu_kb.api.app import create_app as _create_app

        app = _create_app()
        manager = client[1]
        app.dependency_overrides[api_deps.get_manager] = lambda: manager
        loop_chat = LoopToolFakeChat()
        app.dependency_overrides[api_deps.get_chat_model] = lambda: loop_chat
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            events = await self._stream_events(
                http,
                {
                    "kb_id": kb_id,
                    "messages": [{"role": "user", "content": "苹果是什么颜色"}],
                },
            )
        tool_events = [event for event in events if event["type"] == "tool"]
        assert len(tool_events) == 3, "工具循环应在上限 3 轮后停止"
        assert all(event["name"] == "list_kbs" for event in tool_events)
        assert loop_chat.non_stream_calls == 3
        # 轮数用尽后基于工具结果流式生成最终回答
        deltas = [event["content"] for event in events if event["type"] == "delta"]
        assert deltas == list(loop_chat.chunks)
        assert events[-1]["type"] == "done"

    async def test_chat_stream_tools_rejected_falls_back(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(http_client, kb_id)

        from yusu_kb.api import deps as api_deps
        from yusu_kb.api.app import create_app as _create_app

        app = _create_app()
        manager = client[1]
        app.dependency_overrides[api_deps.get_manager] = lambda: manager
        rejecting_chat = ToolsRejectingChat()
        app.dependency_overrides[api_deps.get_chat_model] = lambda: rejecting_chat
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            events = await self._stream_events(
                http,
                {
                    "kb_id": kb_id,
                    "messages": [{"role": "user", "content": "苹果是什么颜色"}],
                },
            )
        assert rejecting_chat.rejected_tools, "探测调用应携带 tools 参数"
        types = [event["type"] for event in events]
        assert "tool" not in types, "降级路径不应执行工具"
        deltas = [event["content"] for event in events if event["type"] == "delta"]
        assert deltas == list(rejecting_chat.chunks)
        assert events[-1]["type"] == "done"

    async def test_chat_stream_missing_kb_returns_sse_error(self, client):
        http_client, _ = client
        events = await self._stream_events(
            http_client,
            {"kb_id": "no-such-kb", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert events[0]["type"] == "error"


class TestSystemPrompt:
    async def test_system_prompt_roundtrip(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)

        initial = await http_client.get(f"/api/knowledge/databases/{kb_id}/system-prompt")
        assert initial.status_code == 200
        assert initial.json()["system_prompt"] == ""

        put = await http_client.put(
            f"/api/knowledge/databases/{kb_id}/system-prompt",
            json={"system_prompt": "你是资深笔录分析师。"},
        )
        assert put.status_code == 200
        assert put.json()["system_prompt"] == "你是资深笔录分析师。"

        fetched = await http_client.get(f"/api/knowledge/databases/{kb_id}/system-prompt")
        assert fetched.json()["system_prompt"] == "你是资深笔录分析师。"

        # 清空也是合法操作
        cleared = await http_client.put(
            f"/api/knowledge/databases/{kb_id}/system-prompt", json={"system_prompt": ""}
        )
        assert cleared.status_code == 200
        assert cleared.json()["system_prompt"] == ""

    async def test_system_prompt_too_long_rejected(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)

        response = await http_client.put(
            f"/api/knowledge/databases/{kb_id}/system-prompt",
            json={"system_prompt": "长" * 8001},
        )
        assert response.status_code == 400

    async def test_system_prompt_missing_kb_404(self, client):
        http_client, _ = client
        get_resp = await http_client.get("/api/knowledge/databases/no-such-kb/system-prompt")
        assert get_resp.status_code == 404
        put_resp = await http_client.put(
            "/api/knowledge/databases/no-such-kb/system-prompt", json={"system_prompt": "x"}
        )
        assert put_resp.status_code == 404


class TestAuth:
    async def test_api_key_enforced(self, client, monkeypatch):
        http_client, _ = client
        monkeypatch.setenv("YUSU_API_KEY", "secret-token")

        # health 是存活探针，不鉴权；知识库端点必须鉴权
        probe = await http_client.get("/api/health")
        assert probe.status_code == 200

        no_key = await http_client.get("/api/knowledge/databases")
        assert no_key.status_code == 401

        bad_key = await http_client.get("/api/knowledge/databases", headers={"Authorization": "Bearer wrong"})
        assert bad_key.status_code == 401

        ok = await http_client.get("/api/knowledge/databases", headers={"Authorization": "Bearer secret-token"})
        assert ok.status_code == 200

        ok_alt = await http_client.get("/api/knowledge/databases", headers={"X-API-Key": "secret-token"})
        assert ok_alt.status_code == 200


class TestGraphApi:
    """Graph build orchestration over the API (configure → build → reset)."""

    @pytest.fixture()
    def fake_chat(self):
        return GraphFakeChat()

    @staticmethod
    async def _wait_build_completed(http_client, kb_id: str, timeout: float = 15.0) -> dict:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            response = await http_client.get(f"/api/knowledge/databases/{kb_id}/graph/status")
            assert response.status_code == 200
            status = response.json()
            if status["build_task_status"] in ("completed", "failed", "cancelled"):
                return status
            if loop.time() > deadline:
                raise AssertionError(f"graph build did not finish within {timeout}s: {status}")
            await asyncio.sleep(0.02)

    async def test_configure_build_status_reset_flow(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(
            http_client, kb_id, content="# 笔记\n\n甲公司开发支付系统，王五负责研发。"
        )

        config = await http_client.get(f"/api/knowledge/databases/{kb_id}/graph/config")
        assert config.status_code == 200
        assert config.json()["config"] is None

        status = await http_client.get(f"/api/knowledge/databases/{kb_id}/graph/status")
        assert status.status_code == 200
        body = status.json()
        assert body["configured"] is False
        assert body["total_chunks"] > 0
        assert body["entity_count"] == 0
        assert body["indexed_chunks"] == 0

        invalid = await http_client.put(
            f"/api/knowledge/databases/{kb_id}/graph/config",
            json={"extractor_type": "llm", "extractor_options": {}},
        )
        assert invalid.status_code == 400

        configured = await http_client.put(
            f"/api/knowledge/databases/{kb_id}/graph/config",
            json={"extractor_type": "llm", "extractor_options": {"model_spec": "fake-model"}},
        )
        assert configured.status_code == 200
        assert configured.json()["locked"] is True

        built = await http_client.post(
            f"/api/knowledge/databases/{kb_id}/graph/build", json={"batch_size": 4}
        )
        assert built.status_code == 200
        assert built.json()["status"]["build_task_status"] == "running"

        final = await self._wait_build_completed(http_client, kb_id)
        assert final["build_task_status"] == "completed"
        assert final["entity_count"] == 2
        assert final["relation_count"] == 1
        assert final["indexed_chunks"] == final["total_chunks"]
        assert final["pending_chunks"] == 0

        stats = await http_client.get(f"/api/knowledge/databases/{kb_id}/graph/stats")
        assert stats.status_code == 200
        storage = stats.json()["storage"]
        assert storage["entities"] == 2
        assert storage["relations"] == 1
        assert storage["chunks"] == final["total_chunks"]

        labels = await http_client.get(f"/api/knowledge/databases/{kb_id}/graph/labels")
        assert labels.status_code == 200
        label_map = {item["label"]: item["count"] for item in labels.json()["labels"]}
        assert label_map["人物"] == 1
        assert label_map["机构"] == 1

        subgraph = await http_client.get(
            f"/api/knowledge/databases/{kb_id}/graph/subgraph",
            params={"entity_ids": "王五", "max_depth": 3},
        )
        assert subgraph.status_code == 200
        sg = subgraph.json()
        node_names = {node["name"] for node in sg["nodes"]}
        assert "王五" in node_names
        assert "甲公司" in node_names
        assert sg["edges"], "子图应包含关系边"

        reset = await http_client.post(
            f"/api/knowledge/databases/{kb_id}/graph/reset",
            json={"clear_extraction_result": True, "clear_config": True},
        )
        assert reset.status_code == 200
        assert reset.json()["reset_chunks"] == final["total_chunks"]

        status = await http_client.get(f"/api/knowledge/databases/{kb_id}/graph/status")
        body = status.json()
        assert body["configured"] is False
        assert body["entity_count"] == 0
        assert body["relation_count"] == 0
        assert body["indexed_chunks"] == 0
        assert body["pending_chunks"] == final["total_chunks"]

    async def test_auto_build_after_index(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        configured = await http_client.put(
            f"/api/knowledge/databases/{kb_id}/graph/config",
            json={"extractor_type": "llm", "extractor_options": {"model_spec": "fake-model"}},
        )
        assert configured.status_code == 200

        await _upload_and_index(http_client, kb_id, content="# 笔记\n\n甲公司开发支付系统，王五负责研发。")

        final = await self._wait_build_completed(http_client, kb_id)
        assert final["build_task_status"] == "completed"
        assert final["entity_count"] == 2
        assert final["indexed_chunks"] == final["total_chunks"]

        rebuilt = await http_client.post(f"/api/knowledge/databases/{kb_id}/graph/build", json={})
        assert rebuilt.status_code == 200
        assert rebuilt.json()["status"]["build_task_status"] in ("running", "completed")

    async def test_update_content_retriggers_auto_build(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        file_id = await _upload_and_index(http_client, kb_id, content="# 笔记\n\n甲公司开发支付系统，王五负责研发。")
        configured = await http_client.put(
            f"/api/knowledge/databases/{kb_id}/graph/config",
            json={"extractor_type": "llm", "extractor_options": {"model_spec": "fake-model"}},
        )
        assert configured.status_code == 200
        await http_client.post(f"/api/knowledge/databases/{kb_id}/graph/build", json={})
        final = await self._wait_build_completed(http_client, kb_id)
        assert final["build_task_status"] == "completed"
        assert final["entity_count"] == 2

        updated = await http_client.post(
            f"/api/knowledge/databases/{kb_id}/files/{file_id}/update-content",
            json={},
        )
        assert updated.status_code == 200

        final = await self._wait_build_completed(http_client, kb_id)
        assert final["build_task_status"] == "completed"
        assert final["entity_count"] == 2
        assert final["indexed_chunks"] == final["total_chunks"]

    async def test_delete_file_cascades_graph(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        file_a = await _upload_and_index(http_client, kb_id, content="# 笔记\n\n甲公司开发支付系统，王五负责研发。")
        await _upload_and_index(http_client, kb_id, name="note2.md", content="# 笔记\n\n乙公司开发风控系统，王五提供咨询。")

        configured = await http_client.put(
            f"/api/knowledge/databases/{kb_id}/graph/config",
            json={"extractor_type": "llm", "extractor_options": {"model_spec": "fake-model"}},
        )
        assert configured.status_code == 200
        await http_client.post(f"/api/knowledge/databases/{kb_id}/graph/build", json={})
        final = await self._wait_build_completed(http_client, kb_id)
        assert final["build_task_status"] == "completed"
        assert final["entity_count"] == 3
        assert final["relation_count"] == 2

        deleted = await http_client.delete(f"/api/knowledge/databases/{kb_id}/files/{file_a}")
        assert deleted.status_code == 200

        stats = await http_client.get(f"/api/knowledge/databases/{kb_id}/graph/stats")
        storage = stats.json()["storage"]
        # 删除文件只清 mentions/chunk/relation 边与向量索引，孤儿实体节点保留（与
        # test_graph_service 的级联删除设计一致），故 entities 不变、relations 减一
        assert storage["entities"] == 3
        assert storage["relations"] == 1
        assert storage["chunks"] == 1
        assert storage["mentions"] == 2

        # 孤儿实体仍可被搜索命中，但子图中已没有任何边
        lone = await http_client.get(
            f"/api/knowledge/databases/{kb_id}/graph/subgraph", params={"entity_ids": "甲公司"}
        )
        assert [node["name"] for node in lone.json()["nodes"]] == ["甲公司"]
        assert lone.json()["edges"] == []

        shared = await http_client.get(
            f"/api/knowledge/databases/{kb_id}/graph/subgraph", params={"entity_ids": "王五"}
        )
        shared_names = {node["name"] for node in shared.json()["nodes"]}
        assert "王五" in shared_names
        assert "乙公司" in shared_names
        assert len(shared.json()["edges"]) == 1

    async def test_build_without_config_returns_400(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(http_client, kb_id)
        response = await http_client.post(f"/api/knowledge/databases/{kb_id}/graph/build", json={})
        assert response.status_code == 400

    async def test_404_on_missing_kb(self, client):
        http_client, _ = client
        missing = await http_client.get("/api/knowledge/databases/no-such-kb/graph/status")
        assert missing.status_code == 404
        missing = await http_client.post("/api/knowledge/databases/no-such-kb/graph/build", json={})
        assert missing.status_code == 404
        missing = await http_client.get(
            "/api/knowledge/databases/no-such-kb/graph/subgraph", params={"entity_ids": "x"}
        )
        assert missing.status_code == 404

    async def test_graph_requires_auth(self, client, monkeypatch):
        http_client, _ = client
        monkeypatch.setenv("YUSU_API_KEY", "secret-token")

        no_key = await http_client.get("/api/knowledge/databases/x/graph/status")
        assert no_key.status_code == 401

        bad_key = await http_client.post(
            "/api/knowledge/databases/x/graph/build", json={}, headers={"Authorization": "Bearer wrong"}
        )
        assert bad_key.status_code == 401

        ok = await http_client.get(
            "/api/knowledge/databases/x/graph/status", headers={"Authorization": "Bearer secret-token"}
        )
        assert ok.status_code == 404  # 鉴权通过，KB 不存在


class TestEvaluation:
    """Evaluation dataset / run endpoints (EvalFakeChat injected via EvaluationService)."""

    # 多行重复内容 → 多个 chunk（chunk_token_num=1200），保证 benchmark 生成可出多个 item
    BENCH_CONTENT = (
        "# 笔录\n\n"
        + "2024年3月张三向李四转账五十万元用于项目投资，该笔转账经财务部审核确认。\n" * 20
        + "\n公司采购自动化生产线设备总价三十万元，采购合同由财务部门审核通过。\n" * 20
    )

    @staticmethod
    async def _wait_dataset_ready(http_client, kb_id: str, dataset_id: str, timeout: float = 20.0) -> dict:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            response = await http_client.get(f"/api/evaluation/datasets/{dataset_id}", params={"kb_id": kb_id})
            # 生成中：service 返回 200 + build_metadata.status=running（items 为空），继续轮询
            assert response.status_code == 200
            body = response.json()["data"]
            status = body["build_metadata"]["status"]
            if status == "completed":
                return body
            if status == "failed":
                raise AssertionError(f"dataset generation failed: {body['build_metadata']}")
            if loop.time() > deadline:
                raise AssertionError(f"dataset generation did not finish: {body['build_metadata']}")
            await asyncio.sleep(0.02)

    @staticmethod
    async def _wait_run_completed(http_client, kb_id: str, run_id: str, timeout: float = 20.0) -> dict:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            response = await http_client.get(f"/api/evaluation/runs/{run_id}", params={"kb_id": kb_id})
            assert response.status_code == 200
            body = response.json()["data"]
            if body["status"] in ("completed", "failed"):
                return body
            if loop.time() > deadline:
                raise AssertionError(f"run did not finish: {body}")
            await asyncio.sleep(0.02)

    async def test_dataset_upload_list_detail_download_delete(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)

        lines = [
            {"query": "2024年3月张三向李四转账的金额是多少？", "gold_answer": "五十万元", "gold_chunk_ids": ["c1"]},
            {"query": "公司采购自动化生产线设备的总价是多少？", "gold_answer": "三十万元"},
        ]
        content = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines).encode("utf-8")
        upload = await http_client.post(
            "/api/evaluation/datasets/upload",
            files={"file": ("笔录测试.jsonl", content, "application/x-ndjson")},
            data={"kb_id": kb_id, "name": "上传数据集"},
        )
        assert upload.status_code == 200
        dataset_id = upload.json()["data"]["dataset_id"]

        listing = await http_client.get("/api/evaluation/datasets", params={"kb_id": kb_id})
        assert listing.status_code == 200
        datasets = listing.json()["data"]
        assert [d["dataset_id"] for d in datasets] == [dataset_id]
        assert datasets[0]["item_count"] == 2
        assert datasets[0]["has_gold_chunks"] is True
        assert datasets[0]["has_gold_answers"] is True
        assert datasets[0]["build_metadata"]["status"] == "completed"

        detail = await http_client.get(
            f"/api/evaluation/datasets/{dataset_id}", params={"kb_id": kb_id, "page": 1, "page_size": 1}
        )
        assert detail.status_code == 200
        data = detail.json()["data"]
        assert len(data["items"]) == 1
        assert data["pagination"]["total_items"] == 2
        assert data["pagination"]["has_next"] is True

        download = await http_client.get(f"/api/evaluation/datasets/{dataset_id}/download")
        assert download.status_code == 200
        assert download.headers["content-type"].startswith("application/x-ndjson")
        assert 'attachment; filename="' in download.headers["content-disposition"]
        assert len(download.text.strip().splitlines()) == 2

        deleted = await http_client.delete(f"/api/evaluation/datasets/{dataset_id}")
        assert deleted.status_code == 200
        missing = await http_client.get(f"/api/evaluation/datasets/{dataset_id}", params={"kb_id": kb_id})
        assert missing.status_code == 404

    async def test_dataset_upload_invalid_jsonl(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        bad = await http_client.post(
            "/api/evaluation/datasets/upload",
            files={"file": ("bad.jsonl", b'{"no_query": 1}\n', "application/x-ndjson")},
            data={"kb_id": kb_id},
        )
        assert bad.status_code == 400
        empty = await http_client.post(
            "/api/evaluation/datasets/upload",
            files={"file": ("empty.jsonl", b"\n", "application/x-ndjson")},
            data={"kb_id": kb_id},
        )
        assert empty.status_code == 400

    async def test_generate_dataset_flow(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(http_client, kb_id, content=self.BENCH_CONTENT)

        response = await http_client.post(
            "/api/evaluation/datasets/generate",
            json={
                "kb_id": kb_id,
                "name": "生成数据集",
                "count": 2,
                "generation_mode": "vector",
                "llm_model_spec": "fake-benchmark",
            },
        )
        assert response.status_code == 200
        dataset_id = response.json()["data"]["dataset_id"]

        ready = await self._wait_dataset_ready(http_client, kb_id, dataset_id)
        assert ready["item_count"] == 2
        assert ready["has_gold_chunks"] is True
        assert ready["build_metadata"]["source"] == "generated"
        assert ready["build_metadata"]["progress"] == 100
        assert ready["items"][0]["gold_chunk_ids"], "生成项应携带 gold_chunk_ids"

    async def test_generate_dataset_graph_mode_requires_index(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(http_client, kb_id, content=self.BENCH_CONTENT)
        # 未建图 → graph_enhanced 必须 400
        response = await http_client.post(
            "/api/evaluation/datasets/generate",
            json={"kb_id": kb_id, "count": 1, "generation_mode": "graph_enhanced", "llm_model_spec": "fake"},
        )
        assert response.status_code == 400
        assert "图索引" in response.json()["detail"]

    async def test_run_evaluation_flow(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        await _upload_and_index(http_client, kb_id, content=self.BENCH_CONTENT)

        lines = [
            {"query": "2024年3月张三向李四转账的金额是多少？", "gold_answer": "五十万元", "gold_chunk_ids": ["missing_1"]},
            {"query": "公司采购自动化生产线设备的总价是多少？", "gold_answer": "三十万元", "gold_chunk_ids": ["missing_2"]},
        ]
        content = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines).encode("utf-8")
        upload = await http_client.post(
            "/api/evaluation/datasets/upload",
            files={"file": ("run.jsonl", content, "application/x-ndjson")},
            data={"kb_id": kb_id, "name": "运行数据集"},
        )
        assert upload.status_code == 200
        dataset_id = upload.json()["data"]["dataset_id"]

        created = await http_client.post(
            "/api/evaluation/runs",
            json={
                "kb_id": kb_id,
                "dataset_id": dataset_id,
                "name": "测试运行",
                "model_config": {
                    "answer_llm": "fake-answer",
                    "judge_llm": "fake-judge",
                    "use_reranker": False,
                    "use_graph_retrieval": False,
                },
            },
        )
        assert created.status_code == 200
        run_id = created.json()["data"]["run_id"]
        assert created.json()["data"]["status"] == "running"

        done = await self._wait_run_completed(http_client, kb_id, run_id)
        assert done["status"] == "completed"
        assert done["completed_items"] == 2
        assert done["total_items"] == 2
        assert done["overall_score"] is not None
        assert done["metrics"]["judge_status"] == "available"
        assert any(key.startswith("recall@") for key in done["metrics"])
        assert len(done["items"]) == 2
        assert all(item["generated_answer"] == "这是评估生成的回答。" for item in done["items"])
        # item 级 answer 指标键为 judge 返回的 score；answer_correctness 只在 run 级聚合
        assert all(item["metrics"].get("score") == 1.0 for item in done["items"])

        error_only = await http_client.get(
            f"/api/evaluation/runs/{run_id}", params={"kb_id": kb_id, "error_only": True}
        )
        assert error_only.status_code == 200
        assert error_only.json()["data"]["pagination"]["error_only"] is True

        listing = await http_client.get("/api/evaluation/runs", params={"kb_id": kb_id})
        assert listing.status_code == 200
        runs = listing.json()["data"]
        assert [r["run_id"] for r in runs] == [run_id]
        assert runs[0]["status"] == "completed"

        deleted = await http_client.delete(f"/api/evaluation/runs/{run_id}", params={"kb_id": kb_id})
        assert deleted.status_code == 200
        missing = await http_client.get(f"/api/evaluation/runs/{run_id}", params={"kb_id": kb_id})
        assert missing.status_code == 404

    async def test_run_validation_errors(self, client):
        http_client, _ = client
        kb_id = await _create_kb(http_client)
        # 不存在的 dataset → 404
        missing = await http_client.post(
            "/api/evaluation/runs", json={"kb_id": kb_id, "dataset_id": "dataset_nope"}
        )
        assert missing.status_code == 404
        # 非法 run_id 格式 → 400
        bad = await http_client.get("/api/evaluation/runs/not-a-run-id", params={"kb_id": kb_id})
        assert bad.status_code == 400

    async def test_eval_requires_auth(self, client, monkeypatch):
        http_client, _ = client
        monkeypatch.setenv("YUSU_API_KEY", "secret-token")
        no_key = await http_client.get("/api/evaluation/datasets", params={"kb_id": "x"})
        assert no_key.status_code == 401
        bad_key = await http_client.post(
            "/api/evaluation/runs", json={"kb_id": "x", "dataset_id": "y"}, headers={"Authorization": "Bearer wrong"}
        )
        assert bad_key.status_code == 401
        ok = await http_client.get(
            "/api/evaluation/datasets", params={"kb_id": "x"}, headers={"Authorization": "Bearer secret-token"}
        )
        assert ok.status_code == 200