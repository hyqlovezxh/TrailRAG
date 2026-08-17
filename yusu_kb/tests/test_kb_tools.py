"""Task 18: KB tool tests (knowledge/tools.py).

Exercises list_kbs / query_kb / open_kb_document / find_kb_document against a
real LocalKB with a fake deterministic embedding, plus the OpenAI-style tool
definitions and the execute_kb_tool dispatcher.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest

from yusu_kb.knowledge import tools as tools_module
from yusu_kb.knowledge.implementations.local_kb import LocalKB
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.models.embed import EmbeddingFunc, wrap_embedding_func_with_attrs
from yusu_kb.repositories import configure_repositories

FAKE_DIM = 64


def _fake_embed_one(text: str, dim: int = FAKE_DIM) -> np.ndarray:
    vec = np.zeros(dim)
    for ch in text:
        # ord() 而非 hash()：Python str hash 受 PYTHONHASHSEED 随机化影响，会导致
        # 向量召回结果跨进程不稳定（flaky）
        vec[ord(ch) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


async def _fake_embed(texts: list[str], **kwargs) -> np.ndarray:
    del kwargs
    return np.array([_fake_embed_one(t) for t in texts])


fake_embedding_func: EmbeddingFunc = wrap_embedding_func_with_attrs(
    embedding_dim=FAKE_DIM, max_token_size=None
)(_fake_embed)


@pytest.fixture(autouse=True)
def default_fake_embedding():
    from yusu_kb.knowledge.implementations.local_kb import set_default_embedding_func

    set_default_embedding_func(fake_embedding_func)
    yield
    set_default_embedding_func(None)


@pytest.fixture()
async def manager(tmp_path, engine):
    configure_repositories(engine)
    kb_manager = KnowledgeBaseManager(str(tmp_path / "kb_work"))
    yield kb_manager
    await kb_manager.close()
    configure_repositories(None)


def get_kb(manager) -> LocalKB:
    instance = manager.get_instance("local")
    assert isinstance(instance, LocalKB)
    return instance


async def create_kb(manager, name: str) -> str:
    created = await manager.create_database(
        name=name,
        description="tools test",
        kb_type="local",
        additional_params={"auto_build_graph": False},
    )
    return created["kb_id"]


async def index_file(manager, kb_id: str, tmp_path: Path, name: str, content: str) -> str:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    record = await manager.add_file_record(kb_id, str(path))
    await manager.parse_file(kb_id, record["file_id"])
    result = await get_kb(manager).index_file(kb_id, record["file_id"])
    assert result["status"] == "indexed"
    return record["file_id"]


CONTENT = "# 水果笔记\n\n苹果是红色的水果，香蕉是黄色的水果。\n\n公司采购自动化生产线设备总价三十万元。"


class TestListKBs:
    async def test_list_kbs_returns_kb_list(self, manager, tmp_path):
        kb_a = await create_kb(manager, "工具库A")
        kb_b = await create_kb(manager, "工具库B")
        result = await tools_module.list_kbs(kb_manager=manager)
        by_id = {item["kb_id"]: item for item in result}
        assert by_id[kb_a]["name"] == "工具库A"
        assert by_id[kb_b]["name"] == "工具库B"
        assert "description" in by_id[kb_a]


class TestQueryKB:
    async def test_query_kb_basic(self, manager, tmp_path):
        kb_id = await create_kb(manager, "查询库")
        file_id = await index_file(manager, kb_id, tmp_path, "fruit.md", CONTENT)

        result = await tools_module.query_kb(kb_id, "苹果是红色还是黄色？", kb_manager=manager)
        assert isinstance(result, dict)
        assert result["kb_id"] == kb_id
        assert result["results"], "应返回检索命中"
        first = result["results"][0]
        assert "苹果" in first["content"]
        assert first["file_id"] == file_id
        # citation_source 统一为 kb://{kb_id}/{file_id}?chunk={chunk_id}（URL 编码）
        prefix = f"kb://{quote(kb_id, safe='')}/{quote(file_id, safe='')}?chunk="
        assert first["citation_source"].startswith(prefix)
        # 每个结果都有 metadata（SearchResultSchema 结构）
        assert isinstance(first["metadata"], dict)

    async def test_query_kb_file_name_filter(self, manager, tmp_path):
        kb_id = await create_kb(manager, "过滤库")
        await index_file(manager, kb_id, tmp_path, "apple.md", "苹果是红色的水果。")
        await index_file(manager, kb_id, tmp_path, "banana.md", "苹果和香蕉都是水果。")

        result = await tools_module.query_kb(kb_id, "苹果", file_name="apple.md", kb_manager=manager)
        file_ids = {item["file_id"] for item in result["results"]}
        assert len(file_ids) == 1, "file_name 过滤后只应命中 apple.md"

    async def test_query_kb_token_budget_truncates(self, manager, tmp_path, monkeypatch):
        kb_id = await create_kb(manager, "预算库")
        await index_file(manager, kb_id, tmp_path, "long.md", CONTENT * 20)
        # 极小预算强制触发截断：首项尾部省略，其余项标记省略
        monkeypatch.setattr(tools_module, "_QUERY_KB_MAX_OUTPUT_TOKENS", 2)

        result = await tools_module.query_kb(kb_id, "苹果", kb_manager=manager)
        contents = [item["content"] for item in result["results"]]
        assert contents, "应返回命中"
        assert any(tools_module._OMITTED_MARKER in content for content in contents)

    async def test_query_kb_validation_and_missing_kb(self, manager):
        assert await tools_module.query_kb("", "q", kb_manager=manager) == "请提供 kb_id"
        assert await tools_module.query_kb("kb", "", kb_manager=manager) == "请提供查询内容"
        missing = await tools_module.query_kb("no-such-kb", "q", kb_manager=manager)
        assert "不存在" in missing

    async def test_query_kb_no_reranker_no_graph_flags(self, manager, tmp_path):
        kb_id = await create_kb(manager, "关闭库")
        await index_file(manager, kb_id, tmp_path, "note.md", CONTENT)
        result = await tools_module.query_kb(
            kb_id, "苹果", use_reranker=False, use_graph_retrieval=False, kb_manager=manager
        )
        assert isinstance(result, dict) and result["results"]


class TestOpenKBDocument:
    async def test_open_kb_document_window(self, manager, tmp_path):
        kb_id = await create_kb(manager, "打开库")
        file_id = await index_file(manager, kb_id, tmp_path, "note.md", CONTENT)

        result = await tools_module.open_kb_document(kb_id, file_id, line=1, window_size=20, kb_manager=manager)
        assert isinstance(result, dict)
        assert result["kb_id"] == kb_id
        assert result["file_id"] == file_id
        assert result["start_line"] == 1
        assert result["end_line"] > 0
        assert "苹果" in result["content"]
        # 行窗口语义：citation_source 使用 lines_{start}_{end}
        assert result["citation_source"].startswith(
            f"kb://{quote(kb_id, safe='')}/{quote(file_id, safe='')}?chunk=lines_"
        )

    async def test_open_kb_document_errors(self, manager, tmp_path):
        kb_id = await create_kb(manager, "错误库")
        file_id = await index_file(manager, kb_id, tmp_path, "note.md", CONTENT)

        assert await tools_module.open_kb_document("", file_id, kb_manager=manager) == "请提供 kb_id"
        assert await tools_module.open_kb_document(kb_id, "", kb_manager=manager) == "请提供 file_id"
        missing_kb = await tools_module.open_kb_document("no-such-kb", file_id, kb_manager=manager)
        assert "不存在" in missing_kb
        missing_file = await tools_module.open_kb_document(kb_id, "no-such-file", kb_manager=manager)
        assert "失败" in missing_file or "不存在" in missing_file


class TestFindKBDocument:
    async def test_find_kb_document_windows(self, manager, tmp_path):
        kb_id = await create_kb(manager, "定位库")
        file_id = await index_file(
            manager, kb_id, tmp_path, "note.md", "# 笔录\n\n张三向李四转账五十万元。\n\n采购设备三十万元。"
        )

        result = await tools_module.find_kb_document(
            kb_id, file_id, ["转账"], window_size=5, kb_manager=manager
        )
        assert isinstance(result, dict)
        assert result["kb_id"] == kb_id
        assert result["file_id"] == file_id
        assert result["total_matches"] >= 1
        assert result["windows"], "应返回命中窗口"
        window = result["windows"][0]
        assert "转账" in window["content"]
        assert window["start_line"] >= 1
        assert window["citation_source"].startswith(
            f"kb://{quote(kb_id, safe='')}/{quote(file_id, safe='')}?chunk=lines_"
        )

    async def test_find_kb_document_regex_and_validation(self, manager, tmp_path):
        kb_id = await create_kb(manager, "正则库")
        file_id = await index_file(
            manager, kb_id, tmp_path, "note.md", "# 笔录\n\n张三的账号是 A-12345。"
        )
        result = await tools_module.find_kb_document(
            kb_id, file_id, [r"A-\d+"], use_regex=True, kb_manager=manager
        )
        assert result["windows"], "正则模式应命中"
        assert result["match_mode"] == "regex"

        assert await tools_module.find_kb_document("", file_id, ["x"], kb_manager=manager) == "请提供 kb_id"
        assert await tools_module.find_kb_document(kb_id, "", ["x"], kb_manager=manager) == "请提供 file_id"
        assert await tools_module.find_kb_document(kb_id, file_id, [], kb_manager=manager) == "请提供 patterns"


class TestToolRegistry:
    def test_get_kb_tool_definitions(self):
        definitions = tools_module.get_kb_tool_definitions()
        names = [tool["function"]["name"] for tool in definitions]
        assert names == ["list_kbs", "query_kb", "open_kb_document", "find_kb_document"]
        for tool in definitions:
            assert tool["type"] == "function"
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
        query_params = definitions[1]["function"]["parameters"]["properties"]
        assert "kb_id" in query_params and "query_text" in query_params
        assert tools_module.get_kb_tool_names() == names

    async def test_execute_kb_tool_dispatch(self, manager, tmp_path):
        kb_id = await create_kb(manager, "分发库")
        file_id = await index_file(manager, kb_id, tmp_path, "note.md", CONTENT)

        raw = await tools_module.execute_kb_tool(
            "query_kb", {"kb_id": kb_id, "query_text": "苹果"}, kb_manager=manager
        )
        parsed = json.loads(raw)
        assert parsed["kb_id"] == kb_id
        assert parsed["results"][0]["citation_source"]

        listed = json.loads(await tools_module.execute_kb_tool("list_kbs", {}, kb_manager=manager))
        assert any(item["kb_id"] == kb_id for item in listed)

        opened = json.loads(
            await tools_module.execute_kb_tool(
                "open_kb_document", {"kb_id": kb_id, "file_id": file_id, "line": 1}, kb_manager=manager
            )
        )
        assert opened["start_line"] == 1

    async def test_execute_kb_tool_errors(self, manager):
        assert "未知工具" in await tools_module.execute_kb_tool("nope", {}, kb_manager=manager)
        # 工具内部失败（缺必填参数）→ 错误字符串而非异常
        result = await tools_module.execute_kb_tool("query_kb", {"query_text": "q"}, kb_manager=manager)
        assert isinstance(result, str) and "失败" in result or "请提供" in result
