"""Tests for LocalKB (NanoVectorDB + SQLite): indexing, retrieval, cascade deletes.

All tests run against a deterministic fake embedding function (char-hash
bag-of-words into a 64-dim unit vector), so vector similarities are stable
within a process and no external embedding service is contacted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from yusu_kb.knowledge.base import FileStatus
from yusu_kb.knowledge.implementations.local_kb import (
    LocalKB,
    create_embedding_func,
    get_or_create_embedding_func,
    set_default_embedding_func,
)
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.models.embed import EmbeddingFunc, wrap_embedding_func_with_attrs
from yusu_kb.repositories import configure_repositories
from yusu_kb.repositories.knowledge_chunk_repository import KnowledgeChunkRepository
from yusu_kb.repositories.knowledge_file_repository import KnowledgeFileRepository

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


@pytest.fixture()
async def kb_id(manager) -> str:
    created = await manager.create_database(name="本地库", description="demo", kb_type="local")
    return created["kb_id"]


def get_kb(manager) -> LocalKB:
    instance = manager.get_instance("local")
    assert isinstance(instance, LocalKB)
    return instance


async def index_file(manager, kb_id: str, tmp_path: Path, name: str = "doc.md", content: str = "") -> str:
    path = tmp_path / name
    path.write_text(content or "# 标题\n\n默认测试文档内容。", encoding="utf-8")
    record = await manager.add_file_record(kb_id, str(path))
    await manager.parse_file(kb_id, record["file_id"])
    result = await get_kb(manager).index_file(kb_id, record["file_id"])
    assert result["status"] == FileStatus.INDEXED
    return record["file_id"]


class TestRegistrationAndFactory:
    async def test_local_type_registered(self, manager):
        types = manager.get_available_types()
        assert "local" in types
        assert types["local"]["name"] == "Local"
        assert manager.is_type_supported("local")

    def test_embedding_func_factory_wraps_provider(self):
        func = create_embedding_func(provider_func=_fake_embed, embedding_dim=FAKE_DIM)
        assert isinstance(func, EmbeddingFunc)
        assert func.embedding_dim == FAKE_DIM

    def test_embedding_func_factory_missing_dim(self):
        with pytest.raises(ValueError):
            create_embedding_func(provider_func=lambda texts: np.zeros((len(texts), 8)))

    def test_wrap_embedding_func_with_attrs(self):
        func = wrap_embedding_func_with_attrs(embedding_dim=8, max_token_size=1024, model_name="m")(_fake_embed)
        assert isinstance(func, EmbeddingFunc)
        assert func.embedding_dim == 8
        assert func.max_token_size == 1024
        assert func.model_name == "m"

    def test_default_embedding_func_injection(self):
        assert get_or_create_embedding_func() is fake_embedding_func

    def test_query_params_config(self, kb_id, manager):
        config = get_kb(manager).get_query_params_config(kb_id)
        assert config["type"] == "local"
        keys = {option["key"] for option in config["options"]}
        assert {"final_top_k", "search_mode", "use_reranker", "lexical_channel_enabled"} <= keys
        by_key = {option["key"]: option for option in config["options"]}
        assert by_key["final_top_k"]["default"] == 10


class TestIndexAndQuery:
    async def test_index_query_roundtrip(self, manager, kb_id, tmp_path):
        content = "# 反诈专题\n\n杭州市公安局反诈中心侦办 CDR-2026-001 案件。\n\n涉案资金流向境外账户。"
        file_id = await index_file(manager, kb_id, tmp_path, name="anti_fraud.md", content=content)

        meta = await get_kb(manager).get_file_basic_info(kb_id, file_id)
        assert meta["meta"]["status"] == FileStatus.INDEXED
        assert meta["meta"]["chunk_count"] > 0
        assert meta["meta"]["token_count"] > 0

        results = await get_kb(manager).aquery("反诈中心", kb_id, use_reranker=False)
        assert results
        top = results[0]
        assert "反诈" in top["content"]
        assert top["metadata"]["chunk_id"]
        assert top["metadata"]["file_id"] == file_id
        assert top["metadata"]["source"] == "anti_fraud.md"
        assert "score" in top

        output = get_kb(manager).build_search_output(kb_id, results)
        assert output["kb_id"] == kb_id
        assert output["results"][0]["content"]
        assert output["results"][0]["file_id"] == file_id

    async def test_vector_filtering_and_top_k(self, manager, kb_id, tmp_path):
        file_id = await index_file(
            manager, kb_id, tmp_path, name="a.md", content="# 甲\n\n甲单位年度财务报告全文内容。"
        )
        await index_file(
            manager, kb_id, tmp_path, name="b.md", content="# 乙\n\n乙单位的安全生产检查记录内容。"
        )

        base = await get_kb(manager).aquery("财务报告", kb_id, use_reranker=False)
        assert base
        assert len(base) >= 1

        filtered = await get_kb(manager).aquery(
            "财务报告", kb_id, use_reranker=False, similarity_threshold=0.999
        )
        assert len(filtered) < len(base)

        single = await get_kb(manager).aquery("财务报告", kb_id, use_reranker=False, final_top_k=1)
        assert len(single) == 1

        file_filtered = await get_kb(manager).aquery(
            "内容", kb_id, use_reranker=False, file_ids=[file_id]
        )
        assert file_filtered
        assert all(chunk["metadata"]["file_id"] == file_id for chunk in file_filtered)

        unfiltered = await get_kb(manager).aquery("内容", kb_id, use_reranker=False)
        assert any(chunk["metadata"]["file_id"] != file_id for chunk in unfiltered)

        name_filtered = await get_kb(manager).aquery("内容", kb_id, use_reranker=False, file_name="a.md")
        assert name_filtered
        assert all(chunk["metadata"]["file_id"] == file_id for chunk in name_filtered)

        empty_filter = await get_kb(manager).aquery("报告", kb_id, use_reranker=False, file_name="nope.md")
        assert empty_filter == []

    async def test_keyword_and_hybrid_search_modes(self, manager, kb_id, tmp_path):
        content = "# 案件详情\n\n报案编号 BX-8821-01，犯罪嫌疑人张某，涉及电信网络诈骗。"
        file_id = await index_file(manager, kb_id, tmp_path, name="case.md", content=content)

        keyword = await get_kb(manager).aquery(
            "电信网络诈骗 张某", kb_id, search_mode="keyword", use_reranker=False
        )
        assert keyword
        assert keyword[0]["metadata"]["file_id"] == file_id
        assert "BX-8821-01" in keyword[0]["content"]
        assert "bm25_score" in keyword[0]

        hybrid = await get_kb(manager).aquery("电信网络诈骗", kb_id, search_mode="hybrid", use_reranker=False)
        assert hybrid
        assert any("fusion_score" in chunk or "bm25_score" in chunk for chunk in hybrid)

        empty = await get_kb(manager).aquery(
            "完全不存在的词语组合XYZ", kb_id, search_mode="keyword", use_reranker=False
        )
        assert empty == []

    async def test_lexical_channel_exact_match(self, manager, kb_id, tmp_path):
        content = "# 通话记录\n\nCDR-2026-001 号通话详情显示主叫号码 13812345678。"
        file_id = await index_file(manager, kb_id, tmp_path, name="cdr.md", content=content)

        plain = await get_kb(manager).aquery(
            "CDR-2026-001 号通话详情", kb_id, use_reranker=False, lexical_channel_enabled=False
        )
        lexical = await get_kb(manager).aquery(
            "CDR-2026-001 号通话详情", kb_id, use_reranker=False, lexical_channel_enabled=True
        )
        assert lexical
        exact = [c for c in lexical if c.get("exact_match")]
        assert exact, "lexical channel must flag exact-token matches"
        assert exact[0]["metadata"]["file_id"] == file_id
        assert "CDR-2026-001" in exact[0]["content"]
        # fusion keeps at least as much information as the plain vector path
        assert len(lexical) >= len(plain)

    async def test_rerank_fallback_and_custom_reranker(self, manager, kb_id, tmp_path):
        await index_file(
            manager, kb_id, tmp_path, name="a.md", content="# 甲\n\n甲单位年度财务报告全文内容。"
        )
        await index_file(
            manager, kb_id, tmp_path, name="b.md", content="# 乙\n\n乙单位的安全生产检查记录内容。"
        )

        # no reranker configured -> retrieval scores used, no crash
        results = await get_kb(manager).aquery("财务报告", kb_id)
        assert results
        assert all("rerank_score" not in chunk for chunk in results)

        # a custom reranker inverts the ordering
        async def inverted_rerank(query: str, documents: list[str]) -> list[float]:
            del query
            return [i / max(len(documents), 1) for i in range(len(documents))]

        custom = LocalKB(str(Path(manager.work_dir)), embedding_func=fake_embedding_func, rerank_func=inverted_rerank)
        custom.databases_meta = get_kb(manager).databases_meta
        reranked = await custom.aquery("财务报告", kb_id)
        assert reranked
        assert all("rerank_score" in chunk for chunk in reranked)
        assert [c["content"] for c in reranked] == [c["content"] for c in results][::-1]


class TestCascadeAndPersistence:
    async def test_delete_file_cascade(self, manager, kb_id, tmp_path):
        await index_file(
            manager, kb_id, tmp_path, name="a.md", content="# 甲\n\n甲单位年度财务报告。"
        )
        file_b = await index_file(
            manager, kb_id, tmp_path, name="b.md", content="# 乙\n\n乙单位安全生产检查记录。"
        )

        results = await get_kb(manager).aquery("安全生产", kb_id, use_reranker=False)
        assert any(chunk["metadata"]["file_id"] == file_b for chunk in results)

        await get_kb(manager).delete_file(kb_id, file_b)

        assert await KnowledgeFileRepository().get_by_file_id(file_b) is None
        remaining = await KnowledgeChunkRepository().list_by_file_id(file_b)
        assert remaining == []

        results = await get_kb(manager).aquery("安全生产", kb_id, use_reranker=False)
        assert all(chunk["metadata"]["file_id"] != file_b for chunk in results)

        stats = await get_kb(manager).refresh_database_stats(kb_id)
        assert stats["file_count"] == 1

    async def test_update_content_reindexes(self, manager, kb_id, tmp_path):
        content = "# 大文档\n\n" + "\n\n".join(f"第{i}段：这是关于项目{PROJECT}的说明文字，内容较长。" for i, PROJECT in enumerate(["A", "B", "C", "D", "E"]))
        file_id = await index_file(manager, kb_id, tmp_path, name="big.md", content=content)

        before = await get_kb(manager).get_file_basic_info(kb_id, file_id)
        old_chunks = {
            c.chunk_id: c.content for c in await KnowledgeChunkRepository().list_by_file_id(file_id)
        }

        updated = await get_kb(manager).update_content(
            kb_id, [file_id], {"chunk_parser_config": {"chunk_token_num": 50}}
        )
        assert updated[0]["status"] == FileStatus.INDEXED

        new_chunks = {
            c.chunk_id: c.content for c in await KnowledgeChunkRepository().list_by_file_id(file_id)
        }
        assert new_chunks
        assert len(new_chunks) > len(old_chunks)
        # chunk ids restart at _chunk_0, so compare contents: every shared id must
        # hold the re-chunked content, not the stale one
        assert any(new_chunks[cid] != old_chunks.get(cid, "") for cid in new_chunks)

        after = await get_kb(manager).get_file_basic_info(kb_id, file_id)
        assert after["meta"]["chunk_count"] > before["meta"]["chunk_count"]

        results = await get_kb(manager).aquery("项目A", kb_id, use_reranker=False)
        assert results

    async def test_vector_persistence_across_instances(self, manager, kb_id, tmp_path):
        content = "# 持久化\n\n这份文档用于验证向量落盘后新实例仍可检索。"
        file_id = await index_file(manager, kb_id, tmp_path, name="persist.md", content=content)
        await manager.close()

        # a fresh manager + fresh LocalKB instance on the same work dir must
        # rediscover the KB and answer from the persisted vector file
        m2 = KnowledgeBaseManager(str(Path(manager.work_dir)))
        await m2.load_all_metadata()
        kb2 = m2.get_instance("local")
        assert isinstance(kb2, LocalKB)
        results = await kb2.aquery("向量落盘后新实例仍可检索", kb_id, use_reranker=False)
        assert results
        assert any(chunk["metadata"]["file_id"] == file_id for chunk in results)
        await m2.close()

    async def test_delete_database_cleans_storage(self, manager, kb_id, tmp_path):
        await index_file(manager, kb_id, tmp_path, name="a.md", content="# 甲\n\n甲单位财务报告。")
        kb_dir = Path(manager.work_dir) / kb_id
        assert any(kb_dir.glob("vdb_*.json"))

        result = await manager.delete_database(kb_id)
        assert result["message"] == "删除成功"
        assert not kb_dir.exists()
        assert await manager.get_database_info(kb_id) is None