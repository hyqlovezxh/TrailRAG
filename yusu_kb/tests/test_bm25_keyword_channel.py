"""Okapi BM25 keyword-channel tests (plan Phase 3).

``_rank_keyword_chunks`` was rewritten from term-hit counting to real Okapi
BM25 (local IDF over the SQLite candidate set, k1=1.5, b=0.75, max-score
normalisation). The acceptance criteria from the plan, asserted here:

- TF saturation: a repeated term no longer adds linearly;
- length normalisation: long chunks no longer win by default;
- long/multi-term queries are not systematically penalised;
- scores are max-normalised into [0, 1];
- the lexical channel's exact-match markers stay compatible.

The ranking helper is pure (records in, chunks out), so most tests exercise
it directly with SimpleNamespace records; one test runs the real SQLite
keyword channel end to end.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from yusu_kb.knowledge.base import FileStatus
from yusu_kb.knowledge.implementations.local_kb import LocalKB
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.models.embed import EmbeddingFunc, wrap_embedding_func_with_attrs
from yusu_kb.repositories import configure_repositories

FAKE_DIM = 64


def _fake_embed_one(text: str, dim: int = FAKE_DIM) -> np.ndarray:
    vec = np.zeros(dim)
    for ch in text:
        vec[ord(ch) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


async def _fake_embed(texts: list[str], **kwargs) -> np.ndarray:
    del kwargs
    return np.array([_fake_embed_one(t) for t in texts])


fake_embedding_func: EmbeddingFunc = wrap_embedding_func_with_attrs(
    embedding_dim=FAKE_DIM, max_token_size=None
)(_fake_embed)


def _rec(content: str, chunk_id: str, chunk_index: int = 0, file_id: str = "f1") -> SimpleNamespace:
    return SimpleNamespace(
        content=content, chunk_id=chunk_id, file_id=file_id, chunk_index=chunk_index
    )


@pytest.fixture()
def kb(tmp_path) -> LocalKB:
    return LocalKB(str(tmp_path / "kb_work"))


@pytest.fixture(autouse=True)
def _default_fake_embedding():
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


class TestOkapiBm25Ranking:
    def test_tf_saturation_repeats_no_longer_linear(self, kb):
        """tf=8 must score higher than tf=1, but far below 8x (saturation)."""
        records = [
            _rec("张三向李四转账", "short"),
            _rec("张三向李四转账 " * 8, "long"),
        ]
        chunks = kb._rank_keyword_chunks(records, ["转账"])
        by_id = {chunk["metadata"]["chunk_id"]: chunk for chunk in chunks}
        ratio = by_id["long"]["bm25_score"] / by_id["short"]["bm25_score"]
        assert ratio > 1.0, "more occurrences must still rank higher"
        assert ratio < 4.0, f"TF must saturate (ratio={ratio:.2f}, linear would be 8)"

    def test_length_normalisation_prefers_concise_chunk(self, kb):
        """Same tf, different lengths: the shorter chunk scores higher."""
        filler = "无关填充内容。" * 300
        records = [
            _rec("张三向李四转账五十万元", "concise"),
            _rec("张三向李四转账五十万元" + filler, "verbose"),
        ]
        chunks = kb._rank_keyword_chunks(records, ["转账"])
        by_id = {chunk["metadata"]["chunk_id"]: chunk for chunk in chunks}
        assert by_id["concise"]["bm25_score"] > by_id["verbose"]["bm25_score"]

    def test_multi_term_query_not_systematically_penalised(self, kb):
        """A long query combining two terms must rank the combined chunk first
        and keep every score inside the same normalised scale."""
        records = [
            _rec("张三向李四转账，款项汇入境外账户", "both"),
            _rec("张三向李四转账", "only_transfer"),
            _rec("款项汇入境外账户", "only_remit"),
        ]
        chunks = kb._rank_keyword_chunks(records, ["转账", "汇入"])
        assert len(chunks) == 3
        assert chunks[0]["metadata"]["chunk_id"] == "both"
        assert all(0.0 < chunk["bm25_score"] <= 1.0 for chunk in chunks)

    def test_rarer_term_gets_higher_idf(self, kb):
        """Same tf and similar length: the term with lower df contributes more."""
        records = [
            _rec("转账记录", "common"),
            _rec("转账汇款", "both"),
            _rec("跨境汇款", "rare"),
            _rec("又一条转账记录", "common2"),
        ]
        chunks = kb._rank_keyword_chunks(records, ["转账", "汇款"])
        by_id = {chunk["metadata"]["chunk_id"]: chunk for chunk in chunks}
        # df(转账)=3 > df(汇款)=2 → 单命中的 rare（汇款）得分高于单命中的 common（转账）
        assert chunks[0]["metadata"]["chunk_id"] == "both"  # 双词命中为峰值
        assert by_id["rare"]["bm25_score"] > by_id["common"]["bm25_score"] > 0.0

    def test_scores_max_normalised_into_unit_range(self, kb):
        records = [
            _rec("张三转账", "a"),
            _rec("张三转账并汇款", "b"),
            _rec("李四汇款", "c"),
        ]
        chunks = kb._rank_keyword_chunks(records, ["转账", "汇款"])
        scores = [chunk["bm25_score"] for chunk in chunks]
        assert scores[0] == pytest.approx(1.0)  # peak normalisation
        assert all(0.0 < score <= 1.0 for score in scores)
        assert scores == sorted(scores, reverse=True)

    def test_tie_breaks_by_chunk_index_ascending(self, kb):
        records = [
            _rec("内容甲转账", "late", chunk_index=7),
            _rec("内容乙转账", "early", chunk_index=2),
        ]
        chunks = kb._rank_keyword_chunks(records, ["转账"])
        assert [chunk["metadata"]["chunk_id"] for chunk in chunks] == ["early", "late"]

    def test_empty_inputs_return_empty(self, kb):
        assert kb._rank_keyword_chunks([], ["转账"]) == []
        assert kb._rank_keyword_chunks([_rec("内容", "a")], []) == []
        assert kb._rank_keyword_chunks([_rec("内容", "a")], ["不存在的词"]) == []

    def test_chunk_shape_carries_bm25_score_and_match_count(self, kb):
        records = [_rec("张三转账", "a", file_id="f9", chunk_index=3)]
        chunks = kb._rank_keyword_chunks(records, ["转账"])
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk["content"] == "张三转账"
        assert chunk["metadata"]["file_id"] == "f9"
        assert chunk["metadata"]["chunk_index"] == 3
        assert chunk["bm25_score"] == chunk["score"] == pytest.approx(1.0)
        assert chunk["match_count"] == 1


class TestKeywordChannelEndToEnd:
    """Real SQLite channel: BM25 scores flow through aquery and lexical marking."""

    @pytest.fixture(autouse=True)
    async def _repo(self, engine):
        configure_repositories(engine)
        yield
        configure_repositories(None)

    @pytest.fixture()
    async def kb_id(self, manager) -> str:
        created = await manager.create_database(name="BM25端到端", description="t", kb_type="local")
        return created["kb_id"]

    async def test_keyword_search_returns_bm25_scores(self, manager, kb_id, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text(
            "# 转账记录\n\n张三向李四转账五十万元，款项随后汇入境外账户。\n\n"
            "第二段：李四又收到一笔汇款，来自王五。",
            encoding="utf-8",
        )
        record = await manager.add_file_record(kb_id, str(path))
        await manager.parse_file(kb_id, record["file_id"])
        result = await manager.get_instance("local").index_file(kb_id, record["file_id"])
        assert result["status"] == FileStatus.INDEXED

        kb = manager.get_instance("local")
        chunks = await kb._retrieve_keyword_chunks(
            "转账 汇款", kb_id, None, {"bm25_top_k": 10}
        )
        assert chunks
        assert all(0.0 < chunk["bm25_score"] <= 1.0 for chunk in chunks)
        assert all("match_count" in chunk for chunk in chunks)

    async def test_lexical_exact_marking_survives_bm25_ranking(self, manager, kb_id, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text(
            "# 涉案人员\n\n嫌疑人张三，手机号 13812345678，名下账户收到多笔转账。",
            encoding="utf-8",
        )
        record = await manager.add_file_record(kb_id, str(path))
        await manager.parse_file(kb_id, record["file_id"])
        await manager.get_instance("local").index_file(kb_id, record["file_id"])

        kb = manager.get_instance("local")
        chunks, error = await kb._retrieve_lexical_chunks(
            "张三 13812345678 转账", kb_id, None, {}
        )
        assert error is None
        assert chunks
        exact_hits = [chunk for chunk in chunks if chunk.get("exact_match")]
        assert exact_hits, "exact token hit must keep the deterministic marker"
        assert any("13812345678" in chunk["matched_exact_tokens"] for chunk in exact_hits)
