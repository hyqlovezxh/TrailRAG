"""Task 15: RAG evaluation tests.

Covers the four eval modules (metrics / evaluator / benchmark_generation /
service) plus the evaluation repository:

- Pure metric math (recall/f1/ndcg, answer judging, overall score).
- Evaluator prompt building, answer generation with 429 retries, and
  ``evaluate_question`` against a FakeKB (mock aquery).
- Benchmark generation in vector and graph_enhanced modes with FakeChat
  injection (no external LLM).
- EvaluationService dataset CRUD / generation / run lifecycle with a real
  LocalKB (fake deterministic embedding) and injected FakeChat model factory.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from yusu_kb.knowledge.eval.benchmark_generation import (
    _validate_question_specificity,
    build_benchmark_generation_prompt,
    dump_benchmark_item,
    iter_generated_benchmark_items,
)
from yusu_kb.knowledge.eval.evaluator import (
    build_answer_prompt,
    evaluate_question,
    generate_answer_if_needed,
    normalize_query_result,
)
from yusu_kb.knowledge.eval.metrics import (
    AnswerMetrics,
    EvaluationMetricsCalculator,
    RetrievalMetrics,
)
from yusu_kb.knowledge.eval.service import EvaluationService, build_evaluation_run_name
from yusu_kb.knowledge.graphs.graph_service import GraphService
from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage
from yusu_kb.knowledge.implementations.local_kb import (
    LocalKB,
    set_default_graph_chat_model_fn,
)
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.models.chat import ChatModelError, GeneralResponse
from yusu_kb.models.embed import EmbeddingFunc, wrap_embedding_func_with_attrs
from yusu_kb.repositories import configure_repositories
from yusu_kb.repositories.evaluation_repository import EvaluationRepository
from yusu_kb.repositories.knowledge_chunk_repository import KnowledgeChunkRepository
from yusu_kb.utils.datetime_utils import utc_now_naive

FAKE_DIM = 64


def _fake_embed_one(text: str, dim: int = FAKE_DIM) -> np.ndarray:
    vec = np.zeros(dim)
    for ch in text:
        # ord() 而非 hash()：Python str hash 受 PYTHONHASHSEED 随机化影响，
        # 会导致向量召回结果跨进程不稳定（flaky）
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


@pytest.fixture(autouse=True)
def reset_graph_chat_fn():
    set_default_graph_chat_model_fn(None)
    yield
    set_default_graph_chat_model_fn(None)


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
        description="eval test",
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


async def _poll_until(condition, timeout: float = 15.0, interval: float = 0.02):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        result = await condition()
        if result:
            return result
        if loop.time() > deadline:
            raise AssertionError("poll timed out")
        await asyncio.sleep(interval)


class FakeChat:
    """Chat stub with a prompt-based responder; records every prompt."""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[str] = []

    @staticmethod
    def _extract_prompt(message) -> str:
        if isinstance(message, list):
            return message[-1]["content"] if message else ""
        return str(message)

    async def call(self, message, stream=False):
        del stream
        prompt = self._extract_prompt(message)
        self.calls.append(prompt)
        return GeneralResponse(self.responder(prompt))


class BlockingChat(FakeChat):
    """FakeChat variant that blocks until released (for task-cancel tests)."""

    def __init__(self, gate: asyncio.Event, responder):
        super().__init__(responder)
        self.gate = gate

    async def call(self, message, stream=False):
        await self.gate.wait()
        return await super().call(message, stream=stream)


class RetryableChat(FakeChat):
    """FakeChat that fails with a retryable error for the first N calls."""

    def __init__(self, responder, fail_first: int = 2):
        super().__init__(responder)
        self.fail_first = fail_first

    async def call(self, message, stream=False):
        prompt = self._extract_prompt(message)
        self.calls.append(prompt)
        if self.fail_first > 0:
            self.fail_first -= 1
            raise ChatModelError("429 rate limit exceeded")
        return GeneralResponse(self.responder(prompt))


class FakeKB:
    """Minimal KB stub: aquery returns a fixed chunk list."""

    kb_type = "local"

    def __init__(self, result_chunks: list[dict]):
        self.result_chunks = result_chunks
        self.queries: list[tuple[str, dict]] = []

    async def aquery(self, query_text: str, kb_id: str, **kwargs) -> list[dict]:
        self.queries.append((query_text, kwargs))
        return self.result_chunks


# ---------------------------------------------------------------------------
# Part 1: metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_precision_recall_f1_ndcg(self):
        retrieved = ["a", "b", "c", "d"]
        relevant = ["b", "d", "e"]
        assert RetrievalMetrics.precision_at_k(retrieved, relevant, 2) == 0.5
        assert RetrievalMetrics.recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)
        f1 = RetrievalMetrics.f1_score_at_k(retrieved, relevant, 2)
        assert f1 == pytest.approx(2 * 0.5 * (1 / 3) / (0.5 + 1 / 3))
        # b 在第 2 位、d 在第 4 位；idcg 按 min(len(relevant), k)=3 项理想排序
        # 实现折扣为 1/log2(rank+2)：rank0->1, rank1->1/log2(3), rank2->0.5, rank3->1/log2(5)
        ndcg = RetrievalMetrics.ndcg_at_k(retrieved, relevant, 4)
        dcg = 1 / math.log2(3) + 1 / math.log2(5)
        idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
        assert ndcg == pytest.approx(dcg / idcg)

    def test_calculate_retrieval_metrics_extracts_ids(self):
        chunks = [
            {"content": "x", "metadata": {"chunk_id": "c1"}},
            {"chunk_id": "c2", "content": "y"},
            {"content": "z", "metadata": {"chunk_id": "c3"}},
        ]
        metrics = EvaluationMetricsCalculator.calculate_retrieval_metrics(chunks, ["c1", "c3"])
        assert metrics["recall@1"] == 0.5
        assert metrics["recall@10"] == 1.0
        # c1 在 rank0、c3 在 rank2（rank1 是无关的 c2）：dcg=1+0.5，idcg=1+1/log2(3)
        assert metrics["ndcg@3"] == pytest.approx(1.5 / (1 + 1 / 1.5849625))
        # k=1 时命中 c1：precision@1=1.0，recall@1=0.5 → f1=2*1*0.5/1.5
        assert metrics["f1@1"] == pytest.approx(2 * 1.0 * 0.5 / (1.0 + 0.5))

    def test_calculate_retrieval_metrics_empty(self):
        assert EvaluationMetricsCalculator.calculate_retrieval_metrics([], ["c1"]) == {}
        assert EvaluationMetricsCalculator.calculate_retrieval_metrics([{"content": "x"}], []) == {}

    async def test_calculate_answer_metrics_with_judge(self):
        judge = FakeChat(lambda prompt: '{"score": 1.0, "reasoning": "一致"}')
        metrics = await EvaluationMetricsCalculator.calculate_answer_metrics(
            "q", "生成的答案", "标准答案", judge_llm=judge
        )
        assert metrics["score"] == 1.0
        assert metrics["reasoning"] == "一致"
        # judge 缺失时不计算
        assert await EvaluationMetricsCalculator.calculate_answer_metrics("q", "a", "g") == {}

    async def test_judge_correctness_empty_answer(self):
        result = await AnswerMetrics.judge_correctness("q", "", "gold", FakeChat(lambda p: ""))
        assert result["score"] == 0.0
        assert result["reasoning"] == "未生成答案"

    def test_calculate_overall_score(self):
        retrieval = [{"recall@10": 0.5}, {"recall@10": 1.0}]
        # 有答案指标 → 用准确率
        answers = [{"score": 1.0}, {"score": 0.0}]
        score = EvaluationMetricsCalculator.calculate_overall_score(retrieval, answers)
        assert score == pytest.approx(0.5)
        # 无答案指标 → 用 recall@10
        score2 = EvaluationMetricsCalculator.calculate_overall_score(retrieval, [])
        assert score2 == pytest.approx(0.75)
        assert EvaluationMetricsCalculator.calculate_overall_score([], []) is None


# ---------------------------------------------------------------------------
# Part 2: evaluator
# ---------------------------------------------------------------------------


class TestEvaluator:
    def test_build_answer_prompt_requirements(self):
        chunks = [
            {"content": "张三向李四转账五十万元。", "metadata": {"source": "a.md"}},
            {"content": "采购合同由财务部门审核通过。", "metadata": {"source": "b.md"}},
        ]
        prompt = build_answer_prompt("转账金额是多少？", chunks)
        assert "上下文信息：" in prompt
        assert "信息不足，无法回答" in prompt
        assert "【文档 1 | 来源：a.md】" in prompt
        assert "【文档 2 | 来源：b.md】" in prompt
        assert "\n\n" in prompt  # Fix-1: 真实换行
        assert "\\n\\n" not in prompt
        assert "按实体匹配记录并读取字段" in prompt

    def test_build_answer_prompt_token_budget(self, monkeypatch):
        # 固定 token 估算为字符数/4，避免 tiktoken 是否安装导致结果不确定
        import yusu_kb.knowledge.eval.evaluator as evaluator_mod

        monkeypatch.setattr(evaluator_mod, "_count_tokens", lambda text: max(1, len(text) // 4))
        chunks = [
            {"content": "A" * 500, "metadata": {"source": "a.md"}},
            {"content": "B" * 500, "metadata": {"source": "b.md"}},
        ]
        # 预算 150：文档1（125 tokens）完整保留，文档2 只能保留 20% 并带截断标记
        prompt = build_answer_prompt("q", chunks, token_budget=150)
        assert "[内容因长度限制已截断]" in prompt
        assert "A" * 500 in prompt
        assert "B" * 200 not in prompt

    def test_normalize_query_result(self):
        assert normalize_query_result({"answer": "a", "retrieved_chunks": [1, 2]}) == ("a", [1, 2])
        assert normalize_query_result([1, 2]) == ("", [1, 2])
        assert normalize_query_result(None) == ("", [])

    async def test_generate_answer_if_needed_passthrough(self):
        answer = await generate_answer_if_needed(
            query="q",
            generated_answer="已有答案",
            retrieved_chunks=[{"content": "x"}],
            retrieval_config={"answer_llm": "fake"},
            answer_llm_fn=lambda model_spec: FakeChat(lambda p: "新答案"),
        )
        assert answer == "已有答案"

    async def test_generate_answer_if_needed_skips_without_chunks_or_config(self):
        assert (
            await generate_answer_if_needed(
                query="q",
                generated_answer="",
                retrieved_chunks=[],
                retrieval_config={"answer_llm": "fake"},
                answer_llm_fn=lambda model_spec: FakeChat(lambda p: "x"),
            )
            == ""
        )
        assert (
            await generate_answer_if_needed(
                query="q",
                generated_answer="",
                retrieved_chunks=[{"content": "x"}],
                retrieval_config={},
                answer_llm_fn=lambda model_spec: FakeChat(lambda p: "x"),
            )
            == ""
        )

    async def test_generate_answer_if_needed_retries_on_429(self):
        chat = RetryableChat(lambda prompt: "重试后的答案", fail_first=2)
        answer = await generate_answer_if_needed(
            query="q",
            generated_answer="",
            retrieved_chunks=[{"content": "x"}],
            retrieval_config={"answer_llm": "fake"},
            answer_llm_fn=lambda model_spec: chat,
        )
        assert answer == "重试后的答案"
        assert len(chat.calls) == 3  # 1 次失败 + 2 次重试成功

    async def test_generate_answer_if_needed_gives_up_after_retries(self):
        chat = RetryableChat(lambda prompt: "x", fail_first=99)
        answer = await generate_answer_if_needed(
            query="q",
            generated_answer="",
            retrieved_chunks=[{"content": "x"}],
            retrieval_config={"answer_llm": "fake"},
            answer_llm_fn=lambda model_spec: chat,
        )
        assert answer == ""

    async def test_evaluate_question_full(self):
        kb = FakeKB([{"chunk_id": "c1", "content": "张三转账", "metadata": {"chunk_id": "c1", "source": "a.md"}}])
        judge = FakeChat(lambda prompt: '{"score": 1.0, "reasoning": "一致"}')
        answer_chat = FakeChat(lambda prompt: "张三向李四转账五十万元")

        result = await evaluate_question(
            kb_instance=kb,
            kb_id="kb1",
            question_data={"query": "转账金额？", "gold_chunk_ids": ["c1"], "gold_answer": "五十万元"},
            retrieval_config={"answer_llm": "fake-answer", "final_top_k": 5},
            has_gold_chunks=True,
            has_gold_answers=True,
            judge_llm=judge,
            answer_llm_fn=lambda model_spec: answer_chat,
        )
        assert result["retrieval_scores"]["recall@1"] == 1.0
        assert result["answer_scores"]["score"] == 1.0
        detail = result["detail"]
        assert detail["generated_answer"] == "张三向李四转账五十万元"
        assert detail["metrics"]["recall@1"] == 1.0
        assert detail["metrics"]["score"] == 1.0
        # aquery 收到了 retrieval_config 透传
        assert kb.queries[0][1]["final_top_k"] == 5

    async def test_evaluate_question_no_gold(self):
        kb = FakeKB([{"chunk_id": "c1", "content": "x", "metadata": {"chunk_id": "c1"}}])
        result = await evaluate_question(
            kb_instance=kb,
            kb_id="kb1",
            question_data={"query": "q", "gold_chunk_ids": [], "gold_answer": ""},
            retrieval_config={"answer_llm": "fake-answer"},
            has_gold_chunks=True,
            has_gold_answers=True,
            judge_llm=None,
            answer_llm_fn=lambda model_spec: FakeChat(lambda p: "答案"),
        )
        assert result["retrieval_scores"] == {}
        assert result["answer_scores"] == {}
        assert result["detail"]["generated_answer"] == "答案"


# ---------------------------------------------------------------------------
# Part 3: benchmark generation
# ---------------------------------------------------------------------------


class TestBenchmarkGeneration:
    def test_validate_question_specificity(self):
        # 数字 / 地点 / 时间 / 姓氏 → 通过
        assert _validate_question_specificity("2024年3月张三转账多少？")
        assert _validate_question_specificity("祥园路903室发生了什么？")
        assert _validate_question_specificity("在3月15日会议中讨论了什么？")
        assert _validate_question_specificity("张三做了什么？")
        # 泛化问句 → 拒绝
        assert not _validate_question_specificity("被讯问人叫什么名字？")
        assert not _validate_question_specificity("案件发生在哪里？")
        assert not _validate_question_specificity("")
        assert not _validate_question_specificity("   ")

    def test_build_benchmark_generation_prompt(self):
        prompt = build_benchmark_generation_prompt([("c1", "内容一"), ("c2", "内容二")])
        assert "片段ID=c1" in prompt
        assert "片段ID=c2" in prompt
        assert "gold_chunk_ids" in prompt
        assert "Fix-4 消歧" in prompt

    def test_dump_benchmark_item(self):
        line = dump_benchmark_item({"query": "q", "gold_chunk_ids": ["c1"], "gold_answer": "a"})
        assert json.loads(line) == {"query": "q", "gold_chunk_ids": ["c1"], "gold_answer": "a"}
        assert line.endswith("\n")

    async def _seed_chunks(self, kb_id: str, file_id: str, contents: list[str]) -> list[str]:
        repo = KnowledgeChunkRepository()
        chunk_ids = []
        for index, content in enumerate(contents):
            chunk_id = f"eval_chunk_{kb_id}_{index}"
            chunk_ids.append(chunk_id)
            await repo.batch_upsert(
                [
                    {
                        "chunk_id": chunk_id,
                        "file_id": file_id,
                        "kb_id": kb_id,
                        "chunk_index": index,
                        "content": content,
                        "graph_indexed": False,
                        "ent_ids": [],
                    }
                ]
            )
        return chunk_ids

    async def test_iter_generated_benchmark_items_vector(self, manager, tmp_path):
        kb_id = await create_kb(manager, "生成基准")
        await index_file(
            manager,
            kb_id,
            tmp_path,
            name="record.md",
            content=(
                "# 转账记录\n\n"
                + "2024年3月张三向李四转账五十万元用于项目投资，该笔转账经财务部审核确认。\n" * 20
                + "\n公司采购自动化生产线设备总价三十万元，采购合同由财务部门审核通过。\n" * 20
            ),
        )
        kb = get_kb(manager)

        # 每次调用生成不同的 query，避免 EVAL-3 去重后只剩 1 题（随机 anchor 可能相同）
        counter = [0]

        def responder(prompt: str) -> str:
            ids = re.findall(r"片段ID=(\S+)", prompt)
            assert ids, "prompt must list context chunk ids"
            counter[0] += 1
            return json.dumps(
                {
                    "query": f"2024年3月第{counter[0]}笔记录中张三向李四转账的金额是多少？",
                    "gold_answer": "五十万元",
                    "gold_chunk_ids": [ids[0]],
                }
            )

        progress_events: list[tuple[int, str]] = []

        async def progress_cb(progress: int, message: str | None) -> None:
            progress_events.append((progress, message or ""))

        items = []
        async for item in iter_generated_benchmark_items(
            kb_instance=kb,
            kb_id=kb_id,
            count=2,
            neighbors_count=1,
            llm_model_spec="fake-gen",
            concurrency_count=2,
            llm_fn=lambda spec: FakeChat(responder),
            progress_cb=progress_cb,
        ):
            items.append(item)

        assert len(items) == 2
        all_ids = {str(item["gold_chunk_ids"][0]) for item in items}
        assert len(all_ids) == 2  # EVAL-3: query 去重后仍生成 2 题
        for item in items:
            assert _validate_question_specificity(item["query"])
            assert item["gold_answer"] == "五十万元"
        assert progress_events and progress_events[-1][0] == 99

    async def test_iter_generated_benchmark_items_graph_enhanced(self, manager, tmp_path):
        kb_id = await create_kb(manager, "图增强基准")
        file_id = await index_file(
            manager,
            kb_id,
            tmp_path,
            name="graph.md",
            content=(
                "# 转账记录\n\n"
                + "张三向李四转账五十万元，该笔转账经财务部审核确认。\n" * 25
                + "\n李四向王五借款三十万元，约定一年后还本付息。\n" * 25
                + "\n公司采购自动化生产线设备总价三十万元，采购合同由财务部门审核通过。\n" * 25
                + "\n张三与李四共同投资成立了项目公司，注册资本一百万元。\n" * 25
            ),
        )
        kb = get_kb(manager)
        # 图索引状态写入 chunk（模拟图构建完成）
        repo = KnowledgeChunkRepository()
        chunks = await repo.list_by_kb_id(kb_id)
        assert chunks, "indexed file must produce chunks"
        entity_map = {
            "ent_zhangsan": "张三",
            "ent_lisi": "李四",
            "ent_wangwu": "王五",
        }
        for index, chunk in enumerate(chunks[:3]):
            await repo.mark_graph_indexed(chunk.chunk_id, ent_ids=[list(entity_map)[index]])

        # 手工写入已知图：张三 -> 资金转账 -> 李四
        await GraphService.evict(kb_id)
        storage = NetworkXGraphStorage(kb_id, kb.work_dir)
        for entity_id, name in entity_map.items():
            storage.upsert_entity(entity_id=entity_id, normalized_name=name, label="人物", name=name)
        for index, chunk in enumerate(chunks[:3]):
            storage.add_chunk(chunk_id=chunk.chunk_id, file_id=file_id, chunk_index=index, content_preview=chunk.content)
        storage.add_mention(entity_id="ent_zhangsan", chunk_id=chunks[0].chunk_id, file_id=file_id)
        storage.add_mention(entity_id="ent_lisi", chunk_id=chunks[1].chunk_id, file_id=file_id)
        storage.add_mention(entity_id="ent_wangwu", chunk_id=chunks[2].chunk_id, file_id=file_id)
        storage.upsert_relation(
            triple_id="triple_1",
            source_id="ent_zhangsan",
            target_id="ent_lisi",
            text="资金转账",
            rtype="资金转账",
            file_ids=[file_id],
        )
        storage.upsert_relation(
            triple_id="triple_2",
            source_id="ent_lisi",
            target_id="ent_wangwu",
            text="借款",
            rtype="借款",
            file_ids=[file_id],
        )
        storage.save()

        def responder(prompt: str) -> str:
            ids = re.findall(r"片段ID=(\S+)", prompt)
            assert ids
            return json.dumps(
                {
                    "query": "在张三向李四转账的记录中，金额是多少？",
                    "gold_answer": "五十万元",
                    "gold_chunk_ids": [ids[0]],
                }
            )

        items = []
        async for item in iter_generated_benchmark_items(
            kb_instance=kb,
            kb_id=kb_id,
            count=1,
            neighbors_count=2,
            llm_model_spec="fake-gen",
            generation_mode="graph_enhanced",
            graph_expand_top_k=1,
            concurrency_count=1,
            llm_fn=lambda spec: FakeChat(responder),
        ):
            items.append(item)
        assert len(items) == 1
        assert items[0]["gold_answer"] == "五十万元"


# ---------------------------------------------------------------------------
# Part 4: evaluation repository
# ---------------------------------------------------------------------------


class TestEvaluationRepository:
    async def test_dataset_crud_with_items(self, manager):
        kb_id = await create_kb(manager, "repo数据集")
        repo = EvaluationRepository()
        row = await repo.create_dataset_with_items(
            {"dataset_id": "dataset_1", "kb_id": kb_id, "name": "d1", "item_count": 2},
            [
                {"item_id": "item_1", "dataset_id": "dataset_1", "kb_id": kb_id, "item_index": 0, "query_text": "q1"},
                {"item_id": "item_2", "dataset_id": "dataset_1", "kb_id": kb_id, "item_index": 1, "query_text": "q2"},
            ],
        )
        assert row.dataset_id == "dataset_1"
        assert await repo.count_dataset_items("dataset_1") == 2
        assert [item.item_index for item in await repo.list_dataset_items("dataset_1", 0, 10)] == [0, 1]
        assert len(await repo.list_all_dataset_items("dataset_1")) == 2
        assert [d.dataset_id for d in await repo.list_datasets(kb_id)] == ["dataset_1"]

        await repo.update_dataset("dataset_1", {"name": "d1-renamed"})
        assert (await repo.get_dataset("dataset_1")).name == "d1-renamed"

        await repo.delete_dataset("dataset_1")
        assert await repo.get_dataset("dataset_1") is None
        assert await repo.count_dataset_items("dataset_1") == 0  # 级联清理

    async def test_run_item_upsert_and_error_filter(self, manager):
        kb_id = await create_kb(manager, "repo运行")
        repo = EvaluationRepository()
        await repo.create_dataset({"dataset_id": "dataset_1", "kb_id": kb_id, "name": "d1", "item_count": 1})
        await repo.create_run(
            {"run_id": "run_12345678", "name": "r1", "kb_id": kb_id, "dataset_id": "dataset_1", "status": "running"}
        )
        await repo.upsert_run_item(
            "run_12345678", 0, {"query_text": "q0", "is_error": False, "metrics": {"recall@1": 1.0}}
        )
        # 冲突更新：(run_id, item_index) 相同 → 覆盖
        await repo.upsert_run_item(
            "run_12345678", 0, {"query_text": "q0-updated", "is_error": True, "metrics": {"recall@1": 0.0}}
        )
        await repo.upsert_run_item("run_12345678", 1, {"query_text": "q1", "is_error": False})

        assert await repo.count_run_items("run_12345678") == 2
        items = await repo.list_run_items("run_12345678", 0, 10)
        assert items[0].query_text == "q0-updated"
        assert items[0].is_error is True

        assert await repo.count_run_items_by_error("run_12345678", is_error=True) == 1
        errors = await repo.list_run_items_by_error("run_12345678", is_error=True, offset=0, limit=10)
        assert [item.item_index for item in errors] == [0]

        await repo.delete_run("run_12345678")
        assert await repo.get_run("run_12345678") is None
        assert await repo.count_run_items("run_12345678") == 0  # 级联清理


# ---------------------------------------------------------------------------
# Part 5: evaluation service
# ---------------------------------------------------------------------------


def _make_service(manager, *, llm_factory=None) -> EvaluationService:
    if llm_factory is None:
        llm_factory = _default_llm_factory()
    return EvaluationService(select_model_fn=llm_factory, kb_manager=manager)


def _default_llm_factory():
    def factory(model_spec: str):
        if model_spec == "fake-answer":
            return FakeChat(lambda prompt: "张三向李四转账五十万元。")
        if model_spec == "fake-judge":
            return FakeChat(lambda prompt: '{"score": 1.0, "reasoning": "一致"}')
        raise AssertionError(f"unexpected model spec: {model_spec}")

    return factory


class TestEvaluationService:
    async def test_build_run_name(self):
        name = build_evaluation_run_name(hash_value="abcdef12")
        assert re.match(r"^eval-\d{8}-abcdef$", name)
        assert build_evaluation_run_name(hash_value="zz") != ""

    async def test_upload_list_detail_export_delete(self, manager):
        kb_id = await create_kb(manager, "评估数据集")
        service = _make_service(manager)
        content = (
            '{"query": "2024年3月张三向李四转账多少钱？", "gold_chunk_ids": ["c1"], "gold_answer": "五十万元"}\n'
            '{"query": "采购合同总价多少？", "gold_chunk_ids": ["c2"], "gold_answer": "三十万元"}\n'
        ).encode()
        created = await service.upload_dataset(
            kb_id, content, filename="ds.jsonl", name="", description="desc", created_by="tester"
        )
        assert created["item_count"] == 2
        assert created["has_gold_chunks"] is True
        assert created["has_gold_answers"] is True

        datasets = await service.list_datasets(kb_id)
        assert len(datasets) == 1
        dataset_id = datasets[0]["dataset_id"]

        detail = await service.get_dataset_detail(kb_id, dataset_id, page=1, page_size=1)
        assert detail["pagination"]["total_items"] == 2
        assert detail["pagination"]["total_pages"] == 2
        assert len(detail["items"]) == 1
        assert detail["items"][0]["query"].startswith("2024年3月")

        exported = await service.export_dataset_jsonl(dataset_id)
        assert exported["filename"] == "ds.jsonl"
        assert exported["content"].count("\n") == 2

        await service.delete_dataset(dataset_id)
        assert await service.list_datasets(kb_id) == []

    async def test_upload_invalid_jsonl(self, manager):
        kb_id = await create_kb(manager, "评估数据集")
        service = _make_service(manager)
        with pytest.raises(ValueError, match="JSON格式错误"):
            await service.upload_dataset(kb_id, b"{bad json\n", "bad.jsonl", "", "", "tester")
        with pytest.raises(ValueError, match="缺少必需的'query'字段"):
            await service.upload_dataset(kb_id, b'{"gold_answer": "x"}\n', "bad.jsonl", "", "", "tester")

    async def test_generate_dataset_vector_completed(self, manager, tmp_path):
        kb_id = await create_kb(manager, "生成数据集")
        await index_file(
            manager,
            kb_id,
            tmp_path,
            name="rec.md",
            content=(
                "# 记录\n\n"
                + "2024年3月张三向李四转账五十万元，该笔转账经财务部审核确认。\n" * 30
                + "\n公司采购自动化生产线设备总价三十万元，采购合同由财务部门审核通过。\n" * 30
            ),
        )

        # 每次调用生成不同的 query，避免 EVAL-3 去重后只剩 1 题
        counter = [0]

        def responder(prompt: str) -> str:
            ids = re.findall(r"片段ID=(\S+)", prompt)
            assert ids
            counter[0] += 1
            return json.dumps(
                {
                    "query": f"2024年3月第{counter[0]}笔记录中张三向李四转账的金额是多少？",
                    "gold_answer": "五十万元",
                    "gold_chunk_ids": [ids[0]],
                }
            )

        service = _make_service(manager, llm_factory=lambda spec: FakeChat(responder))
        result = await service.generate_dataset(
            kb_id=kb_id,
            name="gen-ds",
            description="",
            count=2,
            neighbors_count=1,
            concurrency_count=2,
            llm_model_spec="fake-gen",
            generation_mode="vector",
            created_by="tester",
        )
        assert result["dataset_id"].startswith("dataset_")
        assert result["task_id"].startswith("task_")

        async def dataset_done() -> dict | None:
            datasets = await service.list_datasets(kb_id)
            row = next((d for d in datasets if d["dataset_id"] == result["dataset_id"]), None)
            if row and row["build_metadata"].get("status") == "completed":
                return row
            return None

        row = await _poll_until(dataset_done)
        assert row["item_count"] == 2
        assert row["build_metadata"]["progress"] == 100
        detail = await service.get_dataset_detail(kb_id, result["dataset_id"])
        assert len(detail["items"]) == 2

    async def test_generate_dataset_graph_enhanced_requires_indexed_chunks(self, manager, tmp_path):
        kb_id = await create_kb(manager, "图增强数据集")
        await index_file(manager, kb_id, tmp_path, name="a.md", content="# 记录\n\n张三向李四转账五十万元。")
        service = _make_service(manager)
        with pytest.raises(ValueError, match="尚未完成图索引"):
            await service.generate_dataset(
                kb_id=kb_id,
                name="gen-ds",
                description="",
                count=1,
                neighbors_count=1,
                concurrency_count=1,
                llm_model_spec="fake-gen",
                generation_mode="graph_enhanced",
                created_by="tester",
            )

    async def test_generate_dataset_cancel_on_delete(self, manager, tmp_path):
        kb_id = await create_kb(manager, "取消生成")
        await index_file(
            manager,
            kb_id,
            tmp_path,
            name="rec.md",
            content="# 记录\n\n2024年3月张三向李四转账五十万元。\n\n公司采购设备总价三十万元。",
        )
        gate = asyncio.Event()
        blocked = BlockingChat(gate, lambda prompt: "{}")
        service = _make_service(manager, llm_factory=lambda spec: blocked)
        result = await service.generate_dataset(
            kb_id=kb_id,
            name="gen-ds",
            description="",
            count=2,
            neighbors_count=1,
            concurrency_count=1,
            llm_model_spec="fake-gen",
            created_by="tester",
        )
        assert service._is_task_alive(result["dataset_id"])
        # 任务仍被 gate 阻塞时删除数据集 → 任务取消、数据集删除
        await service.delete_dataset(result["dataset_id"])
        assert not service._is_task_alive(result["dataset_id"])
        assert await service.list_datasets(kb_id) == []
        gate.set()

    async def test_run_evaluation_full_flow(self, manager, tmp_path):
        kb_id = await create_kb(manager, "评估运行")
        await index_file(
            manager,
            kb_id,
            tmp_path,
            name="record.md",
            content="# 转账记录\n\n2024年3月张三向李四转账五十万元，用于项目投资。\n\n张三与李四共同投资了这家公司。",
        )
        await index_file(
            manager,
            kb_id,
            tmp_path,
            name="purchase.md",
            content="# 采购记录\n\n公司采购了一批设备，总价三十万元。\n\n采购合同由财务部门审核通过。",
        )
        chunks = await KnowledgeChunkRepository().list_by_kb_id(kb_id)
        assert len(chunks) >= 2

        service = _make_service(manager)
        content = (
            json.dumps(
                {
                    "query": "2024年3月张三向李四转账多少钱？",
                    "gold_chunk_ids": [chunks[0].chunk_id],
                    "gold_answer": "五十万元",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "query": "采购设备总价是多少？",
                    "gold_chunk_ids": [chunks[1].chunk_id],
                    "gold_answer": "三十万元",
                }
            )
            + "\n"
        ).encode("utf-8")
        dataset = await service.upload_dataset(
            kb_id, content, filename="run.jsonl", name="", description="", created_by="tester"
        )

        run_id = await service.run_evaluation(
            kb_id,
            dataset["dataset_id"],
            model_config={
                "answer_llm": "fake-answer",
                "judge_llm": "fake-judge",
                "use_reranker": False,
                "use_graph_retrieval": False,
            },
            created_by="tester",
        )
        assert re.match(r"^run_[a-f0-9]{8}$", run_id)

        async def run_done() -> dict | None:
            result = await service.get_run_results(kb_id, run_id)
            if result["status"] in ("completed", "failed"):
                return result
            return None

        result = await _poll_until(run_done)
        assert result["status"] == "completed", result
        assert result["completed_items"] == 2
        assert result["total_items"] == 2
        assert result["metrics"]["judge_status"] == "available"
        assert result["overall_score"] is not None
        assert "recall@1" in result["metrics"]
        assert "answer_correctness" in result["metrics"]
        assert result["pagination"]["total"] == 2
        assert len(result["items"]) == 2
        for item in result["items"]:
            assert item["generated_answer"] == "张三向李四转账五十万元。"
            assert any(key.startswith("recall@") for key in item["metrics"])

        # list_runs 展示运行历史
        runs = await service.list_runs(kb_id)
        assert [r["run_id"] for r in runs] == [run_id]
        assert runs[0]["status"] == "completed"
        assert runs[0]["name"].startswith("eval-")

    async def test_run_evaluation_error_only(self, manager, tmp_path):
        kb_id = await create_kb(manager, "错误过滤")
        await index_file(
            manager,
            kb_id,
            tmp_path,
            name="record.md",
            content="# 转账记录\n\n2024年3月张三向李四转账五十万元。",
        )
        service = _make_service(manager)
        # gold_chunk_ids 指向不存在的 chunk → recall 恒为 0 → is_error=True
        content = (
            '{"query": "2024年3月张三向李四转账多少钱？", "gold_chunk_ids": ["missing_chunk"], "gold_answer": "五十万元"}\n'
            '{"query": "采购设备总价是多少？", "gold_chunk_ids": ["missing_chunk"], "gold_answer": "三十万元"}\n'
        ).encode()
        dataset = await service.upload_dataset(
            kb_id, content, filename="err.jsonl", name="", description="", created_by="tester"
        )
        run_id = await service.run_evaluation(
            kb_id,
            dataset["dataset_id"],
            model_config={"answer_llm": "fake-answer", "judge_llm": "fake-judge", "use_reranker": False},
            created_by="tester",
        )

        async def run_done() -> dict | None:
            result = await service.get_run_results(kb_id, run_id)
            if result["status"] in ("completed", "failed"):
                return result
            return None

        result = await _poll_until(run_done)
        assert result["status"] == "completed"
        # error_only 下推过滤（EVAL-2）
        error_page = await service.get_run_results(kb_id, run_id, error_only=True)
        assert error_page["pagination"]["total"] == 2
        assert all(item["metrics"]["recall@1"] == 0.0 for item in error_page["items"])

    async def test_run_zombie_convergence(self, manager):
        kb_id = await create_kb(manager, "僵尸运行")
        service = _make_service(manager)
        await service.eval_repo.create_dataset(
            {"dataset_id": "dataset_zombie", "kb_id": kb_id, "name": "zombie-ds", "item_count": 1}
        )
        await service.eval_repo.create_run(
            {
                "run_id": "run_deadbeef",
                "name": "zombie",
                "kb_id": kb_id,
                "dataset_id": "dataset_zombie",
                "status": "running",
                "started_at": utc_now_naive() - timedelta(seconds=301),
                "total_items": 1,
            }
        )
        runs = await service.list_runs(kb_id)
        assert runs[0]["status"] == "failed"
        # 已收敛：DB 行状态同步更新
        row = await service.eval_repo.get_run("run_deadbeef")
        assert row.status == "failed"

    async def test_run_invalid_id_and_not_found(self, manager):
        kb_id = await create_kb(manager, "运行校验")
        service = _make_service(manager)
        with pytest.raises(ValueError, match="Invalid run_id format"):
            await service.get_run_results(kb_id, "run_bad")
        with pytest.raises(ValueError, match="Run not found"):
            await service.get_run_results(kb_id, "run_00000000")
        with pytest.raises(ValueError, match="Invalid run_id format"):
            await service.delete_run(kb_id, "run_bad")
        with pytest.raises(ValueError, match="Run not found"):
            await service.delete_run(kb_id, "run_00000000")

    async def test_delete_run(self, manager):
        kb_id = await create_kb(manager, "删除运行")
        service = _make_service(manager)
        await service.eval_repo.create_dataset(
            {"dataset_id": "dataset_del", "kb_id": kb_id, "name": "del-ds", "item_count": 1}
        )
        await service.eval_repo.create_run(
            {
                "run_id": "run_12345678",
                "name": "r1",
                "kb_id": kb_id,
                "dataset_id": "dataset_del",
                "status": "failed",
            }
        )
        await service.eval_repo.upsert_run_item("run_12345678", 0, {"query_text": "q0"})
        await service.delete_run(kb_id, "run_12345678")
        assert await service.eval_repo.get_run("run_12345678") is None
        assert await service.eval_repo.count_run_items("run_12345678") == 0
