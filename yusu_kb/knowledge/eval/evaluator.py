"""RAG 评估执行器（逐字移植自 YUSU ``yuxi.knowledge.eval.evaluator``）。

唯一改编：``select_model_fn`` 全局解析改为注入的 ``answer_llm_fn`` 工厂
（按 model_spec 返回 LLM 适配器），测试可注入 FakeChat 而不触网。
"""

import asyncio
from collections.abc import Callable
from typing import Any

from yusu_kb.knowledge.eval.metrics import EvaluationMetricsCalculator
from yusu_kb.utils.logger import logger

# Fix-9: 答案生成 token 预算控制，避免 context window 溢出
# 与 top_k=10 对齐，留 2 个冗余位用于答案生成
_ANSWER_MAX_DOCS = 8
_ANSWER_TOKEN_BUDGET = 12000

# 答案生成 429/瞬时错误重试参数（评估场景串行处理，固定退避即可）
_ANSWER_RETRY_MARKERS = ("429", "rate limit", "too many requests", "timed out", "timeout", "502", "503", "504")
_ANSWER_MAX_RETRIES = 3


def _is_retryable_answer_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _ANSWER_RETRY_MARKERS)


try:
    import tiktoken

    _ANSWER_ENCODING = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ANSWER_ENCODING.encode(text))
except Exception:  # noqa: BLE001 - tiktoken 可选依赖，缺失时用字符数估算
    _ANSWER_ENCODING = None

    def _count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def normalize_query_result(query_result: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(query_result, dict):
        return query_result.get("answer", ""), query_result.get("retrieved_chunks", [])
    if isinstance(query_result, list):
        return "", query_result
    return "", []


def build_answer_prompt(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    max_docs: int = _ANSWER_MAX_DOCS,
    token_budget: int = _ANSWER_TOKEN_BUDGET,
) -> str:
    """构建答案生成 prompt（Fix-1 + Fix-9）。

    - Fix-1: 修复原 `"\\n\\n".join(...)` 字面量 bug，改用真实换行 `"\n\n"`
    - Fix-1: 强化 prompt 约束：仅基于上下文直接陈述，禁止推理动机/外部知识
    - Fix-1: 每个文档加入来源标识（文件名），帮助 LLM 区分多文件上下文
    - Fix-9: max_docs 默认 5→8，与 top_k=10 对齐
    - Fix-9: token 预算控制，累计超 token_budget 时停止拼接，避免 context window 溢出
    """
    context_docs: list[str] = []
    accumulated_tokens = 0
    for idx, chunk in enumerate(retrieved_chunks[:max_docs]):
        content = chunk.get("content", "")
        if not content:
            continue
        source = chunk.get("metadata", {}).get("source", "未知来源")
        doc_tokens = _count_tokens(content)
        # Fix-9 + S1-A3: token 预算控制。超预算的 chunk 保序截断头部内容（带标记）
        # 而非整块丢弃，避免排在靠后位次的目标内容因前位 chunk 占满预算而完全丢失。
        if accumulated_tokens + doc_tokens > token_budget and context_docs:
            keep_ratio = (token_budget - accumulated_tokens) / max(doc_tokens, 1)
            chars_to_keep = int(len(content) * keep_ratio)
            if chars_to_keep > 0:
                content = content[:chars_to_keep] + "\n[内容因长度限制已截断]"
                context_docs.append(f"【文档 {idx + 1} | 来源：{source}】\n{content}")
            logger.debug(
                f"Answer prompt token budget reached: {accumulated_tokens + doc_tokens} > {token_budget}, "
                f"truncated at doc {idx + 1}/{min(len(retrieved_chunks), max_docs)}"
            )
            break
        context_docs.append(f"【文档 {idx + 1} | 来源：{source}】\n{content}")
        accumulated_tokens += doc_tokens

    # Fix-1: 使用真实换行 "\n\n" 而非字面量 "\\n\\n"
    context_text = "\n\n".join(context_docs)
    return (
        f"你是一个严格基于上下文回答问题的助手。请仅根据以下上下文信息回答用户问题。\n\n"
        f"上下文信息：\n{context_text}\n\n"
        f"用户问题：{query}\n\n"
        "回答要求：\n"
        "1. 仅基于上下文中直接陈述的事实回答，严禁使用外部知识或推理动机\n"
        "2. 如果上下文中没有相关信息，回答“信息不足，无法回答”\n"
        "3. 回答应简洁准确，可引用对应文档编号（如“根据文档1...”）\n"
        "4. 禁止猜测、推断或补充上下文中未明确说明的信息\n"
        "5. 当上下文含结构化记录（如“字段：值”键值对、表格、流水/通话记录）时，"
        "应逐条核对记录字段，将问题中的实体（姓名/编号/时间）匹配到对应记录后直接提取目标字段值；"
        "这种“按实体匹配记录并读取字段”属于直接陈述，不属于推断，不要因记录密集而回答“信息不足”\n\n"
    )


async def generate_answer_if_needed(
    *,
    query: str,
    generated_answer: str,
    retrieved_chunks: list[dict[str, Any]],
    retrieval_config: dict[str, Any],
    answer_llm_fn: Callable[..., Any] | None,
) -> str:
    """按需生成答案；``answer_llm_fn(model_spec=...)`` 返回带 ``call()`` 的 LLM 适配器。"""
    if generated_answer:
        return generated_answer
    if not retrieved_chunks or not retrieval_config.get("answer_llm") or answer_llm_fn is None:
        return ""

    prompt = build_answer_prompt(query, retrieved_chunks)
    model_spec = retrieval_config["answer_llm"]
    logger.debug(f"使用 LLM {model_spec} 生成答案...")

    for attempt in range(_ANSWER_MAX_RETRIES + 1):
        try:
            llm = answer_llm_fn(model_spec=model_spec)
            response = await llm.call(prompt, stream=False)
            generated_answer = response.content if response else ""
            logger.debug(f"LLM 生成的答案长度: {len(generated_answer) if generated_answer else 0}")
            return generated_answer
        except Exception as e:  # noqa: BLE001 - 与 YUSU 语义一致：答案生成失败返回空串
            if attempt < _ANSWER_MAX_RETRIES and _is_retryable_answer_error(e):
                wait = 2.0 * (2**attempt)  # 2s, 4s, 8s
                logger.warning(f"答案生成重试 attempt={attempt + 1}/{_ANSWER_MAX_RETRIES}, wait={wait:.1f}s, error={e}")
                await asyncio.sleep(wait)
                continue
            logger.error(f"LLM 生成答案失败（不可重试或已耗尽重试）: {e}")
            return ""
    return ""


async def evaluate_question(
    *,
    kb_instance: Any,
    kb_id: str,
    question_data: dict[str, Any],
    retrieval_config: dict[str, Any],
    has_gold_chunks: bool,
    has_gold_answers: bool,
    judge_llm: Any | None,
    answer_llm_fn: Callable[..., Any] | None,
) -> dict[str, Any]:
    query = question_data["query"]
    query_result = await kb_instance.aquery(query, kb_id, **retrieval_config)
    generated_answer, retrieved_chunks = normalize_query_result(query_result)
    generated_answer = await generate_answer_if_needed(
        query=query,
        generated_answer=generated_answer,
        retrieved_chunks=retrieved_chunks,
        retrieval_config=retrieval_config,
        answer_llm_fn=answer_llm_fn,
    )

    current_metrics = {}
    retrieval_scores = {}
    answer_scores = {}

    if has_gold_chunks and question_data.get("gold_chunk_ids"):
        retrieval_scores = EvaluationMetricsCalculator.calculate_retrieval_metrics(
            retrieved_chunks, question_data["gold_chunk_ids"]
        )
        current_metrics.update(retrieval_scores)

    if has_gold_answers and question_data.get("gold_answer"):
        if judge_llm:
            answer_scores = await EvaluationMetricsCalculator.calculate_answer_metrics(
                query=query,
                generated_answer=generated_answer,
                gold_answer=question_data["gold_answer"],
                judge_llm=judge_llm,
            )
            current_metrics.update(answer_scores)
        else:
            logger.warning("需要计算答案指标但未配置 Judge LLM")

    return {
        "detail": {
            "query_text": query,
            "gold_chunk_ids": question_data.get("gold_chunk_ids"),
            "gold_answer": question_data.get("gold_answer"),
            "generated_answer": generated_answer,
            "retrieved_chunks": retrieved_chunks,
            "metrics": current_metrics,
        },
        "retrieval_scores": retrieval_scores,
        "answer_scores": answer_scores,
    }


def aggregate_metrics(
    retrieval_metrics_list: list[dict[str, float]],
    answer_metrics_list: list[dict[str, Any]],
    *,
    include_overall_score: bool = False,
) -> tuple[dict[str, Any], float | None]:
    overall_metrics = {}

    if retrieval_metrics_list:
        keys = retrieval_metrics_list[0].keys()
        for key in keys:
            overall_metrics[key] = sum(m.get(key, 0) for m in retrieval_metrics_list) / len(retrieval_metrics_list)

    if answer_metrics_list:
        scores = [m.get("score", 0) for m in answer_metrics_list]
        overall_metrics["answer_correctness"] = sum(scores) / len(scores) if scores else 0.0

    overall_score = EvaluationMetricsCalculator.calculate_overall_score(retrieval_metrics_list, answer_metrics_list)
    if include_overall_score:
        overall_metrics["overall_score"] = overall_score

    return overall_metrics, overall_score


__all__ = [
    "aggregate_metrics",
    "build_answer_prompt",
    "evaluate_question",
    "generate_answer_if_needed",
    "normalize_query_result",
]
