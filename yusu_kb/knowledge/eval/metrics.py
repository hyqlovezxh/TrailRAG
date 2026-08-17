"""
RAG评估指标计算工具
简化版：只保留Recall/F1（检索）和 LLM Judge（答案准确性）
"""

import math
import textwrap
from typing import Any

import json_repair

from yusu_kb.utils.logger import logger


class RetrievalMetrics:
    """检索评估指标计算"""

    @staticmethod
    def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        """计算Precision@K"""
        if not retrieved_ids[:k]:
            return 0.0
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_set & relevant_set) / k

    @staticmethod
    def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        """计算Recall@K"""
        if not relevant_ids:
            return 0.0
        retrieved_set = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return len(retrieved_set & relevant_set) / len(relevant_set)

    @staticmethod
    def f1_score_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        """计算F1@K"""
        precision = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
        recall = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
        """计算 nDCG@K（二值相关性）。

        recall 只衡量"是否被召回"，nDCG 额外衡量排序质量（gold 排得越靠前得分越高），
        两者结合才能评估 reranker/融合策略对 top 位次的实际影响。
        """
        if not relevant_ids or not retrieved_ids:
            return 0.0
        relevant_set = set(relevant_ids)
        dcg = sum(
            1.0 / math.log2(rank + 2) for rank, chunk_id in enumerate(retrieved_ids[:k]) if chunk_id in relevant_set
        )
        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0


class AnswerMetrics:
    """答案评估指标计算"""

    @staticmethod
    async def judge_correctness(query: str, generated_answer: str, gold_answer: str, judge_llm: Any) -> dict[str, Any]:
        """
        使用LLM判断生成的答案是否正确
        """
        if not generated_answer:
            return {"score": 0.0, "reasoning": "未生成答案"}
        if not gold_answer:
            return {"score": 0.0, "reasoning": "无参考答案"}

        prompt = textwrap.dedent(f"""你是一个公正的评判者，请评估AI生成的答案相对于标准答案的准确性。

            问题：{query}

            标准答案：
            {gold_answer}

            AI生成的答案：
            {generated_answer}

            请判断AI生成的答案是否在事实层面与标准答案一致。
            忽略措辞、标点符号或格式上的细微差异。
            只关注核心事实是否准确包含。

            请返回以下JSON格式的结果（不要包含其他文本、Markdown 或注释）：
            {{
                "score": 1.0,
                "reasoning": "简要说明判定理由"
            }}
            score 只能是 1.0 或 0.0。
            """)
        try:
            response = await judge_llm.call(prompt, stream=False)
            content = response.content.strip()

            # 尝试清理可能的 markdown 代码块
            content = content.removeprefix("```json")
            content = content.removesuffix("```")
            content = content.strip()

            result = json_repair.loads(content)
            return {"score": float(result.get("score", 0.0)), "reasoning": result.get("reasoning", "")}
        except Exception as e:  # noqa: BLE001 - 评判失败返回 0 分，与 YUSU 语义一致
            logger.error(f"LLM 评判失败: {e}")
            return {"score": 0.0, "reasoning": f"评判出错: {e!s}"}


class EvaluationMetricsCalculator:
    """综合评估指标计算器"""

    @staticmethod
    def calculate_retrieval_metrics(
        retrieved_chunks: list[dict[str, Any]], gold_chunk_ids: list[str], k_values: list[int] = [1, 3, 5, 10]  # noqa: B006 - 与 YUSU 逐字移植保持一致
    ) -> dict[str, float]:
        """计算检索指标 (Recall, F1)"""
        if not retrieved_chunks or not gold_chunk_ids:
            return {}

        # 提取 ID
        retrieved_ids = []
        for chunk in retrieved_chunks:
            chunk_id = chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id")
            retrieved_ids.append(str(chunk_id) if chunk_id else "")

        metrics = {}
        for k in k_values:
            metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(retrieved_ids, gold_chunk_ids, k)
            metrics[f"f1@{k}"] = RetrievalMetrics.f1_score_at_k(retrieved_ids, gold_chunk_ids, k)
            metrics[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(retrieved_ids, gold_chunk_ids, k)

        return metrics

    @staticmethod
    async def calculate_answer_metrics(
        query: str, generated_answer: str, gold_answer: str, judge_llm: Any = None
    ) -> dict[str, Any]:
        """计算答案指标 (LLM Judge)"""
        if not judge_llm:
            return {}

        return await AnswerMetrics.judge_correctness(query, generated_answer, gold_answer, judge_llm)

    @staticmethod
    def calculate_overall_score(
        retrieval_metrics_list: list[dict[str, float]], answer_metrics_list: list[dict[str, Any]]
    ) -> float | None:
        """综合得分：有答案准确率则用准确率，否则用 recall@10。"""
        if answer_metrics_list:
            scores = [m.get("score", 0.0) for m in answer_metrics_list]
            return sum(scores) / len(scores) if scores else None

        recalls = [m["recall@10"] for m in retrieval_metrics_list if m and "recall@10" in m]
        return sum(recalls) / len(recalls) if recalls else None


__all__ = ["AnswerMetrics", "EvaluationMetricsCalculator", "RetrievalMetrics"]
