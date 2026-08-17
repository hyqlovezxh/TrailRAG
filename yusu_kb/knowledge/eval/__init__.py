"""RAG evaluation toolkit (migrated from YUSU ``yuxi.knowledge.eval``).

Demo edition: metrics + evaluator + benchmark generation + service; the
background task system of the source project is replaced by in-process
``asyncio`` tasks tracked by :class:`EvaluationService`.
"""

from .metrics import AnswerMetrics, EvaluationMetricsCalculator, RetrievalMetrics

__all__ = ["AnswerMetrics", "EvaluationMetricsCalculator", "RetrievalMetrics"]
