"""LocalKB: SQLite + VectorStore knowledge base implementation.

Ported from YUSU ``yuxi.knowledge.implementations.milvus`` (demo edition):
- Milvus (vector + BM25) is replaced by a per-KB ``VectorStore``
  (self-contained nano-vectordb wrapper) plus SQLite ``LIKE`` keyword search;
- MinIO markdown/original reads go through the local ``LocalFileStorage``
  (inherited helpers in ``KnowledgeBase``);
- the graph enhancement path is a no-op stub; reranking degrades gracefully
  to retrieval scores when no ``rerank_func`` is configured.

Retrieval flow mirrors the Milvus implementation:
  vector / keyword / hybrid channels -> optional lexical channel (S1-A2
  deterministic exact-identifier recall) -> optional graph fusion -> source
  hydration -> optional rerank -> ``final_top_k`` truncation.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import MISSING, dataclass, field, fields
from typing import Any

from yusu_kb.knowledge.base import FileStatus, KBOperationError, KnowledgeBase
from yusu_kb.knowledge.chunking.dispatcher import chunk_markdown
from yusu_kb.knowledge.chunking.nlp import count_tokens
from yusu_kb.knowledge.factory import KnowledgeBaseFactory
from yusu_kb.knowledge.graphs.graph_service import GraphService
from yusu_kb.knowledge.graphs.keyword_extractor import KeywordExtractor
from yusu_kb.knowledge.graphs.ppr import rank_chunks_by_ppr
from yusu_kb.knowledge.graphs.round_robin_merger import merge_seed_lists_round_robin
from yusu_kb.knowledge.parser.unified import Parser
from yusu_kb.knowledge.retrieval.query_analysis import analyze_query
from yusu_kb.knowledge.utils.kb_utils import (
    resolve_processing_params,
    sanitize_processing_params,
)
from yusu_kb.models.chat import create_chat_model
from yusu_kb.models.embed import EmbeddingFunc, create_default_embedding_func
from yusu_kb.repositories.knowledge_chunk_repository import KnowledgeChunkRepository
from yusu_kb.repositories.knowledge_file_repository import KnowledgeFileRepository
from yusu_kb.storage.vector_store import VectorStore
from yusu_kb.utils.logger import logger

# Per-KB VectorStore config (chunk embedding batch).
LOCAL_CHUNK_EMBED_BATCH_SIZE = 200
# Retrieval candidates fetched when a file filter is active, since filtering
# is applied post-hoc (nano-vectordb has no query-time filter expression).
LOCAL_FILTERED_RECALL_MULTIPLIER = 3

# Deferred embedding singleton for LocalKB instances created without an
# explicit ``embedding_func`` (API layer and tests inject their own via
# ``set_default_embedding_func``).
_default_embedding_func: EmbeddingFunc | None = None
_embedding_func_cache: EmbeddingFunc | None = None

# Deferred graph chat-model fn singleton: the knowledge layer has no chat
# adapter of its own, so the API layer injects the shared adapter's ``call``
# via ``set_default_graph_chat_model_fn`` (same pattern as the embedding
# hook); a lazy env-driven adapter covers standalone / CLI usage.
_default_graph_chat_model_fn = None
_graph_chat_model_fn_cache = None


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def set_default_embedding_func(func: EmbeddingFunc | None) -> None:
    """Set the process-wide default embedding function (API startup / tests)."""
    global _default_embedding_func, _embedding_func_cache
    if func is not None and not isinstance(func, EmbeddingFunc):
        raise KBOperationError("set_default_embedding_func requires an EmbeddingFunc instance")
    _default_embedding_func = func
    _embedding_func_cache = None


def get_or_create_embedding_func() -> EmbeddingFunc:
    """Return the process-wide embedding function, resolving it on first use."""
    global _embedding_func_cache
    if _embedding_func_cache is None:
        if _default_embedding_func is not None:
            _embedding_func_cache = _default_embedding_func
        else:
            _embedding_func_cache = create_embedding_func()
    return _embedding_func_cache


def set_default_graph_chat_model_fn(fn) -> None:
    """Set the process-wide graph chat fn (API startup / tests)."""
    global _default_graph_chat_model_fn, _graph_chat_model_fn_cache
    _default_graph_chat_model_fn = fn
    _graph_chat_model_fn_cache = None


def get_or_create_graph_chat_model_fn():
    """Return the process-wide graph chat fn, resolving it on first use.

    The fn contract matches the graph extractor: an async
    ``(messages: list[dict]) -> GeneralResponse`` callable.
    """
    global _graph_chat_model_fn_cache
    if _graph_chat_model_fn_cache is None:
        if _default_graph_chat_model_fn is not None:
            _graph_chat_model_fn_cache = _default_graph_chat_model_fn
        else:
            _graph_chat_model_fn_cache = create_chat_model().call
    return _graph_chat_model_fn_cache


def create_embedding_func(
    *,
    provider_func=None,
    embedding_dim: int | None = None,
    max_token_size: int | None = None,
    model_name: str | None = None,
) -> EmbeddingFunc:
    """Build an ``EmbeddingFunc`` from the YUSU env contract.

    Env vars: ``YUSU_EMBED_BASE_URL`` (fallback ``YUSU_LLM_BASE_URL``),
    ``YUSU_EMBED_API_KEY``, ``YUSU_EMBED_MODEL``, ``YUSU_EMBED_DIM``
    (optional; probed at first use when absent).

    ``provider_func`` is a test seam: when given, binding resolution is
    skipped and the provided callable is wrapped directly (dimension must be
    resolvable from ``embedding_dim`` or the callable's own attributes).
    """
    return create_default_embedding_func(
        provider_func=provider_func,
        embedding_dim=embedding_dim,
        max_token_size=max_token_size,
        model_name=model_name,
    )


@dataclass(kw_only=True)
class LocalRetrievalConfig:
    """Retrieval options for LocalKB (schema defaults, ported from
    ``MilvusRetrievalConfig``; Milvus-specific fields are dropped)."""

    search_mode: str = field(
        default="vector",
        metadata={
            "label": "检索模式",
            "type": "select",
            "options": [
                {"value": "vector", "label": "向量检索", "description": "仅使用向量相似度检索"},
                {"value": "keyword", "label": "关键词全文检索", "description": "仅使用 SQLite 关键词检索"},
                {"value": "hybrid", "label": "混合检索", "description": "向量检索与关键词检索融合"},
            ],
            "description": "选择检索模式",
        },
    )
    final_top_k: int = field(
        default=10,
        metadata={
            "label": "最终返回 Chunk 数",
            "type": "number",
            "min": 1,
            "max": 100,
            "description": "重排序后返回给前端的文档数量",
        },
    )
    similarity_threshold: float = field(
        default=0.0,
        metadata={
            "label": "相似度阈值（0-1）",
            "type": "number",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "description": "过滤相似度低于此值的结果",
        },
    )
    bm25_top_k: int = field(
        default=50,
        metadata={
            "label": "关键词召回数量",
            "type": "number",
            "min": 1,
            "max": 200,
            "description": "关键词全文检索和混合检索中的关键词候选数量",
        },
    )
    vector_weight: float = field(
        default=0.7,
        metadata={
            "label": "向量检索权重",
            "type": "number",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "description": "混合检索中向量召回结果的融合权重",
        },
    )
    bm25_weight: float = field(
        default=0.3,
        metadata={
            "label": "关键词检索权重",
            "type": "number",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "description": "混合检索中关键词召回结果的融合权重",
        },
    )
    include_distances: bool = field(
        default=True,
        metadata={"label": "显示相似度", "type": "boolean", "description": "在结果中显示相似度分数"},
    )
    use_graph_retrieval: bool = field(
        default=True,
        metadata={
            "label": "启用图检索",
            "type": "boolean",
            "description": "是否启用实体和三元组扩散检索（默认启用，未构建图谱时自动降级为纯向量检索）",
        },
    )
    graph_entity_top_k: int = field(
        default=15,
        metadata={
            "label": "图实体召回数量",
            "type": "number",
            "min": 1,
            "max": 100,
            "depend_on": ("use_graph_retrieval", True),
            "description": "通过 Query 召回的实体数量",
        },
    )
    graph_triple_top_k: int = field(
        default=15,
        metadata={
            "label": "图三元组召回数量",
            "type": "number",
            "min": 1,
            "max": 100,
            "depend_on": ("use_graph_retrieval", True),
            "description": "通过 Query 召回的三元组数量",
        },
    )
    graph_top_k: int = field(
        default=10,
        metadata={
            "label": "图召回 Chunk 数",
            "type": "number",
            "min": 1,
            "max": 200,
            "depend_on": ("use_graph_retrieval", True),
            "description": "PPR 后从图谱路径召回的 Chunk 数量",
        },
    )
    graph_weight: float = field(
        default=0.5,
        metadata={
            "label": "图检索融合权重",
            "type": "number",
            "min": 0.0,
            "max": 2.0,
            "step": 0.1,
            "depend_on": ("use_graph_retrieval", True),
            "description": "排名融合时图检索结果的权重（图应增强而非主导）",
        },
    )
    graph_rrf_k: int = field(
        default=60,
        metadata={
            "label": "RRF 融合参数 K",
            "type": "number",
            "min": 1,
            "max": 200,
            "depend_on": ("use_graph_retrieval", True),
            "description": "RRF 排名融合的 K 参数，值越大图检索权重相对越低",
        },
    )
    ppr_damping: float = field(
        default=0.85,
        metadata={
            "label": "PPR 阻尼系数",
            "type": "number",
            "min": 0.1,
            "max": 0.99,
            "step": 0.01,
            "depend_on": ("use_graph_retrieval", True),
            "description": "Personalized PageRank 的阻尼系数",
        },
    )
    graph_query_max_depth: int = field(
        default=3,
        metadata={
            "label": "图查询多跳深度",
            "type": "number",
            "min": 1,
            "max": 5,
            "depend_on": ("use_graph_retrieval", True),
            "description": "PPR 子图扩展的跳数（1-5）",
        },
    )
    graph_ppr_directed: bool = field(
        default=False,
        metadata={
            "label": "PPR 有向图",
            "type": "boolean",
            "depend_on": ("use_graph_retrieval", True),
            "description": "是否按有向图计算 PPR（默认无向）",
        },
    )
    graph_max_nodes: int = field(
        default=10000,
        metadata={
            "label": "图检索最大节点数",
            "type": "number",
            "min": 100,
            "max": 50000,
            "depend_on": ("use_graph_retrieval", True),
            "description": "2-hop 扩散子图最多读取的节点数量",
        },
    )
    chunk_count_weight: float = field(
        default=0.2,
        metadata={
            "label": "chunk 计数权重",
            "type": "number",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "depend_on": ("use_graph_retrieval", True),
            "description": "PPR 边权中 chunk 出现计数的权重；设为 0 禁用此信号",
        },
    )
    keyword_extractor_enabled: bool = field(
        default=True,
        metadata={
            "label": "启用 HL/LL 关键词抽取",
            "type": "boolean",
            "depend_on": ("use_graph_retrieval", True),
            "description": "关闭后退化为 query embedding 单源检索",
        },
    )
    use_reranker: bool = field(
        default=True,
        metadata={
            "label": "启用重排序",
            "type": "boolean",
            "description": "是否使用精排模型对检索结果进行重排序（默认启用，未配置模型时自动回退为检索分数排序）",
        },
    )
    reranker_model: str = field(
        default="",
        metadata={
            "label": "重排序模型",
            "type": "select",
            "depend_on": ("use_reranker", True),
            "description": "选择用于本次查询的重排序模型",
            "options": [],
        },
    )
    recall_top_k: int = field(
        default=50,
        metadata={
            "label": "召回数量",
            "type": "number",
            "min": 10,
            "max": 200,
            "depend_on": ("use_reranker", True),
            "description": "向量检索或混合检索保留的候选数量（启用重排序时有效）",
        },
    )
    min_rerank_score: float = field(
        default=0.0,
        metadata={
            "label": "最低 rerank 分数",
            "type": "number",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "depend_on": ("use_reranker", True),
            "description": "低于此分数的 chunk 将被过滤；过滤后为空时跳过过滤保证非空",
        },
    )
    lexical_channel_enabled: bool = field(
        default=False,
        metadata={
            "label": "启用精确值词法通道",
            "type": "boolean",
            "description": "对查询中的编号/手机号/证件号等精确标识符做确定性词法召回并逐字校验，"
            "命中 chunk 获得重排序分数下限保障；关闭后完全回退原检索行为",
        },
    )
    lexical_top_k: int = field(
        default=20,
        metadata={
            "label": "词法通道召回数量",
            "type": "number",
            "min": 1,
            "max": 100,
            "depend_on": ("lexical_channel_enabled", True),
            "description": "词法通道关键词候选召回的 chunk 数量",
        },
    )
    exact_score_floor: float = field(
        default=0.85,
        metadata={
            "label": "精确命中分数下限",
            "type": "number",
            "min": 0.0,
            "max": 1.0,
            "step": 0.05,
            "depend_on": ("lexical_channel_enabled", True),
            "description": "精确 token 逐字命中的 chunk 在重排序后获得的分数下限，保证精确值查询的目标 "
            "chunk 进入 top-1；设为 0 禁用该保障",
        },
    )


def _retrieval_config_options() -> list[dict[str, Any]]:
    """Generate the option list from ``LocalRetrievalConfig`` field metadata."""
    options = []
    for config_field in fields(LocalRetrievalConfig):
        metadata = dict(config_field.metadata)
        default = None if config_field.default is MISSING else config_field.default
        options.append({"key": config_field.name, "default": default, **metadata})
    return options


@KnowledgeBaseFactory.register("local")
class LocalKB(KnowledgeBase):
    """基于 NanoVectorDB + SQLite 的轻量本地向量库"""

    kb_type = "local"
    name = "Local"
    description = "轻量本地向量库（NanoVectorDB + SQLite），无需外部服务"

    @classmethod
    def normalize_additional_params(cls, additional_params: dict | None) -> dict:
        """规范化 additional_params，补充 auto_build_graph 默认值。

        与前端保持一致：未显式设置 auto_build_graph 时默认 True，
        确保历史 KB 和 API 创建的 KB 也能自动触发图谱构建（Task 4 接入）。
        用户显式设置为 False 时尊重其选择。
        """
        params = super().normalize_additional_params(additional_params)
        if "auto_build_graph" not in params:
            params["auto_build_graph"] = True
        return params

    def __init__(self, work_dir: str, **kwargs):
        """
        初始化 Local 知识库

        Args:
            work_dir: 工作目录
            **kwargs: 其他配置参数（embedding_func / rerank_func）
        """
        super().__init__(work_dir)

        self.embedding_func = kwargs.get("embedding_func")
        if self.embedding_func is not None and not isinstance(self.embedding_func, EmbeddingFunc):
            raise KBOperationError(
                "embedding_func must be an EmbeddingFunc "
                "(use wrap_embedding_func_with_attrs, see yusu_kb.models.embed)"
            )

        # Optional async reranker: ``await rerank_func(query, documents) -> list[float]``
        # returning [0, 1] scores. When absent, retrieval scores are used.
        self.rerank_func = kwargs.get("rerank_func")

        # 存储集合映射 {kb_id: VectorStore}
        self._vector_stores: dict[str, VectorStore] = {}
        # per-kb 初始化锁，防止并发首访重复创建向量存储
        self._vector_store_locks: dict[str, asyncio.Lock] = {}

        logger.info("LocalKB initialized")

    def _vector_store_lock(self, kb_id: str) -> asyncio.Lock:
        """获取 kb_id 对应的初始化锁（并发安全，按需扩张）"""
        lock = self._vector_store_locks.get(kb_id)
        if lock is None:
            lock = self._vector_store_locks.setdefault(kb_id, asyncio.Lock())
        return lock

    def _get_embedding_function(self, embedding_model_spec: str | None = None):
        """获取 embedding 函数。

        优先使用实例注入的 embedding_func（API 层 / 测试注入）；否则惰性
        创建进程级默认 embedding 函数（首次实际使用索引/查询时才解析配置，
        避免仅实例化/枚举知识库时加载模型 provider）。
        """
        del embedding_model_spec
        if self.embedding_func is not None:
            return self.embedding_func
        self.embedding_func = get_or_create_embedding_func()
        return self.embedding_func

    async def _create_kb_instance(self, kb_id: str, kb_config: dict) -> Any:
        """创建 kb 的 VectorStore 实例。

        working_dir 为 KB 根目录，workspace=kb_id 使落盘路径为
        ``<work_dir>/<kb_id>/vdb_<kb_id>.json``（与 create_database 创建的
        目录一致，随知识库删除一并清理）。
        """
        del kb_config
        if kb_id not in self.databases_meta:
            raise ValueError(f"Database {kb_id} not found")

        storage = VectorStore(
            namespace=kb_id,
            working_dir=self.work_dir,
            embedding_func=self._get_embedding_function(),
            workspace=kb_id,
            embedding_batch_num=LOCAL_CHUNK_EMBED_BATCH_SIZE,
            meta_fields={"content", "file_id", "chunk_index"},
            # 阈值过滤在 aquery 侧按 similarity_threshold 逐条执行（与 Milvus
            # 实现一致），这里设为 0.0 关闭存储层的隐式过滤。
            cosine_better_than_threshold=0.0,
        )
        return storage

    async def _initialize_kb_instance(self, instance: Any) -> None:
        """初始化向量存储（解析 embedding 维度并加载数据文件）。"""
        await instance.initialize()

    async def _get_vector_storage(self, kb_id: str):
        """获取或创建 kb 的 VectorStore。

        per-kb 锁消除并发首访竞态；失败返回 None 并记录错误（与 Milvus
        实现的 collection 处理保持一致）。
        """
        if kb_id in self._vector_stores:
            return self._vector_stores[kb_id]

        if kb_id not in self.databases_meta:
            return None

        async with self._vector_store_lock(kb_id):
            # 双重检查：并发首访中先完成的请求已创建实例
            if kb_id in self._vector_stores:
                return self._vector_stores[kb_id]

            try:
                instance = await self._create_kb_instance(kb_id, {})
                await self._initialize_kb_instance(instance)
                self._vector_stores[kb_id] = instance
                return instance
            except Exception as e:  # noqa: BLE001 - 创建失败须降级为 None（与 Milvus collection 处理一致）
                logger.error(f"Failed to create vector storage for {kb_id}: {e}")
                return None

    def _split_text_into_chunks(self, text: str, file_id: str, filename: str, params: dict) -> list[dict]:
        """将文本分割成块"""
        return chunk_markdown(text, file_id, filename, params)

    def _calculate_chunk_stats(self, chunks: list[dict]) -> dict[str, int]:
        return {
            "chunk_count": len(chunks),
            "token_count": sum(count_tokens(chunk["content"]) for chunk in chunks),
        }

    def _build_chunk_pg_records(self, kb_id: str, chunks: list[dict]) -> list[dict[str, Any]]:
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "file_id": chunk["file_id"],
                "kb_id": kb_id,
                "chunk_index": chunk["chunk_index"],
                "content": chunk["content"],
                "start_char_pos": chunk.get("start_char_pos"),
                "end_char_pos": chunk.get("end_char_pos"),
                "start_token_pos": chunk.get("start_token_pos"),
                "end_token_pos": chunk.get("end_token_pos"),
                "graph_indexed": bool(chunk.get("graph_indexed", False)),
                "ent_ids": chunk.get("ent_ids"),
                "tags": chunk.get("tags"),
                "extraction_result": chunk.get("extraction_result"),
            }
            for chunk in chunks
        ]

    def _build_chunk_vector_data(self, chunks: list[dict]) -> dict[str, dict[str, Any]]:
        return {
            chunk["chunk_id"]: {
                "content": chunk["content"],
                "file_id": chunk["file_id"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        }

    async def _flush_vectors(self, storage: VectorStore, data: dict[str, dict[str, Any]]) -> None:
        """缓冲 upsert + 提交（嵌入 + 落盘）。"""
        if not data:
            return
        await storage.upsert(data)
        await storage.index_done_callback()

    async def _insert_chunks_to_stores(
        self, kb_id: str, file_id: str, storage: VectorStore, chunks: list[dict]
    ) -> None:
        if not chunks:
            return

        chunk_repo = KnowledgeChunkRepository()
        pg_task = chunk_repo.batch_upsert(self._build_chunk_pg_records(kb_id, chunks))
        vector_task = self._flush_vectors(storage, self._build_chunk_vector_data(chunks))
        results = await asyncio.gather(pg_task, vector_task, return_exceptions=True)
        errors = [result for result in results if isinstance(result, Exception)]
        if not errors:
            return

        logger.error(
            f"Chunk double-write failed for file {file_id}, errors: {errors}. "
            f"Data may be partially written (upsert 已覆盖旧数据，不执行删除回滚以避免数据彻底丢失). "
            f"Manual intervention required or re-index the file."
        )
        raise errors[0]

    async def _delete_file_chunks_except(
        self, kb_id: str, file_id: str, storage: VectorStore, keep_chunk_ids: set[str]
    ) -> None:
        """删除文件中不在 keep_chunk_ids 集合中的旧 chunk（SQLite + 向量库），用于重新索引时清理孤儿数据。"""
        del kb_id
        chunk_repo = KnowledgeChunkRepository()
        orphan_ids = await chunk_repo.delete_by_file_id_except(file_id, keep_chunk_ids)
        if orphan_ids:
            try:
                await self._flush_vectors(storage, {})
                await storage.delete(orphan_ids)
                await storage.index_done_callback()
            except Exception as e:  # noqa: BLE001 - 删除失败须告警而非中断（向量为可重建缓存）
                # 不再静默吞异常：SQLite 已删但向量删除失败会产生 chunk 级孤儿向量，
                # 这些孤儿仍可被检索命中（返回过时内容）。
                logger.error(
                    f"Failed to delete {len(orphan_ids)} orphan chunks from vector store for file {file_id}: {e}. "
                    f"Orphan chunk_ids (may cause stale results): {orphan_ids}"
                )

    async def _delete_file_graph_only(self, kb_id: str, file_id: str) -> None:
        """仅删除文件的图谱数据,保留 chunks 数据(用于重索引时避免数据丢失)。"""
        try:
            service = self._get_graph_service(kb_id)
            await service.delete_file_graph(kb_id, file_id)
        except Exception as exc:  # noqa: BLE001 - 图谱删除失败仅记日志，不阻塞重索引/删除
            logger.warning(f"Failed to delete graph data for file {file_id} in kb {kb_id}: {exc}")

    def _get_graph_service(self, kb_id: str) -> GraphService:
        """Resolve the per-KB GraphService singleton with local dependencies."""
        return GraphService.get_instance(
            kb_id=kb_id,
            work_dir=self.work_dir,
            embed_func=self._get_embedding_function(),
            chat_model_fn=get_or_create_graph_chat_model_fn(),
        )

    async def _maybe_auto_build_graph(self, kb_id: str) -> None:
        """Kick off the background graph build after indexing when enabled.

        ``build_pending_chunks`` only starts the background task (no
        blocking), dedupes against a running build itself, and any error here
        (e.g. the graph config is not locked yet) is logged, never surfaced.
        """
        params = self.databases_meta.get(kb_id, {}).get("metadata") or {}
        if not params.get("auto_build_graph"):
            return
        try:
            service = self._get_graph_service(kb_id)
            await service.build_pending_chunks(kb_id, batch_size=100)
        except Exception as exc:  # noqa: BLE001 - 自动构建失败仅记日志，不阻塞索引返回
            logger.warning(f"Auto graph build skipped for kb {kb_id}: {exc}")

    async def index_file(
        self, kb_id: str, file_id: str, operator_id: str | None = None, params: dict | None = None
    ) -> dict:
        """
        Index parsed file (Status: INDEXING -> INDEXED/ERROR_INDEXING)

        Args:
            kb_id: Database ID
            file_id: File ID
            operator_id: ID of the user performing the operation
            params: Override processing params to apply during indexing (merged on top of stored params)

        Returns:
            Updated file metadata
        """
        if kb_id not in self.databases_meta:
            raise ValueError(f"Database {kb_id} not found")

        # Get/Create vector storage
        storage = await self._get_vector_storage(kb_id)
        if not storage:
            raise ValueError(f"Failed to get vector storage for {kb_id}")

        embedding_model_spec = self.databases_meta[kb_id].get("embedding_model_spec")
        embedding_function = self._get_embedding_function(embedding_model_spec)

        file_meta = await self._load_file_meta(kb_id, file_id)
        allowed_statuses = {
            FileStatus.PARSED,
            FileStatus.ERROR_INDEXING,
            FileStatus.INDEXED,
            "done",
        }
        params = resolve_processing_params(
            kb_additional_params=self.databases_meta.get(kb_id, {}).get("metadata"),
            file_processing_params=file_meta.get("processing_params"),
            request_params=params,
        )

        claim_data = {
            "status": FileStatus.INDEXING,
            "processing_params": sanitize_processing_params(params),
            "error_message": None,
        }
        if operator_id:
            claim_data["updated_by"] = operator_id

        claimed_record = await KnowledgeFileRepository().update_fields_if_status(
            kb_id=kb_id,
            file_id=file_id,
            allowed_statuses=allowed_statuses,
            data=claim_data,
        )
        if claimed_record is None:
            current_meta = await self._load_file_meta(kb_id, file_id)
            current_status = current_meta.get("status")
            raise ValueError(
                f"Cannot index file with status '{current_status}'. "
                f"File must be parsed first (status should be one of: {', '.join(allowed_statuses)})"
            )

        file_meta = self._file_record_to_meta(claimed_record)
        if not file_meta.get("markdown_file"):
            await self._mark_file_unparsed(kb_id, file_id, operator_id)
            raise ValueError("File has not been parsed yet (no markdown_file)")

        logger.debug(f"[index_file] file_id={file_id}, processing_params={params}")

        try:
            # Read markdown
            markdown_content = await self._read_markdown(file_meta["markdown_file"])
            filename = file_meta.get("filename")

            # Split
            chunks = await asyncio.to_thread(self._split_text_into_chunks, markdown_content, file_id, filename, params)
            logger.info(
                f"Split {filename} into {len(chunks)} chunks with params: "
                f"chunk_preset_id={params.get('chunk_preset_id')}, "
                f"chunk_parser_config={params.get('chunk_parser_config')}"
            )

            chunk_stats = self._calculate_chunk_stats(chunks)

            # 先 upsert 新数据(失败时旧数据仍可用,避免数据丢失)
            if chunks:
                await self._embed_and_store_chunks(kb_id, file_id, storage, chunks, embedding_function)

            # 新数据写入成功后,清理旧 chunks(本次未覆盖的孤儿)
            new_chunk_ids = {chunk["chunk_id"] for chunk in chunks}
            await self._delete_file_chunks_except(kb_id, file_id, storage, new_chunk_ids)
            await self._delete_file_graph_only(kb_id, file_id)

            logger.info(f"Indexed file {file_id} into vector store")

            # Update status
            update_data = {"status": FileStatus.INDEXED, "error_message": None, **chunk_stats}
            if operator_id:
                update_data["updated_by"] = operator_id
            updated_record = await KnowledgeFileRepository().update_fields(
                file_id=file_id,
                kb_id=kb_id,
                data=update_data,
            )
            result = (
                self._file_record_to_meta(updated_record)
                if updated_record is not None
                else {
                    **file_meta,
                    **chunk_stats,
                    "status": FileStatus.INDEXED,
                    "error": None,
                }
            )

            await self.refresh_database_stats(kb_id)
            await self._maybe_auto_build_graph(kb_id)
            return result

        except Exception as e:
            logger.error(f"Indexing failed for {file_id}: {e}")
            update_data = {"status": FileStatus.ERROR_INDEXING, "error_message": str(e)}
            if operator_id:
                update_data["updated_by"] = operator_id
            await KnowledgeFileRepository().update_fields(file_id=file_id, kb_id=kb_id, data=update_data)
            raise

    async def _embed_and_store_chunks(
        self,
        kb_id: str,
        file_id: str,
        storage: VectorStore,
        chunks: list[dict],
        embedding_function,
        *,
        chunk_batch_size: int = LOCAL_CHUNK_EMBED_BATCH_SIZE,
    ) -> None:
        """对 chunks 进行分批嵌入并存储到向量库和 SQLite"""
        del embedding_function
        if not chunks:
            return

        chunk_batch_size = max(int(chunk_batch_size), 1)
        for start in range(0, len(chunks), chunk_batch_size):
            batch_chunks = chunks[start : start + chunk_batch_size]
            await self._insert_chunks_to_stores(kb_id, file_id, storage, batch_chunks)

    async def update_content(self, kb_id: str, file_ids: list[str], params: dict | None = None) -> list[dict]:
        """更新内容 - 根据file_ids重新解析文件并更新向量库"""
        if kb_id not in self.databases_meta:
            raise ValueError(f"Database {kb_id} not found")

        storage = await self._get_vector_storage(kb_id)
        if not storage:
            raise ValueError(f"Failed to get vector storage for {kb_id}")

        embedding_model_spec = self.databases_meta[kb_id].get("embedding_model_spec")
        embedding_function = self._get_embedding_function(embedding_model_spec)

        # 处理默认参数
        if params is None:
            params = {}
        processed_items_info = []

        for file_id in file_ids:
            try:
                file_meta = await self._load_file_meta(kb_id, file_id)
            except ValueError:
                logger.warning(f"File {file_id} not found in metadata, skipping")
                continue

            file_path = file_meta.get("path")
            filename = file_meta.get("filename")

            if not file_path:
                logger.warning(f"File path not found for {file_id}, skipping")
                continue

            try:
                # 更新状态为处理中
                resolved_params = resolve_processing_params(
                    kb_additional_params=self.databases_meta.get(kb_id, {}).get("metadata"),
                    file_processing_params=file_meta.get("processing_params"),
                    request_params=params,
                )
                file_meta["processing_params"] = resolved_params
                file_meta["status"] = FileStatus.INDEXING
                # 使用 CAS 抢占（update_fields_if_status），与 index_file 对齐，
                # 防止并发重索引竞态导致双写 chunk/向量。
                claimed_record = await KnowledgeFileRepository().update_fields_if_status(
                    kb_id=kb_id,
                    file_id=file_id,
                    allowed_statuses={
                        FileStatus.PARSED,
                        FileStatus.ERROR_INDEXING,
                        FileStatus.INDEXED,
                        "done",
                    },
                    data={
                        "status": FileStatus.INDEXING,
                        "processing_params": sanitize_processing_params(resolved_params),
                    },
                )
                if claimed_record is None:
                    current_status = file_meta.get("status")
                    logger.warning(f"Skip update_content for {file_id}: status '{current_status}' not in allowed set")
                    continue

                # 重新解析文件为 markdown
                images_dir = str(self.file_storage.parsed_dir / "images")
                markdown_content = await Parser.aparse(source=file_path, params=resolved_params, images_dir=images_dir)

                # 与 parse_file 保持同一持久化契约——重解析的 markdown
                # 必须写回本地文件系统并更新 markdown_file，否则后续重索引会读取旧解析
                # 产物，把内容静默回退
                markdown_file_path = await self._save_markdown_file(kb_id, file_id, markdown_content)

                # 重新生成 chunks
                chunks = await asyncio.to_thread(
                    self._split_text_into_chunks, markdown_content, file_id, filename, resolved_params
                )
                logger.info(f"Split {filename} into {len(chunks)} chunks")
                chunk_stats = self._calculate_chunk_stats(chunks)

                # 先 upsert 新数据(失败时旧数据仍可用,避免数据丢失)
                if chunks:
                    await self._embed_and_store_chunks(kb_id, file_id, storage, chunks, embedding_function)

                # 新数据写入成功后,清理旧 chunks(本次未覆盖的孤儿)
                new_chunk_ids = {chunk["chunk_id"] for chunk in chunks}
                await self._delete_file_chunks_except(kb_id, file_id, storage, new_chunk_ids)
                await self._delete_file_graph_only(kb_id, file_id)

                logger.info(f"Updated file {file_path} in vector store. Done.")

                # 更新元数据状态
                file_meta["status"] = FileStatus.INDEXED
                file_meta["markdown_file"] = markdown_file_path
                file_meta.update(chunk_stats)
                await KnowledgeFileRepository().update_fields(
                    file_id=file_id,
                    kb_id=kb_id,
                    data={
                        "status": FileStatus.INDEXED,
                        "error_message": None,
                        "markdown_file": markdown_file_path,
                        **chunk_stats,
                    },
                )
                await self.refresh_database_stats(kb_id)
                await self._maybe_auto_build_graph(kb_id)

                # 返回更新后的文件信息
                updated_file_meta = file_meta.copy()
                updated_file_meta["status"] = FileStatus.INDEXED
                updated_file_meta.update(chunk_stats)
                updated_file_meta["file_id"] = file_id
                processed_items_info.append(updated_file_meta)

            except Exception as e:  # noqa: BLE001 - 单文件失败须记录并继续处理其余文件
                logger.error(f"更新file {file_path} 失败: {e}")
                await KnowledgeFileRepository().update_fields(
                    file_id=file_id,
                    kb_id=kb_id,
                    data={"status": FileStatus.ERROR_INDEXING, "error_message": str(e)},
                )

                # 返回失败的文件信息
                failed_file_meta = file_meta.copy()
                failed_file_meta["status"] = FileStatus.ERROR_INDEXING
                failed_file_meta["error"] = str(e)
                failed_file_meta["file_id"] = file_id
                processed_items_info.append(failed_file_meta)

        return processed_items_info

    def _build_chunk_from_vector(self, dp: dict[str, Any], include_distances: bool) -> dict:
        """将 NanoVectorDB 查询结果转成知识库统一返回的 Chunk 结构。

        NanoVectorDB 的 ``distance``（cosine 相似度，越高越相似）与 Milvus
        COSINE 度量的 hit.distance 语义一致。
        """
        similarity = float(dp.get("distance") or 0.0)
        metadata = {
            "source": "未知来源",
            "chunk_id": dp.get("id"),
            "file_id": dp.get("file_id"),
            "chunk_index": dp.get("chunk_index"),
        }
        chunk = {"content": dp.get("content", ""), "metadata": metadata, "score": similarity}
        if include_distances:
            chunk["distance"] = similarity
        return chunk

    def _build_chunk_from_record(self, chunk: Any, score: float, score_field: str | None = None) -> dict:
        metadata = {
            "source": "未知来源",
            "chunk_id": chunk.chunk_id,
            "file_id": chunk.file_id,
            "chunk_index": chunk.chunk_index,
        }
        result = {"content": chunk.content, "metadata": metadata, "score": float(score or 0.0)}
        if score_field:
            result[score_field] = float(score or 0.0)
        return result

    async def _resolve_file_filter(self, kb_id: str, merged_kwargs: dict[str, Any]) -> set[str] | None:
        """解析 file_ids / file_name 过滤为允许的 file_id 集合（None 表示不过滤）。"""
        file_ids = merged_kwargs.get("file_ids")
        if file_ids:
            allowed = {fid for fid in file_ids if fid}
            return allowed or set()
        file_name = merged_kwargs.get("file_name")
        if file_name:
            matched = await KnowledgeFileRepository().list_file_ids_by_filename_contains(
                kb_id=kb_id, filename_pattern=file_name
            )
            return set(matched)
        return None

    def _rank_keyword_chunks(self, records: list[Any], terms: list[str]) -> list[dict]:
        """按词项命中数降序（并列时按 chunk_index 升序）为关键词命中排序。"""
        chunks = []
        lowered_terms = [term.lower() for term in terms]
        for rec in records:
            content_lower = (rec.content or "").lower()
            match_count = sum(1 for term in lowered_terms if term in content_lower)
            if match_count == 0:
                continue
            score = float(match_count) / max(len(terms), 1)
            chunk = self._build_chunk_from_record(rec, score, score_field="bm25_score")
            chunk["match_count"] = match_count
            chunks.append(chunk)
        chunks.sort(
            key=lambda item: (
                float(item.get("bm25_score") or 0.0),
                -int(item["metadata"].get("chunk_index") or 0),
            ),
            reverse=True,
        )
        return chunks

    async def _retrieve_keyword_chunks(
        self,
        query_text: str,
        kb_id: str,
        allowed_file_ids: set[str] | None,
        merged_kwargs: dict[str, Any],
    ) -> list[dict]:
        """关键词全文检索通道（SQLite LIKE + 词项命中计数排序）。

        使用查询分析产出的 exact tokens 与分词词项作为检索词项；无词项时
        回退 phrase 整句切分。
        """
        analysis = analyze_query(query_text)
        terms = [*analysis.exact_tokens, *analysis.scoring_terms]
        if not terms and analysis.phrase:
            terms = [t for t in re.split(r"\s+", analysis.phrase) if len(t) >= 2][:4]
        if not terms:
            return []

        bm25_top_k = max(int(merged_kwargs.get("bm25_top_k", merged_kwargs.get("recall_top_k", 50)) or 50), 1)
        records = await KnowledgeChunkRepository().search_chunks_by_terms(kb_id=kb_id, terms=terms)
        if allowed_file_ids is not None:
            records = [rec for rec in records if rec.file_id in allowed_file_ids]
        return self._rank_keyword_chunks(records, terms)[:bm25_top_k]

    async def _retrieve_lexical_chunks(
        self,
        query_text: str,
        kb_id: str,
        allowed_file_ids: set[str] | None,
        merged_kwargs: dict[str, Any],
    ) -> tuple[list[dict], str | None]:
        """确定性词法通道（S1-A2，吸收 SAG 词法检索实践）。

        查询分析提取精确 token（编号/手机号/证件号等）与分词词项，经 SQLite
        关键词通道召回候选，再对 exact token 做逐字校验（substring 命中）——
        只有逐字命中的 chunk 标记 exact_match=True，由 rerank 后的分数下限
        保障排序。任何失败返回 ([], error)，调用方降级不影响主检索。

        Returns:
            (chunks, error)：chunks 含 lexical_score 与可选 exact_match/matched_exact_tokens。
        """
        try:
            analysis = analyze_query(query_text)
            if not analysis.exact_tokens and len(analysis.phrase) < 4:
                return [], None

            terms = [*analysis.exact_tokens, *analysis.scoring_terms]
            if not terms:
                return [], None

            lexical_top_k = max(int(merged_kwargs.get("lexical_top_k", 20) or 20), 1)
            records = await KnowledgeChunkRepository().search_chunks_by_terms(kb_id=kb_id, terms=terms)
            if allowed_file_ids is not None:
                records = [rec for rec in records if rec.file_id in allowed_file_ids]

            chunks = self._rank_keyword_chunks(records, terms)[:lexical_top_k]
            exact_tokens_lower = [token.lower() for token in analysis.exact_tokens]
            for chunk in chunks:
                content_lower = chunk["content"].lower()
                matched = [tok for tok in exact_tokens_lower if tok in content_lower]
                if matched:
                    chunk["exact_match"] = True
                    chunk["matched_exact_tokens"] = matched
            return chunks, None
        except Exception as exc:  # noqa: BLE001 - 词法通道失败必须降级而非中断主检索
            return [], f"lexical channel error: {exc}"

    async def _retrieve_graph_chunks(
        self,
        query_text: str,
        kb_id: str,
        base_chunks: list[dict],
        query_params: dict[str, Any],
    ) -> tuple[list[dict], str | None]:
        """检索图增强 chunks（对齐 YUSU `_retrieve_graph_chunks_v2` 管线）。

        流程：HL/LL 关键词抽取（可关闭，失败自动降级 query_text 直查）→
        entity/triple 双路向量召回 seed → round-robin 合并 + 归一化剔除低权重
        → PPR 扩散排序 chunk → 文件过滤 → 组装统一 chunk 结构。任何失败返回
        ([], error)，调用方记录日志并降级纯向量检索，不阻断主流程。
        """
        del base_chunks
        if not self._is_graph_retrieval_enabled(kb_id, query_params):
            return [], None
        try:
            svc = GraphService.get_instance(
                kb_id=kb_id,
                work_dir=self.work_dir,
                embed_func=self._get_embedding_function(),
                chat_model_fn=get_or_create_graph_chat_model_fn(),
            )
            storage = svc.get_storage(kb_id)
            if not storage.is_built():
                logger.debug(f"Graph retrieval for kb={kb_id} skipped: graph not built")
                return [], None

            graph_entity_top_k = max(int(query_params.get("graph_entity_top_k", 15) or 15), 1)
            graph_triple_top_k = max(int(query_params.get("graph_triple_top_k", 15) or 15), 1)
            graph_top_k = max(int(query_params.get("graph_top_k", 10) or 10), 1)
            ppr_damping = float(query_params.get("ppr_damping", 0.85) or 0.85)
            graph_ppr_directed = bool(query_params.get("graph_ppr_directed", False))
            graph_max_nodes = int(query_params.get("graph_max_nodes", 10000) or 10000)
            chunk_count_weight = float(query_params.get("chunk_count_weight", 0.2) or 0.2)

            # 1) HL/LL 关键词：keyword_extractor_enabled=False 或抽取失败时
            #    降级为 query_text 直查（fallback 模式），不依赖 LLM。
            hl_keywords: list[str] = []
            ll_keywords: list[str] = []
            fallback_mode = False
            if bool(query_params.get("keyword_extractor_enabled", True)):
                kb_meta = self.databases_meta.get(kb_id, {})
                additional_params = kb_meta.get("metadata") or {}
                graph_build_config = additional_params.get("graph_build_config") or {}
                extractor_options = graph_build_config.get("extractor_options") or {}
                model_spec = extractor_options.get("model_spec")
                extractor = KeywordExtractor(
                    model_spec=model_spec,
                    chat_model_fn=get_or_create_graph_chat_model_fn(),
                )
                hl_keywords, ll_keywords = await extractor.extract_hl_ll(query_text)
                fallback_mode = not hl_keywords and not ll_keywords
            else:
                fallback_mode = True
            ll_query = query_text if fallback_mode else " ".join(ll_keywords or hl_keywords)
            hl_query = " ".join(hl_keywords) if hl_keywords else query_text

            # 2) 双路 seed：entity store 用 LL 关键词（fallback 模式权重 *0.3
            #    弱化直查信号）；triple store 用 HL 关键词，命中三元组的
            #    source/target 两端各记 *0.6 权重（无 LL 时以 hit.score 为底）。
            entity_store = await svc.get_vector_store(kb_id, "entity")
            triple_store = await svc.get_vector_store(kb_id, "triple")
            entity_weight_scale = 0.3 if fallback_mode else 1.0
            entity_seeds: list[tuple[str, float]] = [
                (hit["id"], float(hit.get("score") or 0.0) * entity_weight_scale)
                for hit in await entity_store.search(ll_query, graph_entity_top_k)
                if hit.get("id")
            ]
            triple_seeds: list[tuple[str, float]] = []
            for hit in await triple_store.search(hl_query, graph_triple_top_k):
                score = float(hit.get("score") or 0.0) * 0.6
                source_id = hit.get("source_id")
                target_id = hit.get("target_id")
                if source_id:
                    triple_seeds.append((str(source_id), score))
                if target_id:
                    triple_seeds.append((str(target_id), score))

            # 3) round-robin 合并 → 归一化，剔除占总权重不足 1% 的弱 seed
            merged = merge_seed_lists_round_robin(entity_seeds, triple_seeds)
            total_weight = sum(merged.values())
            if total_weight <= 0.0:
                logger.debug(f"Graph retrieval for kb={kb_id} skipped: no entity seeds")
                return [], None
            seed_weights = {
                entity_id: weight / total_weight
                for entity_id, weight in merged.items()
                if weight / total_weight >= 0.01
            }
            if not seed_weights:
                return [], None

            # 4) 文件过滤：file_ids 优先，file_name 经 DB 反查（与 aquery 一致）；
            #    有过滤时放大 PPR 候选（*3），过滤后自然收窄。
            allowed_file_ids = await self._resolve_file_filter(kb_id, query_params)
            ppr_top_k = graph_top_k * 3 if allowed_file_ids is not None else graph_top_k

            # 5) PPR 扩散排序（内部含 2hop/1hop 降级与 max 归一化）
            ranked = rank_chunks_by_ppr(
                storage,
                seed_weights,
                top_k=ppr_top_k,
                max_nodes=graph_max_nodes,
                damping=ppr_damping,
                directed=graph_ppr_directed,
                chunk_count_weight=chunk_count_weight,
            )

            # 6) 组装统一 chunk 结构（详情从 repo 取，过滤缺失/文件不匹配项）
            chunk_ids = [chunk_id for chunk_id, _ in ranked]
            records = await KnowledgeChunkRepository().list_by_chunk_ids(chunk_ids)
            record_by_id = {rec.chunk_id: rec for rec in records}
            graph_chunks: list[dict] = []
            for chunk_id, ppr_score in ranked:
                rec = record_by_id.get(chunk_id)
                if rec is None:
                    continue
                if allowed_file_ids is not None and rec.file_id not in allowed_file_ids:
                    continue
                graph_chunks.append(
                    {
                        "content": rec.content,
                        "metadata": {
                            "source": "未知来源",
                            "chunk_id": rec.chunk_id,
                            "file_id": rec.file_id,
                            "chunk_index": rec.chunk_index,
                        },
                        "score": ppr_score,
                        "graph_score": ppr_score,
                    }
                )
            logger.debug(
                f"Graph retrieval for kb={kb_id}: fallback={fallback_mode}, "
                f"seeds={len(seed_weights)}, ranked={len(ranked)}, chunks={len(graph_chunks)}"
            )
            return graph_chunks, None
        except Exception as exc:  # noqa: BLE001 - 图检索任何失败都降级纯向量，不阻断主流程
            logger.warning(f"Graph retrieval failed for kb={kb_id}, degrading to vector-only: {exc}")
            return [], str(exc)

    def _is_graph_retrieval_enabled(self, kb_id: str, query_params: dict[str, Any]) -> bool:
        """判断是否启用图增强检索路径。

        条件：graph_build_config.extractor_options.model_spec 存在（建图时配置
        的 LLM 抽取模型；未配置过视为无图可检索）。注意 ``keyword_extractor_enabled``
        仅控制 HL/LL 关键词抽取是否启用，不作为图检索总开关——关闭时走 query_text
        直查（fallback 模式），图检索本身仍可用（Task 12 依赖此语义）。
        """
        del query_params
        kb_meta = self.databases_meta.get(kb_id, {})
        additional_params = kb_meta.get("metadata") or {}
        graph_build_config = additional_params.get("graph_build_config") or {}
        extractor_options = graph_build_config.get("extractor_options") or {}
        model_spec = extractor_options.get("model_spec")
        return bool(model_spec)

    def _fuse_chunk_rankings(
        self,
        base_chunks: list[dict],
        graph_chunks: list[dict],
        graph_weight: float,
        rrf_k: float = 60.0,
    ) -> list[dict]:
        fused: dict[str, dict[str, Any]] = {}

        def merge_chunk(chunk: dict, rank: int, weight: float, source: str) -> None:
            chunk_id = chunk.get("metadata", {}).get("chunk_id")
            if not chunk_id:
                return
            score = weight / (rrf_k + rank)
            existing = fused.get(chunk_id)
            if existing is None:
                existing = {**chunk, "fusion_score": 0.0, "fusion_sources": []}
                fused[chunk_id] = existing
            existing["fusion_score"] += score
            existing["score"] = existing["fusion_score"]
            existing["fusion_sources"].append(source)
            if source == "graph" and "graph_score" in chunk:
                existing["graph_score"] = chunk["graph_score"]

        for rank, chunk in enumerate(base_chunks, start=1):
            merge_chunk(chunk, rank, 1.0, "chunk")
        for rank, chunk in enumerate(graph_chunks, start=1):
            merge_chunk(chunk, rank, max(graph_weight, 0.0), "graph")

        return sorted(fused.values(), key=lambda item: item.get("fusion_score", 0.0), reverse=True)

    async def _hydrate_chunk_sources(self, kb_id: str, chunks: list[dict]) -> None:
        file_ids = sorted(
            {str(file_id) for chunk in chunks if (file_id := (chunk.get("metadata") or {}).get("file_id"))}
        )
        if not file_ids:
            return

        filenames = await KnowledgeFileRepository().get_filenames_by_file_ids(kb_id=kb_id, file_ids=file_ids)
        for chunk in chunks:
            metadata = chunk.get("metadata")
            if not isinstance(metadata, dict):
                continue
            metadata["source"] = filenames.get(str(metadata.get("file_id") or ""), "") or "未知来源"

    async def _apply_reranker(
        self, query_text: str, chunks: list[dict], merged_kwargs: dict[str, Any]
    ) -> tuple[list[dict], bool]:
        """使用 rerank_func 精排（失败/未配置时返回 (chunks, False) 由调用方兜底）。"""
        if not callable(self.rerank_func):
            logger.debug("No reranker configured, falling back to retrieval scores")
            return chunks, False

        try:
            documents_text = [chunk["content"] for chunk in chunks]
            rerank_scores = await self.rerank_func(query_text, documents_text)

            for chunk, rerank_score in zip(chunks, rerank_scores):
                chunk["rerank_score"] = float(rerank_score)

            # S1-A2: exact 命中的确定性分数下限——精确值查询的目标 chunk 即使被
            # reranker 打分偏低也能稳定进入 top-1。
            exact_floor = float(merged_kwargs.get("exact_score_floor", 0.85) or 0.0)
            if exact_floor > 0.0:
                for chunk in chunks:
                    if chunk.get("exact_match") and chunk.get("rerank_score", -1.0) >= 0.0:
                        chunk["rerank_score"] = max(float(chunk["rerank_score"]), exact_floor)

            # rerank 部分批次失败时 -1.0 哨兵与正常 [0,1] 分双尺度混排。
            # 将正常分与哨兵分离：正常分按 rerank_score 降序，哨兵按原 vector score 降序后追加末尾。
            normal_chunks = [c for c in chunks if c.get("rerank_score", 0.0) >= 0.0]
            sentinel_chunks = [c for c in chunks if c.get("rerank_score", 0.0) < 0.0]
            normal_chunks.sort(key=lambda item: item.get("rerank_score", 0.0), reverse=True)
            sentinel_chunks.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            logger.info("Reranking completed")
            return normal_chunks + sentinel_chunks, True

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Reranking failed: {exc}, falling back to retrieval scores")
            return chunks, False

    async def aquery(self, query_text: str, kb_id: str, agent_call: bool = False, **kwargs) -> list[dict]:
        """异步查询知识库"""
        del agent_call
        storage = await self._get_vector_storage(kb_id)
        if not storage:
            raise ValueError(f"Database {kb_id} not found")

        # 合并查询参数：schema 默认值（底座）→ 已保存 options（用户显式覆盖）→ kwargs（单次临时覆盖）
        # 这样保证 schema 默认值变更能对存量 KB 生效，同时尊重用户显式设置。
        default_options = self._get_default_query_params(kb_id).get("options", {})
        saved_options = self._get_query_params(kb_id)
        merged_kwargs = {**default_options, **saved_options, **kwargs}

        try:
            # 查询参数（从 merged_kwargs 读取）
            logger.debug(f"Query params: {merged_kwargs}")
            final_top_k = int(merged_kwargs.get("final_top_k", 10))
            final_top_k = max(final_top_k, 1)
            similarity_threshold = float(merged_kwargs.get("similarity_threshold", 0.0))
            include_distances = bool(merged_kwargs.get("include_distances", True))
            search_mode = str(merged_kwargs.get("search_mode", "vector")).lower()
            if search_mode not in {"vector", "keyword", "hybrid"}:
                search_mode = "vector"

            use_reranker = bool(merged_kwargs.get("use_reranker", True))
            use_graph_retrieval = bool(merged_kwargs.get("use_graph_retrieval", True))
            if use_reranker or use_graph_retrieval:
                recall_top_k = int(merged_kwargs.get("recall_top_k", 50))
                recall_top_k = max(recall_top_k, final_top_k)
            else:
                recall_top_k = final_top_k

            # file_ids 优先级高于 file_name（更具体，且无需 DB 反查）
            allowed_file_ids = await self._resolve_file_filter(kb_id, merged_kwargs)
            if allowed_file_ids is not None and not allowed_file_ids:
                # 过滤条件存在但无匹配文件，与 Milvus "__no_matching_file__" 表达式语义一致
                return []

            retrieved_chunks: list[dict] = []

            if search_mode == "vector":
                # NanoVectorDB 无查询时过滤表达式，文件过滤在召回后执行；
                # 存在过滤时放大召回候选，避免过滤后数量不足。
                query_top_k = (
                    recall_top_k * LOCAL_FILTERED_RECALL_MULTIPLIER if allowed_file_ids is not None else recall_top_k
                )
                vector_results = await storage.query(query_text, query_top_k)

                for dp in vector_results:
                    if allowed_file_ids is not None and str(dp.get("file_id") or "") not in allowed_file_ids:
                        continue
                    similarity = float(dp.get("distance") or 0.0)
                    if similarity < similarity_threshold:
                        continue
                    retrieved_chunks.append(self._build_chunk_from_vector(dp, include_distances))

                logger.debug(f"Local vector query response: {len(retrieved_chunks)} chunks found (after similarity filtering)")

            elif search_mode == "keyword":
                keyword_chunks = await self._retrieve_keyword_chunks(query_text, kb_id, allowed_file_ids, merged_kwargs)
                retrieved_chunks.extend(keyword_chunks)
                logger.debug(f"Local keyword query response: {len(retrieved_chunks)} chunks found")
            else:
                # hybrid：向量通道为底，关键词通道以权重比例融合（RRF 排名融合）
                query_top_k = (
                    recall_top_k * LOCAL_FILTERED_RECALL_MULTIPLIER if allowed_file_ids is not None else recall_top_k
                )
                vector_results = await storage.query(query_text, query_top_k)

                vector_chunks: list[dict] = []
                for dp in vector_results:
                    if allowed_file_ids is not None and str(dp.get("file_id") or "") not in allowed_file_ids:
                        continue
                    similarity = float(dp.get("distance") or 0.0)
                    if similarity < similarity_threshold:
                        continue
                    vector_chunks.append(self._build_chunk_from_vector(dp, include_distances))

                keyword_chunks = await self._retrieve_keyword_chunks(query_text, kb_id, allowed_file_ids, merged_kwargs)

                # WeightedRanker(vector_weight, bm25_weight) 的排名融合等价形式：
                # base 权重归一化为 1.0，关键词列表权重取相对比例。
                vector_weight = float(merged_kwargs.get("vector_weight", 0.7))
                bm25_weight = float(merged_kwargs.get("bm25_weight", 0.3))
                keyword_weight = bm25_weight / vector_weight if vector_weight > 0.0 else 1.0
                retrieved_chunks = self._fuse_chunk_rankings(vector_chunks, keyword_chunks, keyword_weight)
                logger.debug(
                    f"Local hybrid query response: {len(retrieved_chunks)} chunks found "
                    f"(vector={len(vector_chunks)}, keyword={len(keyword_chunks)})"
                )

            # S1-A2 确定性词法通道：精确值查询（编号/证件号/手机号）的确定性兜底召回。
            # 只叠加不删减：通道未命中或失败时主检索结果完全不受影响。
            if bool(merged_kwargs.get("lexical_channel_enabled", False)):
                lexical_chunks, lexical_error = await self._retrieve_lexical_chunks(
                    query_text, kb_id, allowed_file_ids, merged_kwargs
                )
                if lexical_error:
                    logger.warning(f"Lexical channel degraded for kb={kb_id}: {lexical_error}")
                elif lexical_chunks:
                    exact_matched_ids = {
                        c["metadata"]["chunk_id"]
                        for c in lexical_chunks
                        if c.get("exact_match") and c["metadata"].get("chunk_id")
                    }
                    rrf_k = float(merged_kwargs.get("graph_rrf_k", 60.0))
                    pre_fuse_count = len(retrieved_chunks)
                    retrieved_chunks = self._fuse_chunk_rankings(retrieved_chunks, lexical_chunks, 1.0, rrf_k)
                    # 融合以先入池的 chunk 字典为底，需回写 exact_match 标记供 rerank 后分数下限使用
                    for chunk in retrieved_chunks:
                        if chunk.get("metadata", {}).get("chunk_id") in exact_matched_ids:
                            chunk["exact_match"] = True
                    logger.info(
                        f"Lexical channel fused: kb={kb_id}, pre_fuse={pre_fuse_count}, "
                        f"lexical={len(lexical_chunks)}, exact_matched={len(exact_matched_ids)}, "
                        f"fused={len(retrieved_chunks)}"
                    )

            if use_graph_retrieval:
                # 解包 (chunks, error)，记录图检索错误指标便于运维监控
                graph_chunks, graph_error = await self._retrieve_graph_chunks(
                    query_text, kb_id, retrieved_chunks, merged_kwargs
                )
                if graph_error:
                    # 图检索失败时仅记录日志，不中断主流程（降级到纯向量检索）
                    logger.warning(f"Graph retrieval degraded for kb={kb_id}: {graph_error}")
                if graph_chunks:
                    graph_weight = float(merged_kwargs.get("graph_weight", 0.5))
                    rrf_k = float(merged_kwargs.get("graph_rrf_k", 60.0))
                    pre_fuse_count = len(retrieved_chunks)
                    retrieved_chunks = self._fuse_chunk_rankings(retrieved_chunks, graph_chunks, graph_weight, rrf_k)
                    logger.info(
                        f"Graph retrieval fused: kb={kb_id}, vector_chunks={pre_fuse_count}, "
                        f"graph_chunks={len(graph_chunks)}, fused_chunks={len(retrieved_chunks)}, "
                        f"graph_weight={graph_weight}, rrf_k={rrf_k}"
                    )
                elif not graph_error:
                    logger.info(
                        f"Graph retrieval no contribution: kb={kb_id}, graph_chunks=0, "
                        f"keeping vector-only results={len(retrieved_chunks)}"
                    )

            if not retrieved_chunks:
                return []

            await self._hydrate_chunk_sources(kb_id, retrieved_chunks)

            if not use_reranker:
                return retrieved_chunks[:final_top_k]

            # 使用重排序模型（未配置时自动回退为检索分数排序）
            retrieved_chunks, _ = await self._apply_reranker(query_text, retrieved_chunks, merged_kwargs)

            # rerank min_score 过滤：过滤低分 chunk 提升检索精度
            # 过滤后为空时跳过过滤，返回 rerank 前结果（保证非空，避免检索无结果）
            min_rerank_score = float(merged_kwargs.get("min_rerank_score", 0.0) or 0.0)
            # 仅当所有 chunk 都有有效（非哨兵）rerank_score 时才应用过滤，
            # 避免部分批次失败后 -1.0 哨兵使过滤失效或误杀
            rerank_all_valid = bool(retrieved_chunks) and all(
                "rerank_score" in chunk and chunk["rerank_score"] >= 0.0 for chunk in retrieved_chunks
            )
            if min_rerank_score > 0.0 and retrieved_chunks and rerank_all_valid:
                filtered = [
                    chunk for chunk in retrieved_chunks if float(chunk.get("rerank_score", 1.0)) >= min_rerank_score
                ]
                if filtered:
                    retrieved_chunks = filtered
                    logger.info(
                        f"min_rerank_score filter ({min_rerank_score}) applied for kb={kb_id}: "
                        f"{len(retrieved_chunks)} chunks retained"
                    )
                else:
                    logger.warning(
                        f"min_rerank_score filter ({min_rerank_score}) removed all chunks for kb={kb_id}, "
                        f"skipping filter to guarantee non-empty result"
                    )

            # 统一返回结果
            return retrieved_chunks[:final_top_k]

        except Exception as e:
            logger.error(f"Local query error: {e}")
            raise

    async def delete_file_chunks_only(self, kb_id: str, file_id: str) -> None:
        """仅删除文件的chunks数据，保留元数据（用于更新操作）"""
        chunk_repo = KnowledgeChunkRepository()
        # 先查询文件 chunk 数量，便于清理日志
        chunk_ids = [chunk.chunk_id for chunk in await chunk_repo.list_by_file_id(file_id)]
        logger.info(f"Deleting chunks for file {file_id}, chunk_count={len(chunk_ids)}")

        await self._delete_file_graph_only(kb_id, file_id)

        await chunk_repo.delete_by_file_id(file_id)

        storage = await self._get_vector_storage(kb_id)
        if storage and chunk_ids:
            try:
                await storage.delete(chunk_ids)
                await storage.index_done_callback()
            except Exception as e:  # noqa: BLE001 - 向量删除失败仅告警（SQLite 已删，向量为可重建缓存）
                logger.error(f"Error deleting chunks from vector store for file {file_id}: {e}")

        await KnowledgeFileRepository().update_fields(
            file_id=file_id,
            kb_id=kb_id,
            data={"chunk_count": 0, "token_count": 0},
        )
        await self.refresh_database_stats(kb_id)

    async def delete_file(self, kb_id: str, file_id: str) -> None:
        """删除文件（包括元数据）"""
        # 先删除 chunks 数据
        await self.delete_file_chunks_only(kb_id, file_id)

        await KnowledgeFileRepository().delete(file_id)
        await self.refresh_database_stats(kb_id)

    async def delete_database(self, kb_id: str) -> dict:
        """删除数据库，同时清除向量存储缓存与落盘文件。

        工作目录（含 vdb_*.json）由基类负责删除；这里先 drop 向量存储句柄，
        避免句柄残留指向已删除文件；同时驱逐 GraphService 单例，
        防止 KB 删除后图/向量句柄泄漏（Task 9 修复）。
        """
        storage = self._vector_stores.pop(kb_id, None)
        if storage is not None:
            try:
                await storage.drop()
            except Exception as e:  # noqa: BLE001 - 目录删除由基类兜底，drop 失败仅告警
                logger.warning(f"Failed to drop vector storage for kb {kb_id}: {e}")
        self._vector_store_locks.pop(kb_id, None)
        try:
            await GraphService.evict(kb_id)
        except Exception as e:  # noqa: BLE001 - evict 失败不应阻断 KB 删除主流程
            logger.warning(f"Failed to evict graph service for kb {kb_id}: {e}")
        return await super().delete_database(kb_id)

    async def get_file_basic_info(self, kb_id: str, file_id: str) -> dict:
        """获取文件基本信息（仅元数据）"""
        return {"meta": await self._load_file_meta(kb_id, file_id)}

    async def _get_file_content_from_meta(self, file_id: str, file_meta: dict) -> dict:
        content_info = {"lines": []}
        try:
            chunks = await KnowledgeChunkRepository().list_by_file_id(file_id)
            content_info["lines"] = [
                {
                    "id": chunk.chunk_id,
                    "content": chunk.content,
                    "chunk_order_index": chunk.chunk_index,
                    "start_char_pos": chunk.start_char_pos,
                    "end_char_pos": chunk.end_char_pos,
                    "start_token_pos": chunk.start_token_pos,
                    "end_token_pos": chunk.end_token_pos,
                    "graph_indexed": chunk.graph_indexed,
                    "ent_ids": chunk.ent_ids,
                    "tags": chunk.tags,
                    "extraction_result": chunk.extraction_result,
                }
                for chunk in chunks
            ]
        except Exception as e:  # noqa: BLE001 - 内容读取失败须降级（后续回退 markdown 文件）
            logger.error(f"Failed to get file content from SQLite: {e}")

        if not content_info["lines"]:
            logger.warning(f"No chunks found in SQLite for file {file_id}, file may not have been indexed")

        # Try to read markdown content if available
        if file_meta.get("markdown_file"):
            try:
                content = await self._read_markdown(file_meta["markdown_file"])
                content_info["content"] = content
            except Exception as e:  # noqa: BLE001 - markdown 可缺失，读取失败仅告警
                logger.error(f"Failed to read markdown file for {file_id}: {e}")

        return content_info

    async def get_file_content(self, kb_id: str, file_id: str) -> dict:
        """获取文件内容信息（chunks和lines）"""
        file_meta = await self._load_file_meta(kb_id, file_id)
        return await self._get_file_content_from_meta(file_id, file_meta)

    async def get_file_info(self, kb_id: str, file_id: str) -> dict:
        """获取文件完整信息（基本信息+内容信息）"""
        file_meta = await self._load_file_meta(kb_id, file_id)
        content_info = await self._get_file_content_from_meta(file_id, file_meta)
        return {"meta": file_meta, **content_info}

    def get_query_params_config(self, kb_id: str, **kwargs) -> dict:
        """获取 Local 知识库的查询参数配置。"""
        del kb_id, kwargs
        return {"type": "local", "options": _retrieval_config_options()}
