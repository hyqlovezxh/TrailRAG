"""Per-KB graph build orchestration with an async background task.

Persistence story: ``chunk.extraction_result`` (SQLite) caches extraction,
``chunk.graph_indexed`` marks completion, ``graph_storage.json`` +
``graph_vdb_*.json`` hold the graph. In-memory task state
(progress/enqueued/failed) is lost on restart; ``auto_build_graph`` re-triggers
pending chunks after restart.

Build pipeline (demo-level port of YUSU ``milvus_graph_service`` supply /
worker / flusher pattern, see design docs Task 8):
- supply(): pulls pending chunks into a bounded work queue; the repository
  query excludes already enqueued / failed / in-flight chunk ids at the SQL
  level (``exclude_chunk_ids``, ``NOT IN``), so the ``ORDER BY id LIMIT N``
  window keeps advancing instead of stalling on the same first rows.
- worker() N copies: reuse the cached ``extraction_result`` when it still
  normalizes to non-empty content, otherwise call the extractor, normalize,
  cache non-empty results and enqueue prepared records into the flusher.
- flusher(): a single coroutine accumulates prepared records and batch-writes
  graph_storage + repository rows + vector stores once a threshold or the
  flush timeout is reached; per-chunk failure isolation keeps one bad chunk
  from poisoning the batch.
- Post-processing (only when every pending chunk was written): orphan entity
  cleanup, cross-chunk description merge (LLM only above the 8 fragments /
  1200 tokens thresholds), description write-backs and graph vector re-embed.

The build runs as an ``asyncio.Task`` created by ``build_pending_chunks``;
``reset``, ``configure`` and ``delete_file_graph`` cancel a running build
first so their mutations cannot race the lockless background flusher.
``configure`` merges the build config into the KB's ``additional_params``
(per-KB asyncio lock, replacing YUSU's advisory-lock merge) and clears the
extraction cache when the model/schema/gleaning options materially change.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from yusu_kb.knowledge.graphs.description_merger import DescriptionMerger
from yusu_kb.knowledge.graphs.extractors.base import (
    GraphExtractor,
    normalize_extraction_result,
)
from yusu_kb.knowledge.graphs.extractors.llm import LLMGraphExtractor
from yusu_kb.knowledge.graphs.graph_storage import NetworkXGraphStorage
from yusu_kb.knowledge.graphs.graph_utils import (
    DESC_SEPARATOR,
    build_graph_payload,
    compute_entity_id,
    compute_triple_id,
    normalize_entity_name,
)
from yusu_kb.knowledge.graphs.graph_vector_store import GraphVectorStore
from yusu_kb.knowledge.graphs.token_utils import count_tokens
from yusu_kb.repositories.knowledge_base_repository import KnowledgeBaseRepository
from yusu_kb.repositories.knowledge_chunk_repository import KnowledgeChunkRepository
from yusu_kb.repositories.knowledge_file_repository import KnowledgeFileRepository
from yusu_kb.repositories.knowledge_graph_repository import KnowledgeGraphRepository
from yusu_kb.utils.datetime_utils import utc_isoformat
from yusu_kb.utils.logger import logger

GRAPH_CONFIG_KEY = "graph_build_config"


def _split_description(desc: str) -> list[str]:
    """GKB-10: 拆分拼接的 description 为列表。

    优先按 <SEP> 拆分（新数据），若无 <SEP> 则回退按 "; " 拆分（历史数据兼容）。
    """
    if not desc:
        return []
    if DESC_SEPARATOR in desc:
        return [d.strip() for d in desc.split(DESC_SEPARATOR) if d.strip()]
    # 历史数据兼容：旧分隔符 "; "
    return [d.strip() for d in desc.split("; ") if d.strip()]


def _entity_content(entity: dict[str, Any]) -> str:
    """Rebuild the embedding content of an entity (name|label|description|attrs).

    Shared by the per-chunk record builder and the description-merge re-embed
    so vector rows always carry the latest merged description.
    """
    parts = [str(entity.get("normalized_name") or ""), str(entity.get("label") or "Entity")]
    description = str(entity.get("description") or "").strip()
    if description:
        parts.append(description)
    for attribute in entity.get("attributes") or []:
        attr_text = str(attribute.get("text") or "").strip()
        attr_label = str(attribute.get("label") or "").strip()
        if attr_text:
            parts.append(f"{attr_label}:{attr_text}" if attr_label else attr_text)
    return " | ".join(parts)


def _triple_content(
    source_name: str, relation_type: str, target_name: str, description: str = ""
) -> str:
    """Rebuild the embedding content of a triple (source → type → target | desc)."""
    base = f"{source_name} → {relation_type} → {target_name}"
    return f"{base} | {description}" if description else base


class _PendingGraphWrite:
    """One chunk's prepared records waiting in the flusher queue."""

    __slots__ = ("chunk", "content_preview", "entity_records", "triple_records")

    def __init__(
        self,
        *,
        chunk: Any,
        entity_records: list[dict[str, Any]],
        triple_records: list[dict[str, Any]],
        content_preview: str,
    ) -> None:
        self.chunk = chunk
        self.entity_records = entity_records
        self.triple_records = triple_records
        self.content_preview = content_preview


class _GraphBatchFlusher:
    """Single-coroutine batch writer for prepared chunk records.

    Items accumulate until a threshold (entity/triple/chunk count) or the
    flush timeout forces a batch write; a sentinel drains the queue and exits.
    Demo-level port of YUSU's ``_GraphBatchFlusher``: same thresholds and
    per-chunk failure isolation, no global write semaphore or 429 breaker.
    """

    FLUSH_ENTITY_THRESHOLD = 1000
    FLUSH_TRIPLE_THRESHOLD = 2000
    FLUSH_CHUNK_THRESHOLD = 20
    FLUSH_TIMEOUT = 5.0

    def __init__(self, service: GraphService, kb_id: str, on_flushed=None) -> None:
        self._service = service
        self._kb_id = kb_id
        # Called with the ids of successfully flushed chunks so the build's
        # enqueued set stops growing (graph_indexed filtering is the fallback).
        self._on_flushed = on_flushed
        self._queue: asyncio.Queue[_PendingGraphWrite | None] = asyncio.Queue()
        self._pending: list[_PendingGraphWrite] = []
        self._entity_count = 0
        self._triple_count = 0
        self._processed = 0
        self._failed = 0
        self._failed_chunk_ids: set[str] = set()

    async def enqueue(self, item: _PendingGraphWrite) -> None:
        """Enqueue one chunk's prepared records."""
        await self._queue.put(item)

    async def flusher_loop(self) -> None:
        """Consume the queue, accumulate, and flush at thresholds/timeout."""
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=self.FLUSH_TIMEOUT)
            except TimeoutError:
                # Timeout flush: low-throughput builds must not stall chunks.
                if self._pending:
                    await self._flush_batch_safe()
                continue

            if item is None:  # sentinel: close signal, drain and exit
                if self._pending:
                    await self._flush_batch_safe()
                return

            self._pending.append(item)
            self._entity_count += len(item.entity_records)
            self._triple_count += len(item.triple_records)
            if self._should_flush():
                await self._flush_batch_safe()

    def _should_flush(self) -> bool:
        return (
            self._entity_count >= self.FLUSH_ENTITY_THRESHOLD
            or self._triple_count >= self.FLUSH_TRIPLE_THRESHOLD
            or len(self._pending) >= self.FLUSH_CHUNK_THRESHOLD
        )

    async def _flush_batch_safe(self) -> None:
        """Flush the pending batch; failures count, never raise (keep loop)."""
        if not self._pending:
            return
        batch = self._pending[:]
        self._pending.clear()
        self._entity_count = 0
        self._triple_count = 0
        try:
            succeeded_items, failed_chunk_ids = await self._service._flush_graph_batch(
                self._kb_id, batch
            )
        except Exception as exc:  # noqa: BLE001 - 整批级失败须计 failed 保持 pending，重试幂等自愈
            logger.error(f"Graph batch flush failed for {len(batch)} chunks in kb {self._kb_id}: {exc}")
            self._failed += len(batch)
            self._failed_chunk_ids.update(item.chunk.chunk_id for item in batch)
            return
        self._processed += len(succeeded_items)
        if failed_chunk_ids:
            self._failed += len(failed_chunk_ids)
            self._failed_chunk_ids |= failed_chunk_ids
        if succeeded_items and self._on_flushed is not None:
            self._on_flushed([item.chunk.chunk_id for item in succeeded_items])

    async def close(self) -> None:
        """Send the sentinel; the loop drains pending items before exiting."""
        await self._queue.put(None)


class GraphService:
    """Per-KB graph build orchestration with an async background task.

    Persistence story: ``chunk.extraction_result`` (SQLite) caches extraction,
    ``chunk.graph_indexed`` marks completion, ``graph_storage.json`` +
    ``graph_vdb_*.json`` hold the graph. In-memory task state
    (progress/enqueued/failed) is lost on restart; ``auto_build_graph``
    re-triggers pending chunks after restart.
    """

    _instances: ClassVar[dict[str, GraphService]] = {}
    _locks: ClassVar[dict[str, asyncio.Lock]] = {}

    def __init__(
        self,
        *,
        kb_id: str,
        chunk_repo: KnowledgeChunkRepository | None = None,
        graph_repo: KnowledgeGraphRepository | None = None,
        kb_repo: KnowledgeBaseRepository | None = None,
        work_dir: str | Path,
        embed_func=None,
        chat_model_fn=None,
        default_extractor_options: dict[str, Any] | None = None,
    ) -> None:
        self.kb_id = kb_id
        self.chunk_repo = chunk_repo or KnowledgeChunkRepository()
        self.graph_repo = graph_repo or KnowledgeGraphRepository()
        self.kb_repo = kb_repo or KnowledgeBaseRepository()
        self.work_dir = Path(work_dir)
        self.embed_func = embed_func
        self.chat_model_fn = chat_model_fn
        self.default_extractor_options = dict(default_extractor_options or {})
        # Lazy per-KB resources, cached so status/build/reset share one instance.
        self._storages: dict[str, NetworkXGraphStorage] = {}
        self._vector_stores: dict[str, dict[str, GraphVectorStore]] = {}
        # Per-KB per-kind locks guarding one-time GraphVectorStore creation.
        self._vector_store_locks: dict[str, dict[str, asyncio.Lock]] = {}
        # kb_id -> {"task", "status", "progress", "message"} for get_status.
        self._build_states: dict[str, dict[str, Any]] = {}

    # --- singleton -------------------------------------------------------

    @classmethod
    def get_instance(cls, **deps: Any) -> GraphService:
        """Return the registered instance for ``deps["kb_id"]`` or create one.

        The factory (graph router / local_kb wiring) supplies the deps
        (work_dir, embed_func, chat_model_fn, ...); repositories default
        inside the constructor.
        """
        kb_id = deps["kb_id"]
        instance = cls._instances.get(kb_id)
        if instance is None:
            instance = cls(**deps)
            cls._instances[kb_id] = instance
        return instance

    @classmethod
    async def evict(cls, kb_id: str) -> None:
        """Drop the registered instance (used when a KB is deleted)."""
        instance = cls._instances.get(kb_id)
        if instance is not None:
            # Best effort: a running build's flusher must not keep writing
            # after the instance (and its KB data) is dropped.
            try:
                await instance._cancel_build_if_running(kb_id)
            except Exception as exc:  # noqa: BLE001 - evict 失败不得阻塞 KB 删除
                logger.error(f"Failed to cancel build while evicting kb {kb_id}: {exc}")
            instance._vector_store_locks.pop(kb_id, None)
        cls._instances.pop(kb_id, None)
        # Drop the per-KB lock so deleted KBs do not accumulate entries.
        cls._locks.pop(kb_id, None)

    def _get_lock(self, kb_id: str) -> asyncio.Lock:
        """Per-KB lock serializing configure / build / reset transitions."""
        lock = self._locks.get(kb_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[kb_id] = lock
        return lock

    # --- status / config -------------------------------------------------

    async def get_status(self, kb_id: str) -> dict[str, Any]:
        """Report build configuration, chunk progress and graph counts."""
        kb = await self.kb_repo.get_by_kb_id(kb_id)
        if kb is None:
            raise ValueError(f"知识库 {kb_id} 不存在")
        config = dict(kb.additional_params or {}).get(GRAPH_CONFIG_KEY) or {}
        total, pending, indexed, (entity_count, relation_count) = await asyncio.gather(
            self.chunk_repo.count_by_kb_id(kb_id),
            self.chunk_repo.count_graph_pending_by_kb_id(kb_id),
            self.chunk_repo.count_graph_indexed_by_kb_id(kb_id),
            self.graph_repo.count_by_kb_id(kb_id),
        )
        state = self._build_states.get(kb_id)
        task = state.get("task") if state else None
        build_task_status: str | None = None
        build_task_progress = 0.0
        if task is not None:
            if task.done():
                build_task_status = state.get("status") if state else None
                build_task_progress = float(state.get("progress") or 0.0)
            else:
                build_task_status = "running"
                build_task_progress = float(state.get("progress") or 0.0)
        return {
            "kb_id": kb_id,
            "configured": bool(config),
            "config": self._public_config(config),
            "locked": bool(config.get("locked")),
            "total_chunks": total,
            "pending_chunks": pending,
            "indexed_chunks": indexed,
            "entity_count": entity_count,
            "relation_count": relation_count,
            "build_task_status": build_task_status,
            "build_task_progress": round(build_task_progress, 2),
        }

    async def configure(
        self,
        kb_id: str,
        extractor_type: str = "llm",
        extractor_options: dict[str, Any] | None = None,
        created_by: str = "system",
    ) -> dict[str, Any]:
        """Validate and merge the graph build config into the KB.

        Material changes to model_spec / schema / model_params / gleaning_count
        clear the extraction cache and reset ``graph_indexed`` so the next
        build re-extracts with the new options.
        """
        normalized_type = (extractor_type or "").lower()
        options = {**self.default_extractor_options, **(extractor_options or {})}
        if normalized_type != "llm":
            raise ValueError(f"未知的图谱抽取器类型: {normalized_type}（当前仅支持 llm）")
        self._validate_llm_options(options)

        lock = self._get_lock(kb_id)
        merge_state: dict[str, Any] = {}
        async with lock:
            # The background build runs lockless, so a concurrent flusher could
            # re-mark graph_indexed (or re-write the graph) while we clear the
            # extraction cache below; cancel it first (same pattern as reset).
            await self._cancel_build_if_running(kb_id)
            kb = await self.kb_repo.get_by_kb_id(kb_id)
            if kb is None:
                raise ValueError(f"知识库 {kb_id} 不存在")
            params = dict(kb.additional_params or {})
            existing_config = params.get(GRAPH_CONFIG_KEY) or {}
            if existing_config.get("locked") and normalized_type != (
                existing_config.get("extractor_type") or ""
            ).lower():
                raise ValueError("图谱抽取器类型已锁定，只能修改模型、Schema 等抽取参数")

            # concurrency_count / request_timeout do not change extraction
            # output, so only these four keys trigger a cache clear.
            old_options = existing_config.get("extractor_options") or {}
            merge_state["options_changed"] = any(
                old_options.get(key) != options.get(key)
                for key in ("model_spec", "schema", "model_params", "gleaning_count")
            )
            merge_state["was_locked"] = bool(existing_config.get("locked"))

            config = {
                "locked": True,
                "extractor_type": normalized_type,
                "extractor_options": options,
                "created_at": existing_config.get("created_at") or utc_isoformat(),
                "created_by": existing_config.get("created_by") or created_by,
            }
            if existing_config.get("locked"):
                config["updated_at"] = utc_isoformat()
                config["updated_by"] = created_by
            params[GRAPH_CONFIG_KEY] = config
            await self.kb_repo.update(kb_id, {"additional_params": params})
            merge_state["config"] = config

            if merge_state["options_changed"] and merge_state["was_locked"]:
                cleared = await self.chunk_repo.reset_graph_state_by_kb_id(
                    kb_id, clear_extraction_result=True
                )
                logger.info(
                    f"Extractor options changed for kb {kb_id}, cleared extraction cache for {cleared} chunks"
                )
        return merge_state["config"]

    @staticmethod
    def _validate_llm_options(options: dict[str, Any]) -> None:
        """Configure-time validation: the fields that shape extraction output."""
        if not (options.get("model_spec") or "").strip():
            raise ValueError("LLM 图谱抽取器需要 model_spec")
        if options.get("prompt"):
            raise ValueError("LLM 图谱抽取器不支持自定义完整 Prompt，请使用 schema 配置抽取约束")
        try:
            concurrency_count = int(options.get("concurrency_count") or 8)
        except (TypeError, ValueError) as exc:
            raise ValueError("LLM 抽取器 concurrency_count 必须是整数") from exc
        if concurrency_count < 1 or concurrency_count > 128:
            raise ValueError("LLM 抽取器 concurrency_count 必须在 1 到 128 之间（与执行层 cap 一致）")
        if options.get("model_params") is not None and not isinstance(options["model_params"], dict):
            raise ValueError("LLM 抽取器 model_params 必须是对象")
        try:
            gleaning_count = int(options.get("gleaning_count") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("LLM 抽取器 gleaning_count 必须是整数") from exc
        if gleaning_count < 0 or gleaning_count > 3:
            raise ValueError("LLM 抽取器 gleaning_count 必须在 0 到 3 之间")

    @staticmethod
    def _get_locked_config(additional_params: dict[str, Any]) -> dict[str, Any]:
        config = additional_params.get(GRAPH_CONFIG_KEY) or {}
        if not config.get("locked"):
            raise ValueError("请先确认并锁定图谱抽取配置")
        if not config.get("extractor_type"):
            raise ValueError("图谱抽取配置缺少 extractor_type")
        return config

    @staticmethod
    def _public_config(config: dict[str, Any]) -> dict[str, Any] | None:
        if not config:
            return None
        return {
            "locked": bool(config.get("locked")),
            "extractor_type": config.get("extractor_type"),
            "extractor_options": GraphService._runtime_extractor_options(config),
            "created_at": config.get("created_at"),
            "created_by": config.get("created_by"),
            "updated_at": config.get("updated_at"),
            "updated_by": config.get("updated_by"),
        }

    @staticmethod
    def _runtime_extractor_options(config: dict[str, Any]) -> dict[str, Any]:
        options = dict(config.get("extractor_options") or {})
        options.pop("prompt", None)
        options.setdefault("gleaning_count", 0)
        return options

    @staticmethod
    def _get_worker_count(config: dict[str, Any]) -> int:
        if (config.get("extractor_type") or "").lower() != "llm":
            return 1
        try:
            worker_count = int((config.get("extractor_options") or {}).get("concurrency_count") or 8)
        except (TypeError, ValueError):
            return 8
        return max(1, min(worker_count, 128))

    # --- build -----------------------------------------------------------

    async def build_pending_chunks(self, kb_id: str, batch_size: int = 100) -> dict[str, Any]:
        """Start (or report) the background build task for pending chunks.

        The task is not awaited: callers poll ``get_status``. A second call
        while a build is running returns the running state instead of
        starting another task.
        """
        lock = self._get_lock(kb_id)
        async with lock:
            state = self._build_states.get(kb_id)
            if state is not None and state.get("task") is not None and not state["task"].done():
                remaining = await self.chunk_repo.count_graph_pending_by_kb_id(kb_id)
                return {
                    "kb_id": kb_id,
                    "success": 0,
                    "failed": 0,
                    "remaining": remaining,
                    "graph_built": False,
                    "message": "图谱构建已在后台运行，请稍后查询状态",
                }
            kb = await self.kb_repo.get_by_kb_id(kb_id)
            if kb is None:
                raise ValueError(f"知识库 {kb_id} 不存在")
            config = self._get_locked_config(kb.additional_params or {})
            total_pending = await self.chunk_repo.count_graph_pending_by_kb_id(kb_id)
            state = {
                "task": None,
                "status": "running",
                "progress": 0.0,
                "message": "图谱构建启动中",
            }
            self._build_states[kb_id] = state
            state["task"] = asyncio.create_task(
                self._run_build(kb_id, config, batch_size, total_pending, state)
            )
        return {
            "kb_id": kb_id,
            "success": 0,
            "failed": 0,
            "remaining": total_pending,
            "graph_built": False,
            "message": "图谱构建已在后台启动",
        }

    async def _cancel_build_if_running(self, kb_id: str) -> None:
        """Cancel and await a running build task, then drop its state (no-op
        when idle). The background build runs without the per-KB lock — only
        the state transitions in build/reset/configure hold it — so any
        mutation that must not race the flusher (config clear, reset, file
        delete) cancels the task first; a completed state is dropped too, as
        its reported result is stale after the mutation.
        """
        state = self._build_states.get(kb_id)
        if state is None:
            return
        task = state.get("task")
        if task is not None and not task.done():
            task.cancel()
            # Suppress the CancelledError: the caller proceeds regardless.
            await asyncio.gather(task, return_exceptions=True)
        self._build_states.pop(kb_id, None)

    async def _run_build(
        self,
        kb_id: str,
        config: dict[str, Any],
        batch_size: int,
        total_pending: int,
        state: dict[str, Any],
    ) -> None:
        """Background task wrapper: run the build and record the outcome."""
        try:
            result = await self._build_once(kb_id, config, batch_size, total_pending, state)
            state["status"] = "completed"
            state["progress"] = 100.0
            state["message"] = result["message"]
        except asyncio.CancelledError:
            state["status"] = "cancelled"
            state["message"] = "图谱构建已取消"
            raise
        except Exception as exc:  # noqa: BLE001 - 后台任务失败须记录状态而非杀死 task
            logger.error(f"Graph build failed for kb {kb_id}: {exc}")
            state["status"] = "failed"
            state["message"] = str(exc)

    async def _build_once(
        self,
        kb_id: str,
        config: dict[str, Any],
        batch_size: int,
        total_pending: int,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Run supply/worker/flusher to completion, then post-process."""
        extractor = self._create_extractor(config["extractor_type"], self._runtime_extractor_options(config))
        worker_count = self._get_worker_count(config)
        description_merger = self._create_description_merger(config)

        failed = 0
        failed_chunk_ids: set[str] = set()
        enqueued_chunk_ids: set[str] = set()
        in_flight_chunk_ids: set[str] = set()
        extraction_done = 0

        flusher = _GraphBatchFlusher(
            service=self,
            kb_id=kb_id,
            on_flushed=lambda chunk_ids: enqueued_chunk_ids.difference_update(chunk_ids),
        )
        flusher_task = asyncio.create_task(flusher.flusher_loop())
        work_queue: asyncio.Queue[tuple[Any, str] | None] = asyncio.Queue(maxsize=batch_size * 2)

        async def _cleanup_flusher() -> None:
            """Send the sentinel and wait for the flusher to drain (always)."""
            try:
                await flusher.close()
            except Exception as close_exc:  # noqa: BLE001 - 清理失败不得阻塞构建退出
                logger.error(f"flusher.close() failed during cleanup for kb {kb_id}: {close_exc}")
            try:
                await flusher_task
            except Exception as flusher_exc:  # noqa: BLE001 - 清理失败不得阻塞构建退出
                logger.error(f"Graph batch flusher failed for kb {kb_id}: {flusher_exc}")

        async def supply() -> None:
            """Pull pending chunks into the work queue; sentinels when empty."""
            while True:
                exclude_ids = failed_chunk_ids | enqueued_chunk_ids | in_flight_chunk_ids
                chunks = await self.chunk_repo.list_graph_pending_by_kb_id(
                    kb_id, limit=batch_size, exclude_chunk_ids=exclude_ids or None
                )
                if not chunks:
                    if in_flight_chunk_ids:
                        await asyncio.sleep(0.05)
                        continue
                    break
                file_ids = {c.file_id for c in chunks if c.file_id}
                filenames = (
                    await KnowledgeFileRepository().get_filenames_by_file_ids(kb_id=kb_id, file_ids=list(file_ids))
                    if file_ids
                    else {}
                )
                for chunk in chunks:
                    in_flight_chunk_ids.add(chunk.chunk_id)
                    await work_queue.put((chunk, filenames.get(chunk.file_id, "")))
            for _ in range(worker_count):
                await work_queue.put(None)

        async def worker() -> None:
            nonlocal extraction_done, failed
            while True:
                item = await work_queue.get()
                if item is None:  # sentinel: no more chunks, exit
                    return
                chunk, document_title = item
                try:
                    extraction_result = await self._get_chunk_extraction_result(
                        kb_id, chunk, extractor, document_title=document_title
                    )
                    prepared = self._prepare_chunk_graph_records(kb_id, chunk, extraction_result)
                    await flusher.enqueue(
                        _PendingGraphWrite(
                            chunk=chunk,
                            entity_records=prepared["entity_records"],
                            triple_records=prepared["triple_records"],
                            content_preview=prepared["content_preview"],
                        )
                    )
                    enqueued_chunk_ids.add(chunk.chunk_id)
                    in_flight_chunk_ids.discard(chunk.chunk_id)
                    extraction_done += 1
                except Exception as exc:  # noqa: BLE001 - per-chunk 确定性失败计 failed，不杀 worker
                    failed_chunk_ids.add(chunk.chunk_id)
                    in_flight_chunk_ids.discard(chunk.chunk_id)
                    failed += 1
                    logger.error(f"Chunk 图谱构建失败 chunk_id={chunk.chunk_id}: {exc}")
                finally:
                    completed = extraction_done + failed
                    state["progress"] = 5.0 + min(90.0, completed / max(total_pending, 1) * 90.0)
                    state["message"] = f"图谱构建 {completed}/{total_pending}，失败 {failed}"

        supply_task = asyncio.create_task(supply())
        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        try:
            await asyncio.gather(supply_task, *workers)
        except Exception:
            for task in [supply_task, *workers]:
                task.cancel()
            await asyncio.gather(supply_task, *workers, return_exceptions=True)
            raise
        finally:
            await _cleanup_flusher()

        processed = flusher._processed
        failed += flusher._failed
        failed_chunk_ids |= flusher._failed_chunk_ids

        remaining = await self.chunk_repo.count_graph_pending_by_kb_id(kb_id)
        if remaining > 0:
            if processed == 0:
                raise RuntimeError(
                    f"图谱构建失败：全部 {failed} 个 chunk 处理均失败，无成功记录。"
                    "请检查 LLM 模型配置（model_spec）、API 可用性后重试。"
                )
            raise RuntimeError(
                f"图谱构建未完成：{processed} 成功，{failed} 失败，{remaining} 仍待处理。"
                "已成功的 chunk 不会重复处理，请检查 LLM API 可用性后重试。"
            )

        if processed > 0:
            logger.info(f"[Post-process] Starting full post-processing for kb {kb_id} (all chunks completed)")
            storage = self.get_storage(kb_id)
            try:
                orphan_count = await self._cleanup_orphan_entities(kb_id, storage)
                if orphan_count > 0:
                    logger.info(f"[Post-process] Orphan entities cleaned for kb {kb_id}: {orphan_count}")
            except Exception as exc:  # noqa: BLE001 - 后处理单项失败不阻塞整体完成
                logger.error(f"[Post-process] Orphan entity cleanup failed for kb {kb_id}: {exc}")
            if description_merger is not None:
                try:
                    merged_count = await self._merge_cross_chunk_descriptions(kb_id, storage, description_merger)
                    if merged_count > 0:
                        logger.info(f"[Post-process] Cross-chunk description merged for kb {kb_id}: {merged_count}")
                except Exception as exc:  # noqa: BLE001 - 后处理单项失败不阻塞整体完成
                    logger.error(f"[Post-process] Cross-chunk description merge failed for kb {kb_id}: {exc}")
            storage.save()
            logger.info(f"[Post-process] Full post-processing complete for kb {kb_id}")

        return {
            "kb_id": kb_id,
            "success": processed,
            "failed": failed,
            "remaining": remaining,
            "graph_built": processed > 0 and remaining == 0,
            "message": "图谱构建完成" if processed > 0 else "无 chunk 需要处理",
        }

    def _create_extractor(self, extractor_type: str, options: dict[str, Any]) -> GraphExtractor:
        options = dict(options)
        if extractor_type != "llm":
            raise ValueError(f"未知的图谱抽取器类型: {extractor_type}（当前仅支持 llm）")
        options.setdefault("chat_model_fn", self.chat_model_fn)
        extractor = LLMGraphExtractor(options)
        extractor.validate_options()
        return extractor

    def _create_description_merger(self, config: dict[str, Any]) -> DescriptionMerger | None:
        if not callable(self.chat_model_fn):
            return None
        model_spec = (config.get("extractor_options") or {}).get("model_spec")
        if not model_spec:
            logger.warning("KB 缺少 model_spec，DescriptionMerger 未初始化，description 将不合并")
            return None
        return DescriptionMerger(model_spec=model_spec, chat_model_fn=self.chat_model_fn)

    async def _get_chunk_extraction_result(
        self,
        kb_id: str,
        chunk: Any,
        extractor: GraphExtractor,
        *,
        document_title: str = "",
    ) -> dict[str, Any]:
        """Reuse the cached extraction when it still normalizes non-empty."""
        extractor_type = extractor.extractor_type
        if chunk.extraction_result:
            cached = normalize_extraction_result(chunk.extraction_result, extractor_type)
            if cached.get("entities") or cached.get("relations"):
                return cached
            logger.warning(f"Chunk {chunk.chunk_id} has empty cached extraction_result, re-extracting")

        extraction_result = await extractor.extract(
            chunk.content,
            chunk_metadata={
                "kb_id": kb_id,
                "chunk_id": chunk.chunk_id,
                "file_id": chunk.file_id,
                "chunk_index": chunk.chunk_index,
                "document_title": document_title,
            },
        )
        normalized_result = normalize_extraction_result(extraction_result, extractor_type)
        # Empty results are not cached so later builds retry instead of
        # reusing a permanently empty extraction.
        if not normalized_result.get("entities") and not normalized_result.get("relations"):
            logger.warning(f"LLM returned empty extraction for chunk {chunk.chunk_id}, skipping cache")
            return normalized_result
        await self.chunk_repo.update_extraction_result(chunk.chunk_id, normalized_result)
        return normalized_result

    def _prepare_chunk_graph_records(
        self, kb_id: str, chunk: Any, normalized_result: dict[str, Any]
    ) -> dict[str, Any]:
        """Build deduplicated entity/triple records ready for persistence."""
        graph_payload = build_graph_payload(normalized_result)
        extractor_type = graph_payload["metadata"].get("extractor_type", "unknown")

        entity_records: dict[str, dict[str, Any]] = {}
        entity_by_local_id: dict[str, dict[str, Any]] = {}
        for entity in graph_payload["entities"]:
            label = entity.get("label") or "Entity"
            normalized_name = normalize_entity_name(entity["text"])
            entity_id = compute_entity_id(kb_id, normalized_name, label)
            description = str(entity.get("description") or "").strip()
            record = {
                "entity_id": entity_id,
                "kb_id": kb_id,
                "normalized_name": normalized_name,
                "label": label,
                "name": entity["text"],
                "attributes": list(entity.get("attributes") or []),
                "description": description or None,
                "content": _entity_content(
                    {
                        "normalized_name": normalized_name,
                        "label": label,
                        "description": description,
                        "attributes": entity.get("attributes") or [],
                    }
                ),
            }
            entity_records[entity_id] = record
            entity_by_local_id[entity["id"]] = record

        triple_records: dict[str, dict[str, Any]] = {}
        for relation in graph_payload["relations"]:
            source = entity_by_local_id[relation["source"]]
            target = entity_by_local_id[relation["target"]]
            relation_type = relation.get("label") or "RELATED_TO"
            triple_id = compute_triple_id(
                kb_id,
                source["normalized_name"],
                source["label"],
                relation_type,
                target["normalized_name"],
                target["label"],
            )
            if triple_id in triple_records:
                continue
            description = str(relation.get("description") or "").strip()
            triple_records[triple_id] = {
                "triple_id": triple_id,
                "kb_id": kb_id,
                "source_entity_id": source["entity_id"],
                "target_entity_id": target["entity_id"],
                "relation_type": relation_type,
                "source_name": source["normalized_name"],
                "source_label": source["label"],
                "target_name": target["normalized_name"],
                "target_label": target["label"],
                "content": _triple_content(source["normalized_name"], relation_type, target["normalized_name"], description),
                "text": relation["text"],
                "extractor_type": extractor_type,
                "description": description or None,
            }

        return {
            "entity_records": list(entity_records.values()),
            "triple_records": list(triple_records.values()),
            "content_preview": (chunk.content or "")[:300],
        }

    async def _flush_graph_batch(
        self, kb_id: str, batch: list[_PendingGraphWrite]
    ) -> tuple[list[_PendingGraphWrite], set[str]]:
        """Persist a batch: storage + repository per chunk, vectors in bulk.

        Per-chunk isolation for storage/repo writes; a vector-store failure
        fails the whole batch (chunks stay pending, retry is idempotent).
        """
        storage = self.get_storage(kb_id)
        entity_store = await self.get_vector_store(kb_id, "entity")
        triple_store = await self.get_vector_store(kb_id, "triple")

        succeeded: list[_PendingGraphWrite] = []
        failed_chunk_ids: set[str] = set()
        for item in batch:
            try:
                self._persist_chunk_graph(storage, item)
                await self._persist_chunk_repo(kb_id, item)
                succeeded.append(item)
            except Exception as exc:  # noqa: BLE001 - 单 chunk 失败隔离，不毒化同批其他 chunk
                failed_chunk_ids.add(item.chunk.chunk_id)
                logger.error(f"Graph persist failed for chunk {item.chunk.chunk_id}: {exc}")

        if not succeeded:
            return [], failed_chunk_ids

        entity_records: dict[str, dict[str, Any]] = {}
        triple_records: dict[str, dict[str, Any]] = {}
        for item in succeeded:
            for record in item.entity_records:
                entity_records.setdefault(record["entity_id"], record)
            for record in item.triple_records:
                triple_records.setdefault(record["triple_id"], record)

        await entity_store.upsert(
            [{"id": rec["entity_id"], "content": rec["content"], "label": rec["label"]} for rec in entity_records.values()]
        )
        await triple_store.upsert(
            [
                {
                    "id": rec["triple_id"],
                    "content": rec["content"],
                    "source_id": rec["source_entity_id"],
                    "target_id": rec["target_entity_id"],
                    "type": rec["relation_type"],
                }
                for rec in triple_records.values()
            ]
        )

        for item in succeeded:
            try:
                await self.chunk_repo.mark_graph_indexed(
                    item.chunk.chunk_id,
                    ent_ids=[record["entity_id"] for record in item.entity_records],
                )
            except Exception as mark_exc:  # noqa: BLE001 - mark 失败由下次构建补齐
                logger.warning(
                    f"mark_graph_indexed failed for {item.chunk.chunk_id} "
                    f"(persist succeeded, will be reconciled by next build): {mark_exc}"
                )
        # 图本体与向量库/索引标记同步落盘：这些 chunk 已 mark_graph_indexed，
        # 若此时进程崩溃或构建以失败收尾（存在失败 chunk），内存图尚未保存会导致
        # 已索引 chunk 的图数据不可恢复。每批 flush 即 save，与向量库节奏一致。
        storage.save()
        return succeeded, failed_chunk_ids

    def _persist_chunk_graph(self, storage: NetworkXGraphStorage, item: _PendingGraphWrite) -> None:
        """Merge one chunk's records into the in-memory graph (sync)."""
        chunk = item.chunk
        storage.add_chunk(
            chunk_id=chunk.chunk_id,
            file_id=chunk.file_id,
            chunk_index=chunk.chunk_index,
            content_preview=item.content_preview,
        )
        for record in item.entity_records:
            storage.upsert_entity(
                entity_id=record["entity_id"],
                normalized_name=record["normalized_name"],
                label=record["label"],
                name=record["name"],
                attributes=record.get("attributes"),
                description=record.get("description") or "",
            )
            storage.add_mention(entity_id=record["entity_id"], chunk_id=chunk.chunk_id, file_id=chunk.file_id)
        for record in item.triple_records:
            storage.upsert_relation(
                triple_id=record["triple_id"],
                source_id=record["source_entity_id"],
                target_id=record["target_entity_id"],
                text=record["text"],
                rtype=record["relation_type"],
                file_ids=[chunk.file_id],
                description=record.get("description") or "",
            )

    async def _persist_chunk_repo(self, kb_id: str, item: _PendingGraphWrite) -> None:
        """Persist one chunk's records to the SQLite graph repository."""
        chunk = item.chunk
        await self.graph_repo.upsert_entities(kb_id, item.entity_records)
        await self.graph_repo.upsert_relations(kb_id, item.triple_records)
        await self.graph_repo.upsert_mentions(
            kb_id,
            [
                {"entity_id": record["entity_id"], "file_id": chunk.file_id, "chunk_id": chunk.chunk_id}
                for record in item.entity_records
            ],
        )
        await self.graph_repo.upsert_triple_mentions(
            kb_id,
            [
                {
                    "triple_id": record["triple_id"],
                    "file_id": chunk.file_id,
                    "chunk_id": chunk.chunk_id,
                    "text": record["text"],
                    "extractor_type": record.get("extractor_type"),
                }
                for record in item.triple_records
            ],
        )

    # --- post-processing --------------------------------------------------

    async def _cleanup_orphan_entities(self, kb_id: str, storage: NetworkXGraphStorage) -> int:
        """Delete entities without any RELATION or MENTIONS edge from the vectors.

        Entities kept alive by mentions alone are preserved (they may be
        shared across files), matching YUSU's orphan-cleanup semantics.
        """
        entity_ids = {entity["entity_id"] for entity in storage.iter_entities()}
        related_ids: set[str] = set()
        for relation in storage.iter_relations():
            related_ids.add(relation["source_id"])
            related_ids.add(relation["target_id"])
        for entity_id in entity_ids:
            if storage.chunk_lookup_1hop([entity_id]):
                related_ids.add(entity_id)
        orphans = sorted(entity_ids - related_ids)
        if orphans:
            entity_store = await self.get_vector_store(kb_id, "entity")
            await entity_store.delete_ids(orphans)
            logger.info(f"[Post-process] Orphan entities removed from vectors for kb {kb_id}: {orphans}")
        return len(orphans)

    async def _merge_cross_chunk_descriptions(
        self, kb_id: str, storage: NetworkXGraphStorage, merger: DescriptionMerger
    ) -> int:
        """Merge multi-fragment descriptions (LLM above 8 fragments / 1200 tokens).

        Write-backs: repository rows, storage nodes and the graph vector
        stores (content rebuilt so the merged description enters the index).
        """
        entity_items: list[dict[str, Any]] = []
        for entity in storage.iter_entities():
            description = entity.get("description") or ""
            fragments = _split_description(description)
            if len(fragments) < 8 and count_tokens(description) < 1200:
                continue
            entity_items.append(
                {
                    "entity_id": entity["entity_id"],
                    "name": entity.get("name") or entity.get("normalized_name") or entity["entity_id"],
                    "descriptions": fragments,
                }
            )
        relation_items: list[dict[str, Any]] = []
        for relation in storage.iter_relations():
            description = relation.get("description") or ""
            fragments = _split_description(description)
            if len(fragments) < 8 and count_tokens(description) < 1200:
                continue
            source = storage.get_entity_node(relation["source_id"]) or {}
            target = storage.get_entity_node(relation["target_id"]) or {}
            relation_items.append(
                {
                    "triple_id": relation["triple_id"],
                    "source_name": source.get("name") or relation["source_id"],
                    "relation_type": relation["type"],
                    "target_name": target.get("name") or relation["target_id"],
                    "descriptions": fragments,
                }
            )
        if not entity_items and not relation_items:
            return 0

        merged_entities, merged_triples = await merger.merge_batch(entity_items, relation_items)
        entity_updates = {
            item["entity_id"]: item["description"] for item in merged_entities if item.get("description")
        }
        triple_updates = {
            item["triple_id"]: item["description"] for item in merged_triples if item.get("description")
        }

        if entity_updates:
            await self.graph_repo.update_entity_descriptions(kb_id, entity_updates)
            for entity_id, description in entity_updates.items():
                storage.update_entity_description(entity_id, description)
            entity_store = await self.get_vector_store(kb_id, "entity")
            await entity_store.upsert(
                [
                    {
                        "id": entity_id,
                        "content": _entity_content(storage.get_entity_node(entity_id) or {}),
                        "label": (storage.get_entity_node(entity_id) or {}).get("label", "Entity"),
                    }
                    for entity_id in entity_updates
                ]
            )
        if triple_updates:
            await self.graph_repo.update_triple_descriptions(kb_id, triple_updates)
            for triple_id, description in triple_updates.items():
                storage.update_relation_description(triple_id, description)
            triple_store = await self.get_vector_store(kb_id, "triple")
            records = []
            for relation in storage.iter_relations():
                if relation["triple_id"] in triple_updates:
                    source = storage.get_entity_node(relation["source_id"]) or {}
                    target = storage.get_entity_node(relation["target_id"]) or {}
                    records.append(
                        {
                            "id": relation["triple_id"],
                            "content": _triple_content(
                                source.get("name") or relation["source_id"],
                                relation["type"],
                                target.get("name") or relation["target_id"],
                                triple_updates[relation["triple_id"]],
                            ),
                            "source_id": relation["source_id"],
                            "target_id": relation["target_id"],
                            "type": relation["type"],
                        }
                    )
            await triple_store.upsert(records)
        return len(entity_updates) + len(triple_updates)

    # --- reset / delete / lazy access -------------------------------------

    async def reset(
        self, kb_id: str, *, clear_extraction_result: bool, clear_config: bool
    ) -> dict[str, Any]:
        """Cancel the build task, drop graph data, and reset chunk state."""
        lock = self._get_lock(kb_id)
        async with lock:
            await self._cancel_build_if_running(kb_id)
            self._drop_storage(kb_id)
            for kind in ("entity", "triple"):
                store = self._vector_stores.get(kb_id, {}).pop(kind, None)
                if store is not None:
                    try:
                        await store.drop()
                    except Exception as exc:  # noqa: BLE001 - drop 失败不得阻塞 reset
                        logger.warning(f"Failed to drop {kind} vector store for kb {kb_id}: {exc}")

            await self.graph_repo.delete_by_kb_id(kb_id)
            reset_chunks = await self.chunk_repo.reset_graph_state_by_kb_id(
                kb_id, clear_extraction_result=clear_extraction_result
            )
            if clear_config:
                kb = await self.kb_repo.get_by_kb_id(kb_id)
                if kb is not None:
                    params = dict(kb.additional_params or {})
                    params.pop(GRAPH_CONFIG_KEY, None)
                    await self.kb_repo.update(kb_id, {"additional_params": params})
        return {
            "message": "图谱构建状态已重置",
            "status": "success",
            "reset_chunks": reset_chunks,
            "clear_extraction_result": clear_extraction_result,
            "clear_config": clear_config,
        }

    async def delete_file_graph(self, kb_id: str, file_id: str) -> None:
        """Cascade-delete one file's graph data (mentions, edges, chunks).

        A running build is cancelled first: its flusher could otherwise
        resurrect the deleted file's mentions/edges after
        ``delete_file_references`` removes them. Orphan entities (no residual
        mentions anywhere) and orphan triples are removed from the vector
        stores; shared entities survive.
        """
        await self._cancel_build_if_running(kb_id)
        orphan_refs = await self.graph_repo.delete_file_references(kb_id, file_id)
        storage = self.get_storage(kb_id)
        storage_orphans = storage.delete_file(file_id)
        orphan_entity_ids = sorted(set(storage_orphans) | set(orphan_refs["orphan_entity_ids"]))
        entity_store = await self.get_vector_store(kb_id, "entity")
        await entity_store.delete_ids(orphan_entity_ids)
        triple_store = await self.get_vector_store(kb_id, "triple")
        await triple_store.delete_ids(orphan_refs["orphan_triple_ids"])
        storage.save()

    def get_storage(self, kb_id: str) -> NetworkXGraphStorage:
        """Lazily load (once) the in-memory graph for a KB."""
        storage = self._storages.get(kb_id)
        if storage is None:
            storage = NetworkXGraphStorage(kb_id, self.work_dir)
            storage.load()
            self._storages[kb_id] = storage
        return storage

    async def get_vector_store(self, kb_id: str, kind: str) -> GraphVectorStore:
        """Lazily initialize the graph vector store for a kind (entity/triple)."""
        existing = self._vector_stores.get(kb_id, {}).get(kind)
        if existing is not None:
            return existing
        # Double-checked locking: two concurrent first calls must not build
        # two stores over the same storage file (first-wins, return existing).
        lock = self._vector_store_locks.setdefault(kb_id, {}).setdefault(kind, asyncio.Lock())
        async with lock:
            store = self._vector_stores.get(kb_id, {}).get(kind)
            if store is None:
                if self.embed_func is None:
                    raise ValueError("embed_func 未配置，无法初始化图向量库")
                store = GraphVectorStore(
                    kind=kind, kb_id=kb_id, work_dir=self.work_dir, embedding_func=self.embed_func
                )
                await store.initialize()
                self._vector_stores.setdefault(kb_id, {})[kind] = store
        return store

    def _drop_storage(self, kb_id: str) -> None:
        """Remove the graph storage file and cached instance."""
        storage_file = Path(self.work_dir) / kb_id / "graph_storage.json"
        if storage_file.exists():
            storage_file.unlink()
        self._storages.pop(kb_id, None)


__all__ = ["GraphService"]
