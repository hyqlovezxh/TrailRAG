"""Self-contained vector store wrapping nano-vectordb.

Replaces the LightRAG ``NanoVectorDBStorage`` + shared-storage machinery for
the standalone YUSU knowledge base. Per-KB instance: one JSON data file under
``<working_dir>/<workspace>/vdb_<namespace>.json``, buffered deferred
embedding flushed by ``index_done_callback`` (batched, lock-serialized),
synchronous disk I/O offloaded to a worker thread.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from nano_vectordb import NanoVectorDB

from yusu_kb.utils.logger import logger

# Fields that must never leak into snapshot metadata: heavy payloads (the
# base64 vector blob) and nano-vectordb bookkeeping keys.
_SNAPSHOT_DROP_FIELDS = frozenset(
    {"vector", "__vector__", "__metrics__", "__id__", "__created_at__"}
)


@dataclass(frozen=True)
class VectorSnapshot:
    """Row-aligned, read-only snapshot of a flushed vector corpus.

    ``ids``, ``matrix`` rows and ``metas`` correspond 1:1 by index. Rows are
    L2-normalized by the store, so a plain dot product equals cosine similarity.

    The matrix is a read-only view: the store always *replaces* (never mutates
    in place) a cached snapshot, so a reference handed to a caller stays valid
    even if the corpus is re-indexed while the caller is still ranking.
    """

    ids: list[str] = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))
    metas: list[dict[str, Any]] = field(default_factory=list)


class VectorStore:
    """Vector storage for one knowledge base (chunk embeddings).

    API mirrors the subset of LightRAG's ``NanoVectorDBStorage`` that LocalKB
    uses: ``initialize`` / ``upsert`` / ``index_done_callback`` / ``query`` /
    ``delete`` / ``drop``, plus ``embedding_func`` / ``meta_fields`` /
    ``embedding_batch_num`` attributes.

    Embedding dimension is taken from ``embedding_func.embedding_dim`` and
    probed at ``initialize`` when unset (env-driven default model without
    ``YUSU_EMBED_DIM``).
    """

    def __init__(
        self,
        *,
        namespace: str,
        working_dir: str,
        embedding_func,
        workspace: str | None = None,
        embedding_batch_num: int = 200,
        meta_fields: set[str] | None = None,
        cosine_better_than_threshold: float = 0.2,
    ):
        self.namespace = namespace
        self.workspace = workspace or ""
        self.embedding_func = embedding_func
        self.embedding_batch_num = int(embedding_batch_num or 200)
        self.meta_fields = set(meta_fields or ())
        self.cosine_better_than_threshold = float(cosine_better_than_threshold)

        workspace_dir = (
            os.path.join(working_dir, self.workspace) if self.workspace else working_dir
        )
        os.makedirs(workspace_dir, exist_ok=True)
        self._client_file_name = os.path.join(workspace_dir, f"vdb_{namespace}.json")

        self._client: NanoVectorDB | None = None
        self._pending: dict[str, dict[str, Any]] = {}
        self._flush_lock = asyncio.Lock()
        # Lazily materialised row-aligned snapshot; invalidated on every
        # mutation so callers never observe a half-updated corpus.
        self._snapshot_cache: VectorSnapshot | None = None

    async def initialize(self) -> None:
        """Create/load the nano-vectordb client, resolving the dimension first."""
        dim = self.embedding_func.embedding_dim
        if not dim:
            probe = await self.embedding_func(["__yusu_dim_probe__"])
            dim = int(np.asarray(probe[0]).shape[0])
            if not dim:
                raise ValueError("Embedding probe returned an empty vector; cannot resolve dimension")
        self._client = NanoVectorDB(int(dim), storage_file=self._client_file_name)
        self._snapshot_cache = None

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        """Buffer documents (id -> record) for deferred embedding + flush."""
        if not data:
            return
        for doc_id, record in data.items():
            self._pending[doc_id] = dict(record)

    async def index_done_callback(self) -> None:
        """Embed pending documents and persist them to disk (idempotent)."""
        async with self._flush_lock:
            if not self._pending:
                return
            client = self._require_client()
            pending_items = list(self._pending.items())
            doc_ids = [doc_id for doc_id, _ in pending_items]
            records = [record for _, record in pending_items]

            contents = [record.get("content", "") for record in records]
            batches = [
                contents[i : i + self.embedding_batch_num]
                for i in range(0, len(contents), self.embedding_batch_num)
            ]
            logger.info(
                f"[{self.workspace}] {self.namespace} flush: embedding "
                f"{len(records)} vector(s) in {len(batches)} batch(es) "
                f"(batch_num={self.embedding_batch_num})"
            )
            try:
                embeddings_list = await asyncio.gather(
                    *[self.embedding_func(batch) for batch in batches]
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error embedding pending vector ops "
                    f"(upserts={len(records)}): {e}"
                )
                raise
            embeddings = np.concatenate(
                [np.asarray(emb) for emb in embeddings_list]
            )
            if len(embeddings) != len(records):
                raise RuntimeError(
                    f"[{self.workspace}] embedding is not 1-1 with pending data, "
                    f"{len(embeddings)} != {len(records)}"
                )

            list_data = []
            current_time = int(asyncio.get_event_loop().time())
            for doc_id, record, embedding in zip(doc_ids, records, embeddings):
                vector_f16 = np.asarray(embedding, dtype=np.float32).astype(np.float16)
                compressed = zlib.compress(vector_f16.tobytes())
                data_row = {
                    **record,
                    "__id__": doc_id,
                    "__created_at__": current_time,
                    "vector": base64.b64encode(compressed).decode("utf-8"),
                    "__vector__": np.asarray(embedding, dtype=np.float32),
                }
                list_data.append(data_row)

            await asyncio.to_thread(client.upsert, list_data)
            await asyncio.to_thread(client.save)

            for doc_id in doc_ids:
                self._pending.pop(doc_id, None)
            self._snapshot_cache = None

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] | None = None
    ) -> list[dict[str, Any]]:
        """Similarity search over flushed rows (buffered upserts are invisible)."""
        client = self._require_client()
        if query_embedding is not None:
            embedding = query_embedding
        else:
            embedding = (await self.embedding_func([query]))[0]
        results = await asyncio.to_thread(
            client.query,
            np.asarray(embedding, dtype=np.float32),
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        return [
            {
                **{k: v for k, v in dp.items() if k not in ("vector", "__vector__")},
                "id": dp["__id__"],
                "distance": dp["__metrics__"],
                "created_at": dp.get("__created_at__"),
            }
            for dp in results
        ]

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by id and persist the change."""
        if not ids:
            return
        client = self._require_client()
        await asyncio.to_thread(client.delete, list(ids))
        await asyncio.to_thread(client.save)
        self._snapshot_cache = None

    async def drop(self) -> None:
        """Drop this store: remove the data file (pending buffer discarded)."""
        self._client = None
        self._pending.clear()
        self._snapshot_cache = None
        if os.path.exists(self._client_file_name):
            os.remove(self._client_file_name)

    def snapshot(self) -> VectorSnapshot | None:
        """Row-aligned snapshot of the flushed corpus, or ``None`` when unusable.

        Buffered (not yet flushed) upserts are invisible, matching :meth:`query`.
        Returns ``None`` when the store is uninitialised, empty or its data file
        cannot be decoded -- callers must treat that as "no vector evidence" and
        degrade, never raise.

        The result is cached until the next mutation. It is decoded from this
        store's own JSON file (the same file :meth:`index_done_callback` writes),
        which preserves the exact float32 values and keeps us off
        ``nano-vectordb`` private internals.
        """
        if self._snapshot_cache is None:
            self._snapshot_cache = self._decode_snapshot()
        snap = self._snapshot_cache
        if snap is None or not snap.ids:
            return None
        return snap

    def _decode_snapshot(self) -> VectorSnapshot | None:
        """Decode this store's own data file into a row-aligned snapshot.

        File layout mirrors ``nano_vectordb.save``: ``embedding_dim``, a ``data``
        list of rows (row *i* pairs with matrix row *i*) and a base64-encoded
        float32 ``matrix``.
        """
        if not os.path.exists(self._client_file_name):
            return None
        try:
            with open(self._client_file_name, encoding="utf-8") as handle:
                payload = json.load(handle)
            rows = payload.get("data") or []
            dim = int(payload.get("embedding_dim") or 0)
            encoded = payload.get("matrix") or ""
            if not rows or dim <= 0 or not encoded:
                return VectorSnapshot()
            matrix = np.frombuffer(base64.b64decode(encoded), dtype=np.float32)
            if matrix.size != len(rows) * dim:
                logger.warning(
                    f"[{self.workspace}] {self.namespace} snapshot size mismatch: "
                    f"matrix={matrix.size}, rows={len(rows)}, dim={dim}"
                )
                return None
            matrix = matrix.reshape(len(rows), dim)
            ids: list[str] = []
            metas: list[dict[str, Any]] = []
            for row in rows:
                ids.append(str(row.get("__id__") or ""))
                metas.append({k: v for k, v in row.items() if k not in _SNAPSHOT_DROP_FIELDS})
            return VectorSnapshot(ids=ids, matrix=matrix, metas=metas)
        except (OSError, ValueError, TypeError) as exc:
            # 快照解码失败只影响隐式图回退通道，不阻断主检索，交由调用方降级
            logger.warning(f"[{self.workspace}] {self.namespace} snapshot decode failed: {exc}")
            return None

    def _require_client(self) -> NanoVectorDB:
        if self._client is None:
            raise RuntimeError("VectorStore not initialized; call initialize() first")
        return self._client