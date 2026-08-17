"""nano-vectordb backed store for graph entities/triples (one file per kind).

Replaces the Milvus dual-collection vector index of the source project: each
KB keeps one ``GraphVectorStore`` per kind ("entity" / "triple"), persisted
under ``<work_dir>/<kb_id>/graph_vdb_<kind>.json``. GraphService writes
extracted rows at build time and uses ``search`` hits as retrieval seeds.

Compared with ``yusu_kb.storage.vector_store.VectorStore``:
- Records are embedded and persisted immediately in ``upsert`` (no pending
  buffer / ``index_done_callback`` flush); same-id rows are replaced.
- The compressed ``vector`` field (float16 + zlib + base64) is omitted:
  nano-vectordb consumes ``__vector__`` on upsert and persists the matrix
  itself (base64 float32 in the JSON), so the extra copy would be redundant.
- Query results carry ``score`` (cosine similarity) instead of ``distance``;
  for the cosine metric nano-vectordb's ``__metrics__`` already is the
  similarity (dot product of normalized vectors, see ``_cosine_query`` in
  nano_vectordb/dbs.py), so no conversion is needed and hits are sorted by
  score descending.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from nano_vectordb import NanoVectorDB

from yusu_kb.utils.logger import logger


class GraphVectorStore:
    """Vector store for one graph record kind (entities or triples).

    file: ``<work_dir>/<kb_id>/graph_vdb_{kind}.json``
    Record fields: id (entity_id/triple_id), content (embedding text), **meta
    (e.g. entity label, triple source_id/target_id/type), passed through
    unchanged.
    """

    def __init__(
        self,
        *,
        kind: str,
        kb_id: str,
        work_dir: Path,
        embedding_func,
        cosine_better_than_threshold: float = 0.2,
        embedding_batch_num: int = 200,
    ) -> None:
        if not kind:
            raise ValueError("kind must be a non-empty string")
        if not kb_id:
            raise ValueError("kb_id must be a non-empty string")
        self.kind = kind
        self.kb_id = kb_id
        self.embedding_func = embedding_func
        self.cosine_better_than_threshold = float(cosine_better_than_threshold)
        self.embedding_batch_num = int(embedding_batch_num or 200)

        kb_dir = Path(work_dir) / kb_id
        kb_dir.mkdir(parents=True, exist_ok=True)
        self._storage_file = kb_dir / f"graph_vdb_{kind}.json"

        self._client: NanoVectorDB | None = None
        self._initialized = False
        # Serializes upsert/delete critical sections: nano-vectordb mutations
        # are non-atomic read-modify-writes, and to_thread calls run in a
        # thread pool, so concurrent writers could interleave and lose rows.
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Create/load the nano-vectordb client, resolving the dimension first.

        Lock-free: by contract (GraphService) this runs before any write.
        """
        dim = self.embedding_func.embedding_dim
        if not dim:
            probe = await self.embedding_func(["__yusu_dim_probe__"])
            dim = int(np.asarray(probe[0]).shape[0])
            if not dim:
                raise ValueError(
                    "Embedding probe returned an empty vector; cannot resolve dimension"
                )
        self._client = NanoVectorDB(
            int(dim), metric="cosine", storage_file=str(self._storage_file)
        )
        self._initialized = True

    async def upsert(self, records: list[dict]) -> None:
        """Embed contents and store rows; same-id records are replaced.

        nano-vectordb's ``upsert`` overwrites existing rows for matching
        ``__id__`` values in place (see ``dbs.py``), so re-upserting an
        entity/triple after re-extraction never duplicates it. Duplicate ids
        within one call: last occurrence wins (dict-merge semantics).
        """
        if not records:
            return
        client = self._require_client()

        async with self._write_lock:
            contents = [str(record.get("content", "")) for record in records]
            batches = [
                contents[i : i + self.embedding_batch_num]
                for i in range(0, len(contents), self.embedding_batch_num)
            ]
            try:
                embeddings_list = await asyncio.gather(
                    *[self.embedding_func(batch) for batch in batches]
                )
            except Exception as e:
                logger.error(f"[{self.kb_id}/{self.kind}] embedding failed: {e}")
                raise
            embedded_count = sum(len(np.asarray(emb)) for emb in embeddings_list)
            if embedded_count != len(records):
                raise RuntimeError(
                    f"[{self.kb_id}/{self.kind}] embedding is not 1-1 with records: "
                    f"{embedded_count} != {len(records)}"
                )
            embeddings = np.concatenate([np.asarray(emb) for emb in embeddings_list])

            rows = []
            for record, embedding in zip(records, embeddings):
                # Spread the record first, then override internals so meta
                # keys (e.g. "vector") cannot clobber the internal fields.
                row = dict(record)
                row.pop("id", None)
                row.pop("vector", None)
                row["__id__"] = str(record["id"])
                row["content"] = str(record.get("content", ""))
                row["__vector__"] = np.asarray(embedding, dtype=np.float32)
                rows.append(row)

            await asyncio.to_thread(client.upsert, rows)
            await asyncio.to_thread(client.save)

    async def search(self, query: str, top_k: int) -> list[dict]:
        """Similarity search; returns ``[{id, content, score, **meta}]``.

        ``score`` is the cosine similarity (``__metrics__`` from
        nano-vectordb); hits are sorted by score descending.
        Lock-free: reads the client only; matrix reference swaps keep reads
        consistent while a write is in flight.
        """
        if top_k <= 0:
            return []
        client = self._require_client()
        embedding = (await self.embedding_func([query]))[0]
        results = await asyncio.to_thread(
            client.query,
            np.asarray(embedding, dtype=np.float32),
            top_k=top_k,
            better_than_threshold=self.cosine_better_than_threshold,
        )
        hits = []
        for data_point in results:
            hit = {
                "id": data_point["__id__"],
                "content": data_point.get("content", ""),
                "score": float(data_point["__metrics__"]),
            }
            for key, value in data_point.items():
                if key not in ("__id__", "__metrics__", "__vector__", "content"):
                    hit[key] = value
            hits.append(hit)
        hits.sort(key=lambda hit: hit["score"], reverse=True)
        return hits

    async def delete_ids(self, ids: list[str]) -> None:
        """Delete records by id and persist the change."""
        if not ids:
            return
        client = self._require_client()
        async with self._write_lock:
            await asyncio.to_thread(client.delete, list(ids))
            await asyncio.to_thread(client.save)

    async def drop(self) -> None:
        """Drop this store: remove the data file and reset the state."""
        self._client = None
        self._initialized = False
        if self._storage_file.exists():
            self._storage_file.unlink()

    def is_initialized(self) -> bool:
        """True once ``initialize`` has resolved the dimension and opened the client."""
        return self._initialized

    def _require_client(self) -> NanoVectorDB:
        if self._client is None:
            raise RuntimeError("GraphVectorStore not initialized; call initialize() first")
        return self._client