"""Knowledge base abstract base class.

Ported from YUSU ``yuxi.knowledge.base`` (demo edition):
- file originals and parsed markdown live on the local filesystem
  (``LocalFileStorage``) instead of MinIO;
- folders, preview-to-PDF conversion and share config are not ported;
- stats are stored in the knowledge-base row's ``additional_params`` (SQLite),
  refreshed from the file repository.
"""

from __future__ import annotations

import os
import re
import secrets
import string
from abc import ABC, abstractmethod
from typing import Any

from yusu_kb.knowledge.chunking.presets import (
    ensure_chunk_defaults_in_additional_params,
)
from yusu_kb.knowledge.schemas import (
    FindOutputSchema,
    FindWindowSchema,
    SearchOutputSchema,
    SearchResultSchema,
)
from yusu_kb.utils.datetime_utils import utc_isoformat
from yusu_kb.utils.logger import logger

from .utils.kb_utils import resolve_processing_params, sanitize_processing_params


class FileStatus:
    UPLOADED = "uploaded"
    PARSING = "parsing"
    PARSED = "parsed"
    ERROR_PARSING = "error_parsing"
    INDEXING = "indexing"
    INDEXED = "indexed"
    ERROR_INDEXING = "error_indexing"


INDEXED_STATS_STATUSES = {FileStatus.INDEXED, "done"}


class KnowledgeBaseException(Exception):
    """Unified knowledge base exception base."""


class KBNotFoundError(KnowledgeBaseException):
    """Knowledge base not found."""


class KBOperationError(KnowledgeBaseException):
    """Knowledge base operation error."""


class KnowledgeBase(ABC):
    """Abstract knowledge base class defining the unified interface."""

    kb_type = ""
    name = ""
    description = ""
    requires_embedding_model = True
    supports_documents = True
    apply_chunk_defaults = True

    def __init__(self, work_dir: str):
        """Initialize a knowledge base instance.

        Args:
            work_dir: working directory
        """
        self.work_dir = work_dir
        self.databases_meta: dict[str, dict] = {}
        self.benchmarks_meta: dict[str, dict] = {}
        self._metadata_loaded = False

        from pathlib import Path

        from yusu_kb.storage.files.storage import LocalFileStorage

        self.file_storage = LocalFileStorage(Path(work_dir) / "files")

        os.makedirs(work_dir, exist_ok=True)

    def _ensure_metadata_loaded(self):
        if not self._metadata_loaded:
            logger.warning(f"{self.kb_type}: metadata not loaded yet")

    def _normalize_metadata_state(self) -> None:
        for meta in self.databases_meta.values():
            if "created_at" in meta:
                normalized = self._normalize_timestamp(meta.get("created_at"))
                if normalized:
                    meta["created_at"] = normalized

    @staticmethod
    def _normalize_timestamp(value: Any) -> str | None:
        from yusu_kb.utils.datetime_utils import coerce_any_to_utc_datetime

        try:
            dt_value = coerce_any_to_utc_datetime(value)
        except (TypeError, ValueError) as exc:
            logger.warning(f"Invalid timestamp encountered: {value!r} ({exc})")
            return None

        if not dt_value:
            return None
        return utc_isoformat(dt_value)

    @classmethod
    def get_create_params_config(cls) -> dict[str, Any]:
        """Type-specific create parameter configuration."""
        return {"options": []}

    @classmethod
    def validate_additional_params(cls, additional_params: dict | None) -> dict:
        """Validate and normalize type-specific configuration."""
        return dict(additional_params or {})

    @classmethod
    def normalize_additional_params(cls, additional_params: dict | None) -> dict:
        """Normalize additional_params; document KBs get chunk defaults applied."""
        params = cls.validate_additional_params(additional_params)
        if cls.apply_chunk_defaults:
            return ensure_chunk_defaults_in_additional_params(params)
        return params

    @abstractmethod
    async def _create_kb_instance(self, kb_id: str, config: dict) -> Any:
        """Create the underlying knowledge base instance."""

    @abstractmethod
    async def _initialize_kb_instance(self, instance: Any) -> None:
        """Initialize the underlying knowledge base instance."""

    # ------------------------------------------------------------------
    # Database (knowledge base) lifecycle
    # ------------------------------------------------------------------

    async def create_database(
        self,
        database_name: str,
        description: str,
        embedding_model_spec: str | None = None,
        llm_model_spec: str | None = None,
        record_fields: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """Create a knowledge base."""
        kwargs = self.normalize_additional_params(kwargs)
        kwargs["stats"] = {"file_count": 0, "chunk_count": 0, "token_count": 0}

        alphabet = string.ascii_lowercase + string.digits
        while True:
            kb_id = "kb_" + "".join(secrets.choice(alphabet) for _ in range(10))
            if kb_id not in self.databases_meta:
                break

        self.databases_meta[kb_id] = {
            "name": database_name,
            "description": description,
            "kb_type": self.kb_type,
            "embedding_model_spec": embedding_model_spec,
            "llm_model_spec": llm_model_spec,
            "metadata": kwargs,
            "created_at": utc_isoformat(),
            # query_params.options only records user-explicit overrides; the
            # runtime merges schema defaults dynamically.
            "query_params": {"options": {}},
        }
        await self._persist_kb(kb_id, record_fields=record_fields)

        # Create the working directory
        working_dir = os.path.join(self.work_dir, kb_id)
        os.makedirs(working_dir, exist_ok=True)

        db_dict = self.databases_meta[kb_id].copy()
        db_dict["kb_id"] = kb_id
        db_dict["files"] = {}
        return db_dict

    async def delete_database(self, kb_id: str) -> dict:
        """Delete a knowledge base (metadata + files + working directory)."""
        if kb_id in self.databases_meta:
            from yusu_kb.repositories.knowledge_base_repository import (
                KnowledgeBaseRepository,
            )
            from yusu_kb.repositories.knowledge_file_repository import (
                KnowledgeFileRepository,
            )

            file_repo = KnowledgeFileRepository()
            await file_repo.delete_by_kb_id(kb_id)
            kb_repo = KnowledgeBaseRepository()
            await kb_repo.delete(kb_id)
            del self.databases_meta[kb_id]

        # Delete the working directory
        working_dir = os.path.join(self.work_dir, kb_id)
        if os.path.exists(working_dir):
            import shutil

            try:
                shutil.rmtree(working_dir)
            except OSError as e:
                logger.error(f"Error deleting working directory {working_dir}: {e}")

        return {"message": "删除成功"}

    def update_database(
        self,
        kb_id: str,
        name: str,
        description: str,
        llm_model_spec: str | None = None,
        update_llm_model_spec: bool = False,
    ) -> dict:
        """Update a knowledge base."""
        if kb_id not in self.databases_meta:
            raise ValueError(f"数据库 {kb_id} 不存在")

        self.databases_meta[kb_id]["name"] = name
        self.databases_meta[kb_id]["description"] = description
        if update_llm_model_spec:
            self.databases_meta[kb_id]["llm_model_spec"] = llm_model_spec

        return self.get_database_info(kb_id)

    def get_database_info(self, kb_id: str, include_files: bool = True) -> dict | None:
        """Get detailed knowledge base information."""
        if kb_id not in self.databases_meta:
            return None

        meta = self.databases_meta[kb_id].copy()
        meta["kb_id"] = kb_id

        meta["stats"] = self._get_database_stats(kb_id)
        meta["row_count"] = meta["stats"].get("row_count") or meta["stats"].get("file_count") or 0

        if include_files:
            meta["files"] = {}
            meta["files_truncated"] = True

        meta["status"] = "已连接"
        return meta

    def get_databases(self, include_files: bool = False) -> dict:
        """Get all knowledge bases."""
        self._ensure_metadata_loaded()

        databases = []
        for kb_id, meta in self.databases_meta.items():
            db_dict = meta.copy()
            db_dict["kb_id"] = kb_id
            db_dict["stats"] = self._get_database_stats(kb_id)
            db_dict["row_count"] = db_dict["stats"].get("row_count") or db_dict["stats"].get("file_count") or 0

            if include_files:
                db_dict["files"] = {}
                db_dict["files_truncated"] = True

            db_dict["status"] = "已连接"
            databases.append(db_dict)

        return {"databases": databases}

    # ------------------------------------------------------------------
    # Query params
    # ------------------------------------------------------------------

    @abstractmethod
    def get_query_params_config(self, kb_id: str, **kwargs) -> dict:
        """Get the query parameter configuration for this KB type."""

    def _get_query_params(self, kb_id: str) -> dict:
        """Load saved query params from instance metadata."""
        if kb_id in self.databases_meta:
            query_params_meta = self.databases_meta[kb_id].get("query_params") or {}
            return query_params_meta.get("options", {})
        return {}

    def _get_default_query_params(self, kb_id: str) -> dict[str, Any]:
        """Extract defaults from get_query_params_config, returning {"options": {...}}."""
        config = self.get_query_params_config(kb_id)
        defaults = {}
        for opt in config.get("options", []):
            if "default" in opt:
                defaults[opt["key"]] = opt["default"]
        return {"options": defaults}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _build_database_stats(self, kb_id: str) -> dict[str, int]:
        del kb_id
        return self._normalize_database_stats(None)

    @staticmethod
    def _normalize_database_stats(stats: dict | None) -> dict[str, int]:
        normalized = {
            "file_count": 0,
            "folder_count": 0,
            "row_count": 0,
            "total_size": 0,
            "chunk_count": 0,
            "token_count": 0,
            "pending_parse_count": 0,
            "pending_index_count": 0,
            "processing_count": 0,
        }
        if not isinstance(stats, dict):
            return normalized

        for key in normalized:
            try:
                normalized[key] = max(int(stats.get(key) or 0), 0)
            except (TypeError, ValueError):
                normalized[key] = 0
        return normalized

    def _get_database_stats(self, kb_id: str) -> dict[str, int]:
        metadata = self.databases_meta.get(kb_id, {}).get("metadata") or {}
        stats = metadata.get("stats") if isinstance(metadata, dict) else None
        if isinstance(stats, dict):
            return self._normalize_database_stats(stats)
        return self._build_database_stats(kb_id)

    def _set_database_stats(self, kb_id: str, stats: dict[str, int]) -> None:
        if kb_id not in self.databases_meta:
            raise ValueError(f"Database {kb_id} not found")

        metadata = self.databases_meta[kb_id].setdefault("metadata", {})
        metadata["stats"] = self._normalize_database_stats(stats)

    async def refresh_database_stats(self, kb_id: str) -> dict[str, int]:
        from yusu_kb.repositories.knowledge_file_repository import (
            KnowledgeFileRepository,
        )

        stats = await KnowledgeFileRepository().get_kb_file_stats(kb_id)
        self._set_database_stats(kb_id, stats)
        await self._persist_kb(kb_id)
        return stats

    # ------------------------------------------------------------------
    # File records
    # ------------------------------------------------------------------

    def _file_record_to_meta(self, record: Any) -> dict:
        kb_additional_params = self.databases_meta.get(record.kb_id, {}).get("metadata") or {}
        return {
            "file_id": record.file_id,
            "kb_id": record.kb_id,
            "parent_id": record.parent_id,
            "filename": record.filename,
            "file_type": record.file_type,
            "path": record.path,
            "markdown_file": record.markdown_file,
            "status": record.status,
            "content_hash": record.content_hash,
            "size": record.file_size,
            "chunk_count": int(getattr(record, "chunk_count", 0) or 0),
            "token_count": int(getattr(record, "token_count", 0) or 0),
            "content_type": record.content_type,
            "processing_params": sanitize_processing_params(
                resolve_processing_params(
                    kb_additional_params=kb_additional_params,
                    file_processing_params=record.processing_params,
                )
            ),
            "is_folder": record.is_folder,
            "error": record.error_message,
            "created_by": record.created_by,
            "updated_by": record.updated_by,
            "created_at": utc_isoformat(record.created_at) if record.created_at else None,
            "updated_at": utc_isoformat(record.updated_at) if record.updated_at else None,
            "original_filename": record.original_filename,
        }

    @staticmethod
    def _file_meta_to_record_data(meta: dict) -> dict[str, Any]:
        return {
            "kb_id": meta.get("kb_id"),
            "parent_id": meta.get("parent_id"),
            "filename": meta.get("filename") or "",
            "original_filename": meta.get("original_filename"),
            "file_type": meta.get("file_type"),
            "path": meta.get("path"),
            "markdown_file": meta.get("markdown_file"),
            "status": meta.get("status"),
            "content_hash": meta.get("content_hash"),
            "file_size": meta.get("size"),
            "chunk_count": int(meta.get("chunk_count") or 0),
            "token_count": int(meta.get("token_count") or 0),
            "content_type": meta.get("content_type"),
            "processing_params": sanitize_processing_params(meta.get("processing_params")),
            "is_folder": meta.get("is_folder", False),
            "error_message": meta.get("error"),
            "created_by": meta.get("created_by"),
            "updated_by": meta.get("updated_by"),
        }

    async def _load_file_meta(self, kb_id: str, file_id: str, *, refresh: bool = False) -> dict:
        del refresh

        from yusu_kb.repositories.knowledge_file_repository import (
            KnowledgeFileRepository,
        )

        record = await KnowledgeFileRepository().get_by_file_id(file_id)
        if record is None or record.kb_id != kb_id:
            raise ValueError(f"File {file_id} not found")

        return self._file_record_to_meta(record)

    async def _get_file_meta(self, kb_id: str, file_id: str) -> dict:
        return await self._load_file_meta(kb_id, file_id)

    async def _persist_file_meta(self, file_id: str, meta: dict) -> None:
        from yusu_kb.repositories.knowledge_file_repository import (
            KnowledgeFileRepository,
        )

        data = self._file_meta_to_record_data(meta)
        if not data.get("kb_id"):
            return
        await KnowledgeFileRepository().upsert(file_id=file_id, data=data)

    async def _persist_kb(self, kb_id: str, record_fields: dict[str, Any] | None = None) -> None:
        """Persist a single knowledge base row."""
        from yusu_kb.repositories.knowledge_base_repository import (
            KnowledgeBaseRepository,
        )

        kb_repo = KnowledgeBaseRepository()

        if kb_id not in self.databases_meta:
            return

        meta = self.databases_meta[kb_id]
        existing = await kb_repo.get_by_kb_id(kb_id)
        payload = {
            "kb_id": kb_id,
            "name": meta.get("name") or kb_id,
            "description": meta.get("description"),
            "kb_type": meta.get("kb_type") or self.kb_type,
            "embedding_model_spec": meta.get("embedding_model_spec"),
            "llm_model_spec": meta.get("llm_model_spec"),
            "query_params": meta.get("query_params"),
            "additional_params": meta.get("metadata") or {},
        }
        if record_fields:
            allowed_fields = {"created_by"}
            payload.update({key: value for key, value in record_fields.items() if key in allowed_fields})

        if existing is None:
            await kb_repo.create(payload)
        else:
            update_data = {
                "name": payload["name"],
                "description": payload["description"],
                "kb_type": payload["kb_type"],
                "embedding_model_spec": payload["embedding_model_spec"],
                "llm_model_spec": payload["llm_model_spec"],
                "query_params": payload["query_params"],
                "additional_params": payload["additional_params"],
            }
            await kb_repo.update(kb_id, update_data)

    async def _load_metadata(self) -> None:
        from yusu_kb.repositories.knowledge_base_repository import (
            KnowledgeBaseRepository,
        )

        kb_repo = KnowledgeBaseRepository()

        databases = [kb for kb in await kb_repo.get_all() if kb.kb_type == self.kb_type]
        self.databases_meta = {
            kb.kb_id: {
                "name": kb.name,
                "description": kb.description,
                "kb_type": kb.kb_type,
                "embedding_model_spec": kb.embedding_model_spec,
                "llm_model_spec": kb.llm_model_spec,
                "query_params": kb.query_params or {"options": {}},
                "metadata": self.normalize_additional_params(kb.additional_params),
                "created_at": utc_isoformat(kb.created_at) if kb.created_at else utc_isoformat(),
            }
            for kb in databases
        }

        self.benchmarks_meta = {}
        self._normalize_metadata_state()
        self._metadata_loaded = True

        logger.info(f"Loaded {self.kb_type} metadata from database for {len(self.databases_meta)} databases")

    # ------------------------------------------------------------------
    # File operations shared by all implementations
    # ------------------------------------------------------------------

    async def add_file_record(
        self, kb_id: str, item: str, params: dict | None = None, operator_id: str | None = None
    ) -> dict:
        """Add a file record (Status: UPLOADED).

        Args:
            kb_id: database ID
            item: local file path of the uploaded original
            params: processing parameters
            operator_id: creating user ID
        """
        from yusu_kb.knowledge.parser.unified import is_supported_file_extension
        from yusu_kb.utils.hash_utils import hashstr

        params = params or {}
        content_type = params.get("content_type", "file")
        if content_type != "file":
            raise ValueError(f"Unsupported content_type: {content_type}")

        import time

        file_path = item
        if not os.path.isfile(file_path):
            raise ValueError(f"File source does not exist: {file_path}")

        filename = os.path.basename(file_path)
        if not is_supported_file_extension(filename):
            raise ValueError(f"不支持的文档类型: {filename}")

        file_type = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        file_id = f"file_{hashstr(str(file_path) + str(time.time()), 6)}"

        kb_additional_params = self.databases_meta.get(kb_id, {}).get("metadata") or {}
        metadata = {
            "kb_id": kb_id,
            "filename": filename,
            "file_type": file_type,
            "path": file_path,
            "status": FileStatus.UPLOADED,
            "created_at": utc_isoformat(),
            "file_id": file_id,
            "content_hash": None,
            "size": os.path.getsize(file_path),
            "processing_params": resolve_processing_params(
                kb_additional_params=kb_additional_params,
                file_processing_params=params,
            ),
        }
        if operator_id:
            metadata["created_by"] = operator_id

        await self._persist_file_meta(file_id, metadata)
        await self.refresh_database_stats(kb_id)

        return metadata

    async def parse_file(self, kb_id: str, file_id: str, operator_id: str | None = None) -> dict:
        """Parse a file to markdown (Status: PARSING -> PARSED/ERROR_PARSING)."""
        allowed_statuses = {
            FileStatus.UPLOADED,
            FileStatus.ERROR_PARSING,
            "failed",  # legacy status
        }

        from yusu_kb.repositories.knowledge_file_repository import (
            KnowledgeFileRepository,
        )

        file_repo = KnowledgeFileRepository()
        claim_data = {"status": FileStatus.PARSING, "error_message": None}
        if operator_id:
            claim_data["updated_by"] = operator_id
        claimed_record = await file_repo.update_fields_if_status(
            kb_id=kb_id,
            file_id=file_id,
            allowed_statuses=allowed_statuses,
            data=claim_data,
        )
        if claimed_record is None:
            current_meta = await self._load_file_meta(kb_id, file_id)
            current_status = current_meta.get("status")
            if current_status in (FileStatus.PARSING, FileStatus.PARSED):
                logger.info(f"File {file_id} already {current_status} by another task, skipping")
                return current_meta
            raise ValueError(
                f"Cannot parse file with status '{current_status}'. "
                f"File must be in one of these states: {', '.join(allowed_statuses)}"
            )

        file_meta = self._file_record_to_meta(claimed_record)
        file_path = file_meta.get("path")
        if not file_path:
            message = f"File {file_id} has no valid path in metadata"
            update_data = {"status": FileStatus.ERROR_PARSING, "error_message": message}
            if operator_id:
                update_data["updated_by"] = operator_id
            await file_repo.update_fields(file_id=file_id, kb_id=kb_id, data=update_data)
            raise ValueError(message)

        try:
            from yusu_kb.knowledge.parser.unified import Parser
            from yusu_kb.storage.files.storage import sanitize_filename

            params = file_meta.get("processing_params", {}) or {}
            markdown_path = self.file_storage.parsed_dir / f"{sanitize_filename(file_id)}.md"
            images_dir = str(markdown_path.parent / "images")
            markdown_content = await Parser.aparse(
                source=file_path, params=params, images_dir=images_dir
            )

            markdown_file_path = await self._save_markdown_file(kb_id, file_id, markdown_content)

            file_meta["status"] = FileStatus.PARSED
            file_meta["markdown_file"] = markdown_file_path
            file_meta["error"] = None
            file_meta["updated_at"] = utc_isoformat()
            if operator_id:
                file_meta["updated_by"] = operator_id
            update_data = {
                "status": FileStatus.PARSED,
                "markdown_file": markdown_file_path,
                "error_message": None,
            }
            if operator_id:
                update_data["updated_by"] = operator_id
            await file_repo.update_fields(file_id=file_id, kb_id=kb_id, data=update_data)

            return file_meta

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to parse file {file_id}: {error_msg}")

            file_meta["status"] = FileStatus.ERROR_PARSING
            file_meta["error"] = error_msg
            file_meta["updated_at"] = utc_isoformat()
            if operator_id:
                file_meta["updated_by"] = operator_id
            update_data = {"status": FileStatus.ERROR_PARSING, "error_message": error_msg}
            if operator_id:
                update_data["updated_by"] = operator_id
            await file_repo.update_fields(file_id=file_id, kb_id=kb_id, data=update_data)

            raise

    async def update_file_params(self, kb_id: str, file_id: str, params: dict, operator_id: str | None = None) -> None:
        """Update file processing params."""
        if not params:
            return

        file_meta = await self._load_file_meta(kb_id, file_id)
        current_params = file_meta.get("processing_params", {}) or {}
        kb_additional_params = self.databases_meta.get(kb_id, {}).get("metadata") or {}

        current_params = resolve_processing_params(
            kb_additional_params=kb_additional_params,
            file_processing_params=current_params,
            request_params=params,
        )

        file_meta["processing_params"] = current_params
        file_meta["updated_at"] = utc_isoformat()
        if operator_id:
            file_meta["updated_by"] = operator_id

        from yusu_kb.repositories.knowledge_file_repository import (
            KnowledgeFileRepository,
        )

        update_data = {"processing_params": sanitize_processing_params(current_params)}
        if operator_id:
            update_data["updated_by"] = operator_id
        record = await KnowledgeFileRepository().update_fields(file_id=file_id, kb_id=kb_id, data=update_data)
        if record is None:
            raise ValueError(f"File {file_id} not found")

    async def _mark_file_unparsed(self, kb_id: str, file_id: str, operator_id: str | None = None) -> None:
        from yusu_kb.repositories.knowledge_file_repository import (
            KnowledgeFileRepository,
        )

        update_data = {"status": FileStatus.UPLOADED, "markdown_file": None, "error_message": None}
        if operator_id:
            update_data["updated_by"] = operator_id
        record = await KnowledgeFileRepository().update_fields(file_id=file_id, kb_id=kb_id, data=update_data)
        if record is None:
            raise ValueError(f"File {file_id} not found")

    async def _save_markdown_file(self, kb_id: str, file_id: str, content: str) -> str:
        """Save parsed markdown to the local filesystem and return its path."""
        del kb_id
        path = self.file_storage.save_parsed(file_id, content)
        return str(path)

    async def _read_markdown(self, markdown_file: str) -> str:
        from pathlib import Path

        path = Path(markdown_file)
        if not path.is_file():
            raise FileNotFoundError(markdown_file)
        return path.read_text(encoding="utf-8")

    async def _read_original(self, file_path: str) -> bytes:
        from pathlib import Path

        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(file_path)
        return path.read_bytes()

    async def batch_get_file_statuses(self, kb_id: str, file_ids: list[str]) -> dict[str, str]:
        """Batch query file statuses from DB: {file_id: status}."""
        from yusu_kb.repositories.knowledge_file_repository import (
            KnowledgeFileRepository,
        )

        return await KnowledgeFileRepository().batch_get_file_statuses(kb_id=kb_id, file_ids=file_ids)

    def _knowledge_file_entry(self, kb_id: str, file_id: str, file_meta: dict) -> dict:
        is_dir = bool(file_meta.get("is_folder"))
        original_path = file_meta.get("path")
        path = f"/{file_id}"
        if is_dir:
            path = f"{path}/"
        return {
            "source": "knowledge",
            "kb_id": kb_id,
            "file_id": file_id,
            "parent_id": file_meta.get("parent_id"),
            "path": path,
            "virtual_path": f"/knowledge/{kb_id}/{file_id}",
            "name": file_meta.get("filename") or file_meta.get("original_filename") or file_id,
            "is_dir": is_dir,
            "size": 0 if is_dir else file_meta.get("size") or 0,
            "modified_at": file_meta.get("updated_at") or file_meta.get("created_at") or "",
            "readonly": True,
            "status": file_meta.get("status", "done"),
            "has_original_file": bool(original_path),
            "has_parsed_markdown": bool(file_meta.get("markdown_file")),
        }

    # ------------------------------------------------------------------
    # Preview / download / open / find
    # ------------------------------------------------------------------

    async def read_file_preview(self, kb_id: str, file_id: str) -> dict:
        """Preview a file: parsed markdown when available, otherwise text
        preview of the original (binary originals fall back to unsupported)."""
        file_meta = await self._get_file_meta(kb_id, file_id)
        if file_meta.get("is_folder"):
            raise ValueError("Cannot preview a folder")

        filename = file_meta.get("filename") or file_meta.get("original_filename") or file_id
        response = {
            "source": "knowledge",
            "kb_id": kb_id,
            "file_id": file_id,
            "filename": filename,
            "readonly": True,
        }

        markdown_file = file_meta.get("markdown_file")
        if markdown_file:
            try:
                content = await self._read_markdown(markdown_file)
            except FileNotFoundError:
                content = ""
            if content:
                return {
                    **response,
                    "content": content,
                    "preview_type": "markdown",
                    "supported": True,
                    "message": None,
                }

        original_path = file_meta.get("path")
        if not original_path:
            return {
                **response,
                "content": None,
                "preview_type": "unsupported",
                "supported": False,
                "message": "文件没有可预览的原始内容",
            }

        try:
            raw_content = await self._read_original(original_path)
        except FileNotFoundError:
            return {
                **response,
                "content": None,
                "preview_type": "unsupported",
                "supported": False,
                "message": "原始文件不存在",
            }

        text_extensions = {".txt", ".md", ".csv", ".json", ".html", ".htm"}
        if os.path.splitext(filename)[1].lower() in text_extensions:
            try:
                text = raw_content.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_content.decode("utf-8", errors="replace")
            return {
                **response,
                "content": text,
                "preview_type": "text",
                "supported": True,
                "message": None,
            }

        return {
            **response,
            "content": None,
            "preview_type": "binary",
            "supported": False,
            "message": "该文件类型不支持文本预览，可下载原始文件或查看解析结果",
        }

    async def get_file_download(self, kb_id: str, file_id: str, variant: str = "original") -> dict:
        file_meta = await self._get_file_meta(kb_id, file_id)
        if file_meta.get("is_folder"):
            raise ValueError("Cannot download a folder")
        if variant not in {"original", "parsed"}:
            raise ValueError("Unsupported download variant")

        filename = file_meta.get("filename") or file_meta.get("original_filename") or file_id
        if variant == "parsed":
            markdown_file = file_meta.get("markdown_file")
            if not markdown_file:
                raise ValueError("文件尚未生成解析结果")
            return {
                "filename": f"{filename}.parsed.md",
                "content": (await self._read_markdown(markdown_file)).encode("utf-8"),
                "media_type": "text/markdown; charset=utf-8",
            }

        original_path = file_meta.get("path")
        if not original_path:
            raise ValueError("文件没有可下载的原始内容")
        import mimetypes

        media_type = file_meta.get("content_type") or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return {
            "filename": filename,
            "content": await self._read_original(original_path),
            "media_type": media_type,
        }

    def _build_open_file_window(self, content: str, *, offset: int = 0, limit: int = 800) -> dict[str, Any]:
        lines = content.splitlines()
        total_lines = len(lines)
        start = min(max(int(offset), 0), total_lines)
        window_size = min(max(int(limit), 1), 2000)
        selected = lines[start : start + window_size]
        end = start + len(selected)

        return {
            "start_line": start + 1 if selected else 0,
            "end_line": end,
            "total_lines": total_lines,
            "offset": start,
            "window_size": window_size,
            "has_more_before": start > 0,
            "has_more_after": end < total_lines,
            "next_offset": end if end < total_lines else None,
            "content": "\n".join(f"{start + idx + 1:6d}\t{line}" for idx, line in enumerate(selected)),
        }

    @staticmethod
    def build_search_output(kb_id: str, retrieval_results: Any) -> dict[str, Any] | Any:
        if not isinstance(retrieval_results, list):
            return retrieval_results

        results = []
        for index, chunk in enumerate(retrieval_results):
            if not isinstance(chunk, dict):
                continue

            metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
            metadata = {
                key: value
                for key, value in metadata.items()
                if key not in {"filepath", "parsed_path", "path", "markdown_file"}
            }
            file_id = metadata.get("file_id") or chunk.get("file_id") or chunk.get("full_doc_id") or ""
            chunk_id = metadata.get("chunk_id") or chunk.get("chunk_id") or chunk.get("id")
            chunk_index = metadata.get("chunk_index")
            if chunk_index is None:
                chunk_index = chunk.get("chunk_index")
            if chunk_index is not None:
                metadata.setdefault("chunk_index", chunk_index)
            if chunk.get("score") is not None:
                metadata.setdefault("score", chunk.get("score"))
            if chunk.get("distance") is not None:
                metadata.setdefault("distance", chunk.get("distance"))
            # 图增强检索的 PPR 扩散分随结果透出（供前端/评估标注图来源命中）
            if chunk.get("graph_score") is not None:
                metadata.setdefault("graph_score", chunk.get("graph_score"))

            results.append(
                SearchResultSchema(
                    id=str(chunk_id or f"{file_id}:{index + 1}"),
                    kb_id=str(kb_id),
                    file_id=str(file_id or ""),
                    content=str(chunk.get("content") or ""),
                    metadata=metadata,
                )
            )

        return SearchOutputSchema(kb_id=str(kb_id), results=results).model_dump()

    @staticmethod
    def _build_find_file_windows(
        content: str,
        *,
        patterns: list[str],
        use_regex: bool = False,
        case_sensitive: bool = False,
        max_windows: int = 5,
        window_size: int = 80,
    ) -> dict[str, Any]:
        patterns = [pattern for pattern in patterns if pattern]
        if not patterns:
            raise ValueError("请提供至少一个 pattern")

        lines = content.splitlines()
        flags = 0 if case_sensitive else re.IGNORECASE
        if use_regex:
            matchers = [re.compile(pattern, flags) for pattern in patterns]

            def line_matches(line: str) -> bool:
                return any(matcher.search(line) for matcher in matchers)

        else:
            normalized_patterns = patterns if case_sensitive else [pattern.lower() for pattern in patterns]

            def line_matches(line: str) -> bool:
                haystack = line if case_sensitive else line.lower()
                return any(pattern in haystack for pattern in normalized_patterns)

        matched_indexes = [index for index, line in enumerate(lines) if line_matches(line)]
        windows: list[FindWindowSchema] = []
        covered_until = -1
        normalized_window_size = min(max(int(window_size), 1), 200)
        half_window = normalized_window_size // 2

        for matched_index in matched_indexes:
            if matched_index < covered_until:
                continue
            start = max(matched_index - half_window, 0)
            end = min(start + normalized_window_size, len(lines))
            start = max(end - normalized_window_size, 0)
            matched_lines = [index + 1 for index in matched_indexes if start <= index < end]
            selected = lines[start:end]
            windows.append(
                FindWindowSchema(
                    start_line=start + 1 if selected else 0,
                    end_line=end,
                    matched_lines=matched_lines,
                    content="\n".join(f"{start + idx + 1:6d}\t{line}" for idx, line in enumerate(selected)),
                )
            )
            covered_until = end
            if len(windows) >= max_windows:
                break

        return FindOutputSchema(
            kb_id="",
            file_id="",
            semantic=False,
            match_mode="regex" if use_regex else "keyword",
            total_matches=len(matched_indexes),
            windows=windows,
        ).model_dump(exclude={"kb_id", "file_id"})

    async def open_file_content(self, kb_id: str, file_id: str, offset: int = 0, limit: int = 800) -> dict:
        """Open the parsed markdown of a file as a line window."""
        try:
            file_meta = await self._load_file_meta(kb_id, file_id)
        except ValueError as exc:
            raise KBOperationError(f"文件不存在: {file_id}") from exc
        if file_meta.get("is_folder"):
            raise KBOperationError(f"文件 {file_id} 是文件夹")

        markdown_file = file_meta.get("markdown_file")
        if not markdown_file:
            raise KBOperationError(f"文件 {file_id} 没有解析后的 Markdown 内容")

        content = await self._read_markdown(markdown_file)
        return self._build_open_file_window(content, offset=offset, limit=limit)

    async def find_file_content(
        self,
        kb_id: str,
        file_id: str,
        patterns: list[str],
        *,
        use_regex: bool = False,
        case_sensitive: bool = False,
        max_windows: int = 5,
        window_size: int = 80,
    ) -> dict:
        """Find keyword/regex patterns inside a file's parsed markdown."""
        try:
            file_meta = await self._load_file_meta(kb_id, file_id)
        except ValueError as exc:
            raise KBOperationError(f"文件不存在: {file_id}") from exc
        if file_meta.get("is_folder"):
            raise KBOperationError(f"文件 {file_id} 是文件夹")

        markdown_file = file_meta.get("markdown_file")
        if not markdown_file:
            raise KBOperationError(f"文件 {file_id} 没有解析后的 Markdown 内容")

        content = await self._read_markdown(markdown_file)
        return self._build_find_file_windows(
            content,
            patterns=patterns,
            use_regex=use_regex,
            case_sensitive=case_sensitive,
            max_windows=max_windows,
            window_size=window_size,
        )

    def get_retrievers(self) -> dict[str, dict]:
        """Get retrieval functions for every database (used by the chat loop)."""
        retrievers = {}
        for kb_id, meta in self.databases_meta.items():

            def make_retriever(kb_id):
                async def retriever(query_text, **kwargs):
                    results = await self.aquery(query_text, kb_id, agent_call=True, **kwargs)
                    return self.build_search_output(kb_id, results)

                return retriever

            retrievers[kb_id] = {
                "name": meta["name"],
                "description": meta["description"],
                "retriever": make_retriever(kb_id),
                "metadata": meta,
            }
        return retrievers

    # ------------------------------------------------------------------
    # Abstract methods implemented by concrete KB types
    # ------------------------------------------------------------------

    @abstractmethod
    async def index_file(self, kb_id: str, file_id: str, operator_id: str | None = None) -> dict:
        """Index a parsed file (Status: INDEXING -> INDEXED/ERROR_INDEXING)."""

    @abstractmethod
    async def update_content(self, kb_id: str, file_ids: list[str], params: dict | None = None) -> list[dict]:
        """Update content — re-parse and re-index the given files."""

    @abstractmethod
    async def aquery(self, query_text: str, kb_id: str, **kwargs) -> list[dict]:
        """Query the knowledge base; returns a list of retrieved chunk dicts."""

    @abstractmethod
    async def delete_file(self, kb_id: str, file_id: str) -> None:
        """Delete a file (vectors, chunks, parsed markdown, original)."""

    @abstractmethod
    async def get_file_basic_info(self, kb_id: str, file_id: str) -> dict:
        """Get basic file info (metadata only)."""

    @abstractmethod
    async def get_file_content(self, kb_id: str, file_id: str) -> dict:
        """Get file content info (chunks and lines)."""

    @abstractmethod
    async def get_file_info(self, kb_id: str, file_id: str) -> dict:
        """Get full file info (basic info + content info)."""

    async def export_data(self, kb_id: str, format: str = "zip", **kwargs) -> str:
        raise NotImplementedError("export_data not implemented")