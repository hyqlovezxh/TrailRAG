"""Knowledge base manager — unified entry point for KB operations.

Ported from YUSU ``yuxi.knowledge.manager`` (demo edition): folders,
permissions, MinIO and mindmap are not ported.
"""

from __future__ import annotations

import asyncio
from typing import Any

from yusu_kb.utils.logger import logger

from .base import KBNotFoundError, KBOperationError, KnowledgeBase
from .factory import KnowledgeBaseFactory


class KnowledgeBaseManager:
    """Manage knowledge bases of all registered types."""

    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self._instances: dict[str, KnowledgeBase] = {}

    def get_instance(self, kb_type: str) -> KnowledgeBase:
        """Get (or create) the shared instance for a knowledge base type."""
        if kb_type not in self._instances:
            self._instances[kb_type] = KnowledgeBaseFactory.create(kb_type, self.work_dir)
        return self._instances[kb_type]

    def is_type_supported(self, kb_type: str) -> bool:
        return KnowledgeBaseFactory.is_type_supported(kb_type)

    def get_available_types(self) -> dict:
        return KnowledgeBaseFactory.get_available_types()

    async def load_all_metadata(self) -> None:
        """Load metadata for every registered type (call once at startup)."""
        for kb_type in list(KnowledgeBaseFactory.get_available_types()):
            try:
                await self.get_instance(kb_type)._load_metadata()
            except Exception as exc:  # noqa: BLE001 - one type failing must not block others
                logger.error(f"Failed to load metadata for type {kb_type}: {exc}")

    async def close(self) -> None:
        """Release resources of all instances."""
        for kb in self._instances.values():
            closer = getattr(kb, "close", None)
            if closer is not None:
                try:
                    await closer()
                except Exception as exc:  # noqa: BLE001 - shutdown must not raise
                    logger.error(f"Failed to close KB instance {kb.kb_type}: {exc}")
        self._instances.clear()

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------

    async def create_database(
        self,
        name: str,
        description: str = "",
        kb_type: str = "local",
        embedding_model_spec: str | None = None,
        llm_model_spec: str | None = None,
        additional_params: dict | None = None,
        created_by: str | None = None,
    ) -> dict:
        """Create a knowledge base."""
        if not self.is_type_supported(kb_type):
            raise KBOperationError(f"不支持的知识库类型: {kb_type}")
        if not name or not name.strip():
            raise KBOperationError("知识库名称不能为空")

        return await self.get_instance(kb_type).create_database(
            database_name=name.strip(),
            description=description or "",
            embedding_model_spec=embedding_model_spec,
            llm_model_spec=llm_model_spec,
            record_fields={"created_by": created_by} if created_by else None,
            **(additional_params or {}),
        )

    async def delete_database(self, kb_id: str) -> dict:
        """Delete a knowledge base and all its data."""
        kb = await self._require_kb_for_database(kb_id)
        return await kb.delete_database(kb_id)

    async def update_database(
        self,
        kb_id: str,
        name: str | None = None,
        description: str | None = None,
        llm_model_spec: str | None = None,
        update_llm_model_spec: bool = False,
    ) -> dict:
        """Update knowledge base information."""
        kb = await self._require_kb_for_database(kb_id)
        current = await self.get_database_info(kb_id, include_files=False)
        if current is None:
            raise KBNotFoundError(f"知识库不存在: {kb_id}")

        return kb.update_database(
            kb_id,
            name=name if name is not None else current.get("name") or "",
            description=description if description is not None else current.get("description") or "",
            llm_model_spec=llm_model_spec,
            update_llm_model_spec=update_llm_model_spec,
        )

    async def get_database_info(self, kb_id: str, include_files: bool = False) -> dict | None:
        """Get detailed information about a knowledge base, or ``None`` when missing."""
        try:
            kb = await self._require_kb_for_database(kb_id)
        except KBNotFoundError:
            return None
        return kb.get_database_info(kb_id, include_files=include_files)

    async def get_databases(self, include_files: bool = False) -> dict:
        """List all knowledge bases across supported types."""
        databases = []
        for kb_type in KnowledgeBaseFactory.get_available_types():
            kb = self.get_instance(kb_type)
            try:
                await kb._load_metadata()
            except Exception as exc:  # noqa: BLE001 - one type failing must not block others
                logger.error(f"Failed to load metadata for type {kb_type}: {exc}")
                continue
            result = kb.get_databases(include_files=include_files)
            databases.extend(result.get("databases", []))
        return {"databases": databases}

    async def get_query_params_config(self, kb_id: str, **kwargs) -> dict:
        kb = await self._require_kb_for_database(kb_id)
        return kb.get_query_params_config(kb_id, **kwargs)

    async def get_query_params(self, kb_id: str) -> dict:
        """Get effective query params (defaults merged with saved overrides)."""
        kb = await self._require_kb_for_database(kb_id)
        config = kb.get_query_params_config(kb_id)
        saved = kb._get_query_params(kb_id) or {}
        merged: dict[str, Any] = {}
        for option in config.get("options", []):
            key = option.get("key")
            if not key:
                continue
            if key in saved:
                merged[key] = saved[key]
            elif "default" in option:
                merged[key] = option["default"]
        return {"options": merged}

    async def update_query_params(self, kb_id: str, params: dict | None) -> dict:
        """Persist user query-param overrides for a knowledge base."""
        kb = await self._require_kb_for_database(kb_id)
        if kb_id not in kb.databases_meta:
            raise KBNotFoundError(f"知识库不存在: {kb_id}")

        config = kb.get_query_params_config(kb_id)
        allowed_keys = {option.get("key") for option in config.get("options", []) if option.get("key")}
        saved = kb._get_query_params(kb_id) or {}
        for key, value in (params or {}).items():
            if key in allowed_keys:
                saved[key] = value
            else:
                logger.warning(f"Ignoring unknown query param {key!r}")

        kb.databases_meta[kb_id]["query_params"] = {"options": saved}
        await kb._persist_kb(kb_id)
        return await self.get_query_params(kb_id)

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    async def add_file_record(
        self,
        kb_id: str,
        item: str,
        params: dict | None = None,
        operator_id: str | None = None,
    ) -> dict:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.add_file_record(kb_id, item, params=params, operator_id=operator_id)

    async def parse_file(self, kb_id: str, file_id: str, operator_id: str | None = None) -> dict:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.parse_file(kb_id, file_id, operator_id=operator_id)

    async def index_file(self, kb_id: str, file_id: str, operator_id: str | None = None) -> dict:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.index_file(kb_id, file_id, operator_id=operator_id)

    async def update_content(self, kb_id: str, file_ids: list[str], params: dict | None = None) -> list[dict]:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.update_content(kb_id, file_ids, params=params)

    async def update_file_params(
        self, kb_id: str, file_id: str, params: dict, operator_id: str | None = None
    ) -> None:
        kb = await self._require_kb_for_database(kb_id)
        await kb.update_file_params(kb_id, file_id, params, operator_id=operator_id)

    async def delete_file(self, kb_id: str, file_id: str) -> None:
        kb = await self._require_kb_for_database(kb_id)
        await kb.delete_file(kb_id, file_id)

    async def get_file_basic_info(self, kb_id: str, file_id: str) -> dict:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.get_file_basic_info(kb_id, file_id)

    async def get_file_content(self, kb_id: str, file_id: str) -> dict:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.get_file_content(kb_id, file_id)

    async def get_file_info(self, kb_id: str, file_id: str) -> dict:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.get_file_info(kb_id, file_id)

    async def get_file_download(self, kb_id: str, file_id: str, variant: str = "original") -> dict:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.get_file_download(kb_id, file_id, variant=variant)

    async def read_file_preview(self, kb_id: str, file_id: str) -> dict:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.read_file_preview(kb_id, file_id)

    async def open_file_content(self, kb_id: str, file_id: str, offset: int = 0, limit: int = 800) -> dict:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.open_file_content(kb_id, file_id, offset=offset, limit=limit)

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
        kb = await self._require_kb_for_database(kb_id)
        return await kb.find_file_content(
            kb_id,
            file_id,
            patterns,
            use_regex=use_regex,
            case_sensitive=case_sensitive,
            max_windows=max_windows,
            window_size=window_size,
        )

    async def refresh_database_stats(self, kb_id: str) -> dict[str, int]:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.refresh_database_stats(kb_id)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def aquery(self, query_text: str, kb_id: str, **kwargs) -> list[dict]:
        kb = await self._require_kb_for_database(kb_id)
        return await kb.aquery(query_text, kb_id, **kwargs)

    async def search(self, query_text: str, kb_id: str, top_k: int = 10, score_threshold: float = 0.0) -> dict:
        """Agent-facing search returning a serializable output schema."""
        kb = await self._require_kb_for_database(kb_id)
        results = await kb.aquery(query_text, kb_id, top_k=top_k, agent_call=True)
        output = kb.build_search_output(kb_id, results)
        if isinstance(output, dict):
            output["score_threshold"] = score_threshold
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _require_kb_for_database(self, kb_id: str) -> KnowledgeBase:
        """Resolve the instance that owns a database id (without loading)."""
        for kb_type, kb in self._instances.items():
            if kb_id in kb.databases_meta:
                return kb

        # Unknown so far — try loading metadata for each type.
        for kb_type in KnowledgeBaseFactory.get_available_types():
            kb = self.get_instance(kb_type)
            if kb._metadata_loaded:
                continue
            try:
                await kb._load_metadata()
            except Exception as exc:  # noqa: BLE001 - best-effort probe per type
                logger.error(f"Failed to load metadata for type {kb_type}: {exc}")
            if kb_id in kb.databases_meta:
                return kb

        raise KBNotFoundError(f"知识库不存在: {kb_id}")


async def wait_for_tasks(tasks: list[asyncio.Task]) -> None:
    """Wait for a batch of background tasks to complete."""
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)