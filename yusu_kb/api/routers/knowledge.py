"""Knowledge base management endpoints: KB CRUD, files, parsing, indexing, retrieval."""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from yusu_kb.api.deps import ManagerDep, verify_api_key
from yusu_kb.knowledge.base import KBNotFoundError, KBOperationError
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.repositories.knowledge_file_repository import KnowledgeFileRepository
from yusu_kb.storage.sqlite.engine import get_data_dir

router = APIRouter(
    prefix="/api/knowledge",
    tags=["knowledge"],
    dependencies=[Depends(verify_api_key)],
)


class CreateDatabaseRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    description: str = ""
    additional_params: dict[str, Any] | None = None


class UpdateDatabaseRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class QueryParamsRequest(BaseModel):
    params: dict[str, Any] | None = None


class UpdateContentRequest(BaseModel):
    params: dict[str, Any] | None = None


class QueryRequest(BaseModel):
    kb_id: str
    query: str = Field(..., min_length=1)
    params: dict[str, Any] | None = None


class SystemPromptRequest(BaseModel):
    system_prompt: str = ""


def _require_kb(manager: KnowledgeBaseManager, kb_id: str):
    """Resolve the local KB instance, validating the kb_id exists."""
    kb = manager.get_instance("local")
    if kb_id not in kb.databases_meta:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb


async def _list_files(manager: KnowledgeBaseManager, kb_id: str) -> list[dict]:
    """Serialize file records of a knowledge base via the owning KB instance."""
    records = await KnowledgeFileRepository().list_documents(kb_id=kb_id)
    kb = manager.get_instance("local")
    return [kb._file_record_to_meta(record) for record in records]


@router.get("/databases")
async def list_databases(manager: ManagerDep) -> dict:
    """List all knowledge bases."""
    return await manager.get_databases(include_files=False)


@router.post("/databases")
async def create_database(payload: CreateDatabaseRequest, manager: ManagerDep) -> dict:
    """Create a knowledge base."""
    try:
        return await manager.create_database(
            name=payload.name,
            description=payload.description,
            additional_params=payload.additional_params,
        )
    except KBOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/databases/{kb_id}")
async def get_database(kb_id: str, manager: ManagerDep) -> dict:
    """Get knowledge base details (including file listing)."""
    info = await manager.get_database_info(kb_id, include_files=False)
    if info is None:
        raise HTTPException(status_code=404, detail="知识库不存在")
    info["files"] = await _list_files(manager, kb_id)
    return info


@router.delete("/databases/{kb_id}")
async def delete_database(kb_id: str, manager: ManagerDep) -> dict:
    """Delete a knowledge base and all its data."""
    try:
        return await manager.delete_database(kb_id)
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KBOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.put("/databases/{kb_id}")
async def update_database(kb_id: str, payload: UpdateDatabaseRequest, manager: ManagerDep) -> dict:
    """Update knowledge base name / description."""
    try:
        return await manager.update_database(kb_id, name=payload.name, description=payload.description)
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


@router.post("/databases/{kb_id}/files")
async def upload_file(
    kb_id: str,
    file: Annotated[UploadFile, File()],
    manager: ManagerDep,
) -> dict:
    """Upload a document into the knowledge base (record created; parse separately)."""
    _require_kb(manager, kb_id)
    filename = file.filename or "upload.bin"

    upload_dir = get_data_dir() / "uploads" / kb_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    target = upload_dir / filename

    # Non-blocking write: file.read() is async but write_bytes() is synchronous
    # and would otherwise stall the event loop on large files.
    data = await file.read()
    await asyncio.to_thread(target.write_bytes, data)

    # Re-uploading the same file path should reuse the existing record instead
    # of inserting a duplicate (which would otherwise become an orphan).
    for record in await _list_files(manager, kb_id):
        if record.get("path") == str(target):
            return record

    try:
        return await manager.add_file_record(kb_id, str(target))
    except KBOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/databases/{kb_id}/files")
async def list_files(kb_id: str, manager: ManagerDep) -> dict:
    """List files of a knowledge base."""
    _require_kb(manager, kb_id)
    return {"files": await _list_files(manager, kb_id)}


@router.post("/databases/{kb_id}/files/{file_id}/parse")
async def parse_file(kb_id: str, file_id: str, manager: ManagerDep) -> dict:
    """Parse a file into markdown (Status: PARSING -> PARSED/ERROR_PARSING)."""
    try:
        return await manager.parse_file(kb_id, file_id)
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KBOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/databases/{kb_id}/files/{file_id}/index")
async def index_file(kb_id: str, file_id: str, manager: ManagerDep) -> dict:
    """Index a parsed file into chunks + vectors (Status: INDEXING -> INDEXED)."""
    try:
        return await manager.index_file(kb_id, file_id)
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KBOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/databases/{kb_id}/files/{file_id}/update-content")
async def update_content(
    kb_id: str, file_id: str, payload: UpdateContentRequest, manager: ManagerDep
) -> dict:
    """Re-parse and re-index a file (optionally with new chunking params)."""
    try:
        results = await manager.update_content(kb_id, [file_id], params=payload.params)
        return {"updated": results}
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KBOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/databases/{kb_id}/files/{file_id}")
async def get_file(kb_id: str, file_id: str, manager: ManagerDep) -> dict:
    """Get full file info (basic + preview)."""
    try:
        return await manager.get_file_info(kb_id, file_id)
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/databases/{kb_id}/files/{file_id}")
async def delete_file(kb_id: str, file_id: str, manager: ManagerDep) -> dict:
    """Delete a file (vectors, chunks, markdown, original)."""
    try:
        await manager.delete_file(kb_id, file_id)
        return {"deleted": True}
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KBOperationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Query params + retrieval
# ---------------------------------------------------------------------------


@router.get("/databases/{kb_id}/query-params")
async def get_query_params(kb_id: str, manager: ManagerDep) -> dict:
    """Get effective retrieval parameters for a knowledge base."""
    try:
        config = await manager.get_query_params_config(kb_id)
        effective = await manager.get_query_params(kb_id)
        return {"config": config, "effective": effective}
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.put("/databases/{kb_id}/query-params")
async def update_query_params(kb_id: str, payload: QueryParamsRequest, manager: ManagerDep) -> dict:
    """Persist retrieval parameter overrides."""
    try:
        return await manager.update_query_params(kb_id, payload.params)
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/databases/{kb_id}/system-prompt")
async def get_system_prompt(kb_id: str, manager: ManagerDep) -> dict:
    """Get the custom system prompt for a knowledge base ('' when unset)."""
    kb = _require_kb(manager, kb_id)
    additional = kb.databases_meta.get(kb_id, {}).get("metadata") or {}
    return {"system_prompt": additional.get("system_prompt") or ""}


@router.put("/databases/{kb_id}/system-prompt")
async def update_system_prompt(
    kb_id: str, payload: SystemPromptRequest, manager: ManagerDep
) -> dict:
    """Persist a custom system prompt appended to the chat prompt (>8000 chars rejected)."""
    kb = _require_kb(manager, kb_id)
    if len(payload.system_prompt) > 8000:
        raise HTTPException(status_code=400, detail="系统提示词不能超过 8000 字符")
    kb.databases_meta.setdefault(kb_id, {}).setdefault("metadata", {})[
        "system_prompt"
    ] = payload.system_prompt
    await kb._persist_kb(kb_id)
    return {"system_prompt": payload.system_prompt}


@router.post("/query")
async def query_kb(payload: QueryRequest, manager: ManagerDep) -> dict:
    """Retrieval-only query (used by the retrieval test page)."""
    try:
        kb = await manager._require_kb_for_database(payload.kb_id)
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    results = await kb.aquery(payload.query, payload.kb_id, **payload.params or {})
    return kb.build_search_output(payload.kb_id, results)


@router.post("/databases/{kb_id}/refresh-stats")
async def refresh_stats(kb_id: str, manager: ManagerDep) -> dict:
    """Recalculate knowledge base statistics (file/chunk/token counts)."""
    try:
        return await manager.refresh_database_stats(kb_id)
    except KBNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc