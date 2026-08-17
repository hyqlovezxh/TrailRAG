"""RAG evaluation endpoints.

Prefix ``/api/evaluation``: dataset upload / list / detail / download /
generate / delete, and evaluation-run create / list / detail (error_only) /
delete. All heavy work runs in the shared EvaluationService's in-process
background tasks; clients poll the dataset/run rows until completion.
"""

from __future__ import annotations

from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from yusu_kb.api.deps import EvaluationServiceDep, verify_api_key

router = APIRouter(
    prefix="/api/evaluation",
    tags=["evaluation"],
    dependencies=[Depends(verify_api_key)],
)


def _http_error(exc: ValueError) -> HTTPException:
    message = str(exc)
    if "not found" in message.lower() or "不存在" in message:
        return HTTPException(status_code=404, detail=message)
    return HTTPException(status_code=400, detail=message)


class GenerateDatasetRequest(BaseModel):
    kb_id: str
    name: str = ""
    description: str = ""
    count: int = Field(default=5, ge=1, le=50)
    neighbors_count: int = Field(default=3, ge=1, le=20)
    concurrency_count: int = Field(default=4, ge=1, le=16)
    llm_model_spec: str = ""
    generation_mode: str = Field(default="vector", pattern="^(vector|graph_enhanced)$")
    graph_expand_top_k: int = Field(default=1, ge=0, le=10)


class CreateRunRequest(BaseModel):
    """Request body for starting an evaluation run.

    ``model_config`` is a pydantic reserved name, so the field is stored as
    ``config`` with a wire alias of ``model_config``.
    """

    model_config = ConfigDict(populate_by_name=True)

    kb_id: str
    dataset_id: str
    name: str | None = None
    config: dict[str, Any] = Field(default_factory=dict, alias="model_config")
    created_by: str = "api"


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@router.post("/datasets/upload")
async def upload_dataset(
    file: Annotated[UploadFile, File()],
    kb_id: Annotated[str, Form()],
    name: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
    service: EvaluationServiceDep = None,
) -> dict:
    """Upload a JSONL dataset (one question per line)."""
    try:
        data = await service.upload_dataset(
            kb_id=kb_id,
            file_content=await file.read(),
            filename=file.filename or "dataset.jsonl",
            name=name,
            description=description,
            created_by="api",
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {"success": True, "data": data}


@router.get("/datasets")
async def list_datasets(kb_id: str = Query(...), service: EvaluationServiceDep = None) -> dict:
    """List datasets of a knowledge base (newest first)."""
    data = await service.list_datasets(kb_id)
    return {"success": True, "data": data}


@router.get("/datasets/{dataset_id}")
async def get_dataset_detail(
    dataset_id: str,
    kb_id: str = Query(...),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=100),
    service: EvaluationServiceDep = None,
) -> dict:
    """Dataset detail with paginated items."""
    try:
        data = await service.get_dataset_detail(kb_id, dataset_id, page=page, page_size=page_size)
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {"success": True, "data": data}


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(dataset_id: str, service: EvaluationServiceDep = None):
    """Download the dataset as a JSONL file attachment."""
    try:
        exported = await service.export_dataset_jsonl(dataset_id)
    except ValueError as exc:
        raise _http_error(exc) from exc
    filename = quote(exported["filename"])
    return Response(
        content=exported["content"],
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/datasets/generate")
async def generate_dataset(request: GenerateDatasetRequest, service: EvaluationServiceDep = None) -> dict:
    """Submit a benchmark-generation task (background; poll the dataset row)."""
    try:
        data = await service.generate_dataset(
            kb_id=request.kb_id,
            name=request.name,
            description=request.description,
            count=request.count,
            neighbors_count=request.neighbors_count,
            concurrency_count=request.concurrency_count,
            llm_model_spec=request.llm_model_spec,
            generation_mode=request.generation_mode,
            graph_expand_top_k=request.graph_expand_top_k,
            created_by="api",
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {"success": True, "data": data}


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, service: EvaluationServiceDep = None) -> dict:
    """Delete a dataset (cancels a running generation task first)."""
    try:
        await service.delete_dataset(dataset_id)
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {"success": True, "data": {"deleted": dataset_id}}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@router.post("/runs")
async def create_run(request: CreateRunRequest, service: EvaluationServiceDep = None) -> dict:
    """Start an evaluation run over a dataset (background; poll the run row)."""
    try:
        run_id = await service.run_evaluation(
            kb_id=request.kb_id,
            dataset_id=request.dataset_id,
            name=request.name,
            model_config=request.config,
            created_by=request.created_by,
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {"success": True, "data": {"run_id": run_id, "status": "running"}}


@router.get("/runs")
async def list_runs(kb_id: str = Query(...), service: EvaluationServiceDep = None) -> dict:
    """List evaluation runs of a knowledge base (newest first)."""
    data = await service.list_runs(kb_id)
    return {"success": True, "data": data}


@router.get("/runs/{run_id}")
async def get_run_detail(
    run_id: str,
    kb_id: str = Query(...),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    error_only: bool = Query(default=False),
    service: EvaluationServiceDep = None,
) -> dict:
    """Run detail: aggregated metrics + paginated per-item results."""
    try:
        data = await service.get_run_results(
            kb_id, run_id, page=page, page_size=page_size, error_only=error_only
        )
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {"success": True, "data": data}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, kb_id: str = Query(...), service: EvaluationServiceDep = None) -> dict:
    """Delete a run (cancels the in-process task if still running)."""
    try:
        await service.delete_run(kb_id, run_id)
    except ValueError as exc:
        raise _http_error(exc) from exc
    return {"success": True, "data": {"deleted": run_id}}
