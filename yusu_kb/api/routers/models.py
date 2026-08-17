"""Model provider management endpoints.

Prefix ``/api/system/model-providers``: provider CRUD, live remote-model
fetching, model cache refresh, model status probes and default-spec config.
All responses use the ``{"success": True, "data": ...}`` envelope.
"""

from __future__ import annotations

from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from yusu_kb.api.deps import (
    get_httpx_transport,
    refresh_model_cache,
    verify_api_key,
)
from yusu_kb.models.providers.cache import model_cache
from yusu_kb.models.providers.service import (
    check_credential_status,
    create_provider_config,
    delete_provider_config,
    fetch_remote_models,
    get_all_model_providers,
    get_model_provider_by_id,
    test_model_status_by_spec,
    update_provider_config,
)
from yusu_kb.repositories.model_provider_repository import AppConfigRepository
from yusu_kb.storage.sqlite.models_knowledge import ModelProvider

router = APIRouter(
    prefix="/api/system/model-providers",
    tags=["model-providers"],
    dependencies=[Depends(verify_api_key)],
)

TransportDep = Annotated[Any, Depends(get_httpx_transport)]

DEFAULT_SPEC_KEYS = ("default_chat_model_spec", "default_embedding_model_spec", "default_rerank_model_spec")


class DefaultsRequest(BaseModel):
    default_chat_model_spec: str | None = None
    default_embedding_model_spec: str | None = None
    default_rerank_model_spec: str | None = None


def _provider_dict(provider: ModelProvider) -> dict:
    return {
        "provider_id": provider.provider_id,
        "display_name": provider.display_name,
        "provider_type": provider.provider_type,
        "default_protocol": provider.default_protocol,
        "base_url": provider.base_url,
        "embedding_base_url": provider.embedding_base_url,
        "rerank_base_url": provider.rerank_base_url,
        "models_endpoint": provider.models_endpoint,
        "embedding_models_endpoint": provider.embedding_models_endpoint,
        "rerank_models_endpoint": provider.rerank_models_endpoint,
        "api_key_env": provider.api_key_env,
        "capabilities": provider.capabilities or [],
        "enabled_models": provider.enabled_models or [],
        "headers_json": provider.headers_json or {},
        "extra_json": provider.extra_json or {},
        "is_enabled": bool(provider.is_enabled),
        "is_builtin": bool(provider.is_builtin),
        "credential_status": check_credential_status(provider),
    }


def _http_400(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@router.get("")
async def list_providers() -> dict:
    """List all model providers with credential status."""
    providers = await get_all_model_providers()
    return {"success": True, "data": [_provider_dict(p) for p in providers]}


@router.get("/defaults")
async def get_defaults() -> dict:
    """Read the three default model spec keys from AppConfig."""
    rows = await AppConfigRepository().get_all()
    values = {row.config_key: row.config_value for row in rows}
    return {"success": True, "data": {key: values.get(key) for key in DEFAULT_SPEC_KEYS}}


@router.put("/defaults")
async def put_defaults(request: DefaultsRequest) -> dict:
    """Persist default model specs and rebuild the model cache."""
    repository = AppConfigRepository()
    for key, value in request.model_dump(exclude_none=True).items():
        await repository.set(key, value, updated_by="api")
    count = await refresh_model_cache()
    return {"success": True, "message": f"默认模型配置已保存，缓存模型 {count} 个", "data": request.model_dump()}


@router.post("/models/cache/refresh")
async def refresh_cache() -> dict:
    """Rebuild the in-process model cache from the DB."""
    count = await refresh_model_cache()
    return {"success": True, "message": f"模型缓存已刷新，共 {count} 个模型", "model_count": count}


@router.get("/models/v2")
async def list_models_v2(model_type: str = Query(default="chat", pattern="^(chat|embedding|rerank)$")) -> dict:
    """Return cached specs grouped by provider."""
    grouped = model_cache.get_specs_grouped_by_provider(model_type)
    providers = {p.provider_id: p for p in await get_all_model_providers()}
    data: dict[str, dict] = {}
    for provider_id, infos in grouped.items():
        provider = providers.get(provider_id)
        data[provider_id] = {
            "provider_id": provider_id,
            "provider_display_name": provider.display_name if provider else provider_id,
            "models": [
                {
                    "spec": info.spec,
                    "model_id": info.model_id,
                    "display_name": info.display_name,
                    "dimension": info.dimension,
                    "batch_size": info.batch_size,
                }
                for info in infos
            ],
        }
    return {"success": True, "data": data}


@router.get("/models/status")
async def model_status(spec: str = Query(...), transport: TransportDep = None) -> dict:
    """Probe a model spec's connectivity (embedding/rerank/chat)."""
    result = await test_model_status_by_spec(spec, transport=transport)
    return {"success": True, "data": result}


@router.get("/{provider_id}")
async def get_provider(provider_id: str) -> dict:
    provider = await get_model_provider_by_id(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail="供应商不存在")
    return {"success": True, "data": _provider_dict(provider)}


@router.get("/{provider_id}/remote-models")
async def get_remote_models(provider_id: str, transport: TransportDep = None) -> dict:
    """Live-fetch the provider's remote model list (401 -> 502)."""
    provider = await get_model_provider_by_id(provider_id)
    if provider is None:
        raise HTTPException(status_code=404, detail="供应商不存在")
    try:
        models = await fetch_remote_models(provider, transport=transport)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise HTTPException(status_code=502, detail="远端 API 认证失败，请检查 API Key 配置") from exc
        raise HTTPException(status_code=502, detail=f"远端 API 请求失败: {exc}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"远端 API 请求失败: {exc}") from exc
    except ValueError as exc:
        # Covers malformed/non-JSON provider responses (e.g. proxy error pages)
        # so the UI gets a 502 with a clear message instead of a 500.
        raise HTTPException(status_code=502, detail=f"远端返回格式异常: {exc}") from exc
    return {"success": True, "data": models}


@router.post("")
async def create_provider(payload: dict[str, Any]) -> dict:
    try:
        provider = await create_provider_config(payload, username="api")
    except ValueError as exc:
        raise _http_400(exc) from exc
    await refresh_model_cache()
    return {"success": True, "data": _provider_dict(provider)}


@router.put("/{provider_id}")
async def update_provider(provider_id: str, payload: dict[str, Any]) -> dict:
    try:
        provider = await update_provider_config(provider_id, payload, username="api")
    except ValueError as exc:
        raise _http_400(exc) from exc
    if provider is None:
        raise HTTPException(status_code=404, detail="供应商不存在")
    await refresh_model_cache()
    return {"success": True, "data": _provider_dict(provider)}


@router.delete("/{provider_id}")
async def delete_provider(provider_id: str) -> dict:
    deleted = await delete_provider_config(provider_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="供应商不存在")
    await refresh_model_cache()
    return {"success": True, "data": {"deleted": provider_id}}
