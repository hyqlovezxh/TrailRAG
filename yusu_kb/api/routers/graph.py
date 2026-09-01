"""Graph endpoints: build orchestration, status, stats and visualization data.

The router owns the GraphService dependency resolution: ``work_dir`` and the
embedding function come from the owning LocalKB instance (same lazy defaults
as chunk indexing), and the chat model fn is the shared chat adapter's
``call`` — the same object the /api/chat router serves.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from yusu_kb.api.deps import ChatModelDep, ManagerDep, verify_api_key
from yusu_kb.knowledge.graphs.connectivity import evaluate_gate
from yusu_kb.knowledge.graphs.graph_service import GRAPH_CONFIG_KEY, GraphService
from yusu_kb.knowledge.implementations.local_kb import LocalKB
from yusu_kb.models.chat import OpenAIChatAdapter
from yusu_kb.repositories.knowledge_base_repository import KnowledgeBaseRepository

router = APIRouter(
    prefix="/api/knowledge/databases/{kb_id}/graph",
    tags=["graph"],
    dependencies=[Depends(verify_api_key)],
)


class BuildRequest(BaseModel):
    batch_size: int | None = Field(default=None, ge=1, le=1000)


class ResetRequest(BaseModel):
    clear_extraction_result: bool = True
    clear_config: bool = False


class ConfigureRequest(BaseModel):
    extractor_type: str = "event"
    extractor_options: dict[str, Any] | None = None


def _require_kb(manager, kb_id: str) -> LocalKB:
    """Resolve the local KB instance, validating the kb_id exists."""
    kb = manager.get_instance("local")
    if kb_id not in kb.databases_meta:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return kb


def _resolve_extractor_options(
    extractor_type: str,
    options: dict[str, Any] | None,
    chat_model,
) -> dict[str, Any] | None:
    """Fill ``model_spec`` from the active chat model when the caller omits it.

    The LLM extractor never builds a model object from ``model_spec`` — the
    actual call goes through the injected ``chat_model_fn``, and ``model_spec``
    only serves as identity/LLM-cache key. Requiring it verbatim would block
    the documented ``.env``-only setup, where no provider row exists in the DB
    yet. Defaulting it to the chat model actually in use keeps the cache key
    correct while letting env-only deployments build graphs.
    """
    if extractor_type not in ("llm", "event"):
        return options
    resolved = dict(options or {})
    if str(resolved.get("model_spec") or "").strip():
        return resolved
    # 只对真实聊天适配器补全：测试注入的 mock（如 FakeChat）没有 env/DB 模型
    # 身份，补全会绕过 service 层"LLM 图谱抽取器需要 model_spec"的 400 校验，
    # 破坏配置契约；真实 .env-only 部署的 chat_model 一定是 OpenAIChatAdapter。
    if not isinstance(chat_model, OpenAIChatAdapter):
        return resolved
    model_id = getattr(chat_model, "model", None) or getattr(chat_model, "model_name", None)
    if not model_id:
        return resolved
    resolved["model_spec"] = f"env:{model_id}"
    return resolved


def _get_graph_service(kb: LocalKB, kb_id: str, chat_model) -> GraphService:
    """Resolve the per-KB GraphService singleton with the shared model deps."""
    return GraphService.get_instance(
        kb_id=kb_id,
        work_dir=kb.work_dir,
        embed_func=kb._get_embedding_function(),
        chat_model_fn=chat_model.call_collect,
    )


async def _sync_graph_config_to_kb(kb: LocalKB, kb_id: str) -> None:
    """Mirror the DB graph_build_config into the LocalKB in-memory metadata.

    The LocalKB owns ``additional_params`` in memory and re-persists it from
    there on every stats refresh; GraphService writes the config to the DB
    row directly. Without the mirror, the next index/update would overwrite
    the DB row with stale metadata and silently drop the config, breaking
    the auto-build trigger.
    """
    row = await KnowledgeBaseRepository().get_by_kb_id(kb_id)
    if row is None:
        return
    config = dict(row.additional_params or {}).get(GRAPH_CONFIG_KEY)
    metadata = kb.databases_meta.setdefault(kb_id, {}).setdefault("metadata", {})
    if config is None:
        metadata.pop(GRAPH_CONFIG_KEY, None)
    else:
        metadata[GRAPH_CONFIG_KEY] = config


@router.get("/status")
async def get_graph_status(
    kb_id: str,
    manager: ManagerDep,
    chat_model: ChatModelDep,
) -> dict:
    """Report build configuration, chunk progress and graph counts."""
    kb = _require_kb(manager, kb_id)
    service = _get_graph_service(kb, kb_id, chat_model)
    return await service.get_status(kb_id)


@router.post("/build")
async def build_graph(
    kb_id: str,
    payload: BuildRequest,
    manager: ManagerDep,
    chat_model: ChatModelDep,
) -> dict:
    """Start the background graph build for pending chunks; return status."""
    kb = _require_kb(manager, kb_id)
    service = _get_graph_service(kb, kb_id, chat_model)
    try:
        result = await service.build_pending_chunks(kb_id, batch_size=payload.batch_size or 100)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result["status"] = await service.get_status(kb_id)
    return result


@router.post("/reset")
async def reset_graph(
    kb_id: str,
    payload: ResetRequest,
    manager: ManagerDep,
    chat_model: ChatModelDep,
) -> dict:
    """Cancel the build and drop all graph data (chunks stay indexed)."""
    kb = _require_kb(manager, kb_id)
    service = _get_graph_service(kb, kb_id, chat_model)
    result = await service.reset(
        kb_id,
        clear_extraction_result=payload.clear_extraction_result,
        clear_config=payload.clear_config,
    )
    await _sync_graph_config_to_kb(kb, kb_id)
    return result


@router.get("/config")
async def get_graph_config(
    kb_id: str,
    manager: ManagerDep,
    chat_model: ChatModelDep,
) -> dict:
    """Return the locked graph extraction config (or null)."""
    kb = _require_kb(manager, kb_id)
    service = _get_graph_service(kb, kb_id, chat_model)
    status = await service.get_status(kb_id)
    return {"config": status["config"]}


@router.put("/config")
async def update_graph_config(
    kb_id: str,
    payload: ConfigureRequest,
    manager: ManagerDep,
    chat_model: ChatModelDep,
) -> dict:
    """Validate and lock the graph extraction config."""
    kb = _require_kb(manager, kb_id)
    service = _get_graph_service(kb, kb_id, chat_model)
    try:
        result = await service.configure(
            kb_id,
            extractor_type=payload.extractor_type,
            extractor_options=_resolve_extractor_options(
                payload.extractor_type, payload.extractor_options, chat_model
            ),
            created_by="api",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _sync_graph_config_to_kb(kb, kb_id)
    return result


@router.get("/stats")
async def get_graph_stats(
    kb_id: str,
    manager: ManagerDep,
    chat_model: ChatModelDep,
) -> dict:
    """Return status counts plus in-memory storage statistics."""
    kb = _require_kb(manager, kb_id)
    service = _get_graph_service(kb, kb_id, chat_model)
    result = await service.get_status(kb_id)
    storage = service.get_storage(kb_id)
    result["storage"] = storage.get_stats() if storage.is_built() else None
    # 连通性一等指标 + 门禁（事件化重构的验收口径）
    if result["storage"] is not None:
        connectivity = storage.get_connectivity()
        gate_passed, gate_failures = evaluate_gate(connectivity)
        result["connectivity"] = connectivity
        result["gate"] = {"passed": gate_passed, "failures": gate_failures}
    return result


@router.get("/labels")
async def get_graph_labels(
    kb_id: str,
    manager: ManagerDep,
    chat_model: ChatModelDep,
) -> dict:
    """Return entity label counts (visualization data source)."""
    kb = _require_kb(manager, kb_id)
    service = _get_graph_service(kb, kb_id, chat_model)
    return {"labels": service.get_storage(kb_id).get_labels()}


@router.get("/subgraph")
async def get_graph_subgraph(
    kb_id: str,
    manager: ManagerDep,
    chat_model: ChatModelDep,
    entity_ids: str = Query(..., min_length=1),
    max_depth: int = Query(3, ge=1, le=10),
    max_nodes: int = Query(5000, ge=1, le=50000),
) -> dict:
    """Return the entity subgraph reachable from the requested ids/names.

    Each comma-separated ``entity_ids`` entry is first matched by entity
    name; when nothing matches, the raw string is treated as an entity id.
    """
    kb = _require_kb(manager, kb_id)
    service = _get_graph_service(kb, kb_id, chat_model)
    storage = service.get_storage(kb_id)
    seeds: list[str] = []
    for raw in entity_ids.split(","):
        raw = raw.strip()
        if not raw:
            continue
        matched = storage.search_entities_by_name(raw, limit=1)
        if matched:
            seeds.append(matched[0]["entity_id"])
        elif storage.get_entity_node(raw) is not None:
            seeds.append(raw)
    return storage.bfs_subgraph(seeds, max_depth=max_depth, max_nodes=max_nodes)


@router.get("/full")
async def get_graph_full(
    kb_id: str,
    manager: ManagerDep,
    chat_model: ChatModelDep,
    limit: int = Query(2000, ge=1, le=20000),
) -> dict:
    """Return the whole built graph (capped to ``limit`` entities).

    Unlike ``/subgraph`` this does not require a seed entity, so the UI can
    render the already-built graph immediately after selecting a KB.
    """
    kb = _require_kb(manager, kb_id)
    service = _get_graph_service(kb, kb_id, chat_model)
    storage = service.get_storage(kb_id)
    entities = storage.iter_entities()
    if len(entities) > limit:
        entities = entities[:limit]
    entity_ids = {entity["entity_id"] for entity in entities}
    edges = [
        relation
        for relation in storage.iter_relations()
        if relation.get("source_id") in entity_ids and relation.get("target_id") in entity_ids
    ]
    return {"nodes": entities, "edges": edges}
