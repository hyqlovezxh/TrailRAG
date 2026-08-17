"""API dependencies: env loading, shared instances, optional auth."""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Any

from dotenv import load_dotenv
from fastapi import Depends, Header, HTTPException

from yusu_kb.knowledge.implementations.local_kb import set_default_graph_chat_model_fn
from yusu_kb.knowledge.manager import KnowledgeBaseManager
from yusu_kb.models.chat import OpenAIChatAdapter, create_chat_model
from yusu_kb.models.rerank import BaseReranker, create_reranker
from yusu_kb.repositories import configure_repositories
from yusu_kb.storage.sqlite.engine import create_engine, dispose, get_data_dir, init_db

# Load the local .env (gitignored) once at import time.
load_dotenv()

_manager: KnowledgeBaseManager | None = None
_engine = None
_init_lock = asyncio.Lock()
_chat_model: OpenAIChatAdapter | None = None
_chat_model_lock = asyncio.Lock()
_evaluation_service = None


async def get_manager() -> KnowledgeBaseManager:
    """Lazily create the shared manager (engine + repositories + metadata)."""
    global _manager, _engine
    if _manager is None:
        async with _init_lock:
            if _manager is None:
                data_dir = get_data_dir()
                engine = create_engine()
                await init_db(engine)
                configure_repositories(engine)
                manager = KnowledgeBaseManager(str(data_dir))
                await manager.load_all_metadata()
                # 启动即重建 ModelCache：builtin 种子落库 + 按 is_enabled 装载模型。
                # 否则重启后缓存为空，任何 provider_id:model_id spec 解析都会失败，
                # 直至用户手动触发 refresh（models/cache/refresh）。
                await refresh_model_cache()
                _inject_defaults(manager)
                _manager = manager
                _engine = engine
    return _manager


def _inject_defaults(manager: KnowledgeBaseManager) -> None:
    """Wire the env-configured reranker into the local KB instance (if any)."""
    rerank_func = build_rerank_func()
    if rerank_func is None:
        return
    kb = manager.get_instance("local")
    kb.rerank_func = rerank_func


def build_rerank_func():
    """Build an async ``(query, documents) -> list[float]`` callable from env, or None."""
    reranker: BaseReranker | None = create_reranker()
    if reranker is None:
        return None

    async def rerank_func(query: str, documents) -> list[float]:
        return await reranker.acompute_score([query, list(documents)])

    return rerank_func


async def _get_default_model_spec(model_type: str) -> str | None:
    """Read ``default_*_model_spec`` from the AppConfig table (best-effort)."""
    from yusu_kb.repositories.model_provider_repository import AppConfigRepository

    try:
        row = await AppConfigRepository().get(f"default_{model_type}_model_spec")
    except Exception:  # noqa: BLE001 - DB 未就绪/表不存在时回退 env 路径
        return None
    if row is None:
        return None
    value = row.config_value
    return value if isinstance(value, str) and value else None


async def get_chat_model() -> OpenAIChatAdapter:
    """Lazily create the shared chat model adapter (AppConfig spec first, env fallback)."""
    global _chat_model
    if _chat_model is None:
        async with _chat_model_lock:
            if _chat_model is None:
                _chat_model = create_chat_model(default_spec=await _get_default_model_spec("chat"))
                # KG-9: 图谱抽取/合并必须注入流式收集器 call_collect（read timeout 随每个
                # token 重置），非流式 call 的 120s 硬超时会在大 JSON 生成时误杀长请求
                set_default_graph_chat_model_fn(_chat_model.call_collect)
    return _chat_model


async def refresh_model_cache() -> int:
    """Rebuild the ModelCache from DB (builtin seed + provider rows) and
    invalidate the cached chat adapter so the next request picks up new specs.
    Returns the number of cached model entries."""
    from yusu_kb.models.providers.cache import model_cache
    from yusu_kb.models.providers.service import (
        ensure_builtin_model_providers_in_db,
        get_all_model_providers,
    )

    global _chat_model
    await ensure_builtin_model_providers_in_db()
    providers = await get_all_model_providers()
    model_cache.rebuild(providers)
    _chat_model = None
    return len(model_cache.get_all_specs())


def get_httpx_transport():
    """httpx transport for outbound HTTP calls (test seam; ``None`` in production)."""
    return None  # noqa: RET501 - 显式返回 None 表示生产环境不使用注入 transport


async def get_evaluation_service():
    """Lazily create the shared RAG evaluation service (model factory resolves
    specs through the ModelCache; KB access reuses the global manager)."""
    from yusu_kb.knowledge.eval.service import EvaluationService

    global _evaluation_service
    if _evaluation_service is None:
        manager = await get_manager()
        _evaluation_service = EvaluationService(kb_manager=manager)
    return _evaluation_service


async def shutdown() -> None:
    """Release shared resources (manager, engine)."""
    global _manager, _engine, _chat_model, _evaluation_service
    if _manager is not None:
        await _manager.close()
        _manager = None
    if _engine is not None:
        await dispose(_engine)
        _engine = None
    _chat_model = None
    _evaluation_service = None


def verify_api_key(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """Optional bearer-token auth; enforced only when ``YUSU_API_KEY`` is set."""
    expected = os.getenv("YUSU_API_KEY")
    if not expected:
        return
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[len("Bearer ") :].strip()
    if not token:
        token = (x_api_key or "").strip()
    if token != expected:
        raise HTTPException(status_code=401, detail="无效的 API Key")


# Annotated dependency aliases for FastAPI route signatures.
ManagerDep = Annotated[KnowledgeBaseManager, Depends(get_manager)]
ChatModelDep = Annotated[OpenAIChatAdapter, Depends(get_chat_model)]
EvaluationServiceDep = Annotated[Any, Depends(get_evaluation_service)]