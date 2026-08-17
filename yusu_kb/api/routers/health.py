"""Health endpoints: liveness and model connectivity probes."""

from __future__ import annotations

from fastapi import APIRouter

from yusu_kb.api.deps import ChatModelDep, ManagerDep
from yusu_kb.models.embed import create_embedding_model
from yusu_kb.models.rerank import create_reranker

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("")
async def health() -> dict:
    """Liveness probe (no external dependencies)."""
    return {"status": "ok", "service": "yusu-kb", "version": "0.1.0"}


@router.get("/models")
async def model_status(
    manager: ManagerDep,
    chat_model: ChatModelDep,
) -> dict:
    """Probe connectivity of the configured embedding / chat / rerank models."""
    del manager

    embed = create_embedding_model()
    try:
        embed_ok, embed_message = await embed.test_connection()
    except Exception as exc:  # noqa: BLE001 - probe must report instead of raising
        embed_ok, embed_message = False, str(exc)
    finally:
        await embed.close_async_client()

    try:
        response = await chat_model.call([{"role": "user", "content": "Say 1"}], stream=False)
        chat_status = "available" if response and response.content else "unavailable"
        chat_message = "连接正常" if chat_status == "available" else "响应无效"
    except Exception as exc:  # noqa: BLE001 - probe must report instead of raising
        chat_status, chat_message = "error", str(exc)

    reranker = create_reranker()
    if reranker is None:
        rerank_status = {"configured": False}
    else:
        try:
            rerank_ok, rerank_message = await reranker.test_connection()
            rerank_status = {
                "configured": True,
                "status": "available" if rerank_ok else "unavailable",
                "message": rerank_message,
            }
        except Exception as exc:  # noqa: BLE001 - probe must report instead of raising
            rerank_status = {"configured": True, "status": "error", "message": str(exc)}
        finally:
            await reranker.aclose()

    return {
        "embedding": {"status": "available" if embed_ok else "error", "message": embed_message},
        "chat": {"status": chat_status, "message": chat_message},
        "rerank": rerank_status,
    }