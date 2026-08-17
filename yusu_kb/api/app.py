"""FastAPI application factory + lifespan."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from yusu_kb.api import deps
from yusu_kb.api.routers import chat, evaluation, graph, health, knowledge, models


@asynccontextmanager
async def _lifespan(app: FastAPI):
    yield
    await deps.shutdown()


def _mount_webui(app: FastAPI) -> None:
    """Serve the built frontend (yusu_webui/dist) when present."""
    # app.py 位于 yusu_kb/api/，parents[2] 即 current_repository_code/；
    # 前端构建产物在 yusu_webui/dist/。
    dist = Path(__file__).resolve().parents[2] / "yusu_webui" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="webui")


def create_app() -> FastAPI:
    """Create the YUSU API application."""
    app = FastAPI(title="语溯开源版 YUSU API", version="0.1.0", lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(knowledge.router)
    app.include_router(graph.router)
    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(evaluation.router)
    _mount_webui(app)
    return app


app = create_app()