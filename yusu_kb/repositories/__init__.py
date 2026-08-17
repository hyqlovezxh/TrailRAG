"""Repository layer over SQLite (replaces YUSU's PostgreSQL repositories).

Repositories resolve their engine through :func:`get_session_factory`; tests
and the API server call :func:`configure_repositories` with a dedicated
engine before use.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from yusu_kb.storage.sqlite.engine import create_engine, create_session_factory

_factory: async_sessionmaker | None = None


def configure_repositories(engine: AsyncEngine | None) -> None:
    """Point all repositories at a specific engine (or reset to default)."""
    global _factory
    _factory = create_session_factory(engine) if engine is not None else None


def get_session_factory() -> async_sessionmaker:
    """Get the repository session factory (lazily bound to the default DB)."""
    global _factory
    if _factory is None:
        _factory = create_session_factory(create_engine())
    return _factory


__all__ = ["configure_repositories", "get_session_factory"]