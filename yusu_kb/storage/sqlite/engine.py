"""SQLite storage: async engine, schema initialization and session factory.

Replaces YUSU's PostgreSQL (asyncpg) layer. SQLite specifics:
- WAL journal mode for concurrent readers,
- ``PRAGMA foreign_keys=ON`` so ``ondelete="CASCADE"`` constraints work,
- naive UTC datetimes (SQLite has no timezone support).
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

DEFAULT_DATA_DIR_NAME = "yusu_data"
ENV_DATA_DIR = "YUSU_DATA_DIR"


class Base(DeclarativeBase):
    """Declarative base for all YUSU knowledge tables."""


def get_data_dir() -> Path:
    """Resolve the YUSU data directory (env ``YUSU_DATA_DIR`` or ``./yusu_data``)."""
    env = os.getenv(ENV_DATA_DIR)
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd() / DEFAULT_DATA_DIR_NAME


def get_db_path(data_dir: Path | None = None) -> Path:
    """Resolve the SQLite database file path."""
    return (data_dir or get_data_dir()) / "yusu.db"


def _sqlite_on_connect(dbapi_connection, connection_record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


def create_engine(db_path: Path | None = None) -> AsyncEngine:
    """Create an async SQLAlchemy engine over the given SQLite file.

    ``db_path`` is used by tests for isolation; default resolves via
    :func:`get_db_path`.
    """
    path = db_path or get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite+aiosqlite:///{path.as_posix()}"
    engine = create_async_engine(url, echo=False)
    event.listen(engine.sync_engine, "connect", _sqlite_on_connect)
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to ``engine``."""
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine: AsyncEngine) -> None:
    """Create all tables (idempotent)."""
    from . import models_knowledge  # noqa: F401  (register models on Base)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose(engine: AsyncEngine) -> None:
    """Dispose the engine (close connection pool)."""
    await engine.dispose()


__all__ = [
    "ENV_DATA_DIR",
    "Base",
    "create_engine",
    "create_session_factory",
    "dispose",
    "get_data_dir",
    "get_db_path",
    "init_db",
]