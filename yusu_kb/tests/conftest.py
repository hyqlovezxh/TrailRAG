"""Shared fixtures for YUSU KB tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

from yusu_kb.storage.files.storage import LocalFileStorage
from yusu_kb.storage.sqlite.engine import (
    create_engine,
    create_session_factory,
    init_db,
)


@pytest.fixture()
def tmp_workdir(tmp_path: Path) -> Path:
    """A throwaway directory that also avoids the default ``./yusu_data``."""
    return tmp_path / "yusu_work"


@pytest.fixture()
async def engine(tmp_workdir: Path) -> AsyncEngine:
    eng = create_engine(tmp_workdir / "yusu.db")
    await init_db(eng)
    yield eng
    await eng.dispose()


@pytest.fixture()
def session_factory(engine: AsyncEngine):
    return create_session_factory(engine)


@pytest.fixture()
def file_storage(tmp_workdir: Path) -> LocalFileStorage:
    return LocalFileStorage(tmp_workdir / "files")
