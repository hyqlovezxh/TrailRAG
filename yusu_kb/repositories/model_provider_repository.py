"""Model provider & app config repository (SQLite).

Ported from YUSU ``yuxi.models.providers.repository`` (which was trimmed from
the reference tree); follows the same self-contained session pattern as the
other SQLite repositories.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select

from yusu_kb.storage.sqlite.models_knowledge import AppConfig, ModelProvider

from . import get_session_factory

_PROVIDER_FIELDS = {
    "provider_id",
    "display_name",
    "provider_type",
    "default_protocol",
    "base_url",
    "embedding_base_url",
    "rerank_base_url",
    "models_endpoint",
    "embedding_models_endpoint",
    "rerank_models_endpoint",
    "api_key_env",
    "capabilities",
    "enabled_models",
    "headers_json",
    "extra_json",
    "is_enabled",
    "is_builtin",
    "created_by",
    "updated_by",
}

_CONFIG_FIELDS = {"config_key", "config_value", "updated_by"}


def _sanitize(data: dict[str, Any], allowed: set[str]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key in allowed and value is not None}


class ModelProviderRepository:
    """SQLite-backed model provider repository."""

    async def get_all(self) -> list[ModelProvider]:
        async with get_session_factory()() as session:
            result = await session.execute(select(ModelProvider).order_by(ModelProvider.id))
            return list(result.scalars().all())

    async def get_by_provider_id(self, provider_id: str) -> ModelProvider | None:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(ModelProvider).where(ModelProvider.provider_id == provider_id)
            )
            return result.scalar_one_or_none()

    async def create(self, data: dict[str, Any]) -> ModelProvider:
        async with get_session_factory()() as session:
            record = ModelProvider(**_sanitize(data, _PROVIDER_FIELDS))
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def update(self, provider_id: str, data: dict[str, Any]) -> ModelProvider | None:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(ModelProvider).where(ModelProvider.provider_id == provider_id)
            )
            record = result.scalar_one_or_none()
            if record is None:
                return None
            for key, value in _sanitize(data, _PROVIDER_FIELDS).items():
                setattr(record, key, value)
            await session.commit()
            await session.refresh(record)
            return record

    async def delete(self, provider_id: str) -> bool:
        async with get_session_factory()() as session:
            result = await session.execute(
                select(ModelProvider.id).where(ModelProvider.provider_id == provider_id)
            )
            if result.scalar_one_or_none() is None:
                return False
            await session.execute(delete(ModelProvider).where(ModelProvider.provider_id == provider_id))
            await session.commit()
            return True


class AppConfigRepository:
    """SQLite-backed app config repository (key/value JSON)."""

    async def get(self, config_key: str) -> AppConfig | None:
        async with get_session_factory()() as session:
            result = await session.execute(select(AppConfig).where(AppConfig.config_key == config_key))
            return result.scalar_one_or_none()

    async def get_all(self) -> list[AppConfig]:
        async with get_session_factory()() as session:
            result = await session.execute(select(AppConfig).order_by(AppConfig.config_key))
            return list(result.scalars().all())

    async def set(self, config_key: str, config_value: Any, updated_by: str | None = None) -> AppConfig:
        async with get_session_factory()() as session:
            result = await session.execute(select(AppConfig).where(AppConfig.config_key == config_key))
            record = result.scalar_one_or_none()
            if record is None:
                record = AppConfig(config_key=config_key, config_value=config_value, updated_by=updated_by)
                session.add(record)
            else:
                record.config_value = config_value
                if updated_by is not None:
                    record.updated_by = updated_by
            await session.commit()
            await session.refresh(record)
            return record

    async def delete(self, config_key: str) -> bool:
        async with get_session_factory()() as session:
            result = await session.execute(select(AppConfig.id).where(AppConfig.config_key == config_key))
            if result.scalar_one_or_none() is None:
                return False
            await session.execute(delete(AppConfig).where(AppConfig.config_key == config_key))
            await session.commit()
            return True
