"""进程内模型缓存（无 Redis）。

Ported from YUSU ``yuxi.models.providers.cache`` with the Redis layer
removed: the cache is a plain in-process snapshot that ``rebuild()``
replaces in place after provider config changes (single-process server, so
no cross-process consistency is needed). Model spec format:
``provider_id:model_id`` (colon-separated; model_id may contain slashes).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from yusu_kb.utils.logger import logger

_CACHE_TTL_SECONDS = 5


@dataclass(frozen=True)
class ModelInfo:
    """不可变的模型信息，供运行时使用。"""

    provider_id: str
    model_id: str
    model_type: str  # chat / embedding / rerank
    display_name: str

    # 运行时配置
    api_key: str
    base_url: str
    provider_type: str  # openai / anthropic / gemini / openrouter

    # 可选配置
    headers: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    # 能力标记
    supports_vision: bool = False

    # Embedding 专属
    dimension: int | None = None
    batch_size: int = 40

    @property
    def spec(self) -> str:
        return f"{self.provider_id}:{self.model_id}"

    def to_dict(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "model_id": self.model_id,
            "model_type": self.model_type,
            "display_name": self.display_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "provider_type": self.provider_type,
            "headers": self.headers,
            "extra": self.extra,
            "supports_vision": self.supports_vision,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ModelInfo:
        return cls(
            provider_id=data["provider_id"],
            model_id=data["model_id"],
            model_type=data["model_type"],
            display_name=data["display_name"],
            api_key=data["api_key"],
            base_url=data["base_url"],
            provider_type=data["provider_type"],
            headers=data.get("headers", {}),
            extra=data.get("extra", {}),
            supports_vision=data.get("supports_vision", False),
            dimension=data.get("dimension"),
            batch_size=data.get("batch_size", 40),
        )


class ModelCache:
    """进程内模型缓存：rebuild 后立即替换快照，读取无 IO。"""

    def __init__(self) -> None:
        self._cache: dict[str, ModelInfo] = {}
        self._cache_at: float = 0.0

    def _load_cache(self) -> dict[str, ModelInfo]:
        # 进程内缓存没有外部数据源（无 Redis），快照由 rebuild() 显式重建；
        # _CACHE_TTL_SECONDS 保留以对齐 YUSU 结构，实际读取始终返回当前快照。
        return self._cache

    def get_model_info(self, spec: str) -> ModelInfo | None:
        cache = self._load_cache()
        return cache.get(spec)

    def get_all_specs(self, model_type: str | None = None) -> list[ModelInfo]:
        cache = self._load_cache()
        if model_type is None:
            return list(cache.values())
        return [info for info in cache.values() if info.model_type == model_type]

    def get_specs_grouped_by_provider(self, model_type: str = "chat") -> dict[str, list[ModelInfo]]:
        cache = self._load_cache()
        grouped: dict[str, list[ModelInfo]] = {}
        for info in cache.values():
            if info.model_type != model_type:
                continue
            grouped.setdefault(info.provider_id, []).append(info)
        return grouped

    def rebuild(self, providers: list[Any]) -> None:
        from yusu_kb.models.providers.service import resolve_api_key

        new_cache: dict[str, ModelInfo] = {}

        for provider in providers:
            if not provider.is_enabled:
                continue

            api_key = resolve_api_key(provider)

            for model in provider.enabled_models or []:
                model_type = model.get("type", "chat")
                base_url = model.get("base_url_override") or self._get_base_url_for_type(provider, model_type)

                supports_vision = model.get("supports_vision")
                if supports_vision is None:
                    model_extra = dict(model.get("extra") or {})
                    input_modalities = model_extra.get("input_modalities") or []
                    supports_vision = "image" in input_modalities

                info = ModelInfo(
                    provider_id=provider.provider_id,
                    model_id=model["id"],
                    model_type=model_type,
                    display_name=model.get("display_name", model["id"]),
                    api_key=api_key or "",
                    base_url=base_url,
                    provider_type=provider.provider_type,
                    headers=dict(provider.headers_json or {}),
                    extra=dict(provider.extra_json or {}),
                    supports_vision=bool(supports_vision),
                    dimension=model.get("dimension"),
                    batch_size=model.get("batch_size", 40),
                )
                new_cache[info.spec] = info

        self._cache = new_cache
        self._cache_at = time.monotonic()
        logger.info(f"Model cache rebuilt: {len(new_cache)} models")

    @staticmethod
    def _get_base_url_for_type(provider: Any, model_type: str) -> str:
        if model_type == "embedding" and provider.embedding_base_url:
            return provider.embedding_base_url
        if model_type == "rerank" and provider.rerank_base_url:
            return provider.rerank_base_url
        return provider.base_url


model_cache = ModelCache()


def resolve_model_spec(spec: str) -> ModelInfo:
    """根据 spec 返回 ModelInfo（provider_id:model_id）。"""
    if not spec:
        raise ValueError("model spec 不能为空")

    info = model_cache.get_model_info(spec)
    if info:
        return info

    all_specs = model_cache.get_all_specs()
    available = [item.spec for item in all_specs[:10]]
    raise ValueError(f"未找到模型: '{spec}'。可用模型 ({len(all_specs)}): {available}")
