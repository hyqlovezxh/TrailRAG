"""Self-contained embedding layer.

Ported from YUSU ``yuxi.models.embed`` (model registry removed; configuration
is env-driven) plus a lightweight ``EmbeddingFunc`` replacement for the
LightRAG ``wrap_embedding_func_with_attrs`` contract used by LocalKB.

Env contract:
- ``YUSU_EMBED_BASE_URL`` (fallback ``YUSU_LLM_BASE_URL``)
- ``YUSU_EMBED_API_KEY``
- ``YUSU_EMBED_MODEL``
- ``YUSU_EMBED_DIM`` (optional; probed at first use when absent)
- ``YUSU_EMBED_BATCH_SIZE`` (default 40)
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from yusu_kb.utils.logger import logger

EMBEDDING_RATE_LIMIT_MAX_RETRIES = 10
EMBEDDING_TRANSIENT_MAX_RETRIES = 2
EMBEDDING_RETRY_MAX_DELAY_SECONDS = 10.0
EMBEDDING_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_PROBE_TEXT = "__yusu_embed_dim_probe__"


@dataclass
class EmbeddingFunc:
    """Lightweight replacement for the LightRAG ``EmbeddingFunc`` contract.

    Wraps an async embed callable ``(texts: list[str], **kwargs) -> vectors``
    together with its dimensionality metadata.
    """

    func: Callable[..., Awaitable[Any]]
    embedding_dim: int | None = None
    max_token_size: int | None = None
    model_name: str | None = None

    async def __call__(self, texts: list[str] | str, **kwargs) -> Any:
        return await self.func(texts, **kwargs)


def wrap_embedding_func_with_attrs(
    embedding_dim: int | None = None,
    max_token_size: int | None = None,
    model_name: str | None = None,
) -> Callable[[Callable[..., Awaitable[Any]]], EmbeddingFunc]:
    """Decorator: wrap an async embed callable into an ``EmbeddingFunc``."""

    def decorator(func: Callable[..., Awaitable[Any]]) -> EmbeddingFunc:
        return EmbeddingFunc(
            func=func,
            embedding_dim=embedding_dim,
            max_token_size=max_token_size,
            model_name=model_name,
        )

    return decorator


class BaseEmbeddingModel(ABC):
    def __init__(
        self,
        model=None,
        name=None,
        dimension=None,
        url=None,
        base_url=None,
        api_key=None,
        model_id=None,
        batch_size=40,
    ):
        base_url = base_url or url
        self.model = model or name or model_id
        self.dimension = dimension
        self.base_url = base_url
        self.api_key = api_key
        self.batch_size = int(batch_size or 40)
        # 限制 embed_state 大小，避免长期运行内存泄漏
        self.embed_state: dict[str, dict] = {}
        self._embed_state_max_entries = 100

    @abstractmethod
    async def aencode(self, message: list[str] | str) -> list[list[float]]:
        raise NotImplementedError("Subclasses must implement this method")

    async def aencode_queries(self, queries: list[str] | str) -> list[list[float]]:
        return await self.aencode(queries)

    async def abatch_encode(self, messages: list[str], batch_size: int | None = None) -> list[list[float]]:
        batch_size = batch_size or self.batch_size
        data = []
        task_id = None
        if len(messages) > batch_size:
            task_id = self._task_id(messages)
            self.embed_state[task_id] = {"status": "in-progress", "total": len(messages), "progress": 0}

        try:
            for i in range(0, len(messages), batch_size):
                group_msg = messages[i : i + batch_size]
                logger.info(f"Async encoding [{i}/{len(messages)}] messages (bsz={batch_size})")
                res = await self.aencode(group_msg)
                data.extend(res)
                if task_id:
                    self.embed_state[task_id]["progress"] = i + len(group_msg)

            if task_id:
                self.embed_state[task_id]["status"] = "completed"
        except Exception:
            if task_id:
                self.embed_state[task_id]["status"] = "failed"
            raise
        finally:
            if task_id:
                self._cleanup_embed_state()

        return data

    @staticmethod
    def _task_id(messages: list[str]) -> str:
        import hashlib

        return hashlib.sha256(repr(messages).encode("utf-8")).hexdigest()[:16]

    def _cleanup_embed_state(self) -> None:
        """清理已完成的 embed_state 条目，限制最大数量避免内存泄漏。"""
        if len(self.embed_state) <= self._embed_state_max_entries:
            return
        completed = [k for k, v in self.embed_state.items() if v.get("status") == "completed"]
        for k in completed:
            self.embed_state.pop(k, None)
            if len(self.embed_state) <= self._embed_state_max_entries:
                return
        while len(self.embed_state) > self._embed_state_max_entries:
            oldest_key = next(iter(self.embed_state))
            self.embed_state.pop(oldest_key, None)

    async def test_connection(self) -> tuple[bool, str]:
        try:
            embeddings = await self.aencode(["Hello world"])
            if self.dimension not in (None, ""):
                actual_dimension = len(embeddings[0]) if embeddings else 0
                expected_dimension = int(self.dimension)
                if actual_dimension != expected_dimension:
                    return False, f"Embedding 维度不一致：配置 {expected_dimension}，实际 {actual_dimension}"
            return True, "连接正常"
        except Exception as e:  # noqa: BLE001 - 连通性探测须返回结果而非抛出
            error_msg = str(e)
            error_msg += f", maybe you can check the `{self.base_url}` end with /embeddings as examples."
            logger.error(error_msg)
            return False, error_msg

    async def probe_dimension(self) -> int:
        """探测 embedding 维度（用于未显式配置 YUSU_EMBED_DIM 的场景）。"""
        embeddings = await self.aencode([_PROBE_TEXT])
        if not embeddings:
            raise ValueError("Embedding probe returned no vectors")
        return len(embeddings[0])


class OtherEmbedding(BaseEmbeddingModel):
    """OpenAI-compatible ``/embeddings`` client (async, httpx)."""

    def __init__(self, *, transport=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        # Test seam: an httpx transport to inject (e.g. MockTransport).
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    def build_payload(self, message: list[str] | str) -> dict:
        return {"model": self.model, "input": message}

    @staticmethod
    def _retry_delay_seconds(retry_index: int, retry_after: str | None = None) -> float:
        if retry_after:
            try:
                return min(float(retry_after), EMBEDDING_RETRY_MAX_DELAY_SECONDS)
            except ValueError:
                pass
        return min(float(2 ** (retry_index - 1)), EMBEDDING_RETRY_MAX_DELAY_SECONDS)

    def _prepare_retry(
        self,
        message: list[str] | str,
        *,
        retry_index: int,
        response=None,
        error: Exception | None = None,
    ) -> tuple[int, float] | None:
        status_code = getattr(response, "status_code", None)
        response_text = str(getattr(response, "text", "") or "")
        messages = [message] if isinstance(message, str) else message

        if status_code == 400 and response is not None:
            logger.warning(
                "Embedding request returned 400 Bad Request: "
                f"model={self.model}, base_url={self.base_url}, input_count={len(messages)}, "
                f"input_lengths={[len(item) for item in messages]}, body={response_text[:2000]}"
            )

        if status_code == 429:
            max_retries = EMBEDDING_RATE_LIMIT_MAX_RETRIES
        elif status_code in EMBEDDING_RETRYABLE_STATUS_CODES or status_code is None:
            max_retries = EMBEDDING_TRANSIENT_MAX_RETRIES
        else:
            max_retries = 0
        if retry_index >= max_retries:
            return None

        next_retry_index = retry_index + 1
        retry_after = response.headers.get("Retry-After") if response is not None else None
        delay = self._retry_delay_seconds(next_retry_index, retry_after)
        reason = f"status={status_code}" if status_code is not None else f"error={type(error).__name__}"
        logger.warning(
            "Retrying embedding request: "
            f"{reason}, model={self.model}, base_url={self.base_url}, "
            f"retry={next_retry_index}/{max_retries}, delay={delay:.1f}s, "
            f"input_count={len(messages)}, body={response_text[:1000]}"
        )
        return next_retry_index, delay

    @staticmethod
    def _extract_embeddings(result: dict) -> list[list[float]]:
        if not isinstance(result, dict) or "data" not in result:
            raise ValueError(f"Embedding failed: Invalid response format {result}")
        return [item["embedding"] for item in result["data"]]

    def _get_async_client(self) -> httpx.AsyncClient:
        """获取或创建复用的 AsyncClient，共享 TCP 连接池"""
        if self._client is None or self._client.is_closed:
            kwargs = {"timeout": 60}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def close_async_client(self) -> None:
        """关闭复用 AsyncClient，在应用关闭时调用"""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def aencode(self, message: list[str] | str) -> list[list[float]]:
        payload = self.build_payload(message)
        client = self._get_async_client()
        retry_index = 0
        while True:
            try:
                response = await client.post(self.base_url, json=payload, headers=self.headers, timeout=60)
                response.raise_for_status()
                return self._extract_embeddings(response.json())
            except httpx.HTTPStatusError as e:
                retry = self._prepare_retry(
                    message,
                    retry_index=retry_index,
                    response=e.response,
                    error=e,
                )
                if retry:
                    retry_index, delay = retry
                    await asyncio.sleep(delay)
                    continue
                raise
            except httpx.RequestError as e:
                retry = self._prepare_retry(message, retry_index=retry_index, error=e)
                if retry:
                    retry_index, delay = retry
                    await asyncio.sleep(delay)
                    continue
                raise ValueError(f"Embedding async request failed: {e}, {payload}, {self.base_url=}")


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _build_embedding_model_from_spec(default_spec: str) -> OtherEmbedding:
    """Build the embedding model from a ``provider_id:model_id`` spec via ModelCache."""
    from yusu_kb.models.providers.cache import model_cache

    info = model_cache.get_model_info(default_spec)
    if info is None:
        available = [item.spec for item in model_cache.get_all_specs("embedding")[:10]]
        raise ValueError(
            f"未找到模型: '{default_spec}'。可用 embedding 模型 ({len(model_cache.get_all_specs('embedding'))}): {available}"
        )
    if info.model_type != "embedding":
        raise ValueError(f"模型 '{default_spec}' 的类型是 {info.model_type}，不是 embedding")
    return OtherEmbedding(
        model=info.model_id,
        base_url=info.base_url,
        api_key=info.api_key,
        dimension=info.dimension,
        batch_size=info.batch_size,
    )


def create_embedding_model(*, default_spec: str | None = None) -> OtherEmbedding:
    """两级解析：AppConfig 默认 spec 优先（经 ModelCache），无配置回退 env 路径。

    ``default_spec`` 为 ``provider_id:model_id``；为 ``None`` 时行为与旧版一致
    （读取 ``YUSU_EMBED_*`` / ``YUSU_LLM_*`` 环境变量）。
    """
    if default_spec:
        return _build_embedding_model_from_spec(default_spec)
    base_url = _env_value("YUSU_EMBED_BASE_URL", "YUSU_LLM_BASE_URL")
    if not base_url:
        raise ValueError("YUSU_EMBED_BASE_URL / YUSU_LLM_BASE_URL 未配置")
    model = _env_value("YUSU_EMBED_MODEL")
    if not model:
        raise ValueError("YUSU_EMBED_MODEL 未配置")
    api_key = _env_value("YUSU_EMBED_API_KEY", "YUSU_LLM_API_KEY") or ""
    dimension = _env_value("YUSU_EMBED_DIM")
    batch_size = int(_env_value("YUSU_EMBED_BATCH_SIZE", default="40") or 40)
    if not base_url.endswith("/embeddings"):
        base_url = base_url.rstrip("/") + "/embeddings"
    return OtherEmbedding(
        model=model,
        base_url=base_url,
        api_key=api_key,
        dimension=int(dimension) if dimension else None,
        batch_size=batch_size,
    )


def create_default_embedding_func(
    *,
    provider_func=None,
    embedding_dim: int | None = None,
    max_token_size: int | None = None,
    model_name: str | None = None,
) -> EmbeddingFunc:
    """Build the process-wide default ``EmbeddingFunc``.

    ``provider_func`` is a test seam: when given, the provided async callable
    is wrapped directly (dimension must be resolvable from ``embedding_dim``
    or the callable's own attributes). Otherwise the env-driven
    ``OtherEmbedding`` model is used; the dimension is probed lazily at first
    use when not configured (``YUSU_EMBED_DIM`` or explicit ``embedding_dim``).
    """
    if provider_func is not None:
        provider_dim = None
        provider_max = None
        if isinstance(provider_func, EmbeddingFunc):
            provider_dim = provider_func.embedding_dim
            provider_max = provider_func.max_token_size
            provider_func = provider_func.func
        final_dim = int(embedding_dim or provider_dim or 0) or None
        if final_dim is None:
            raise ValueError("无法确定 embedding 维度；请显式传入 embedding_dim")
        return EmbeddingFunc(
            func=provider_func,
            embedding_dim=final_dim,
            max_token_size=max_token_size or provider_max,
            model_name=model_name,
        )

    model = create_embedding_model()
    final_dim = int(embedding_dim or model.dimension or 0) or None

    async def _embed(texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return await model.abatch_encode(texts)

    return EmbeddingFunc(
        func=_embed,
        embedding_dim=final_dim,
        max_token_size=max_token_size,
        model_name=model_name or model.model,
    )