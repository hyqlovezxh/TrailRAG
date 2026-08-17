"""Self-contained reranker layer.

Ported from YUSU ``yuxi.models.rerank`` (aiohttp swapped for httpx; the
model registry is replaced by env-driven ``create_reranker``). Env contract:

- ``YUSU_RERANK_BASE_URL``
- ``YUSU_RERANK_API_KEY``
- ``YUSU_RERANK_MODEL``
- ``YUSU_RERANK_PROTOCOL`` (optional: ``openai`` (default) | ``dashscope``)

Reranking is optional: when ``YUSU_RERANK_MODEL`` is unset, retrieval runs
without a reranker.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

import httpx

from yusu_kb.utils.logger import logger


class BaseReranker(ABC):
    def __init__(self, model_name, api_key, base_url, **kwargs):
        self.url = base_url
        self.model = model_name
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.timeout = 30.0
        self.parameters: dict[str, Any] = dict(kwargs.get("parameters", {}))
        # Test seam: an httpx transport to inject (e.g. MockTransport).
        self._transport = kwargs.get("transport")
        self._session: httpx.AsyncClient | None = None

    async def _ensure_session(self) -> httpx.AsyncClient:
        if self._session is None or self._session.is_closed:
            kwargs = {"headers": self.headers, "timeout": self.timeout}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._session = httpx.AsyncClient(**kwargs)
        return self._session

    @abstractmethod
    def _build_payload(self, query: str, documents: list[str], max_length: int) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _extract_results(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def acompute_score(
        self,
        sentence_pairs: Sequence[Sequence[str]],
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
    ) -> list[float]:
        if not sentence_pairs or len(sentence_pairs) < 2:
            return []

        query, sentences = sentence_pairs[0], sentence_pairs[1]
        documents = [sentences] if isinstance(sentences, str) else list(sentences)

        if not documents:
            return []

        all_scores: list[float] = []
        batch_size = max(1, int(batch_size))
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for batch_no, start in enumerate(range(0, len(documents), batch_size), start=1):
            batch = documents[start : start + batch_size]
            try:
                scores = await self._batch_rerank(query, batch, max_length=max_length)
                all_scores.extend(scores)
                logger.debug(f"Reranking batch {batch_no}/{total_batches} completed")
            except Exception as exc:  # noqa: BLE001 - 沉底哨兵须捕获全部异常
                # 失败批次重试一次，仍失败则用 -1.0 沉底哨兵（排到末尾，被 min_rerank_score 自然过滤）
                logger.error(f"Reranking batch {batch_no} failed (attempt 1): {exc}")
                try:
                    scores = await self._batch_rerank(query, batch, max_length=max_length)
                    all_scores.extend(scores)
                    logger.info(f"Reranking batch {batch_no} succeeded on retry")
                except Exception as retry_exc:  # noqa: BLE001 - 沉底哨兵须捕获全部异常
                    logger.error(f"Reranking batch {batch_no} retry failed: {retry_exc}")
                    all_scores.extend([-1.0] * len(batch))

        # R2 修复：主流 rerank API 返回的 relevance_score 已是 [0,1] 归一化值，
        # 不再做二次 sigmoid（会将分数错误压缩到 [0.5, 0.73]，使 min_rerank_score 阈值失效）。
        # normalize 参数保留以兼容调用方签名，但仅在分数超出 [0,1] 时做 min-max 归一化。
        # R3：-1.0 是失败哨兵，不参与 min-max 计算（否则会把正常分数错误拉伸）。
        # RR-1 修复：valid_scores 已过滤 >= 0，min(valid_scores) < 0.0 恒假，移除该死代码分支。
        if normalize:
            valid_scores = [s for s in all_scores if s >= 0.0]
            if valid_scores and max(valid_scores) > 1.0:
                lo, hi = min(valid_scores), max(valid_scores)
                if hi > lo:
                    all_scores = [(s - lo) / (hi - lo) if s >= 0.0 else s for s in all_scores]

        return all_scores

    async def _batch_rerank(self, query: str, documents: Iterable[str], max_length: int) -> list[float]:
        docs = list(documents)
        if not docs:
            return []

        payload = self._build_payload(query, docs, max_length)
        session = await self._ensure_session()

        try:
            response = await session.post(self.url, json=payload)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
        except httpx.TimeoutException:
            logger.error(f"Reranking request timeout after {self.timeout:.1f}s")
            raise
        except httpx.HTTPError as exc:
            logger.error(f"Reranking request failed: {exc}")
            raise

        processed = sorted(self._extract_results(result), key=lambda item: item.get("index", 0))
        return [float(entry.get("relevance_score", 0.0)) for entry in processed]

    def compute_score(self, sentence_pairs, batch_size=256, max_length=512, normalize=False):
        try:
            _ = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.acompute_score(sentence_pairs, batch_size, max_length, normalize))
        raise RuntimeError("compute_score cannot be used while an event loop is running. Use acompute_score instead.")

    async def test_connection(self) -> tuple[bool, str]:
        try:
            scores = await self._batch_rerank("test query", ["test document"], max_length=128)
            if scores:
                return True, "连接正常"
            return False, "响应无效"
        except Exception as e:  # noqa: BLE001 - 连通性探测须返回结果而非抛出
            error_msg = str(e)
            logger.error(f"Rerank connection test failed: {error_msg}")
            return False, error_msg
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        if self._session is not None and not self._session.is_closed:
            await self._session.aclose()
            self._session = None


class OpenAIReranker(BaseReranker):
    def _build_payload(self, query: str, documents: list[str], max_length: int) -> dict[str, Any]:
        return {
            "model": self.model,
            "query": query,
            "documents": documents,
            "max_chunks_per_doc": max_length,
        }

    def _extract_results(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        return list(result.get("results", []))


class DashscopeReranker(BaseReranker):
    def _build_payload(self, query: str, documents: list[str], max_length: int) -> dict[str, Any]:
        params = {"top_n": len(documents), "return_documents": False}
        instruct = self.parameters.get("instruct")
        if instruct:
            params["instruct"] = instruct
        return {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": params,
        }

    def _extract_results(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        return list(result.get("output", {}).get("results", []))


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _build_reranker_from_spec(default_spec: str) -> OpenAIReranker:
    """Build the reranker from a ``provider_id:model_id`` spec via ModelCache."""
    from yusu_kb.models.providers.cache import model_cache

    info = model_cache.get_model_info(default_spec)
    if info is None:
        available = [item.spec for item in model_cache.get_all_specs("rerank")[:10]]
        raise ValueError(
            f"未找到模型: '{default_spec}'。可用 rerank 模型 ({len(model_cache.get_all_specs('rerank'))}): {available}"
        )
    if info.model_type != "rerank":
        raise ValueError(f"模型 '{default_spec}' 的类型是 {info.model_type}，不是 rerank")
    return OpenAIReranker(model_name=info.model_id, api_key=info.api_key, base_url=info.base_url)


def create_reranker(*, default_spec: str | None = None) -> BaseReranker | None:
    """两级解析：AppConfig 默认 spec 优先（经 ModelCache），无配置回退 env 路径。

    ``default_spec`` 为 ``provider_id:model_id``；为 ``None`` 时行为与旧版一致
    （读取 ``YUSU_RERANK_*`` 环境变量；未配置返回 ``None``）。
    """
    if default_spec:
        return _build_reranker_from_spec(default_spec)
    model = _env_value("YUSU_RERANK_MODEL")
    if not model:
        return None
    base_url = _env_value("YUSU_RERANK_BASE_URL")
    if not base_url:
        raise ValueError("YUSU_RERANK_MODEL 已配置但缺少 YUSU_RERANK_BASE_URL")
    api_key = _env_value("YUSU_RERANK_API_KEY")
    if not api_key:
        raise ValueError(f"YUSU_RERANK_API_KEY is required for reranker {model}")
    protocol = _env_value("YUSU_RERANK_PROTOCOL", default="openai")
    if protocol == "dashscope":
        return DashscopeReranker(model_name=model, api_key=api_key, base_url=base_url)
    return OpenAIReranker(model_name=model, api_key=api_key, base_url=base_url)


_reranker_cache: dict[str, BaseReranker] = {}
_reranker_cache_lock = asyncio.Lock()


async def get_cached_reranker(model_id: str) -> BaseReranker:
    """获取或创建按 model_id 缓存的 reranker 实例（复用 session，不关闭）。"""
    async with _reranker_cache_lock:
        cached = _reranker_cache.get(model_id)
        if cached is not None and cached._session is not None and not cached._session.is_closed:
            return cached
        reranker = create_reranker()
        if reranker is None:
            raise ValueError(f"No reranker configured for model: {model_id}")
        reranker.model = model_id
        _reranker_cache[model_id] = reranker
        return reranker


async def close_cached_rerankers() -> None:
    """进程退出时统一关闭所有缓存的 reranker session。"""
    async with _reranker_cache_lock:
        for reranker in _reranker_cache.values():
            try:
                await reranker.aclose()
            except Exception:  # noqa: BLE001, S110 - 退出清理吞掉关闭错误
                pass
        _reranker_cache.clear()