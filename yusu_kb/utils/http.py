"""Shared HTTP utilities for model providers."""

from __future__ import annotations


def _normalize_endpoint(base_url: str, suffix: str) -> str:
    """Normalize a provider base URL to a full OpenAI-style endpoint.

    Accepts a base with or without a ``/v1`` suffix; always produces
    ``<base>/v1/<suffix>`` (unchanged when the URL already ends with
    ``suffix``). ``.env`` entries may specify just the bare host, and the
    code will still hit each vendor's ``/v1`` route.
    """
    base = (base_url or "").rstrip("/")
    if not base:
        return base
    if base.endswith(suffix):
        return base
    if base.endswith("/v1"):
        return f"{base}/{suffix}"
    return f"{base}/v1/{suffix}"