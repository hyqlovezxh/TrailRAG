"""Datetime helpers for consistent UTC handling.

Adapted from YUSU ``yuxi.utils.datetime_utils`` — timestamps are stored in UTC
(naive, matching SQLite's timezone-less storage) and exposed as ISO-8601 UTC.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from typing import Any
from zoneinfo import ZoneInfo

UTC = dt.UTC
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai")
_ISO_Z_SUFFIX = "+00:00"


def utc_now() -> dt.datetime:
    """Return the current UTC time as an aware datetime."""
    return dt.datetime.now(UTC)


def utc_now_naive() -> dt.datetime:
    """Return the current UTC time as a naive datetime (for DB fields without timezone)."""
    return dt.datetime.now(UTC).replace(tzinfo=None)


def ensure_utc(value: dt.datetime) -> dt.datetime:
    """Convert a datetime to UTC; naive values are assumed to be Asia/Shanghai."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=SHANGHAI_TZ)
    return value.astimezone(UTC)


def utc_isoformat(value: dt.datetime | None = None) -> str:
    """Return an ISO 8601 string in UTC with a trailing Z suffix."""
    value = ensure_utc(value or utc_now())
    iso_string = value.isoformat()
    if iso_string.endswith(_ISO_Z_SUFFIX):
        return iso_string.replace(_ISO_Z_SUFFIX, "Z")
    return iso_string


def format_utc_datetime(value: dt.datetime | None) -> str | None:
    """Format a datetime to a UTC ISO-8601 string, handling naive datetimes (assumed UTC)."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return utc_isoformat(value)


def coerce_any_to_utc_datetime(value: Any) -> dt.datetime | None:
    """Best-effort conversion of strings / datetimes to a UTC-aware datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return ensure_utc(value)
    if isinstance(value, dt.date):
        return ensure_utc(dt.datetime.combine(value, dt.time.min))
    if isinstance(value, str):
        text = value.strip().rstrip("Z")
        parsed = dt.datetime.fromisoformat(text)
        return ensure_utc(parsed)
    raise TypeError(f"Cannot coerce {type(value).__name__} to datetime")


def normalize_iterable_to_utc(values: Iterable[dt.datetime | None]) -> list[dt.datetime | None]:
    """Normalize each datetime in an iterable to UTC."""
    return [ensure_utc(item) if item is not None and item.tzinfo is not None else item for item in values]


__all__ = [
    "SHANGHAI_TZ",
    "UTC",
    "coerce_any_to_utc_datetime",
    "ensure_utc",
    "format_utc_datetime",
    "normalize_iterable_to_utc",
    "utc_isoformat",
    "utc_now",
    "utc_now_naive",
]
