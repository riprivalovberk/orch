"""ID and time helpers. All timestamps are UTC ISO-8601 strings with millisecond precision."""

from __future__ import annotations

from datetime import UTC, datetime

from ulid import ULID


def new_id(prefix: str) -> str:
    """A sortable, prefixed id such as ``tsk_01JB...``."""
    return f"{prefix}_{ULID()}"


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="milliseconds")


def now_iso() -> str:
    return iso(utcnow())
