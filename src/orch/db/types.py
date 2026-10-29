"""Column types that store Pydantic values as JSON and hand them back typed."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter
from sqlalchemy import JSON
from sqlalchemy.types import TypeDecorator


class PydanticJSON(TypeDecorator[Any]):
    """JSON column for any Pydantic-validatable type: a model, ``list[Model]``, etc.

    Writes are validated, so a bad value fails at flush instead of on a later read.
    Values are treated as immutable: reassign the attribute to change it (in-place
    mutation of a JSON column is not tracked by SQLAlchemy).
    """

    impl = JSON
    cache_ok = True

    def __init__(self, tp: Any) -> None:
        super().__init__()
        self.tp = tp
        self._adapter: TypeAdapter[Any] = TypeAdapter(tp)

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return self._adapter.dump_python(self._adapter.validate_python(value), mode="json")

    def process_result_value(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        return self._adapter.validate_python(value)
