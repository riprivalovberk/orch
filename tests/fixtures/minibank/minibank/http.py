"""Just enough HTTP: a handler takes a Request and returns a Response."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Request:
    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""

    def header(self, name: str) -> str | None:
        """Case-insensitive header lookup."""
        for k, v in self.headers.items():
            if k.lower() == name.lower():
                return v
        return None

    def json(self) -> Any:
        return json.loads(self.body) if self.body else None


@dataclass(frozen=True)
class Response:
    status: int
    body: dict[str, Any] = field(default_factory=dict)


def error(status: int, message: str) -> Response:
    return Response(status, {"error": message})
