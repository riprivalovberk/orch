from __future__ import annotations

import pytest
from pydantic import ValidationError

from orch.core.scope import can_write, protected_grants
from orch.domain.schemas import AgentConfig

P = frozenset({"payments"})
DEV = ["minibank/**"]
PAY = ["minibank/payments/**"]


@pytest.mark.parametrize(
    ("scope", "path", "ok"),
    [
        (DEV, "minibank/transfers.py", True),
        (DEV, "minibank/payments/charge.py", False),  # broad glob never reaches payments
        (DEV, "tests/test_transfers.py", False),  # dev cannot edit the acceptance tests
        (PAY, "minibank/payments/charge.py", True),
        (PAY, "minibank/api.py", False),
        (DEV, "../etc/passwd", False),
        (DEV, "/minibank/x.py", False),
    ],
)
def test_can_write(scope, path, ok):
    assert can_write(scope, path, P) is ok


def test_protected_grants():
    assert protected_grants(DEV, P) == set()
    assert protected_grants(PAY, P) == {"payments"}


def agent(**kw):
    base = {"name": "dev", "description": "writes code for tests", "capabilities": ["implement"]}
    return AgentConfig(**{**base, **kw})


def test_writer_needs_scope():
    with pytest.raises(ValidationError, match="no write_paths"):
        agent(tools=["write_file"])


def test_scope_must_be_relative():
    with pytest.raises(ValidationError, match="repo-relative"):
        agent(tools=["write_file"], write_paths=["../outside/**"])
