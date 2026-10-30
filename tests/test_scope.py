from __future__ import annotations

import pytest
from pydantic import ValidationError

from orch.core.scope import can_write, protected_grants, task_can_write, touches
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


def test_task_can_write_needs_agent_and_task_scope():
    # payments_developer on a task the plan scoped to refunds.py only
    assert task_can_write(PAY, ["minibank/payments/refunds.py"], "minibank/payments/refunds.py", P)
    assert not task_can_write(PAY, ["minibank/payments/refunds.py"], "minibank/payments/x.py", P)
    # a broad task glob never grants payments: the agent's scope still decides
    assert not task_can_write(DEV, ["minibank/**"], "minibank/payments/refunds.py", P)
    assert task_can_write(DEV, ["minibank/**"], "minibank/api/transfers.py", P)


def test_touches():
    assert touches(["minibank/payments/refunds.py"], P) == {"payments"}
    assert touches(["minibank/**", "tests/test_x.py"], P) == set()
