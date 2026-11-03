"""A scripted team for the transfers objective: no network, same tools, real code.

Used by ``orch run --offline`` and the end-to-end tests. Each role is recognized by the
tools it was given (submit_plan, submit_review, write_file) and the task line in its brief.
The test engineer and developer write working code, so the gates run for real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

ROOT = Path(__file__).resolve().parents[3]

TRANSFERS_OBJECTIVE = "Add an idempotent POST /transfers endpoint in minibank/api/transfers.py"
TRANSFERS_CRITERIA = {
    "AC-1": "A repeated Idempotency-Key never double-posts",
    "AC-2": "Amounts are Decimal, never float",
    "AC-3": "Overdrafts are rejected with 422",
    "AC-4": "Logs never contain full account numbers",
}

TESTS = '''"""POST /transfers acceptance tests, one or more per criterion."""

import json
import logging
from decimal import Decimal

import pytest

from minibank.api.transfers import Transfers
from minibank.http import Request
from minibank.ledger import Ledger

A, B = "GB0011112222", "GB0033334444"


@pytest.fixture
def api() -> Transfers:
    led = Ledger()
    led.open(A, Decimal("100.00"))
    led.open(B, Decimal("0.00"))
    return Transfers(led)


def post(api: Transfers, key: str | None, **body: object):
    headers = {"Idempotency-Key": key} if key else {}
    return api.post(Request("POST", "/transfers", headers, json.dumps(body)))


def test_ac1_repeated_key_posts_once(api: Transfers):
    first = post(api, "k1", source=A, target=B, amount="10.00")
    again = post(api, "k1", source=A, target=B, amount="10.00")
    assert first.status == again.status == 201
    assert api.ledger.balance(B) == Decimal("10.00")


def test_ac1_key_is_required(api: Transfers):
    assert post(api, None, source=A, target=B, amount="1.00").status == 400


def test_ac2_amount_must_be_a_decimal_string(api: Transfers):
    assert post(api, "k2", source=A, target=B, amount=0.1).status == 422
    assert post(api, "k3", source=A, target=B, amount="0.10").status == 201
    assert api.ledger.balance(B) == Decimal("0.10")


def test_ac3_overdraft_is_422(api: Transfers):
    assert post(api, "k4", source=B, target=A, amount="1.00").status == 422
    assert api.ledger.balance(A) == Decimal("100.00")


def test_ac4_logs_mask_account_numbers(api: Transfers, caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.INFO):
        post(api, "k5", source=A, target=B, amount="1.00")
        post(api, "k6", source=B, target=A, amount="500.00")
    assert caplog.records
    assert A not in caplog.text and B not in caplog.text
'''

TRANSFERS = '''"""POST /transfers: move money between accounts, at most once per Idempotency-Key."""

from __future__ import annotations

import logging
from decimal import Decimal, InvalidOperation

from minibank.http import Request, Response, error
from minibank.ledger import InsufficientFunds, Ledger, LedgerError, mask

log = logging.getLogger(__name__)


class Transfers:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self._done: dict[str, Response] = {}  # Idempotency-Key -> the response we sent

    def post(self, req: Request) -> Response:
        key = req.header("Idempotency-Key")
        if not key:
            return error(400, "Idempotency-Key header is required")
        if key in self._done:
            return self._done[key]
        resp = self._transfer(req)
        if resp.status < 500:
            self._done[key] = resp
        return resp

    def _transfer(self, req: Request) -> Response:
        body = req.json() or {}
        source, target, raw = body.get("source"), body.get("target"), body.get("amount")
        if not isinstance(source, str) or not isinstance(target, str):
            return error(422, "source and target are required")
        if not isinstance(raw, str):
            return error(422, "amount must be a decimal string, e.g. \\"10.00\\"")
        try:
            amount = Decimal(raw)
        except InvalidOperation:
            return error(422, "amount is not a number")
        try:
            self.ledger.move(source, target, amount)
        except InsufficientFunds:
            log.info("transfer refused: insufficient funds in %s", mask(source))
            return error(422, "insufficient funds")
        except LedgerError as e:
            return error(422, str(e))
        log.info("moved %s from %s to %s", amount, mask(source), mask(target))
        return Response(201, {"source": source, "target": target, "amount": str(amount)})
'''


def _prompt(messages: list[ModelMessage]) -> str:
    for m in messages:
        if isinstance(m, ModelRequest):
            for p in m.parts:
                if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                    return p.content
    return ""


def _returned(messages: list[ModelMessage]) -> list[str]:
    return [
        p.tool_name
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]


def _call(name: str, args: dict[str, Any]) -> ModelResponse:
    return ModelResponse([ToolCallPart(name, args)])


def scripted_team() -> FunctionModel:
    plan = yaml.safe_load((ROOT / "examples" / "plan_transfers.yaml").read_text())
    plan["objective"] = TRANSFERS_OBJECTIVE
    plan["acceptance_criteria"] = TRANSFERS_CRITERIA

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools = {t.name for t in info.function_tools}
        done = _returned(messages)
        prompt = _prompt(messages)
        finished = {"submit_plan", "complete_task", "submit_review"} & set(done)
        if finished:
            return ModelResponse([TextPart("Done.")])
        if "submit_plan" in tools:
            return _call("list_agents", {}) if not done else _call("submit_plan", plan)
        if "submit_review" in tools:
            return _call(
                "submit_review",
                {"verdict": "approve", "summary": "Every criterion is met and tested."},
            )
        steps: list[tuple[str, dict[str, Any]]]
        if "(test)" in prompt:
            steps = [
                ("write_file", {"path": "tests/test_transfers.py", "content": TESTS}),
                ("complete_task", {"summary": "Acceptance tests for AC-1..AC-4."}),
            ]
        else:
            steps = [
                ("write_file", {"path": "minibank/api/transfers.py", "content": TRANSFERS}),
                ("run_tests", {}),
                ("complete_task", {"summary": "POST /transfers with idempotency keys."}),
            ]
        return _call(*steps[min(len(done), len(steps) - 1)])

    return FunctionModel(respond, model_name="scripted-team")
