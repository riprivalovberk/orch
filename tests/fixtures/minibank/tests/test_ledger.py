from decimal import Decimal

import pytest

from minibank.api.accounts import get_balance
from minibank.http import Request
from minibank.ledger import InsufficientFunds, Ledger, mask


@pytest.fixture
def ledger() -> Ledger:
    led = Ledger()
    led.open("GB0011112222", Decimal("100.00"))
    led.open("GB0033334444", Decimal("5.00"))
    return led


def test_move_is_all_or_nothing(ledger: Ledger):
    with pytest.raises(InsufficientFunds):
        ledger.move("GB0033334444", "GB0011112222", Decimal("10.00"))
    assert ledger.balance("GB0033334444") == Decimal("5.00")


def test_move(ledger: Ledger):
    ledger.move("GB0011112222", "GB0033334444", Decimal("0.10"))
    assert ledger.balance("GB0033334444") == Decimal("5.10")


def test_float_amounts_are_refused(ledger: Ledger):
    with pytest.raises(TypeError):
        ledger.move("GB0011112222", "GB0033334444", 0.1)  # type: ignore[arg-type]


def test_mask():
    assert mask("GB0011112222") == "********2222"


def test_get_balance(ledger: Ledger):
    r = get_balance(ledger, Request("GET", "/accounts/GB0011112222"))
    assert r.status == 200 and r.body["balance"] == "100.00"
    assert get_balance(ledger, Request("GET", "/accounts/nope")).status == 404
