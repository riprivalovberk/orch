"""Accounts and balances. Money is always Decimal, never float."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


class LedgerError(Exception):
    pass


class InsufficientFunds(LedgerError):
    pass


class UnknownAccount(LedgerError):
    pass


def mask(account: str) -> str:
    """For logs: only the last four characters of an account number."""
    return "*" * max(len(account) - 4, 0) + account[-4:]


@dataclass
class Ledger:
    balances: dict[str, Decimal] = field(default_factory=dict)

    def open(self, account: str, balance: Decimal = Decimal("0")) -> None:
        self.balances[account] = Decimal(balance)

    def balance(self, account: str) -> Decimal:
        try:
            return self.balances[account]
        except KeyError:
            raise UnknownAccount(mask(account)) from None

    def move(self, source: str, target: str, amount: Decimal) -> None:
        """Debit source and credit target, or change nothing."""
        if not isinstance(amount, Decimal):
            raise TypeError("amount must be a Decimal")
        if amount <= 0:
            raise LedgerError("amount must be positive")
        if self.balance(source) < amount:
            raise InsufficientFunds(mask(source))
        self.balance(target)  # unknown target: fail before debiting
        self.balances[source] -= amount
        self.balances[target] += amount
