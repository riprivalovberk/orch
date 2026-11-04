"""Card captures. Card numbers are never stored or logged in full."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal

log = logging.getLogger(__name__)


def mask_card(pan: str) -> str:
    return "**** " + pan[-4:]


@dataclass
class Capture:
    id: str
    card_last4: str
    amount: Decimal


@dataclass
class CardProcessor:
    captures: dict[str, Capture] = field(default_factory=dict)

    def capture(self, capture_id: str, pan: str, amount: Decimal) -> Capture:
        if not isinstance(amount, Decimal) or amount <= 0:
            raise ValueError("amount must be a positive Decimal")
        c = Capture(capture_id, pan[-4:], amount)
        self.captures[capture_id] = c
        log.info("captured %s on %s", amount, mask_card(pan))
        return c
