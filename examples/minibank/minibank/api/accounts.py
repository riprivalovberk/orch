"""GET /accounts/<id>: an account's balance."""

from __future__ import annotations

import logging

from minibank.http import Request, Response, error
from minibank.ledger import Ledger, UnknownAccount, mask

log = logging.getLogger(__name__)


def get_balance(ledger: Ledger, req: Request) -> Response:
    account = req.path.rsplit("/", 1)[-1]
    try:
        balance = ledger.balance(account)
    except UnknownAccount:
        log.info("balance lookup for unknown account %s", mask(account))
        return error(404, "unknown account")
    return Response(200, {"account": account, "balance": str(balance)})
