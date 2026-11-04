# minibank

A tiny bank service that FinOrch's agents work on. No web framework: a handler takes a
`Request` and returns a `Response` (`minibank/http.py`).

- `minibank/ledger.py`: accounts and balances, always `Decimal`
- `minibank/api/`: HTTP handlers (`accounts.py`)
- `minibank/payments/`: card payments. Protected: only `payments_developer` may change it,
  and every change needs a security review (FinOrch rule PAY-001)
- `tests/`: pytest

Run the tests with `python -m pytest`.
