"""Policy packs: domain rules applied to every plan on top of the structural rules."""

from __future__ import annotations

from orch.core.plan import Rule
from orch.policies import fintech_sdlc

PACKS: dict[str, tuple[Rule, ...]] = {
    "fintech_sdlc": fintech_sdlc.RULES,
}
STANDARDS: dict[str, dict[str, str]] = {
    "fintech_sdlc": fintech_sdlc.STANDARDS,
}


def get_pack(name: str) -> tuple[Rule, ...]:
    try:
        return PACKS[name]
    except KeyError:
        raise ValueError(f"unknown policy pack '{name}'; available: {sorted(PACKS)}") from None


def get_standards(name: str) -> dict[str, str]:
    """Engineering standards reviewers may cite in a blocking finding (STD-001, ...)."""
    get_pack(name)
    return STANDARDS.get(name, {})
