"""The task state machine. Every status change goes through ``assert_transition``."""

from __future__ import annotations

from orch.domain.enums import TaskStatus as S

TRANSITIONS: dict[S, frozenset[S]] = {
    S.PENDING: frozenset({S.READY, S.CANCELED}),
    S.READY: frozenset({S.RUNNING, S.AWAITING_HUMAN, S.CANCELED}),
    S.RUNNING: frozenset(
        {S.READY, S.WAITING, S.AWAITING_HUMAN, S.VERIFYING, S.DONE, S.FAILED, S.CANCELED}
    ),
    S.WAITING: frozenset({S.DONE, S.FAILED, S.CANCELED}),  # plan task, closed by the orchestrator
    S.AWAITING_HUMAN: frozenset({S.READY, S.DONE, S.FAILED, S.CANCELED}),
    S.VERIFYING: frozenset({S.READY, S.DONE, S.AWAITING_HUMAN, S.FAILED, S.CANCELED}),
    # A verifier (gate/review) is reopened when the work it checked is revised.
    # The orchestrator refuses this edge for every other kind of task.
    S.DONE: frozenset({S.PENDING}),
    S.FAILED: frozenset(),
    S.CANCELED: frozenset(),
}


class IllegalTransition(ValueError):
    pass


def can_transition(src: S | str, dst: S | str) -> bool:
    return S(dst) in TRANSITIONS[S(src)]


def assert_transition(src: S | str, dst: S | str) -> None:
    if not can_transition(src, dst):
        raise IllegalTransition(f"illegal transition {src} -> {dst}")
