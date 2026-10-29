"""Closed vocabularies shared by the database, the validator and the MCP tools."""

from __future__ import annotations

from enum import StrEnum


class TaskStatus(StrEnum):
    PENDING = "PENDING"  # dependencies not met yet
    READY = "READY"  # can be claimed
    RUNNING = "RUNNING"  # claimed by the runner; has an open run
    WAITING = "WAITING"  # plan accepted; waiting for its planned tasks
    AWAITING_HUMAN = "AWAITING_HUMAN"
    VERIFYING = "VERIFYING"  # work submitted; verifier tasks are running
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


TERMINAL_STATUSES = frozenset({TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELED})


class TaskKind(StrEnum):
    PLAN = "plan"
    TEST = "test"
    IMPLEMENT = "implement"
    GATE = "gate"
    REVIEW = "review"
    SECURITY_REVIEW = "security_review"
    APPROVAL = "approval"


VERIFIER_KINDS = frozenset({TaskKind.GATE, TaskKind.REVIEW, TaskKind.SECURITY_REVIEW})
REVIEW_KINDS = frozenset({TaskKind.REVIEW, TaskKind.SECURITY_REVIEW})
#: Kinds run by the system (gate) or a human (approval), never by an agent.
UNASSIGNED_KINDS = frozenset({TaskKind.GATE, TaskKind.APPROVAL})
#: Capabilities an agent may declare in the team file.
AGENT_CAPABILITIES = frozenset(k.value for k in TaskKind if k not in UNASSIGNED_KINDS)


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WAIVED = "waived"  # a human overrode failed verification


PASSING_OUTCOMES = frozenset({Outcome.PASS, Outcome.WAIVED})


class EdgeKind(StrEnum):
    BLOCKS = "blocks"  # target waits until the source is DONE
    VERIFIES = "verifies"  # target starts when the source enters VERIFYING


class Verdict(StrEnum):
    APPROVE = "approve"
    REQUEST_CHANGES = "request_changes"


class GateName(StrEnum):
    PYTEST = "pytest"
    RUFF = "ruff"
    POLICY = "policy"


class HumanRequestKind(StrEnum):
    QUESTION = "question"
    APPROVAL = "approval"
    ESCALATION = "escalation"
