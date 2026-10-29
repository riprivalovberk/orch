"""SQLAlchemy models. ``tasks`` holds current state; ``events`` is the append-only history.

Rule for this file: a column exists only if some code reads it. Anything that is just
"nice to know" goes in an event payload instead, which is free and already hash-chained.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    JSON,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from orch.db.types import PydanticJSON
from orch.domain.schemas import Finding, Plan, PlanError, ProjectLimits
from orch.ids import now_iso


class Base(DeclarativeBase):
    # Plain str/int/float/bool columns need no mapped_column(); JSON shapes are mapped here.
    type_annotation_map = {
        list[str]: JSON,
        dict[str, str]: JSON,
        dict[str, Any]: JSON,
        Any: JSON,
    }


def _created() -> Mapped[str]:
    return mapped_column(default=now_iso)


class Project(Base):
    """One repo + one team + one policy pack. Objectives live on root tasks, not here."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str]
    repo_path: Mapped[str | None]
    policy_pack: Mapped[str]
    pm_agent: Mapped[str | None]
    limits: Mapped[ProjectLimits] = mapped_column(PydanticJSON(ProjectLimits))
    spent_usd: Mapped[float] = mapped_column(default=0.0)
    spent_tokens: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[str] = _created()


class Agent(Base):
    """A row per agent in the team YAML. Configuration, not a running process."""

    __tablename__ = "agents"
    __table_args__ = (UniqueConstraint("project_id", "name"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str]
    description: Mapped[str]
    capabilities: Mapped[list[str]]
    tools: Mapped[list[str]]  # MCP allow-list, enforced server-side
    write_paths: Mapped[list[str]]  # globs; see core/scope.py
    model: Mapped[str]
    prompt: Mapped[str] = mapped_column(default="")
    created_at: Mapped[str] = _created()


class PlanRecord(Base):
    """Every plan the PM submits, valid or not. Rejections are part of the evidence."""

    __tablename__ = "plans"
    __table_args__ = (UniqueConstraint("root_task_id", "version"),)

    id: Mapped[str] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    root_task_id: Mapped[str] = mapped_column(index=True)  # the plan task this attempt answers
    version: Mapped[int]
    spec: Mapped[Plan] = mapped_column(PydanticJSON(Plan))
    valid: Mapped[bool]  # at most one valid plan per root task
    errors: Mapped[list[PlanError]] = mapped_column(PydanticJSON(list[PlanError]), default=list)
    created_at: Mapped[str] = _created()


class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = (Index("ix_tasks_project_status", "project_id", "status"),)

    # identity and place in the tree
    id: Mapped[str] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    # the plan task that created it
    parent_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"), index=True)
    root_id: Mapped[str] = mapped_column(index=True)
    # which node in the parent's plan this task came from, e.g. "write_tests"
    plan_node_id: Mapped[str | None]
    depth: Mapped[int] = mapped_column(default=0)

    # what the work is
    kind: Mapped[str]  # TaskKind
    title: Mapped[str]
    objective: Mapped[str] = mapped_column(default="")
    acceptance_criteria: Mapped[dict[str, str] | None]  # root task only
    gates: Mapped[list[str]] = mapped_column(default=list)  # gate tasks only: which checks to run
    assignee: Mapped[str | None]  # agent name; None for gate/approval tasks

    # state
    status: Mapped[str]  # TaskStatus
    outcome: Mapped[str | None]  # pass | fail | waived, set once DONE
    result_summary: Mapped[str | None]
    transcript: Mapped[Any | None]  # Pydantic AI message history; next run resumes from it

    # retry and revision bookkeeping
    attempt: Mapped[int] = mapped_column(default=0)  # claims so far (crash retries)
    revision_round: Mapped[int] = mapped_column(default=0)  # review rejections so far
    plan_attempts: Mapped[int] = mapped_column(default=0)  # rejected plans so far

    created_at: Mapped[str] = _created()


class TaskDep(Base):
    """``task_id`` depends on ``depends_on_id``. Plan edge ``a -> b`` is stored as row (b, a)."""

    __tablename__ = "task_deps"

    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), primary_key=True)
    depends_on_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), primary_key=True, index=True)
    kind: Mapped[str]  # blocks | verifies


class Run(Base):
    """One execution attempt: open while its task is RUNNING. Holds the MCP token and the cost."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    run_token_hash: Mapped[str | None] = mapped_column(unique=True)  # raw token is never stored
    started_at: Mapped[str] = _created()
    ended_at: Mapped[str | None]  # set when the task leaves RUNNING; token dies with it
    end_reason: Mapped[str | None]
    tokens_in: Mapped[int] = mapped_column(default=0)
    tokens_out: Mapped[int] = mapped_column(default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)


class ToolCall(Base):
    """Written before a tool acts and after it returns; a replay returns the stored result."""

    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    run_id: Mapped[str]
    idempotency_key: Mapped[str] = mapped_column(unique=True)  # task_id:model_call_id
    tool: Mapped[str]
    args: Mapped[Any]
    status: Mapped[str]
    result: Mapped[Any | None]
    created_at: Mapped[str] = _created()


class Artifact(Base):
    """Index of files agents produced (diffs, reports). The bytes live on disk by hash."""

    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"))
    run_id: Mapped[str | None]
    name: Mapped[str]
    kind: Mapped[str]
    sha256: Mapped[str]
    path: Mapped[str]
    created_at: Mapped[str] = _created()


class Review(Base):
    """A reviewer's verdict on one work task in one round."""

    __tablename__ = "reviews"

    id: Mapped[str] = mapped_column(primary_key=True)
    review_task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"))
    target_task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    reviewer: Mapped[str]
    round: Mapped[int]  # target's revision_round at review time
    verdict: Mapped[str]
    findings: Mapped[list[Finding]] = mapped_column(PydanticJSON(list[Finding]), default=list)
    summary: Mapped[str] = mapped_column(default="")


class GateResult(Base):
    """One deterministic check (pytest/ruff/policy) on one work task in one round."""

    __tablename__ = "gate_results"

    id: Mapped[str] = mapped_column(primary_key=True)
    gate_task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"))
    target_task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    round: Mapped[int]  # target's revision_round at check time
    gate: Mapped[str]
    passed: Mapped[bool]
    output: Mapped[str] = mapped_column(default="")
    duration_ms: Mapped[int] = mapped_column(default=0)


class HumanRequest(Base):
    """A question, approval or escalation waiting on a person."""

    __tablename__ = "human_requests"

    id: Mapped[str] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    kind: Mapped[str]
    prompt: Mapped[str]
    options: Mapped[list[str]] = mapped_column(default=list)  # [] = free-text answer
    status: Mapped[str] = mapped_column(default="open")
    response: Mapped[str | None]  # injected into the agent's next run
    created_at: Mapped[str] = _created()


class Event(Base):
    """Append-only, hash-chained per project: hash = sha256(prev_hash + canonical(fields))."""

    __tablename__ = "events"

    seq: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[str] = mapped_column(index=True)
    task_id: Mapped[str | None] = mapped_column(index=True)
    run_id: Mapped[str | None]
    actor: Mapped[str]  # agent name, "human" or "system"
    type: Mapped[str]
    payload: Mapped[dict[str, Any]]
    ts: Mapped[str]
    prev_hash: Mapped[str]
    hash: Mapped[str]

    def hashed_fields(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "actor": self.actor,
            "type": self.type,
            "payload": self.payload,
            "ts": self.ts,
        }
