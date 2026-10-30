"""The PM's tools, calling the orchestrator in-process.

Spike stand-in for the MCP server: same orchestrator calls, same return shapes, but the
caller's identity comes from PMDeps instead of a run token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import ModelRetry, RunContext, Tool

from orch.core.orchestrator import OrchestratorError, TaskOrchestrator
from orch.domain.schemas import Plan, PlanError


@dataclass
class PMDeps:
    orch: TaskOrchestrator
    project_id: str
    task_id: str
    run_id: str


def _errors(errors: list[PlanError]) -> list[dict[str, Any]]:
    return [e.model_dump(exclude_defaults=True) for e in errors]


def list_agents(ctx: RunContext[PMDeps]) -> list[dict[str, Any]]:
    """List the team: each agent's name, description, capabilities and write_paths.

    Assign test, implement, review and security_review nodes only to agents with that
    capability. Gate and approval nodes have no assignee.
    """
    return ctx.deps.orch.list_agents(ctx.deps.project_id)


def validate_plan(ctx: RunContext[PMDeps], plan: Plan) -> dict[str, Any]:
    """Check a draft plan against the team and policy rules without submitting it.

    Returns valid=true, or errors that each name a rule code, the node and a hint.
    """
    try:
        errors = ctx.deps.orch.validate_plan(ctx.deps.task_id, plan)
    except OrchestratorError as e:
        raise ModelRetry(str(e)) from e
    return {"valid": not errors, "errors": _errors(errors)}


def submit_plan(ctx: RunContext[PMDeps], plan: Plan) -> dict[str, Any]:
    """Submit the whole plan. It is validated again; if accepted, the team starts work.

    Edges: `blocks` means the target waits for the source to finish; `verifies` points
    from a work node to a gate/review/security_review node that checks it. Test and
    implement nodes list `paths`: the files or globs they will write.
    """
    d = ctx.deps
    try:
        r = d.orch.submit_plan(d.task_id, plan, run_id=d.run_id)
    except OrchestratorError as e:
        raise ModelRetry(str(e)) from e
    if r.valid:
        return {
            "accepted": True,
            "version": r.version,
            "tasks": sorted(r.task_ids),
            "next": "Plan accepted. You are done: reply with one line and make no more calls.",
        }
    if r.escalated:
        return {
            "accepted": False,
            "escalated": True,
            "errors": _errors(r.errors),
            "next": "A human has been asked to decide. Do not resubmit: reply with one line.",
        }
    return {
        "accepted": False,
        "errors": _errors(r.errors),
        "attempts_left": r.attempts_left,
        "next": "Fix every error, then call submit_plan again with the whole plan.",
    }


def pm_tools(*, with_validate: bool = True) -> list[Tool[PMDeps]]:
    fns = [list_agents, validate_plan, submit_plan] if with_validate else [list_agents, submit_plan]
    return [Tool(f, takes_ctx=True) for f in fns]
