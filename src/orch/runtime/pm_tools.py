"""The PM's tools: one implementation, two ways in.

Each tool is a function of PMDeps (who is calling, on which task). The spike calls them
in-process through Pydantic AI; the agent MCP server (orch.mcp.server) calls the same
functions for a caller identified by its run token, and publishes the same names,
descriptions and JSON schemas. So the two can't drift.
"""

from __future__ import annotations

from collections.abc import Callable
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
    run_token: str = ""  # only used when the tools are reached over MCP


def _errors(errors: list[PlanError]) -> list[dict[str, Any]]:
    return [e.model_dump(exclude_defaults=True) for e in errors]


def _call[**P, T](fn: Callable[P, T], *args: P.args, **kw: P.kwargs) -> T:
    """Orchestrator errors go back to the model as a retry, with the message."""
    try:
        return fn(*args, **kw)
    except OrchestratorError as e:
        raise ModelRetry(str(e)) from e


# ------------------------------------------------------------------------- implementations


def do_list_agents(d: PMDeps) -> list[dict[str, Any]]:
    return d.orch.list_agents(d.project_id)


def do_validate_plan(d: PMDeps, plan: Plan) -> dict[str, Any]:
    errors = _call(d.orch.validate_plan, d.task_id, plan)
    return {"valid": not errors, "errors": _errors(errors)}


def do_submit_plan(d: PMDeps, plan: Plan) -> dict[str, Any]:
    r = _call(d.orch.submit_plan, d.task_id, plan, run_id=d.run_id)
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


def do_get_task_status(d: PMDeps, task_id: str) -> dict[str, Any]:
    for t in d.orch.task_tree(d.task_id):
        if t["id"] == task_id:
            return t
    raise ModelRetry(f"{task_id} is not a task in your objective; see get_task_tree")


def do_get_task_tree(d: PMDeps) -> list[dict[str, Any]]:
    return d.orch.task_tree(d.task_id)


def do_get_remaining_budget(d: PMDeps) -> dict[str, float]:
    return d.orch.remaining_budget(d.project_id)


def do_request_human_input(
    d: PMDeps, prompt: str, options: list[str] | None = None
) -> dict[str, Any]:
    rid = _call(d.orch.request_human, d.task_id, prompt, options)
    return {
        "request_id": rid,
        "next": "Your task is paused until a person answers. Reply with one line and stop.",
    }


# ------------------------------------------------------------------------- Pydantic AI tools
# The docstrings below are the tool descriptions the model sees, in-process and over MCP.


def list_agents(ctx: RunContext[PMDeps]) -> list[dict[str, Any]]:
    """List the team: each agent's name, description, capabilities and write_paths.

    Assign test, implement, review and security_review nodes only to agents with that
    capability. Gate and approval nodes have no assignee.
    """
    return do_list_agents(ctx.deps)


def validate_plan(ctx: RunContext[PMDeps], plan: Plan) -> dict[str, Any]:
    """Check a draft plan against the team and policy rules without submitting it.

    Returns valid=true, or errors that each name a rule code, the node and a hint.
    """
    return do_validate_plan(ctx.deps, plan)


def submit_plan(ctx: RunContext[PMDeps], plan: Plan) -> dict[str, Any]:
    """Submit the whole plan. It is validated again; if accepted, the team starts work.

    Edges: `blocks` means the target waits for the source to finish; `verifies` points
    from a work node to a gate/review/security_review node that checks it; a work node
    finishes only once its verifiers pass, so never have one verifier check both a node
    and work that node blocks (point it at the later work only). A pytest gate runs every
    test, so it may only run once all implement work is written: prefer one implement node,
    or one pytest gate over implement nodes that do not block each other. Test and
    implement nodes list `paths`: the files or globs they will write.
    """
    return do_submit_plan(ctx.deps, plan)


def get_task_status(ctx: RunContext[PMDeps], task_id: str) -> dict[str, Any]:
    """Status, outcome and summary of one task in your objective."""
    return do_get_task_status(ctx.deps, task_id)


def get_task_tree(ctx: RunContext[PMDeps]) -> list[dict[str, Any]]:
    """Every task in your objective with its status, assignee and dependencies."""
    return do_get_task_tree(ctx.deps)


def get_remaining_budget(ctx: RunContext[PMDeps]) -> dict[str, float]:
    """The project's budget in USD: total, spent, remaining and the estimate per task."""
    return do_get_remaining_budget(ctx.deps)


def request_human_input(
    ctx: RunContext[PMDeps], prompt: str, options: list[str] | None = None
) -> dict[str, Any]:
    """Ask a person a question. Give options when the answer is a choice.

    Your task pauses until they answer; the answer arrives in your next run.
    """
    return do_request_human_input(ctx.deps, prompt, options)


ALL_PM_TOOLS = [
    list_agents,
    validate_plan,
    submit_plan,
    get_task_status,
    get_task_tree,
    get_remaining_budget,
    request_human_input,
]
SPIKE_TOOLS = ["list_agents", "validate_plan", "submit_plan"]


def pm_tools(*, with_validate: bool = True) -> list[Tool[PMDeps]]:
    """The spike's tool set: list_agents, validate_plan (optional) and submit_plan."""
    names = SPIKE_TOOLS if with_validate else ["list_agents", "submit_plan"]
    return [Tool(f, takes_ctx=True) for f in ALL_PM_TOOLS if f.__name__ in names]
