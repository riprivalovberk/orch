"""The PM in the chat: answers questions about the project, or starts an objective.

Separate from the planning PM (which runs as a task over MCP): this one has no task and
cannot change anything except by start_objective. It reads the project's git repo through
refs, so it sees main and every objective's branch, merged or not:

    where = "main"            what is merged
    where = "<objective id>"  that objective's branch (obj/<id>), work in progress included
"""

from __future__ import annotations

import fnmatch
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from orch.core.orchestrator import TaskOrchestrator
from orch.core.scope import is_safe_relative
from orch.paths import data_dir

MAX_FILES = 500
MAX_HITS = 100
MAX_CHARS = 40_000

INSTRUCTIONS = """\
You are the project manager of a software team at a regulated financial institution,
talking with the person who owns this project. Answer what they ask, briefly and
concretely, from the tools: project_status for objectives, tasks, branches, worktrees,
changed files and costs; list_files, read_file and search_code to look at code and data,
on main or on an objective's branch (its code may not be merged yet). Name files by path
and say which branch they are on and whether they are merged into main.

Only when the person asks for new work or a change to the code, call start_objective with
one clear objective in their words (name the files and data it concerns); the team then
drafts criteria with them and plans. Do not write acceptance criteria, plan, estimate or
design here (the team sends the criteria next), and do not start work for a question. If
the request is unclear, ask one short question instead. If the person asks to change work
that is still waiting for their approval, do not start new work: a new objective starts
from main, without that unapproved code. Tell them to reject it and then ask again for the
whole change with their tweak included, since a rejected change is not kept; suggest
that request in one sentence.

You cannot approve, reject or answer a decision the team is waiting on, so never say that
something was approved, rejected or merged. If the person wants to decide, tell them to
click Approve or Reject on the decision above, or to reply with just "approve" or "reject".
"""


@dataclass
class ChatDeps:
    orch: TaskOrchestrator
    project_id: str
    repo: Path
    start: Callable[[str], str]  # objective text -> root task id; raises ValueError if busy


def _git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=30)
    return p.stdout


def _ref(ctx: RunContext[ChatDeps], where: str) -> str:
    if where in ("", "main"):
        return "main"
    ids = {o["id"] for o in ctx.deps.orch.objectives(ctx.deps.project_id)}
    if where not in ids:
        raise ModelRetry(
            f"{where!r} is not an objective of this project; use 'main' or one of {sorted(ids)}"
        )
    return f"obj/{where}"


def _files(ctx: RunContext[ChatDeps], ref: str) -> list[str]:
    out = _git(ctx.deps.repo, "ls-tree", "-r", "-z", "--name-only", ref)
    return [f for f in out.split("\0") if f]


def project_status(ctx: RunContext[ChatDeps]) -> dict[str, Any]:
    """The project at a glance: its repo, budget, and every objective with its status,
    criteria, tasks, branch, worktree folder, files changed against main and whether it
    is merged."""
    d = ctx.deps
    p = d.orch.get_project(d.project_id)
    objectives = []
    for o in d.orch.objectives(d.project_id):
        branch = f"obj/{o['id']}"
        changed = _git(d.repo, "diff", "--name-only", "main", branch).split()
        merged = bool(_git(d.repo, "branch", "--merged", "main", "--list", branch).strip())
        objectives.append(
            {
                "id": o["id"],
                "objective": o["objective"],
                "status": o["status"],
                "criteria": o["criteria"],
                "branch": branch,
                "worktree": str(data_dir() / "worktrees" / o["id"]),
                "merged_into_main": merged,
                "files_changed_vs_main": changed,
                "tasks": [
                    {k: t[k] for k in ("node", "kind", "assignee", "status", "outcome", "paths")}
                    for t in d.orch.task_tree(o["id"])[1:]
                ],
                "open_requests": [
                    r["prompt"][:300]
                    for r in d.orch.open_human_requests(d.project_id)
                    if d.orch.get_task(r["task_id"])["root_id"] == o["id"]
                ],
            }
        )
    return {
        "repo": p["repo_path"],
        "budget_usd": p["budget_usd"],
        "spent_usd": p["spent_usd"],
        "objectives": objectives,
    }


def list_files(ctx: RunContext[ChatDeps], where: str = "main", glob: str = "**/*") -> list[str]:
    """Files in the repo matching a glob.

    Args:
        where: "main", or an objective id for that objective's branch
        glob: e.g. "**/*.py" or "data/*"
    """
    top = glob.removeprefix("**/")  # "**/*.py" also matches top-level files
    files = [
        f
        for f in _files(ctx, _ref(ctx, where))
        if fnmatch.fnmatch(f, glob) or fnmatch.fnmatch(f, top)
    ]
    return files[:MAX_FILES]


def read_file(ctx: RunContext[ChatDeps], path: str, where: str = "main") -> str:
    """One file's text (the first 40,000 characters).

    Args:
        path: repo-relative path
        where: "main", or an objective id for that objective's branch
    """
    if not is_safe_relative(path):
        raise ModelRetry(f"{path!r}: use a repo-relative path without '..'")
    ref = _ref(ctx, where)
    if path not in _files(ctx, ref):
        raise ModelRetry(f"{path} is not on {ref}; call list_files")
    text = _git(ctx.deps.repo, "show", f"{ref}:{path}")
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n... cut"


def search_code(
    ctx: RunContext[ChatDeps], pattern: str, where: str = "main", glob: str = "*"
) -> list[str]:
    """Lines matching a regular expression, as path:line:text.

    Args:
        pattern: a regular expression
        where: "main", or an objective id for that objective's branch
        glob: limit to these files, e.g. "*.py"
    """
    ref = _ref(ctx, where)
    out = _git(ctx.deps.repo, "grep", "-n", "-E", "-e", pattern, ref, "--", glob)
    hits = [line.removeprefix(f"{ref}:") for line in out.splitlines()]
    return hits[:MAX_HITS]


def start_objective(ctx: RunContext[ChatDeps], objective: str) -> dict[str, str]:
    """Start new work: the team drafts acceptance criteria with the person, plans and builds.
    Only call this when the person asked for work to be done.

    Args:
        objective: what to build or change, in the person's words, naming files and data
    """
    try:
        return {"objective_id": ctx.deps.start(objective)}
    except ValueError as e:
        raise ModelRetry(f"could not start it: {e}. Tell the person.") from e


TOOLS = [project_status, list_files, read_file, search_code, start_objective]


@dataclass
class Reply:
    text: str
    started: str | None
    tokens: int
    usage: RunUsage


async def answer(
    deps: ChatDeps, history: list[tuple[str, str]], message: str, *, model: Model
) -> Reply:
    """One turn: the PM reads what it needs and replies. history is (role, text), oldest first."""
    agent: Agent[ChatDeps, str] = Agent(deps_type=ChatDeps, instructions=INSTRUCTIONS, tools=TOOLS)
    started: list[str] = []
    real_start = deps.start

    def start(text: str) -> str:
        root = real_start(text)
        started.append(root)
        return root

    deps = ChatDeps(deps.orch, deps.project_id, deps.repo, start)
    convo = "\n".join(f"{'Person' if r == 'user' else 'You'}: {t}" for r, t in history[-20:])
    prompt = (f"Conversation so far:\n{convo}\n\n" if convo else "") + f"Person: {message}"
    usage = RunUsage()
    result = await agent.run(
        prompt, deps=deps, model=model, usage=usage, usage_limits=UsageLimits(request_limit=15)
    )
    return Reply(result.output, started[0] if started else None, usage.total_tokens, usage)
