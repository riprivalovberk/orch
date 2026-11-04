"""The runner: drives one objective from plan to approved merge.

Each tick: merge finished work into the objective branch -> claim READY tasks (up to
``parallel`` at once) -> run each as an agent over MCP, or run its gates directly.
When nothing can move, it asks the human (``ask``) or stops and says what it waits on.

Agents never touch the orchestrator directly: each gets a run token and reaches the agent
MCP server, which this process hosts on 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.mcp import CallToolFunc, MCPToolset
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits

from orch.core.orchestrator import Claim, TaskOrchestrator
from orch.core.team import TeamRegistry
from orch.domain.enums import (
    PASSING_OUTCOMES,
    REVIEW_KINDS,
    TERMINAL_STATUSES,
    GateName,
    TaskKind,
    TaskStatus,
)
from orch.domain.schemas import GateResultIn
from orch.mcp.server import CALL_ID_META, serve
from orch.policies import get_standards
from orch.policies.fintech_sdlc import PROTECTED_DIRS
from orch.runtime import gates
from orch.runtime.model import DEFAULT_MODEL, cost_usd
from orch.runtime.workspace import Workspace

log = logging.getLogger(__name__)
S = TaskStatus
WRITING = (TaskKind.TEST, TaskKind.IMPLEMENT)
# What a writing task's brief shows of the repo so it need not spend turns reading it.
MAX_BRIEF_FILES = 5
CSV_PREVIEW_LINES = 4
MAX_BRIEF_FILE_CHARS = 30_000

#: What each kind of task must call to finish; used to nudge an agent that stopped early.
FINISH = {
    TaskKind.PLAN: "submit_plan (or request_human_input if you must ask something)",
    TaskKind.TEST: "complete_task",
    TaskKind.IMPLEMENT: "complete_task",
    TaskKind.REVIEW: "submit_review",
    TaskKind.SECURITY_REVIEW: "submit_review",
}

Ask = Callable[[dict[str, Any]], str | None]
"""Put a human request to a person; return their answer, or None to stop and wait."""


@dataclass
class RunReport:
    root_id: str
    status: str
    waiting_on: list[dict[str, Any]] = field(default_factory=list)
    merged_into_main: str | None = None
    note: str = ""


@dataclass
class _Deps:
    run_token: str


async def _with_call_id(
    ctx: RunContext[_Deps], call: CallToolFunc, name: str, args: dict[str, Any]
) -> Any:
    """Send the model's tool-call id, so a retried request can't act twice."""
    return await call(name, args, metadata={CALL_ID_META: ctx.tool_call_id})


class Runner:
    def __init__(
        self,
        orch: TaskOrchestrator,
        team: TeamRegistry,
        ws: Workspace,
        *,
        model: Model,
        price_as: str = DEFAULT_MODEL,
        parallel: int = 3,
        settings: OpenAIChatModelSettings | None = None,
        agent_timeout_s: float = 1800,
        request_limit: int = 40,
    ) -> None:
        self.orch = orch
        self.team = team
        self.ws = ws
        self.model = model
        self.price_as = price_as
        self.parallel = max(parallel, 1)
        self.settings = settings or OpenAIChatModelSettings()
        self.agent_timeout_s = agent_timeout_s
        self.request_limit = request_limit
        self.url = ""

    # ------------------------------------------------------------------ main loop

    async def run(self, root_id: str, *, ask: Ask | None = None) -> RunReport:
        self.ws.start_objective(root_id)
        running: dict[str, asyncio.Task[None]] = {}
        async with serve(self.orch, self.ws) as self.url:
            while True:
                if note := self._merge_finished(root_id):
                    await self._drain(running)
                    return RunReport(root_id, self._status(root_id), note=note)
                root = self.orch.get_task(root_id)
                if root["status"] in TERMINAL_STATUSES:
                    await self._drain(running)
                    return self._finish(root_id, root["status"])
                if self._over_budget(root_id) and not running:
                    return RunReport(root_id, root["status"], note="project budget used up")
                for tid in self._ready(root_id):
                    if len(running) >= self.parallel or self._over_budget(root_id):
                        break
                    if tid in running:
                        continue
                    claim = self.orch.claim(tid)
                    if claim is not None:
                        running[tid] = asyncio.create_task(self._execute(claim))
                if running:
                    done, _ = await asyncio.wait(
                        running.values(), return_when=asyncio.FIRST_COMPLETED
                    )
                    for tid in [t for t, fut in running.items() if fut in done]:
                        running.pop(tid).result()  # re-raise a runner bug, not an agent error
                    continue
                waiting = self._open_requests(root_id)
                if not waiting:
                    return RunReport(root_id, root["status"], note="nothing left to run")
                for req in waiting:
                    answer = ask(req) if ask else None
                    if answer is None:
                        return RunReport(
                            root_id, root["status"], waiting_on=waiting, note="waiting on a human"
                        )
                    self.orch.resolve_human(req["id"], answer)

    async def _drain(self, running: dict[str, asyncio.Task[None]]) -> None:
        if running:
            await asyncio.gather(*running.values(), return_exceptions=True)

    def _status(self, root_id: str) -> str:
        return self.orch.get_task(root_id)["status"]

    def _tree(self, root_id: str) -> list[dict[str, Any]]:
        return self.orch.task_tree(root_id)

    def _ready(self, root_id: str) -> list[str]:
        return [
            t["id"]
            for t in self._tree(root_id)
            if t["status"] == S.READY and t["kind"] != TaskKind.APPROVAL
        ]

    def _open_requests(self, root_id: str) -> list[dict[str, Any]]:
        tree = self._tree(root_id)
        ids = {t["id"] for t in tree}
        reqs = self.orch.open_human_requests(tree[0]["project_id"])
        return [r for r in reqs if r["task_id"] in ids]

    def _over_budget(self, root_id: str) -> bool:
        pid = self.orch.get_task(root_id)["project_id"]
        return self.orch.remaining_budget(pid)["remaining_usd"] <= 0

    def _merge_finished(self, root_id: str) -> str:
        """Merge every passed work task not yet in the objective branch. A conflict stops the
        run (PAR-001 should make this rare): returns what went wrong, or ""."""
        for t in self._tree(root_id):
            if t["kind"] not in WRITING or t["status"] != S.DONE:
                continue
            if t["outcome"] not in PASSING_OUTCOMES or self.ws.is_merged(root_id, t["id"]):
                continue
            if not self.ws.task_dir(t["id"]).exists():
                continue
            m = self.ws.merge_task(root_id, t["id"])
            self.orch.log_event(
                t["id"],
                "workspace.merged" if m.ok else "workspace.merge_conflict",
                {"branch": self.ws.task_branch(t["id"]), "detail": m.detail[-2000:]},
            )
            if not m.ok:
                return (
                    f"merge conflict merging {t['node']} into {self.ws.objective_branch(root_id)}"
                    f"; resolve it in {self.ws.objective_dir(root_id)}:\n{m.detail}"
                )
        return ""

    def _finish(self, root_id: str, status: str) -> RunReport:
        if status != S.DONE:
            return RunReport(root_id, status, note="objective did not pass")
        kind, detail = merge_into_main(self.ws, root_id)
        self.orch.log_event(
            root_id,
            kind,
            {"branch": self.ws.objective_branch(root_id), "detail": detail[-2000:]},
            actor="system",
        )
        if kind != "workspace.merged_to_main":
            return RunReport(root_id, status, note=f"could not merge into main:\n{detail}")
        return RunReport(root_id, status, merged_into_main=detail)

    # ------------------------------------------------------------------ one task

    async def _execute(self, claim: Claim) -> None:
        task = self.orch.get_task(claim.task_id)
        try:
            if task["kind"] == TaskKind.GATE:
                await asyncio.to_thread(self._run_gates, task)
            else:
                if task["kind"] in WRITING:
                    await asyncio.to_thread(self.ws.open_task, task["root_id"], task["id"])
                await self._run_agent(claim, task)
        except Exception as e:  # an agent or gate crash: give the task back (or fail it)
            log.exception("task %s crashed", task["id"])
            if self.orch.get_task(task["id"])["status"] == S.RUNNING:
                self.orch.retry(task["id"], f"{type(e).__name__}: {e}"[:500])

    def _run_gates(self, gate: dict[str, Any]) -> None:
        results = []
        for target_id in self.orch.verified_targets(gate["id"]):
            target = self.orch.get_task(target_id)
            agent = self.team.get(target["assignee"]) if target["assignee"] else None
            results.append(
                gates.run_gates(
                    gate_names(gate),
                    self.ws.task_dir(target_id),
                    changed=self.ws.changed_files(target_id),
                    agent_paths=agent.write_paths if agent else [],
                    task_paths=target["paths"],
                    protected=PROTECTED_DIRS,
                    security_reviewed=TaskKind.SECURITY_REVIEW
                    in self.orch.verifier_kinds(target_id),
                )
            )
        merged = combine(results)
        self.orch.record_gate_results(gate["id"], merged)

    async def _run_agent(self, claim: Claim, task: dict[str, Any]) -> None:
        cfg = self.team.get(task["assignee"])
        toolset: MCPToolset[_Deps] = MCPToolset(
            self.url,
            auth=claim.run_token,
            process_tool_call=_with_call_id,
            include_return_schema=False,
        )
        agent: Agent[_Deps, str] = Agent(
            deps_type=_Deps,
            instructions=cfg.prompt,
            toolsets=[toolset],
            retries=3,
        )
        settings = OpenAIChatModelSettings(**self.settings)
        if cfg.effort:
            settings["openai_reasoning_effort"] = cfg.effort
        usage = RunUsage()
        limits = UsageLimits(request_limit=self.request_limit)
        deps = _Deps(claim.run_token)
        try:
            async with asyncio.timeout(self.agent_timeout_s):
                result = await agent.run(
                    self.brief(task),
                    deps=deps,
                    model=self.model,
                    model_settings=settings,
                    usage=usage,
                    usage_limits=limits,
                )
                if self._still_running(task["id"]):  # stopped to chat: nudge once
                    await agent.run(
                        f"You have not finished. Call {FINISH[TaskKind(task['kind'])]} now.",
                        deps=deps,
                        model=self.model,
                        model_settings=settings,
                        usage=usage,
                        usage_limits=limits,
                        message_history=result.all_messages(),
                    )
        finally:
            self.orch.record_usage(
                claim.run_id,
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                cost_usd=cost_usd(usage, self.price_as),
            )
        if self._still_running(task["id"]):
            self.orch.retry(
                task["id"], f"agent ended without calling {FINISH[TaskKind(task['kind'])]}"
            )

    def _still_running(self, task_id: str) -> bool:
        return self.orch.get_task(task_id)["status"] == S.RUNNING

    # ------------------------------------------------------------------ prompts

    def _verified_nodes(self, task: dict[str, Any]) -> list[str]:
        """Plan node ids of the work a review task checks (none for other tasks)."""
        if task["kind"] not in REVIEW_KINDS:
            return []
        return [self.orch.get_task(t)["node"] for t in self.orch.verified_targets(task["id"])]

    def _repo_context(self, task: dict[str, Any]) -> list[str]:
        """What a writing task would otherwise spend turns reading: the header and first rows
        of each CSV in the repo, and for an implement task the test files it must pass."""
        root = self.ws.task_dir(task["id"])
        if not root.is_dir():
            return []
        out = []
        csvs = sorted(p for p in root.rglob("*.csv") if ".git" not in p.parts)
        for p in csvs[:MAX_BRIEF_FILES]:
            head = p.read_text(errors="replace").splitlines()[:CSV_PREVIEW_LINES]
            out.append(f"{p.relative_to(root)} (first lines):\n" + "\n".join(head))
        if task["kind"] == TaskKind.IMPLEMENT:
            tests = sorted(p for p in (root / "tests").rglob("test_*.py"))[:MAX_BRIEF_FILES]
            for p in tests:
                text = p.read_text(errors="replace")[:MAX_BRIEF_FILE_CHARS]
                out.append(f"{p.relative_to(root)}:\n```python\n{text}\n```")
        return out

    def brief(self, task: dict[str, Any]) -> str:
        """The task as the agent reads it: objective, criteria, scope, and any feedback."""
        root = self.orch.evidence(task["root_id"])
        kind = TaskKind(task["kind"])
        criteria = "\n".join(f"- {k}: {v}" for k, v in root["criteria"].items())
        out = [f"Objective: {root['objective']}"]
        if kind == TaskKind.PLAN:
            if criteria:
                out.append(
                    f"Acceptance criteria (already agreed with the human; do not re-confirm):\n"
                    f"{criteria}"
                )
            answers = self.orch.human_answers(task["id"])
            for qa in answers:
                out.append(f"You asked: {qa['question']}\nThe human answered: {qa['answer']}")
            if not criteria and not answers:
                out.append(
                    "No acceptance criteria were given. Read the repo, write the criteria and "
                    "confirm them with the human before you plan (step 1 of your instructions)."
                )
            elif not criteria:
                out.append(
                    "If the human approved the criteria, or only dropped some, put them in the "
                    "plan's acceptance_criteria and plan. If they changed or added criteria, "
                    "send the revised list back with request_human_input once and plan only "
                    "after they approve it."
                )
            else:
                out.append("Plan the work for the team now.")
            return "\n\n".join(out)

        out.append(f"Acceptance criteria:\n{criteria}")
        accepted = [p["spec"] for p in root["plans"] if p["valid"]]
        if accepted and accepted[-1]["interface"]:
            out.append(
                "Interface (agreed in the plan; tests and code use exactly these names):\n"
                + accepted[-1]["interface"]
            )
        if kind == TaskKind.IMPLEMENT or kind in REVIEW_KINDS:
            pack = self.orch.get_project(root["project_id"])["policy_pack"]
            out.append(
                "Engineering standards (reviewers block on these):\n"
                + "\n".join(f"- {k}: {v}" for k, v in get_standards(pack).items())
            )
        out.append(f"Your task: {task['title']} ({kind}).")
        if accepted and (kind == TaskKind.IMPLEMENT or kind in REVIEW_KINDS):
            mine = {task["node"], *self._verified_nodes(task)}
            others = [
                f"- {n['id']}: {n.get('title') or n.get('objective') or ''} "
                f"(writes {', '.join(n.get('paths') or []) or 'unspecified files'})"
                for n in accepted[-1]["nodes"]
                if n["kind"] == TaskKind.IMPLEMENT and n["id"] not in mine
            ]
            if others:
                out.append(
                    "Other steps of the plan build the rest; their work is not part of this "
                    "task, so do not write it or ask for it here:\n" + "\n".join(others)
                )
        if kind in REVIEW_KINDS:
            paths = sorted(
                {
                    p
                    for t in self.orch.verified_targets(task["id"])
                    for p in self.orch.get_task(t)["paths"]
                }
            )
            if paths:
                out.append(
                    f"The work you review may write only {', '.join(paths)}. A change it needs "
                    "outside those files is a GAP finding (severity minor) to disclose, never a "
                    "blocker: the author cannot make it."
                )
        if accepted and kind == TaskKind.TEST:
            code = [
                f"- {n['id']}: writes {', '.join(n.get('paths') or []) or 'unspecified files'}"
                for n in accepted[-1]["nodes"]
                if n["kind"] == TaskKind.IMPLEMENT
            ]
            if code:
                out.append(
                    "The code these tests check is written by the steps below, only in their "
                    "files. Look for modules and files there; a test that expects a file "
                    "anywhere else can never pass:\n" + "\n".join(code)
                )
        if kind in WRITING:
            out.append(
                "Tests run in an environment with exactly the packages the project declares "
                "(requirements.txt, or dependencies in pyproject.toml) plus pytest, so import "
                "them normally: never stub or fake a package outside tests/. A package your "
                "code needs goes in requirements.txt."
            )
            out.append(
                "Files this task may write: "
                + ", ".join(task["paths"])
                + ". The repo's data files and (for an implement task) the tests are below, "
                "so you need not read them; read other files only if you need them."
            )
            out.extend(self._repo_context(task))
            if task["attempt"] > 1 and (done := self.ws.changed_files(task["id"])):
                out.append(
                    "Your previous run on this task stopped before it finished (time limit or "
                    "crash), but what it wrote is kept: " + ", ".join(done) + ". Read those "
                    "files and carry on from them; do not start over."
                )
            fb = self.orch.feedback(task["id"])
            if fb["reviews"] or fb["gates"]:
                out.append(
                    f"This is revision round {task['revision_round']}. Fix every point below "
                    "before calling complete_task."
                )
                for r in fb["reviews"]:
                    lines = [
                        f"- [{f['severity']}] {f['issue']}"
                        + (f" Assume: {f['assumption']}" if f.get("assumption") else "")
                        for f in r["findings"]
                    ]
                    out.append(
                        f"{r['reviewer']} ({r['verdict']}): {r['summary']}\n" + "\n".join(lines)
                    )
                for g in fb["gates"]:
                    out.append(f"Gate {g['gate']} failed:\n{g['output']}")
        elif kind in REVIEW_KINDS:
            for target_id in self.orch.verified_targets(task["id"]):
                target = self.orch.get_task(target_id)
                out.append(
                    f"Change to review: {target['title']} by {target['assignee']} "
                    f"(round {target['revision_round']}).\nTheir summary: {target['summary']}\n\n"
                    f"Diff:\n{self.ws.diff(target_id)}"
                )
            out.append(
                "Check the change against every criterion and every standard, then call "
                "submit_review."
            )
        return "\n\n".join(out)


def merge_into_main(
    ws: Workspace,
    root_id: str,
    tests: Callable[[Any], tuple[bool, str, int]] = gates.pytest,
) -> tuple[str, str]:
    """Merge an approved objective into main, as the event to log and its detail.

    If main moved since the objective was built, main is brought into it and its tests run
    again first: two changes that each passed can still fail together, and main must not
    break. A failure leaves both main and the objective's branch as they were.
    """
    caught = ws.catch_up(root_id)
    if caught is not None and not caught.ok:
        return "workspace.merge_conflict", caught.detail
    if caught is not None:
        try:
            ok, out, _ = tests(ws.objective_dir(root_id))
        except Exception as e:  # e.g. the package index is unreachable: don't merge blind
            ok, out = False, f"{type(e).__name__}: {e}"
        if not ok:
            ws.undo_catch_up(root_id)
            return "workspace.merge_blocked", out
    m = ws.finish_objective(root_id)
    return ("workspace.merged_to_main" if m.ok else "workspace.merge_conflict"), m.detail


def gate_names(gate: dict[str, Any]) -> list[str]:
    return list(gate["gates"]) or [g.value for g in GateName]


def combine(per_target: list[list[GateResultIn]]) -> list[GateResultIn]:
    """One result per gate across all targets: passes only if it passed for every one."""
    if len(per_target) == 1:
        return per_target[0]
    by_gate: dict[str, list[GateResultIn]] = {}
    for rs in per_target:
        for r in rs:
            by_gate.setdefault(r.gate, []).append(r)
    return [
        GateResultIn(
            gate=rs[0].gate,
            passed=all(r.passed for r in rs),
            output="\n---\n".join(r.output for r in rs),
            duration_ms=sum(r.duration_ms for r in rs),
        )
        for rs in by_gate.values()
    ]
